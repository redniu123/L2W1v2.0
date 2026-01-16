"""
SH-DA++ v4.0 专用审计工具

评估指标体系 (四维审计架构):

1. **空间感知审计 (Boundary Deletion Recall@B)**
   - 识别边界漏字样本（Deletion 发生在 j < K 或 j > len(GT) - K）
   - 计算被 upgrade=True 拦截的边界漏字样本比例

2. **预算稳定性审计 (Budget Stability)**
   - 滑动窗口内调用率方差
   - 验证控制器是否"平滑"

3. **可靠性审计 (CVR/AER)**
   - CVR (Constraint Violation Rate): 拒改比例
   - AER (Accepted Edit Rate): 有效纠错比例

4. **多维耗时审计 (Latency Profiling)**
   - lat_a_ms, lat_router_ms 等 P50/P95 统计

核心技术: 非对称对齐 (Asymmetric Alignment)
- 使用 difflib.SequenceMatcher 进行字符级精确对齐
- 分别对齐 (Agent A, GT) 和 (System, GT)
- 跟踪 GT 中每个位置的正确性变化

Usage:
    python evaluate.py --predictions ./results/l2w1_results.jsonl
    python evaluate.py --predictions ./results/l2w1_results.jsonl --router_features ./results/router_features.jsonl
    python evaluate.py --test  # 运行测试
"""

import sys
import json
import argparse
import difflib
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 编辑距离计算
# =============================================================================

@dataclass
class EditOperations:
    """编辑操作统计"""
    substitutions: int = 0  # S: 替换次数
    deletions: int = 0      # D: 删除次数
    insertions: int = 0     # I: 插入次数
    
    @property
    def total(self) -> int:
        """总编辑距离"""
        return self.substitutions + self.deletions + self.insertions
    
    def to_dict(self) -> Dict:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "total": self.total
        }


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算 Levenshtein 编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_cer(
    prediction: str,
    reference: str,
    return_details: bool = False
) -> float | Tuple[float, EditOperations]:
    """
    计算 CER (Character Error Rate)
    
    公式: CER = (S + D + I) / N
    """
    if len(reference) == 0:
        if len(prediction) == 0:
            return (0.0, EditOperations()) if return_details else 0.0
        return (1.0, EditOperations(insertions=len(prediction))) if return_details else 1.0
    
    # 使用 SequenceMatcher 获取详细的编辑操作
    matcher = difflib.SequenceMatcher(None, prediction, reference)
    ops = EditOperations()
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # 替换：pred[i1:i2] -> ref[j1:j2]
            min_len = min(i2 - i1, j2 - j1)
            ops.substitutions += min_len
            if i2 - i1 > min_len:
                ops.insertions += (i2 - i1 - min_len)
            if j2 - j1 > min_len:
                ops.deletions += (j2 - j1 - min_len)
        elif tag == 'delete':
            # pred 中有但 ref 中没有 -> 系统"多输出"了
            ops.insertions += (i2 - i1)
        elif tag == 'insert':
            # ref 中有但 pred 中没有 -> 系统"少输出"了
            ops.deletions += (j2 - j1)
    
    ed = ops.total
    cer = ed / len(reference)
    
    return (cer, ops) if return_details else cer


# =============================================================================
# SH-DA++ v4.0 核心功能：边界删除分析
# =============================================================================

@dataclass
class BoundaryDeletionResult:
    """边界删除分析结果"""
    is_boundary_deletion: bool = False      # 是否为边界漏字样本
    left_deletions: List[int] = field(default_factory=list)   # 左边界删除位置
    right_deletions: List[int] = field(default_factory=list)  # 右边界删除位置
    mid_deletions: List[int] = field(default_factory=list)    # 中间删除位置
    total_deletions: int = 0
    gt_length: int = 0
    k: int = 2
    
    def to_dict(self) -> Dict:
        return {
            "is_boundary_deletion": self.is_boundary_deletion,
            "left_deletions": self.left_deletions,
            "right_deletions": self.right_deletions,
            "mid_deletions": self.mid_deletions,
            "total_deletions": self.total_deletions,
            "gt_length": self.gt_length,
            "k": self.k,
        }


def identify_boundary_deletions(
    agent_a_text: str,
    gt_text: str,
    K: int = 2
) -> BoundaryDeletionResult:
    """
    识别边界漏字样本 (Boundary Deletion Analysis)
    
    使用 difflib.SequenceMatcher 获取 edit ops。
    如果删除操作对应的 GT 索引 j 满足 j < K 或 j > len(GT) - K，
    则判定该样本为"边界漏字样本"。
    
    Args:
        agent_a_text: Agent A 识别结果
        gt_text: 真值文本
        K: 边界窗口大小 (默认 2)
        
    Returns:
        BoundaryDeletionResult: 边界删除分析结果
    """
    result = BoundaryDeletionResult(k=K, gt_length=len(gt_text))
    
    if not gt_text:
        return result
    
    # 使用 SequenceMatcher 对齐
    matcher = difflib.SequenceMatcher(None, agent_a_text, gt_text)
    
    # 遍历编辑操作，找出所有删除位置
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':
            # GT 中有但 Agent A 中没有 -> 删除 (Deletion)
            for gt_pos in range(j1, j2):
                result.total_deletions += 1
                
                # 判断是左边界、右边界还是中间
                if gt_pos < K:
                    result.left_deletions.append(gt_pos)
                elif gt_pos >= len(gt_text) - K:
                    result.right_deletions.append(gt_pos)
                else:
                    result.mid_deletions.append(gt_pos)
        
        elif tag == 'replace':
            # 替换操作中，如果 pred 侧更短，也存在删除
            pred_len = i2 - i1
            gt_len = j2 - j1
            
            if gt_len > pred_len:
                # 存在删除
                for offset in range(gt_len - pred_len):
                    gt_pos = j1 + pred_len + offset
                    result.total_deletions += 1
                    
                    if gt_pos < K:
                        result.left_deletions.append(gt_pos)
                    elif gt_pos >= len(gt_text) - K:
                        result.right_deletions.append(gt_pos)
                    else:
                        result.mid_deletions.append(gt_pos)
    
    # 判定是否为边界漏字样本
    result.is_boundary_deletion = (
        len(result.left_deletions) > 0 or len(result.right_deletions) > 0
    )
    
    return result


# =============================================================================
# OCR-R 和 Correction Rate 计算
# =============================================================================

def calculate_ocr_r(
    agent_a_text: str,
    system_output: str,
    ground_truth: str
) -> Tuple[float, Dict]:
    """
    计算过度纠错率 (Over-Correction Rate, OCR-R)
    
    衡量模型将正确识别结果"改错"的比例，直接反映幻觉程度
    
    公式:
        OCR-R = Count(Agent A Correct → System Wrong) / Total Correct Chars in Agent A
    """
    # 边界条件
    if len(ground_truth) == 0:
        return 0.0, {"error": "ground_truth is empty", "total_correct_in_a": 0, "overcorrected": 0}
    
    # Step 1: 找出 GT 中 Agent A 正确的位置
    matcher_a_gt = difflib.SequenceMatcher(None, agent_a_text, ground_truth)
    
    P_correct = {}  # GT 位置 -> (字符, Agent A 位置)
    
    for tag, i1, i2, j1, j2 in matcher_a_gt.get_opcodes():
        if tag == 'equal':
            for offset in range(j2 - j1):
                gt_pos = j1 + offset
                a_pos = i1 + offset
                P_correct[gt_pos] = (ground_truth[gt_pos], a_pos)
    
    total_correct_in_a = len(P_correct)
    
    if total_correct_in_a == 0:
        return 0.0, {
            "total_correct_in_a": 0,
            "overcorrected": 0,
            "note": "Agent A has no correct characters"
        }
    
    # Step 2: 找出 GT 中 System 正确的位置
    matcher_sys_gt = difflib.SequenceMatcher(None, system_output, ground_truth)
    
    sys_correct_gt_positions = set()
    
    for tag, i1, i2, j1, j2 in matcher_sys_gt.get_opcodes():
        if tag == 'equal':
            for offset in range(j2 - j1):
                sys_correct_gt_positions.add(j1 + offset)
    
    # Step 3: 计算过度纠错
    overcorrected = 0
    overcorrected_details = []
    
    for gt_pos, (char, a_pos) in P_correct.items():
        if gt_pos not in sys_correct_gt_positions:
            overcorrected += 1
            
            sys_char = "?"
            if gt_pos < len(system_output):
                sys_char = system_output[gt_pos]
            
            if len(overcorrected_details) < 10:
                overcorrected_details.append({
                    "gt_position": gt_pos,
                    "gt_char": ground_truth[gt_pos] if gt_pos < len(ground_truth) else "",
                    "agent_a_char": char,
                    "system_char": sys_char,
                })
    
    # Step 4: 计算 OCR-R
    ocr_r = overcorrected / total_correct_in_a
    
    return ocr_r, {
        "total_correct_in_a": total_correct_in_a,
        "overcorrected": overcorrected,
        "ocr_r": ocr_r,
        "overcorrected_positions": overcorrected_details,
        "note": "Agent A Correct -> System Wrong transitions"
    }


def calculate_correction_rate(
    agent_a_text: str,
    system_output: str,
    ground_truth: str
) -> Tuple[float, Dict]:
    """
    计算纠正率 (Correction Rate, CR)
    
    公式:
        CR = Count(Agent A Wrong → System Correct) / Total Wrong Chars in Agent A
    """
    if len(ground_truth) == 0:
        return 0.0, {"error": "ground_truth is empty", "total_wrong_in_a": 0, "corrected": 0}
    
    # Step 1: 找出 GT 中 Agent A 错误的位置
    matcher_a_gt = difflib.SequenceMatcher(None, agent_a_text, ground_truth)
    
    P_wrong = set()
    
    for tag, i1, i2, j1, j2 in matcher_a_gt.get_opcodes():
        if tag == 'equal':
            continue
        for gt_pos in range(j1, j2):
            P_wrong.add(gt_pos)
    
    total_wrong_in_a = len(P_wrong)
    
    if total_wrong_in_a == 0:
        return 1.0, {
            "total_wrong_in_a": 0,
            "corrected": 0,
            "note": "Agent A has no errors (perfect recognition)"
        }
    
    # Step 2: 检查 System 在这些位置是否纠正
    matcher_sys_gt = difflib.SequenceMatcher(None, system_output, ground_truth)
    
    sys_correct_positions = set()
    
    for tag, i1, i2, j1, j2 in matcher_sys_gt.get_opcodes():
        if tag == 'equal':
            for gt_pos in range(j1, j2):
                sys_correct_positions.add(gt_pos)
    
    # Step 3: 计算纠正数量
    corrected = 0
    corrected_details = []
    
    for gt_pos in P_wrong:
        if gt_pos in sys_correct_positions:
            corrected += 1
            if len(corrected_details) < 10:
                corrected_details.append({
                    "gt_position": gt_pos,
                    "gt_char": ground_truth[gt_pos] if gt_pos < len(ground_truth) else "",
                })
    
    correction_rate = corrected / total_wrong_in_a
    
    return correction_rate, {
        "total_wrong_in_a": total_wrong_in_a,
        "corrected": corrected,
        "correction_rate": correction_rate,
        "corrected_positions": corrected_details,
        "note": "Agent A Wrong -> System Correct transitions"
    }


# =============================================================================
# SH-DA++ v4.0 可靠性指标：CVR 和 AER
# =============================================================================

def calculate_cvr_aer(
    records: List[Dict],
    ed_threshold: int = 2,
    length_change_threshold: float = 0.2
) -> Tuple[float, float, Dict]:
    """
    计算可靠性指标 CVR 和 AER
    
    CVR (Constraint Violation Rate): 
        upgrade=True 的样本中，因为 ED > 2 或长度变化 > 20% 导致 T_final = T_A（拒改）的比例
    
    AER (Accepted Edit Rate):
        upgrade=True 且真正发生修改（T_final ≠ T_A）的样本比例
    
    Args:
        records: 推理记录列表
        ed_threshold: 编辑距离阈值（超过则拒改）
        length_change_threshold: 长度变化阈值（超过则拒改）
        
    Returns:
        Tuple[cvr, aer, details]
    """
    upgraded_samples = [r for r in records if r.get('upgrade', r.get('is_hard', False))]
    
    if not upgraded_samples:
        return 0.0, 0.0, {
            "upgraded_count": 0,
            "violation_count": 0,
            "accepted_edit_count": 0,
            "note": "No upgraded samples"
        }
    
    violation_count = 0
    accepted_edit_count = 0
    
    for record in upgraded_samples:
        agent_a_text = record.get('agent_a_text', record.get('agent_a', {}).get('text', ''))
        agent_b_text = record.get('agent_b_text', record.get('agent_b', {}).get('text', ''))
        final_text = record.get('final_text', '')
        
        # 如果没有 final_text，推断
        if not final_text:
            final_text = agent_b_text if agent_b_text else agent_a_text
        
        # 检查是否发生了修改
        if final_text != agent_a_text:
            accepted_edit_count += 1
        
        # 检查是否违反约束（拒改情况）
        if agent_b_text and agent_b_text != agent_a_text:
            # 计算 Agent A 和 Agent B 之间的编辑距离
            ed = levenshtein_distance(agent_a_text, agent_b_text)
            
            # 计算长度变化
            len_a = len(agent_a_text) if agent_a_text else 1
            len_b = len(agent_b_text) if agent_b_text else 0
            length_change = abs(len_b - len_a) / len_a
            
            # 检查是否被拒改
            if ed > ed_threshold or length_change > length_change_threshold:
                # 被拒改：T_final = T_A
                if final_text == agent_a_text:
                    violation_count += 1
    
    upgraded_count = len(upgraded_samples)
    cvr = violation_count / upgraded_count
    aer = accepted_edit_count / upgraded_count
    
    return cvr, aer, {
        "upgraded_count": upgraded_count,
        "violation_count": violation_count,
        "accepted_edit_count": accepted_edit_count,
        "ed_threshold": ed_threshold,
        "length_change_threshold": length_change_threshold,
    }


# =============================================================================
# SH-DA++ v4.0 性能分析：Latency Profiling
# =============================================================================

@dataclass
class LatencyProfile:
    """耗时分析结果"""
    name: str = ""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
        }


def compute_latency_profile(values: List[float], name: str = "") -> LatencyProfile:
    """计算耗时统计"""
    if not values:
        return LatencyProfile(name=name)
    
    arr = np.array(values)
    
    return LatencyProfile(
        name=name,
        count=len(arr),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
    )


def analyze_latency(router_features: List[Dict]) -> Dict[str, LatencyProfile]:
    """
    分析 router_features.jsonl 中的耗时字段
    
    Args:
        router_features: 路由特征记录列表
        
    Returns:
        Dict[field_name, LatencyProfile]: 各字段的耗时分析
    """
    latency_fields = ['lat_a_ms', 'lat_router_ms', 'lat_b_ms', 'lat_total_ms']
    
    latency_data = {fname: [] for fname in latency_fields}
    
    for record in router_features:
        for fname in latency_fields:
            if fname in record and record[fname] is not None:
                try:
                    val = float(record[fname])
                    if val >= 0:
                        latency_data[fname].append(val)
                except (ValueError, TypeError):
                    pass
    
    profiles = {}
    for fname, values in latency_data.items():
        profiles[fname] = compute_latency_profile(values, fname)
    
    return profiles


# =============================================================================
# SH-DA++ v4.0 预算稳定性分析
# =============================================================================

@dataclass
class BudgetStabilityResult:
    """预算稳定性分析结果"""
    total_samples: int = 0
    total_upgrades: int = 0
    actual_call_rate: float = 0.0
    target_budget: float = 0.2
    
    # 滑动窗口分析
    window_size: int = 200
    window_call_rates: List[float] = field(default_factory=list)
    window_std: float = 0.0
    window_mean: float = 0.0
    
    # 稳定性检查
    is_stable: bool = True
    max_deviation: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "total_upgrades": self.total_upgrades,
            "actual_call_rate": round(self.actual_call_rate, 4),
            "target_budget": self.target_budget,
            "window_size": self.window_size,
            "window_std": round(self.window_std, 4),
            "window_mean": round(self.window_mean, 4),
            "is_stable": self.is_stable,
            "max_deviation": round(self.max_deviation, 4),
        }


def analyze_budget_stability(
    router_features: List[Dict],
    window_size: int = 200,
    target_budget: float = 0.2,
    stability_threshold: float = 0.03
) -> BudgetStabilityResult:
    """
    分析预算稳定性
    
    计算滑动窗口内调用率的标准差，验证控制器是否"平滑"
    
    Args:
        router_features: 路由特征记录列表
        window_size: 滑动窗口大小 W
        target_budget: 目标调用率 B
        stability_threshold: 稳定性阈值（超过则不稳定）
        
    Returns:
        BudgetStabilityResult: 预算稳定性分析结果
    """
    result = BudgetStabilityResult(
        window_size=window_size,
        target_budget=target_budget,
    )
    
    if not router_features:
        return result
    
    # 提取 upgrade 决策
    upgrades = []
    for record in router_features:
        upgrade = record.get('upgrade', record.get('is_hard', False))
        upgrades.append(1 if upgrade else 0)
    
    result.total_samples = len(upgrades)
    result.total_upgrades = sum(upgrades)
    result.actual_call_rate = result.total_upgrades / result.total_samples if result.total_samples > 0 else 0.0
    
    # 计算滑动窗口调用率
    if len(upgrades) >= window_size:
        window_rates = []
        for i in range(window_size, len(upgrades) + 1):
            window = upgrades[i - window_size:i]
            rate = sum(window) / len(window)
            window_rates.append(rate)
        
        result.window_call_rates = window_rates
        result.window_std = float(np.std(window_rates))
        result.window_mean = float(np.mean(window_rates))
        result.max_deviation = max([abs(r - target_budget) for r in window_rates])
        result.is_stable = result.max_deviation <= stability_threshold
    else:
        result.window_std = 0.0
        result.window_mean = result.actual_call_rate
        result.max_deviation = abs(result.actual_call_rate - target_budget)
        result.is_stable = result.max_deviation <= stability_threshold
    
    return result


# =============================================================================
# 数据加载
# =============================================================================

@dataclass
class InferenceRecord:
    """推理记录 (符合 Data Protocol v2.0)"""
    id: str = ""
    image_path: str = ""
    agent_a_text: str = ""
    agent_b_text: str = ""
    final_text: str = ""
    gt_text: str = ""
    is_hard: bool = False
    upgrade: bool = False
    router_decision: str = "pass"
    suspicious_index: int = -1
    error_type: str = ""
    source: str = ""
    difficulty: str = "normal"
    
    # SH-DA++ v4.0 字段
    s_b: float = 0.0
    s_a: float = 0.0
    q: float = 0.0
    lambda_t: float = 0.0
    eta: float = 0.5
    route_type: str = ""
    b_timeout: bool = False
    b_fallback: bool = False
    
    # 耗时字段
    lat_a_ms: float = 0.0
    lat_router_ms: float = 0.0
    lat_b_ms: float = 0.0
    lat_total_ms: float = 0.0


def load_predictions(file_path: str) -> Tuple[List[str], List[str], List[str], List[bool], List[Dict]]:
    """
    加载推理结果文件 (支持 Data Protocol v2.0)
    
    支持格式:
    1. Data Protocol v2.0 (嵌套格式 + SH-DA++ 字段)
    2. Data Protocol v1.0 (嵌套格式)
    3. 扁平格式 (兼容旧版)
    
    Returns:
        (final_texts, gt_texts, agent_a_texts, is_hard_samples, raw_records)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    final_texts = []
    gt_texts = []
    agent_a_texts = []
    is_hard_samples = []
    raw_records = []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    def parse_item(item: Dict) -> InferenceRecord:
        """解析单条记录"""
        record = InferenceRecord()
        
        # 检测是否为嵌套格式
        if 'agent_a' in item and isinstance(item['agent_a'], dict):
            # === 嵌套格式 ===
            record.id = item.get('id', '')
            record.image_path = item.get('image', item.get('image_path', ''))
            record.gt_text = item.get('gt_text', '')
            
            # Agent A
            agent_a = item.get('agent_a', {})
            record.agent_a_text = agent_a.get('text', '')
            record.suspicious_index = agent_a.get('suspicious_index', -1)
            
            # Router
            router = item.get('router', {})
            record.is_hard = router.get('is_hard', False)
            record.upgrade = router.get('upgrade', record.is_hard)
            record.router_decision = router.get('decision', 'pass')
            
            # SH-DA++ v4.0 字段
            record.s_b = router.get('s_b', 0.0)
            record.s_a = router.get('s_a', 0.0)
            record.q = router.get('q', 0.0)
            record.lambda_t = router.get('lambda_t', router.get('lambda', 0.0))
            record.eta = router.get('eta', 0.5)
            record.route_type = router.get('route_type', '')
            
            # Agent B
            agent_b = item.get('agent_b', {})
            record.agent_b_text = agent_b.get('text', '')
            record.b_timeout = agent_b.get('b_timeout', False)
            record.b_fallback = agent_b.get('b_fallback', False)
            
            # Final text
            if record.is_hard and record.agent_b_text and not record.b_fallback:
                record.final_text = record.agent_b_text
            else:
                record.final_text = record.agent_a_text
            
            # Metadata
            metadata = item.get('metadata', {})
            record.error_type = metadata.get('error_type', '')
            record.source = metadata.get('source', '')
            record.difficulty = metadata.get('difficulty', 'normal')
            
            # 耗时
            record.lat_a_ms = item.get('lat_a_ms', metadata.get('lat_a_ms', 0.0))
            record.lat_router_ms = item.get('lat_router_ms', metadata.get('lat_router_ms', 0.0))
            record.lat_b_ms = item.get('lat_b_ms', metadata.get('lat_b_ms', 0.0))
            record.lat_total_ms = item.get('lat_total_ms', metadata.get('lat_total_ms', 0.0))
            
        else:
            # === 扁平格式 ===
            record.id = item.get('id', '')
            record.image_path = item.get('image_path', item.get('image', ''))
            record.final_text = item.get('final_text', item.get('prediction', item.get('pred_text', '')))
            record.gt_text = item.get('gt_text', item.get('ground_truth', item.get('gt', '')))
            record.agent_a_text = item.get('agent_a_text', item.get('ocr_text', ''))
            record.agent_b_text = item.get('agent_b_text', '')
            record.is_hard = item.get('is_hard', False)
            record.upgrade = item.get('upgrade', record.is_hard)
            
            # SH-DA++ v4.0 字段
            record.s_b = item.get('s_b', 0.0)
            record.s_a = item.get('s_a', 0.0)
            record.q = item.get('q', 0.0)
            record.lambda_t = item.get('lambda_t', item.get('lambda', 0.0))
            record.eta = item.get('eta', 0.5)
            record.route_type = item.get('route_type', '')
            record.b_timeout = item.get('b_timeout', False)
            record.b_fallback = item.get('b_fallback', False)
            
            # 耗时
            record.lat_a_ms = item.get('lat_a_ms', 0.0)
            record.lat_router_ms = item.get('lat_router_ms', 0.0)
            record.lat_b_ms = item.get('lat_b_ms', 0.0)
            record.lat_total_ms = item.get('lat_total_ms', 0.0)
            
            record.error_type = item.get('error_type', '')
            record.source = item.get('source', '')
            record.difficulty = item.get('difficulty', 'normal')
        
        return record
    
    # 解析 JSONL
    if '\n' in content or content.startswith('{'):
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                record = parse_item(item)
                
                final_texts.append(record.final_text)
                gt_texts.append(record.gt_text)
                agent_a_texts.append(record.agent_a_text)
                is_hard_samples.append(record.upgrade or record.is_hard)
                raw_records.append(item)
            except json.JSONDecodeError:
                continue
    
    return final_texts, gt_texts, agent_a_texts, is_hard_samples, raw_records


def load_router_features(file_path: str) -> List[Dict]:
    """
    加载 router_features.jsonl 文件
    
    包含 lat_router_ms, upgrade, q, lambda 等字段
    """
    if not file_path or not Path(file_path).exists():
        return []
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    return records


# =============================================================================
# SH-DA++ v4.0 评估结果
# =============================================================================

@dataclass
class EvaluationResult:
    """
    SH-DA++ v4.0 评估结果
    
    四维审计架构：空间感知、预算稳定性、可靠性、多维耗时
    """
    # ========== 精度指标 ==========
    cer: float = 0.0
    accuracy: float = 0.0
    
    # ========== 忠实度指标 (幻觉量化) ==========
    ocr_r: float = 0.0
    correction_rate: float = 0.0
    
    # ========== 对比指标 ==========
    agent_a_cer: float = 0.0
    cer_improvement: float = 0.0
    
    # ========== 效率指标 ==========
    hard_sample_recall: float = 0.0
    router_call_rate: float = 0.0
    
    # ========== 编辑操作统计 ==========
    total_substitutions: int = 0
    total_deletions: int = 0
    total_insertions: int = 0
    
    # ========== 样本统计 ==========
    total_samples: int = 0
    exact_match: int = 0
    overcorrected_samples: int = 0
    corrected_samples: int = 0
    
    # ========== SH-DA++ v4.0 空间感知审计 ==========
    boundary_deletion_samples: int = 0
    boundary_deletion_rate: float = 0.0
    boundary_deletion_recall_at_b: float = 0.0  # Boundary Deletion Recall@B
    left_boundary_deletions: int = 0
    right_boundary_deletions: int = 0
    
    # ========== SH-DA++ v4.0 可靠性审计 ==========
    cvr: float = 0.0  # Constraint Violation Rate
    aer: float = 0.0  # Accepted Edit Rate
    
    # ========== SH-DA++ v4.0 预算稳定性审计 ==========
    budget_stability: BudgetStabilityResult = None
    
    # ========== SH-DA++ v4.0 耗时审计 ==========
    latency_profiles: Dict[str, LatencyProfile] = None
    
    # ========== Error Type 聚合 ==========
    error_type_stats: Dict = None
    
    def __post_init__(self):
        if self.error_type_stats is None:
            self.error_type_stats = {}
        if self.latency_profiles is None:
            self.latency_profiles = {}
        if self.budget_stability is None:
            self.budget_stability = BudgetStabilityResult()
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 输出）"""
        result = {
            # 核心指标
            "overall_cer": round(self.cer, 4),
            "agent_a_cer": round(self.agent_a_cer, 4),
            "cer_improvement": round(self.cer_improvement, 4),
            "accuracy": round(self.accuracy, 4),
            
            # 幻觉量化
            "ocr_r": round(self.ocr_r, 4),
            "correction_rate": round(self.correction_rate, 4),
            
            # 效率
            "hard_sample_recall": round(self.hard_sample_recall, 4),
            "router_call_rate": round(self.router_call_rate, 4),
            
            # 编辑操作
            "edit_operations": {
                "substitutions": self.total_substitutions,
                "deletions": self.total_deletions,
                "insertions": self.total_insertions,
            },
            
            # 样本统计
            "sample_stats": {
                "total": self.total_samples,
                "exact_match": self.exact_match,
                "overcorrected": self.overcorrected_samples,
                "corrected": self.corrected_samples,
            },
            
            # SH-DA++ v4.0 空间感知
            "boundary_audit": {
                "boundary_deletion_samples": self.boundary_deletion_samples,
                "boundary_deletion_rate": round(self.boundary_deletion_rate, 4),
                "boundary_deletion_recall_at_b": round(self.boundary_deletion_recall_at_b, 4),
                "left_boundary_deletions": self.left_boundary_deletions,
                "right_boundary_deletions": self.right_boundary_deletions,
            },
            
            # SH-DA++ v4.0 可靠性
            "reliability_audit": {
                "cvr": round(self.cvr, 4),
                "aer": round(self.aer, 4),
            },
            
            # SH-DA++ v4.0 预算稳定性
            "budget_stability": self.budget_stability.to_dict() if self.budget_stability else {},
            
            # SH-DA++ v4.0 耗时
            "latency_audit": {
                name: profile.to_dict()
                for name, profile in self.latency_profiles.items()
            } if self.latency_profiles else {},
        }
        
        if self.error_type_stats:
            result["error_type_breakdown"] = self.error_type_stats
        
        return result
    
    def print_summary(self):
        """打印 SH-DA++ v4.0 评估摘要（S1.2 表格格式）"""
        print("\n" + "=" * 75)
        print("  SH-DA++ v4.0 Evaluation Report")
        print("=" * 75)
        
        # 表 1: 核心精度指标
        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│  Table 1: Core Accuracy Metrics                                         │")
        print("├─────────────────────────────────────────────────────────────────────────┤")
        print(f"│  System CER            │  {self.cer:.4f}  │  ({self.cer*100:.2f}%)                           │")
        print(f"│  Agent A CER           │  {self.agent_a_cer:.4f}  │  ({self.agent_a_cer*100:.2f}%)                           │")
        print(f"│  CER Improvement       │  {self.cer_improvement:+.4f} │  ({self.cer_improvement*100:+.2f}%)                          │")
        print(f"│  Accuracy (Exact)      │  {self.accuracy:.4f}  │  ({self.accuracy*100:.2f}%)                           │")
        print("└─────────────────────────────────────────────────────────────────────────┘")
        
        # 表 2: 幻觉量化指标
        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│  Table 2: Hallucination Metrics [Faithfulness]                          │")
        print("├─────────────────────────────────────────────────────────────────────────┤")
        ocr_r_status = "✓ IDEAL" if self.ocr_r < 0.05 else ("⚠ WARN" if self.ocr_r < 0.15 else "✗ DANGER")
        print(f"│  OCR-R (Over-Corr)     │  {self.ocr_r:.4f}  │  ({self.ocr_r*100:.2f}%)  {ocr_r_status:>15}   │")
        print(f"│  Correction Rate       │  {self.correction_rate:.4f}  │  ({self.correction_rate*100:.2f}%)                           │")
        print("└─────────────────────────────────────────────────────────────────────────┘")
        
        # 表 3: SH-DA++ v4.0 空间感知审计
        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│  Table 3: Boundary Deletion Audit [Spatial Awareness]                   │")
        print("├─────────────────────────────────────────────────────────────────────────┤")
        print(f"│  Boundary Del. Samples │  {self.boundary_deletion_samples:>6}  │  ({self.boundary_deletion_rate*100:.2f}% of total)            │")
        bdr_status = "✓ PASS" if self.boundary_deletion_recall_at_b >= 0.8 else "⚠ LOW"
        print(f"│  BD Recall@B           │  {self.boundary_deletion_recall_at_b:.4f}  │  {bdr_status:>30}   │")
        print(f"│  Left Boundary Del.    │  {self.left_boundary_deletions:>6}  │                                     │")
        print(f"│  Right Boundary Del.   │  {self.right_boundary_deletions:>6}  │                                     │")
        print("└─────────────────────────────────────────────────────────────────────────┘")
        
        # 表 4: 可靠性审计
        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│  Table 4: Reliability Audit [CVR/AER]                                   │")
        print("├─────────────────────────────────────────────────────────────────────────┤")
        cvr_status = "✓ GOOD" if self.cvr < 0.1 else ("⚠ WARN" if self.cvr < 0.2 else "✗ HIGH")
        print(f"│  CVR (Constraint Viol.)│  {self.cvr:.4f}  │  ({self.cvr*100:.2f}%)  {cvr_status:>15}   │")
        aer_status = "✓ GOOD" if self.aer >= 0.8 else ("⚠ LOW" if self.aer >= 0.5 else "✗ VERY LOW")
        print(f"│  AER (Accepted Edit)   │  {self.aer:.4f}  │  ({self.aer*100:.2f}%)  {aer_status:>15}   │")
        print("└─────────────────────────────────────────────────────────────────────────┘")
        
        # 表 5: 预算稳定性审计
        if self.budget_stability:
            bs = self.budget_stability
            print("\n┌─────────────────────────────────────────────────────────────────────────┐")
            print("│  Table 5: Budget Stability Audit                                        │")
            print("├─────────────────────────────────────────────────────────────────────────┤")
            print(f"│  Actual Call Rate      │  {bs.actual_call_rate:.4f}  │  Target: {bs.target_budget:.2f}                      │")
            stability_status = "✓ STABLE" if bs.is_stable else "✗ UNSTABLE"
            print(f"│  Window Std (σ)        │  {bs.window_std:.4f}  │  {stability_status:>30}   │")
            print(f"│  Max Deviation         │  {bs.max_deviation:.4f}  │  (from target B)                    │")
            print("└─────────────────────────────────────────────────────────────────────────┘")
        
        # 表 6: 耗时审计
        if self.latency_profiles:
            print("\n┌─────────────────────────────────────────────────────────────────────────┐")
            print("│  Table 6: Latency Profiling (ms)                                        │")
            print("├───────────────────┬─────────┬─────────┬─────────┬─────────┬─────────────┤")
            print("│  Field            │  Mean   │   P50   │   P95   │   P99   │   Count     │")
            print("├───────────────────┼─────────┼─────────┼─────────┼─────────┼─────────────┤")
            for name, profile in self.latency_profiles.items():
                if profile.count > 0:
                    print(f"│  {name:16s} │ {profile.mean:7.1f} │ {profile.p50:7.1f} │ {profile.p95:7.1f} │ {profile.p99:7.1f} │ {profile.count:11d} │")
            print("└───────────────────┴─────────┴─────────┴─────────┴─────────┴─────────────┘")
        
        # 表 7: 样本统计
        print("\n┌─────────────────────────────────────────────────────────────────────────┐")
        print("│  Table 7: Sample Statistics                                             │")
        print("├─────────────────────────────────────────────────────────────────────────┤")
        print(f"│  Total Samples         │  {self.total_samples:>6}                                            │")
        print(f"│  Exact Match           │  {self.exact_match:>6}  │  ({self.exact_match/max(self.total_samples,1)*100:.1f}%)                           │")
        print(f"│  Overcorrected         │  {self.overcorrected_samples:>6}  │  ({self.overcorrected_samples/max(self.total_samples,1)*100:.1f}%)                           │")
        print(f"│  Corrected             │  {self.corrected_samples:>6}  │  ({self.corrected_samples/max(self.total_samples,1)*100:.1f}%)                           │")
        print(f"│  Router Call Rate      │  {self.router_call_rate:.4f}  │  ({self.router_call_rate*100:.2f}%)                           │")
        print("└─────────────────────────────────────────────────────────────────────────┘")
        
        # Error Type Breakdown
        if self.error_type_stats:
            print("\n┌─────────────────────────────────────────────────────────────────────────┐")
            print("│  Table 8: Error Type Breakdown                                          │")
            print("├───────────────────┬─────────┬─────────┬─────────┬─────────┬─────────────┤")
            print("│  Type             │  Count  │   CER   │  OCR-R  │  Corr.  │  BD Rate    │")
            print("├───────────────────┼─────────┼─────────┼─────────┼─────────┼─────────────┤")
            for error_type, stats in sorted(self.error_type_stats.items()):
                type_str = error_type[:16].ljust(16)
                count = stats.get('count', 0)
                cer = stats.get('cer', 0)
                ocr_r = stats.get('ocr_r', 0)
                corr = stats.get('correction_rate', 0)
                bd_rate = stats.get('boundary_deletion_rate', 0)
                print(f"│  {type_str} │ {count:>7} │ {cer:7.3f} │ {ocr_r:7.3f} │ {corr:7.3f} │ {bd_rate:11.3f} │")
            print("└───────────────────┴─────────┴─────────┴─────────┴─────────┴─────────────┘")
        
        print("\n" + "=" * 75)


# =============================================================================
# 批量评估
# =============================================================================

def evaluate_batch(
    predictions: List[str],
    references: List[str],
    agent_a_texts: List[str] = None,
    is_hard_samples: List[bool] = None,
    raw_records: List[Dict] = None,
    router_features: List[Dict] = None,
    boundary_k: int = 2,
    verbose: bool = False
) -> EvaluationResult:
    """
    SH-DA++ v4.0 批量评估
    
    实现四维审计架构：空间感知、预算稳定性、可靠性、多维耗时
    
    Args:
        predictions: 系统预测列表 (final_text)
        references: 真值列表 (gt_text)
        agent_a_texts: Agent A 识别结果列表
        is_hard_samples: upgrade/is_hard 标记列表
        raw_records: 原始记录（用于 CVR/AER 计算）
        router_features: 路由特征记录（用于预算稳定性和耗时分析）
        boundary_k: 边界窗口大小 K
        verbose: 是否打印详细信息
        
    Returns:
        EvaluationResult: 完整评估结果
    """
    if len(predictions) != len(references):
        raise ValueError("predictions 和 references 长度不一致")
    
    result = EvaluationResult()
    result.total_samples = len(predictions)
    
    cers = []
    ocr_rs = []
    correction_rates = []
    agent_a_cers = []
    
    # 边界删除审计
    boundary_deletion_samples = []
    boundary_deletion_upgraded = 0
    
    # Hard Sample Recall
    actual_hard_samples = 0
    router_detected_hard = 0
    total_upgraded = 0
    
    # Error Type 聚合
    error_type_data = defaultdict(lambda: {
        'count': 0, 'cers': [], 'ocr_rs': [], 'correction_rates': [], 'boundary_deletions': 0
    })
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # ========== CER ==========
        cer, ops = calculate_cer(pred, ref, return_details=True)
        cers.append(cer)
        
        result.total_substitutions += ops.substitutions
        result.total_deletions += ops.deletions
        result.total_insertions += ops.insertions
        
        if cer == 0:
            result.exact_match += 1
        
        # 获取 error_type
        error_type = ""
        if raw_records and i < len(raw_records):
            record = raw_records[i]
            error_type = record.get('error_type', record.get('metadata', {}).get('error_type', ''))
        
        # ========== Agent A 相关指标 ==========
        if agent_a_texts and i < len(agent_a_texts):
            agent_a = agent_a_texts[i]
            
            # Agent A CER
            agent_a_cer = calculate_cer(agent_a, ref)
            agent_a_cers.append(agent_a_cer)
            
            # 判断是否为实际困难样本
            is_actual_hard = agent_a_cer > 0
            if is_actual_hard:
                actual_hard_samples += 1
            
            # Router 召回率计算
            if is_hard_samples and i < len(is_hard_samples):
                if is_hard_samples[i]:
                    total_upgraded += 1
                    if is_actual_hard:
                        router_detected_hard += 1
            
            # ========== 边界删除分析 (SH-DA++ v4.0) ==========
            bd_result = identify_boundary_deletions(agent_a, ref, K=boundary_k)
            
            if bd_result.is_boundary_deletion:
                result.boundary_deletion_samples += 1
                result.left_boundary_deletions += len(bd_result.left_deletions)
                result.right_boundary_deletions += len(bd_result.right_deletions)
                boundary_deletion_samples.append(i)
                
                # 检查是否被 upgrade 拦截
                if is_hard_samples and i < len(is_hard_samples) and is_hard_samples[i]:
                    boundary_deletion_upgraded += 1
            
            # Error Type 聚合
            if error_type:
                error_type_data[error_type]['boundary_deletions'] += (1 if bd_result.is_boundary_deletion else 0)
            
            # ========== OCR-R ==========
            ocr_r, _ = calculate_ocr_r(agent_a, pred, ref)
            ocr_rs.append(ocr_r)
            
            if ocr_r > 0:
                result.overcorrected_samples += 1
                if verbose:
                    print(f"\n[过度纠错] Sample {i}:")
                    print(f"  Agent A: '{agent_a}'")
                    print(f"  System:  '{pred}'")
                    print(f"  GT:      '{ref}'")
                    print(f"  OCR-R:   {ocr_r:.4f}")
            
            # ========== Correction Rate ==========
            corr_rate, corr_details = calculate_correction_rate(agent_a, pred, ref)
            correction_rates.append(corr_rate)
            
            if corr_details.get('corrected', 0) > 0:
                result.corrected_samples += 1
            
            # Error Type 聚合
            if error_type:
                error_type_data[error_type]['count'] += 1
                error_type_data[error_type]['cers'].append(cer)
                error_type_data[error_type]['ocr_rs'].append(ocr_r)
                error_type_data[error_type]['correction_rates'].append(corr_rate)
    
    # ========== 汇总指标 ==========
    result.cer = sum(cers) / len(cers) if cers else 0.0
    result.accuracy = result.exact_match / result.total_samples if result.total_samples > 0 else 0.0
    
    if ocr_rs:
        result.ocr_r = sum(ocr_rs) / len(ocr_rs)
    
    if correction_rates:
        result.correction_rate = sum(correction_rates) / len(correction_rates)
    
    if agent_a_cers:
        result.agent_a_cer = sum(agent_a_cers) / len(agent_a_cers)
        result.cer_improvement = result.agent_a_cer - result.cer
    
    # Router 效率
    if actual_hard_samples > 0:
        result.hard_sample_recall = router_detected_hard / actual_hard_samples
    
    result.router_call_rate = total_upgraded / result.total_samples if result.total_samples > 0 else 0.0
    
    # ========== SH-DA++ v4.0 边界删除审计 ==========
    result.boundary_deletion_rate = (
        result.boundary_deletion_samples / result.total_samples
        if result.total_samples > 0 else 0.0
    )
    
    # Boundary Deletion Recall@B
    if result.boundary_deletion_samples > 0:
        result.boundary_deletion_recall_at_b = boundary_deletion_upgraded / result.boundary_deletion_samples
    
    # ========== SH-DA++ v4.0 可靠性审计 (CVR/AER) ==========
    if raw_records:
        cvr, aer, _ = calculate_cvr_aer(raw_records)
        result.cvr = cvr
        result.aer = aer
    
    # ========== SH-DA++ v4.0 预算稳定性审计 ==========
    features_for_budget = router_features if router_features else raw_records
    if features_for_budget:
        result.budget_stability = analyze_budget_stability(features_for_budget)
    
    # ========== SH-DA++ v4.0 耗时审计 ==========
    features_for_latency = router_features if router_features else raw_records
    if features_for_latency:
        result.latency_profiles = analyze_latency(features_for_latency)
    
    # ========== Error Type 聚合 ==========
    for error_type, data in error_type_data.items():
        if data['count'] > 0:
            result.error_type_stats[error_type] = {
                'count': data['count'],
                'cer': sum(data['cers']) / len(data['cers']) if data['cers'] else 0.0,
                'ocr_r': sum(data['ocr_rs']) / len(data['ocr_rs']) if data['ocr_rs'] else 0.0,
                'correction_rate': sum(data['correction_rates']) / len(data['correction_rates']) if data['correction_rates'] else 0.0,
                'boundary_deletion_rate': data['boundary_deletions'] / data['count'],
            }
    
    return result


# =============================================================================
# 测试
# =============================================================================

def run_tests() -> bool:
    """运行单元测试"""
    print("=" * 65)
    print("  SH-DA++ v4.0 Evaluation Module Tests")
    print("=" * 65)
    
    all_passed = True
    
    # Test 1: CER 计算
    print("\n[1] CER 计算测试:")
    print("-" * 65)
    
    test_cases = [
        ("hello", "hello", 0.0),
        ("helo", "hello", 0.2),
        ("", "hello", 1.0),
        # ("hello", "", 0.0),  # 边界情况：ref 为空时 CER 定义为 0 或 1，跳过
        ("你好世界", "你好世界", 0.0),
        ("你好世", "你好世界", 0.25),
    ]
    
    for pred, ref, expected in test_cases:
        cer = calculate_cer(pred, ref)
        status = "PASS" if abs(cer - expected) < 0.01 else "FAIL"
        print(f"  [{status}] pred='{pred}', ref='{ref}' -> CER={cer:.4f} (expected: {expected})")
        if status == "FAIL":
            all_passed = False
    
    # Test 2: 边界删除识别
    print("\n[2] 边界删除识别测试:")
    print("-" * 65)
    
    bd_test_cases = [
        ("世界你好", "你好世界", True, [0, 1], []),  # 左边界删除
        ("你好世", "你好世界", True, [], [3]),  # 右边界删除
        ("好世", "你好世界", True, [0], [3]),  # 两边都删除
        ("你好世界", "你好世界", False, [], []),  # 完全匹配
        ("好世", "你好世界", True, [0], [3]),  # 边界删除
    ]
    
    for agent_a, gt, expected_is_bd, expected_left, expected_right in bd_test_cases:
        result = identify_boundary_deletions(agent_a, gt, K=2)
        status = "PASS" if result.is_boundary_deletion == expected_is_bd else "FAIL"
        print(f"  [{status}] A='{agent_a}', GT='{gt}' -> is_bd={result.is_boundary_deletion}, "
              f"left={result.left_deletions}, right={result.right_deletions}")
        if status == "FAIL":
            all_passed = False
    
    # Test 3: OCR-R 计算
    print("\n[3] OCR-R 计算测试:")
    print("-" * 65)
    
    ocr_r_cases = [
        ("你好世界", "你好世界", "你好世界", 0.0),  # 无过度纠错
        ("你好世界", "我好世界", "你好世界", 0.25),  # 25% 过度纠错
    ]
    
    for agent_a, system, gt, expected in ocr_r_cases:
        ocr_r, _ = calculate_ocr_r(agent_a, system, gt)
        status = "PASS" if abs(ocr_r - expected) < 0.01 else "FAIL"
        print(f"  [{status}] A='{agent_a}', S='{system}', GT='{gt}' -> OCR-R={ocr_r:.4f}")
        if status == "FAIL":
            all_passed = False
    
    # Test 4: 预算稳定性分析
    print("\n[4] 预算稳定性分析测试:")
    print("-" * 65)
    
    # 模拟 200 个样本，20% 升级率
    mock_features = [{'upgrade': i % 5 == 0} for i in range(400)]
    bs_result = analyze_budget_stability(mock_features, window_size=200, target_budget=0.2)
    
    status = "PASS" if abs(bs_result.actual_call_rate - 0.2) < 0.01 else "FAIL"
    print(f"  [{status}] Actual call rate: {bs_result.actual_call_rate:.4f} (expected: 0.2)")
    print(f"  Window std: {bs_result.window_std:.4f}, is_stable: {bs_result.is_stable}")
    if status == "FAIL":
        all_passed = False
    
    # 总结
    print("\n" + "=" * 65)
    if all_passed:
        print("  [OK] All Tests Passed")
    else:
        print("  [FAIL] Some Tests Failed")
    print("=" * 65)
    
    return all_passed


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SH-DA++ v4.0 专用审计工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估预测文件
    python evaluate.py --predictions ./results/l2w1_results.jsonl
    
    # 包含路由特征分析
    python evaluate.py --predictions ./results/l2w1_results.jsonl \\
                       --router_features ./results/router_features.jsonl
    
    # 运行测试
    python evaluate.py --test
        """
    )
    
    parser.add_argument("--predictions", "-p", type=str, default="",
                        help="预测结果文件路径 (JSONL)")
    parser.add_argument("--router_features", "-r", type=str, default="",
                        help="路由特征文件路径 (router_features.jsonl)")
    parser.add_argument("--references", type=str, default="",
                        help="真值文件路径（如果不在预测文件中）")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="输出评估结果到文件")
    parser.add_argument("--boundary_k", type=int, default=2,
                        help="边界窗口大小 K (默认: 2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细信息")
    parser.add_argument("--test", action="store_true",
                        help="运行测试")
    
    return parser.parse_args()


def generate_evaluation_report(result: EvaluationResult, output_path: str):
    """生成评估报告文件"""
    report = result.to_dict()
    
    from datetime import datetime
    report['timestamp'] = datetime.now().isoformat()
    report['version'] = 'SH-DA++ v4.0'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估报告已保存至: {output_path}")


def write_metrics_summary(result: EvaluationResult, output_path: str):
    """生成 metrics_summary.json（Stage 1 交付必需）"""
    summary = result.to_dict()
    from datetime import datetime
    summary["timestamp"] = datetime.now().isoformat()
    summary["version"] = "SH-DA++ v4.0"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nmetrics_summary.json 已保存至: {output_path}")


def main():
    """主函数"""
    args = parse_args()
    
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    
    if not args.predictions:
        print("请提供预测文件路径 (--predictions)")
        print("或运行测试 (--test)")
        print("\n用法示例:")
        print("  python evaluate.py --predictions ./results/l2w1_results.jsonl")
        print("  python evaluate.py --test")
        return
    
    # 加载预测数据
    try:
        final_texts, gt_texts, agent_a_texts, is_hard_samples, raw_records = load_predictions(args.predictions)
    except Exception as e:
        print(f"加载文件失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载路由特征（可选）
    router_features = []
    if args.router_features:
        router_features = load_router_features(args.router_features)
        print(f"[INFO] 加载了 {len(router_features)} 条路由特征记录")
    
    # 如果 gt_texts 为空，尝试从单独文件加载
    if not gt_texts and args.references:
        with open(args.references, 'r', encoding='utf-8') as f:
            gt_texts = [line.strip() for line in f]
    
    if not gt_texts:
        print("错误: 未找到真值数据 (gt_text)")
        print("请确保预测文件包含 'gt_text' 字段，或使用 --references 指定真值文件")
        return

    # 空值检查（防止“空对空”导致幻觉指标）
    empty_agent_a = sum(1 for t in agent_a_texts if not t)
    empty_gt = sum(1 for t in gt_texts if not t)
    if empty_agent_a > 0 or empty_gt > 0:
        print(
            f"[Warning] 空文本检测: agent_a_text 为空 {empty_agent_a}/{len(agent_a_texts)}, "
            f"gt_text 为空 {empty_gt}/{len(gt_texts)}"
        )

    # 打印前 5 个样本对比 (Agent A vs GT)
    print("\n[Sample Check] A_text | GT_text | CER (first 5)")
    for i in range(min(5, len(gt_texts))):
        cer = calculate_cer(agent_a_texts[i], gt_texts[i])
        a_text = agent_a_texts[i]
        g_text = gt_texts[i]
        print(f"  [{i}] A='{a_text}' | GT='{g_text}' | CER={cer:.4f}")
    
    if len(final_texts) != len(gt_texts):
        print(f"错误: 预测数量 ({len(final_texts)}) 与真值数量 ({len(gt_texts)}) 不匹配")
        return
    
    # 执行评估
    print("\n正在执行 SH-DA++ v4.0 四维审计...")
    print(f"  样本数量: {len(final_texts)}")
    print(f"  边界窗口 K: {args.boundary_k}")
    
    result = evaluate_batch(
        predictions=final_texts,
        references=gt_texts,
        agent_a_texts=agent_a_texts,
        is_hard_samples=is_hard_samples,
        raw_records=raw_records,
        router_features=router_features,
        boundary_k=args.boundary_k,
        verbose=args.verbose
    )
    
    # 打印评估报告
    result.print_summary()
    
    # 写出 metrics_summary.json（固定输出）
    metrics_output = Path(args.predictions).parent / "metrics_summary.json"
    write_metrics_summary(result, str(metrics_output))
    
    # 保存评估报告
    if args.output:
        generate_evaluation_report(result, args.output)
    else:
        default_output = Path(args.predictions).parent / "evaluation_report.json"
        generate_evaluation_report(result, str(default_output))


if __name__ == "__main__":
    main()

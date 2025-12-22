"""
L2W1 核心评价指标计算脚本 - 非对称编辑距离分析

评估指标体系 (精度-忠实度-效率三维):

1. **CER (Character Error Rate)**: 字符错误率
   公式: CER = (S + D + I) / N
   其中 S=替换, D=删除, I=插入, N=真值长度

2. **OCR-R (Over-Correction Rate)**: 过度纠错率 [核心创新指标]
   公式: OCR-R = Count(Agent A Correct → System Wrong) / Total Correct Chars in Agent A
   衡量模型将正确识别结果"改错"的比例，直接反映幻觉程度
   - OCR-R = 0: 无过度纠错（理想）
   - OCR-R > 0: 存在"瞎改"行为（幻觉）

3. **Correction Rate (CR)**: 纠正率
   公式: CR = Count(Agent A Wrong → System Correct) / Total Wrong Chars in Agent A
   衡量模型"救回"了多少 Agent A 识别错的字

4. **Hard Sample Recall**: Router 召回率
   衡量 Router 成功识别并拦截困难样本的能力

核心技术: 非对称对齐 (Asymmetric Alignment)
- 使用 difflib.SequenceMatcher 进行字符级精确对齐
- 分别对齐 (Agent A, GT) 和 (System, GT)
- 跟踪 GT 中每个位置的正确性变化

Usage:
    python evaluate.py --predictions ./outputs/inference_results.jsonl
    python evaluate.py --test  # 运行测试
"""

import os
import sys
import json
import argparse
import difflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

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
    
    Args:
        s1, s2: 两个字符串
        
    Returns:
        编辑距离
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


def get_edit_operations_detailed(pred: str, gt: str) -> EditOperations:
    """
    获取详细的编辑操作统计 (S, D, I)
    
    核心逻辑: 使用 difflib.SequenceMatcher 精确计算各类操作
    
    Args:
        pred: 预测文本
        gt: 真值文本 (Ground Truth)
        
    Returns:
        EditOperations: 包含 S, D, I 的详细统计
    """
    ops = EditOperations()
    matcher = difflib.SequenceMatcher(None, pred, gt)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            # 替换: pred[i1:i2] -> gt[j1:j2]
            pred_len = i2 - i1
            gt_len = j2 - j1
            # 替换数量取较小值，多余部分算删除或插入
            ops.substitutions += min(pred_len, gt_len)
            if pred_len > gt_len:
                ops.deletions += (pred_len - gt_len)
            elif gt_len > pred_len:
                ops.insertions += (gt_len - pred_len)
        elif tag == 'delete':
            # 删除: pred 中多余的字符
            ops.deletions += (i2 - i1)
        elif tag == 'insert':
            # 插入: gt 中存在但 pred 中缺失
            ops.insertions += (j2 - j1)
    
    return ops


def get_edit_operations(s1: str, s2: str) -> List[Tuple[str, int, str, str]]:
    """
    获取详细的编辑操作列表
    
    Returns:
        List[(operation, position, char_from, char_to)]
    """
    matcher = difflib.SequenceMatcher(None, s1, s2)
    operations = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            for offset in range(min(i2 - i1, j2 - j1)):
                operations.append((
                    'replace',
                    i1 + offset,
                    s1[i1 + offset] if i1 + offset < len(s1) else '',
                    s2[j1 + offset] if j1 + offset < len(s2) else ''
                ))
        elif tag == 'delete':
            for offset in range(i2 - i1):
                operations.append((
                    'delete',
                    i1 + offset,
                    s1[i1 + offset],
                    ''
                ))
        elif tag == 'insert':
            operations.append((
                'insert',
                i1,
                '',
                s2[j1:j2]
            ))
    
    return operations


# =============================================================================
# 核心指标计算
# =============================================================================

def calculate_cer(pred: str, gt: str, return_details: bool = False):
    """
    计算字符错误率 (Character Error Rate)
    
    公式: CER = (S + D + I) / N
    其中:
    - S = 替换 (Substitution)
    - D = 删除 (Deletion)
    - I = 插入 (Insertion)
    - N = 真值总长度
    
    Args:
        pred: 预测文本
        gt: 真值文本 (Ground Truth)
        return_details: 是否返回 S, D, I 详细统计
    
    Returns:
        如果 return_details=False: CER 值 (0-1)
        如果 return_details=True: Tuple[CER, EditOperations]
    """
    if len(gt) == 0:
        cer = 1.0 if len(pred) > 0 else 0.0
        if return_details:
            ops = EditOperations(insertions=len(pred))
            return cer, ops
        return cer
    
    # 获取详细的编辑操作
    ops = get_edit_operations_detailed(pred, gt)
    
    # CER = (S + D + I) / N
    cer = min(ops.total / len(gt), 1.0)
    
    if return_details:
        return cer, ops
    return cer


def calculate_ocr_r(
    agent_a_text: str,
    system_output: str,
    ground_truth: str
) -> Tuple[float, Dict]:
    """
    计算过度纠错率 (OCR-R / Over-Correction Rate)
    
    核心幻觉抑制指标：衡量模型将正确识别结果改错的比例
    这是论文的核心创新指标！
    
    公式:
        OCR-R = Count(Agent A Correct → System Wrong) / Total Correct Chars in Agent A
    
    非对称对齐算法 (Asymmetric Alignment):
    1. 对齐 Agent A 与 GT，标记 Agent A 正确的字符在 GT 中的位置集合 P_correct
    2. 对齐 System 与 GT，检查 P_correct 中的位置是否仍为 'equal'
    3. 若原本正确的位置变为 'replace' 或 'delete'，则计入过度纠错
    
    Args:
        agent_a_text: Agent A 的识别结果
        system_output: 系统最终输出（经过 Agent B 处理）
        ground_truth: 真值文本
        
    Returns:
        Tuple[ocr_r, details]:
            - ocr_r: 过度纠错率 (0-1)，越低越好
            - details: 详细信息字典
    """
    # ========== 边界条件处理 ==========
    if len(ground_truth) == 0:
        return 0.0, {"error": "ground_truth is empty", "total_correct_in_a": 0, "overcorrected": 0}
    
    if len(agent_a_text) == 0:
        return 0.0, {"error": "agent_a_text is empty", "total_correct_in_a": 0, "overcorrected": 0}
    
    if len(system_output) == 0:
        # System 输出为空，所有 Agent A 正确的字符都被"删除"了
        matcher_a_gt = difflib.SequenceMatcher(None, agent_a_text, ground_truth)
        total_correct = sum(i2 - i1 for tag, i1, i2, j1, j2 in matcher_a_gt.get_opcodes() if tag == 'equal')
        return 1.0 if total_correct > 0 else 0.0, {
            "total_correct_in_a": total_correct, 
            "overcorrected": total_correct,
            "note": "system_output is empty"
        }
    
    # ========== Step 1: 定位 Agent A 正确区域 ==========
    # 使用 difflib.SequenceMatcher 对齐 Agent A 和 GT
    # 记录所有 'equal' 操作码对应的 GT 位置索引集合 P_correct
    
    matcher_a_gt = difflib.SequenceMatcher(None, agent_a_text, ground_truth)
    
    # P_correct: GT 中 Agent A 正确识别的位置集合
    # 格式: {gt_position: (char_in_a, pos_in_a)}
    P_correct = {}
    
    for tag, i1, i2, j1, j2 in matcher_a_gt.get_opcodes():
        if tag == 'equal':
            for offset in range(i2 - i1):
                gt_pos = j1 + offset
                a_pos = i1 + offset
                P_correct[gt_pos] = (agent_a_text[a_pos], a_pos)
    
    total_correct_in_a = len(P_correct)
    
    if total_correct_in_a == 0:
        # Agent A 全错，无法计算 OCR-R（分母为0）
        return 0.0, {
            "total_correct_in_a": 0,
            "overcorrected": 0,
            "note": "Agent A has no correct characters (all wrong)"
        }
    
    # ========== Step 2: 检测 System 改动 ==========
    # 对齐 System 输出和 GT
    # 检查 P_correct 中的位置在 System 中是否仍为 'equal'
    
    matcher_sys_gt = difflib.SequenceMatcher(None, system_output, ground_truth)
    
    # 构建 GT 中正确位置的集合（System 输出正确的位置）
    sys_correct_gt_positions = set()
    
    for tag, i1, i2, j1, j2 in matcher_sys_gt.get_opcodes():
        if tag == 'equal':
            for offset in range(j2 - j1):
                sys_correct_gt_positions.add(j1 + offset)
    
    # ========== Step 3: 计算过度纠错 ==========
    # P_correct 中存在但 sys_correct_gt_positions 中不存在的位置
    # 即 Agent A 正确 → System 错误
    
    overcorrected = 0
    overcorrected_details = []
    
    for gt_pos, (char, a_pos) in P_correct.items():
        if gt_pos not in sys_correct_gt_positions:
            # 过度纠错：原本正确，现在错了
            overcorrected += 1
            
            # 尝试找出 System 在这个位置附近输出了什么
            sys_char = "?"
            # 由于对齐可能导致位置偏移，这里简化处理
            if gt_pos < len(system_output):
                sys_char = system_output[gt_pos]
            
            if len(overcorrected_details) < 10:  # 只记录前10个
                overcorrected_details.append({
                    "gt_position": gt_pos,
                    "gt_char": ground_truth[gt_pos] if gt_pos < len(ground_truth) else "",
                    "agent_a_char": char,
                    "system_char": sys_char,
                })
    
    # ========== Step 4: 计算 OCR-R ==========
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
    
    衡量 Agent B 成功"救回"了多少 Agent A 识别错的字
    目标：越高越好，代表纠错能力强
    
    公式:
        CR = Count(Agent A Wrong → System Correct) / Total Wrong Chars in Agent A
    
    非对称对齐逻辑:
    1. 找出 GT 中 Agent A 错误的位置集合 P_wrong
    2. 检查这些位置在 System 输出中是否变为正确
    
    Args:
        agent_a_text: Agent A 识别结果
        system_output: 系统输出
        ground_truth: 真值
        
    Returns:
        Tuple[correction_rate, details]
    """
    # ========== 边界条件 ==========
    if len(ground_truth) == 0:
        return 0.0, {"error": "ground_truth is empty", "total_wrong_in_a": 0, "corrected": 0}
    
    # ========== Step 1: 找出 GT 中 Agent A 错误的位置 ==========
    matcher_a_gt = difflib.SequenceMatcher(None, agent_a_text, ground_truth)
    
    # P_wrong: GT 中 Agent A 识别错误的位置集合
    P_wrong = set()
    
    for tag, i1, i2, j1, j2 in matcher_a_gt.get_opcodes():
        if tag == 'equal':
            continue
        # replace, delete (in agent_a), insert (in gt)
        for gt_pos in range(j1, j2):
            P_wrong.add(gt_pos)
    
    total_wrong_in_a = len(P_wrong)
    
    if total_wrong_in_a == 0:
        # Agent A 完全正确
        return 1.0, {
            "total_wrong_in_a": 0,
            "corrected": 0,
            "note": "Agent A has no errors (perfect recognition)"
        }
    
    # ========== Step 2: 检查 System 在这些位置是否纠正 ==========
    matcher_sys_gt = difflib.SequenceMatcher(None, system_output, ground_truth)
    
    # System 正确的 GT 位置
    sys_correct_positions = set()
    
    for tag, i1, i2, j1, j2 in matcher_sys_gt.get_opcodes():
        if tag == 'equal':
            for gt_pos in range(j1, j2):
                sys_correct_positions.add(gt_pos)
    
    # ========== Step 3: 计算纠正数量 ==========
    # Agent A Wrong → System Correct
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
    
    # ========== Step 4: 计算纠正率 ==========
    correction_rate = corrected / total_wrong_in_a
    
    return correction_rate, {
        "total_wrong_in_a": total_wrong_in_a,
        "corrected": corrected,
        "correction_rate": correction_rate,
        "corrected_positions": corrected_details,
        "note": "Agent A Wrong -> System Correct transitions"
    }


# =============================================================================
# 批量评估
# =============================================================================

@dataclass
class EvaluationResult:
    """
    评估结果 (Evaluation Report)
    
    包含三维指标体系：精度-忠实度-效率
    支持按 error_type 聚合统计 (Data Protocol v1.0)
    """
    # ========== 精度指标 ==========
    cer: float = 0.0               # 系统最终 CER
    accuracy: float = 0.0          # 完全正确率
    
    # ========== 忠实度指标 (幻觉量化) ==========
    ocr_r: float = 0.0             # 过度纠错率 (越低越好)
    correction_rate: float = 0.0   # 纠正率 (越高越好)
    
    # ========== 对比指标 ==========
    agent_a_cer: float = 0.0       # Agent A 原始 CER
    cer_improvement: float = 0.0   # CER 改进 (agent_a_cer - system_cer)
    
    # ========== 效率指标 ==========
    hard_sample_recall: float = 0.0  # Router 召回率 (如果有 Router 标记)
    router_call_rate: float = 0.0    # Agent B 触发比例
    
    # ========== 编辑操作统计 ==========
    total_substitutions: int = 0   # 总替换次数
    total_deletions: int = 0       # 总删除次数
    total_insertions: int = 0      # 总插入次数
    
    # ========== 样本统计 ==========
    total_samples: int = 0
    exact_match: int = 0
    overcorrected_samples: int = 0
    corrected_samples: int = 0     # 成功纠错的样本数
    
    # ========== Error Type 聚合 (Data Protocol v1.0) ==========
    error_type_stats: Dict = None  # 按错误类型聚合的统计
    
    def __post_init__(self):
        if self.error_type_stats is None:
            self.error_type_stats = {}
    
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
            }
        }
        
        # 添加 error_type 聚合统计 (如果有)
        if self.error_type_stats:
            result["error_type_breakdown"] = self.error_type_stats
        
        return result
    
    def print_summary(self):
        """打印评估摘要表"""
        print("\n" + "=" * 65)
        print("  L2W1 Evaluation Report (Data Protocol v1.0)")
        print("=" * 65)
        
        print("\n+-------------------------------------------------------------+")
        print("|  Accuracy Metrics                                           |")
        print("+-------------------------------------------------------------+")
        print(f"|  System CER:         {self.cer:.4f} ({self.cer*100:.2f}%)                        |")
        print(f"|  Agent A CER:        {self.agent_a_cer:.4f} ({self.agent_a_cer*100:.2f}%)                        |")
        print(f"|  CER Improvement:    {self.cer_improvement:.4f} ({self.cer_improvement*100:+.2f}%)                       |")
        print(f"|  Accuracy:           {self.accuracy:.4f} ({self.accuracy*100:.2f}%)                        |")
        print("+-------------------------------------------------------------+")
        
        print("\n+-------------------------------------------------------------+")
        print("|  Hallucination Metrics [Core Innovation]                    |")
        print("+-------------------------------------------------------------+")
        ocr_r_status = "[IDEAL]" if self.ocr_r < 0.05 else ("[WARN]" if self.ocr_r < 0.15 else "[DANGER]")
        print(f"|  OCR-R (Over-Corr):  {self.ocr_r:.4f} ({self.ocr_r*100:.2f}%) {ocr_r_status:>12} |")
        print(f"|  Correction Rate:    {self.correction_rate:.4f} ({self.correction_rate*100:.2f}%)                        |")
        print("+-------------------------------------------------------------+")
        
        print("\n+-------------------------------------------------------------+")
        print("|  Sample Statistics                                          |")
        print("+-------------------------------------------------------------+")
        print(f"|  Total Samples:      {self.total_samples:>6}                                   |")
        print(f"|  Exact Match:        {self.exact_match:>6} ({self.exact_match/max(self.total_samples,1)*100:.1f}%)                           |")
        print(f"|  Overcorrected:      {self.overcorrected_samples:>6} ({self.overcorrected_samples/max(self.total_samples,1)*100:.1f}%)                           |")
        print(f"|  Corrected:          {self.corrected_samples:>6} ({self.corrected_samples/max(self.total_samples,1)*100:.1f}%)                           |")
        print("+-------------------------------------------------------------+")
        
        if self.total_substitutions + self.total_deletions + self.total_insertions > 0:
            print("\n+-------------------------------------------------------------+")
            print("|  Edit Operations                                            |")
            print("+-------------------------------------------------------------+")
            print(f"|  Substitutions (S):  {self.total_substitutions:>6}                                   |")
            print(f"|  Deletions (D):      {self.total_deletions:>6}                                   |")
            print(f"|  Insertions (I):     {self.total_insertions:>6}                                   |")
            print("+-------------------------------------------------------------+")
        
        # Error Type Breakdown (Data Protocol v1.0)
        if self.error_type_stats:
            print("\n+-------------------------------------------------------------+")
            print("|  Error Type Breakdown                                       |")
            print("+-------------------------------------------------------------+")
            print("|  Type                 Count   CER     OCR-R   Corr.Rate     |")
            print("|  -----------------------------------------------------------+")
            for error_type, stats in sorted(self.error_type_stats.items()):
                type_str = error_type[:18].ljust(18)
                count = stats.get('count', 0)
                cer = stats.get('cer', 0)
                ocr_r = stats.get('ocr_r', 0)
                corr = stats.get('correction_rate', 0)
                print(f"|  {type_str}  {count:>5}   {cer:.3f}   {ocr_r:.3f}   {corr:.3f}         |")
            print("+-------------------------------------------------------------+")
        
        print("\n" + "=" * 65)


def evaluate_batch(
    predictions: List[str],
    references: List[str],
    agent_a_texts: List[str] = None,
    is_hard_samples: List[bool] = None,
    metadata_list: List[Dict] = None,
    verbose: bool = False
) -> EvaluationResult:
    """
    批量评估
    
    实现非对称编辑距离分析，计算三维指标体系
    
    Args:
        predictions: 系统预测列表 (final_text)
        references: 真值列表 (gt_text)
        agent_a_texts: Agent A 识别结果列表（用于计算 OCR-R）
        is_hard_samples: Router 标记的困难样本（用于计算召回率）
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
    
    # 用于 Hard Sample Recall
    actual_hard_samples = 0  # 实际困难样本（Agent A 有错误）
    router_detected_hard = 0  # Router 正确检测的困难样本
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # ========== CER with detailed S, D, I ==========
        cer, ops = calculate_cer(pred, ref, return_details=True)
        cers.append(cer)
        
        result.total_substitutions += ops.substitutions
        result.total_deletions += ops.deletions
        result.total_insertions += ops.insertions
        
        if cer == 0:
            result.exact_match += 1
        
        # ========== Agent A 相关指标 ==========
        if agent_a_texts and i < len(agent_a_texts):
            agent_a = agent_a_texts[i]
            
            # Agent A CER
            agent_a_cer = calculate_cer(agent_a, ref)
            agent_a_cers.append(agent_a_cer)
            
            # 判断是否为实际困难样本（Agent A 有错误）
            is_actual_hard = agent_a_cer > 0
            if is_actual_hard:
                actual_hard_samples += 1
            
            # Router 召回率计算
            if is_hard_samples and i < len(is_hard_samples):
                if is_hard_samples[i] and is_actual_hard:
                    router_detected_hard += 1
            
            # ========== OCR-R (过度纠错率) ==========
            ocr_r, ocr_details = calculate_ocr_r(agent_a, pred, ref)
            ocr_rs.append(ocr_r)
            
            if ocr_r > 0:
                result.overcorrected_samples += 1
                if verbose:
                    print(f"\n[过度纠错] Sample {i}:")
                    print(f"  Agent A: '{agent_a}'")
                    print(f"  System:  '{pred}'")
                    print(f"  GT:      '{ref}'")
                    print(f"  OCR-R:   {ocr_r:.4f}")
                    if ocr_details.get('overcorrected_positions'):
                        for pos_info in ocr_details['overcorrected_positions'][:3]:
                            print(f"    位置 {pos_info['gt_position']}: "
                                  f"'{pos_info['agent_a_char']}' -> '{pos_info['system_char']}' "
                                  f"(GT: '{pos_info['gt_char']}')")
            
            # ========== Correction Rate (纠正率) ==========
            corr_rate, corr_details = calculate_correction_rate(agent_a, pred, ref)
            correction_rates.append(corr_rate)
            
            # 统计成功纠错的样本
            if corr_details.get('corrected', 0) > 0:
                result.corrected_samples += 1
                
                if verbose and agent_a_cer > 0 and cer < agent_a_cer:
                    print(f"\n[成功纠错] Sample {i}:")
                    print(f"  Agent A: '{agent_a}' (CER: {agent_a_cer:.4f})")
                    print(f"  System:  '{pred}' (CER: {cer:.4f})")
                    print(f"  GT:      '{ref}'")
                    print(f"  Corrected: {corr_details.get('corrected', 0)}/{corr_details.get('total_wrong_in_a', 0)}")
    
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
    
    # ========== Router 效率指标 ==========
    if is_hard_samples:
        result.router_call_rate = sum(is_hard_samples) / len(is_hard_samples)
        if actual_hard_samples > 0:
            result.hard_sample_recall = router_detected_hard / actual_hard_samples
    
    # ========== Error Type 聚合 (Data Protocol v1.0) ==========
    if metadata_list:
        error_type_stats = aggregate_by_error_type(
            predictions=predictions,
            references=references,
            agent_a_texts=agent_a_texts,
            metadata_list=metadata_list
        )
        result.error_type_stats = error_type_stats
    
    return result


def aggregate_by_error_type(
    predictions: List[str],
    references: List[str],
    agent_a_texts: List[str],
    metadata_list: List[Dict]
) -> Dict:
    """
    按 error_type 聚合统计指标
    
    Args:
        predictions: 系统预测列表
        references: 真值列表
        agent_a_texts: Agent A 识别结果列表
        metadata_list: 包含 error_type 的元数据列表
        
    Returns:
        Dict: 按 error_type 分组的统计信息
        {
            "similar_char": {"count": 10, "cer": 0.05, "ocr_r": 0.02, "correction_rate": 0.8},
            "grammar_omission": {"count": 5, "cer": 0.08, ...},
            ...
        }
    """
    # 按 error_type 分组
    grouped = defaultdict(lambda: {
        'predictions': [],
        'references': [],
        'agent_a_texts': []
    })
    
    for i, meta in enumerate(metadata_list):
        error_type = meta.get('error_type', 'unknown') or 'unknown'
        
        if i < len(predictions):
            grouped[error_type]['predictions'].append(predictions[i])
        if i < len(references):
            grouped[error_type]['references'].append(references[i])
        if agent_a_texts and i < len(agent_a_texts):
            grouped[error_type]['agent_a_texts'].append(agent_a_texts[i])
    
    # 计算每个 error_type 的指标
    stats = {}
    
    for error_type, data in grouped.items():
        preds = data['predictions']
        refs = data['references']
        a_texts = data['agent_a_texts']
        
        if not refs:
            continue
        
        count = len(refs)
        
        # 计算 CER
        cers = [calculate_cer(p, r) for p, r in zip(preds, refs)]
        avg_cer = sum(cers) / len(cers) if cers else 0.0
        
        # 计算 OCR-R 和 Correction Rate
        ocr_rs = []
        corr_rates = []
        
        if a_texts:
            for pred, ref, a_text in zip(preds, refs, a_texts):
                ocr_r, _ = calculate_ocr_r(a_text, pred, ref)
                corr_rate, _ = calculate_correction_rate(a_text, pred, ref)
                ocr_rs.append(ocr_r)
                corr_rates.append(corr_rate)
        
        stats[error_type] = {
            "count": count,
            "cer": round(avg_cer, 4),
            "ocr_r": round(sum(ocr_rs) / len(ocr_rs), 4) if ocr_rs else 0.0,
            "correction_rate": round(sum(corr_rates) / len(corr_rates), 4) if corr_rates else 0.0,
            "exact_match": sum(1 for p, r in zip(preds, refs) if p == r),
        }
    
    return stats


# =============================================================================
# 文件加载
# =============================================================================

@dataclass
class InferenceRecord:
    """
    推理记录 (符合 Data Protocol v1.0)
    """
    id: str = ""
    image_path: str = ""
    agent_a_text: str = ""
    agent_b_text: str = ""
    final_text: str = ""
    gt_text: str = ""
    is_hard: bool = False
    router_decision: str = "pass"
    suspicious_index: int = -1  # 0-indexed
    error_type: str = ""
    source: str = ""
    difficulty: str = "normal"


def load_predictions(file_path: str) -> Tuple[List[str], List[str], List[str], List[bool], List[Dict]]:
    """
    加载推理结果文件 (支持 Data Protocol v1.0 嵌套格式)
    
    支持格式:
    1. Data Protocol v1.0 (嵌套格式):
       {
         "id": str,
         "image": str,
         "gt_text": str,
         "agent_a": {"text": str, ...},
         "router": {"is_hard": bool, ...},
         "agent_b": {"text": str, ...},
         "metadata": {"error_type": str, ...}
       }
    
    2. 扁平格式 (兼容旧版):
       {"agent_a_text": str, "final_text": str, "gt_text": str, ...}
    
    Returns:
        (final_texts, gt_texts, agent_a_texts, is_hard_samples, metadata_list)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    final_texts = []
    gt_texts = []
    agent_a_texts = []
    is_hard_samples = []
    metadata_list = []  # 新增: 存储 metadata 用于 error_type 聚合
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    def parse_item(item: Dict) -> InferenceRecord:
        """解析单条记录 (支持嵌套和扁平格式)"""
        record = InferenceRecord()
        
        # 检测是否为嵌套格式 (Data Protocol v1.0)
        if 'agent_a' in item and isinstance(item['agent_a'], dict):
            # === 嵌套格式 ===
            record.id = item.get('id', '')
            record.image_path = item.get('image', '')
            record.gt_text = item.get('gt_text', '')
            
            # Agent A
            agent_a = item.get('agent_a', {})
            record.agent_a_text = agent_a.get('text', '')
            record.suspicious_index = agent_a.get('suspicious_index', -1)
            
            # Router
            router = item.get('router', {})
            record.is_hard = router.get('is_hard', False)
            record.router_decision = router.get('decision', 'pass')
            
            # Agent B
            agent_b = item.get('agent_b', {})
            record.agent_b_text = agent_b.get('text', '')
            
            # Final text: 优先使用 agent_b.text (如果 is_hard)，否则使用 agent_a.text
            if record.is_hard and record.agent_b_text:
                record.final_text = record.agent_b_text
            else:
                record.final_text = record.agent_a_text
            
            # Metadata
            metadata = item.get('metadata', {})
            record.error_type = metadata.get('error_type', '')
            record.source = metadata.get('source', '')
            record.difficulty = metadata.get('difficulty', 'normal')
        else:
            # === 扁平格式 (兼容旧版) ===
            record.id = item.get('id', '')
            record.image_path = item.get('image_path', item.get('image', ''))
            
            # Final text
            record.final_text = item.get('final_text', 
                                item.get('prediction', 
                                item.get('pred', '')))
            
            # GT text
            record.gt_text = item.get('gt_text',
                             item.get('reference',
                             item.get('gt',
                             item.get('ground_truth', ''))))
            
            # Agent A text
            record.agent_a_text = item.get('agent_a_text',
                                  item.get('agent_a', ''))
            
            # Router
            record.is_hard = item.get('is_hard', False)
            
            # Metadata
            record.error_type = item.get('error_type', '')
            record.source = item.get('source', '')
        
        return record
    
    # 解析内容
    items = []
    
    # 尝试 JSON 数组格式
    if content.startswith('['):
        items = json.loads(content)
    # 尝试 JSONL 格式
    elif content.startswith('{'):
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效 JSON 行: {e}")
    # TXT 格式 (仅预测)
    else:
        final_texts = [line for line in content.split('\n') if line.strip()]
        return (final_texts, None, None, None, None)
    
    # 解析所有记录
    for item in items:
        record = parse_item(item)
        
        final_texts.append(record.final_text)
        gt_texts.append(record.gt_text)
        agent_a_texts.append(record.agent_a_text)
        is_hard_samples.append(record.is_hard)
        metadata_list.append({
            'id': record.id,
            'error_type': record.error_type,
            'source': record.source,
            'difficulty': record.difficulty,
            'suspicious_index': record.suspicious_index,
        })
    
    return (
        final_texts, 
        gt_texts if any(gt_texts) else None, 
        agent_a_texts if any(agent_a_texts) else None,
        is_hard_samples if any(is_hard_samples) else None,
        metadata_list if any(m.get('error_type') for m in metadata_list) else None
    )


# =============================================================================
# 测试函数
# =============================================================================

def run_tests():
    """
    运行完整测试套件
    
    验证核心指标计算的正确性
    """
    print("=" * 65)
    print("  L2W1 评估指标测试 (Evaluation Metrics Test Suite)")
    print("=" * 65)
    
    all_passed = True
    
    # ========== Test 1: CER with S, D, I ==========
    print("\n[1] CER 详细计算测试 (S, D, I):")
    print("-" * 65)
    
    cer_cases = [
        # (pred, gt, expected_cer, expected_S, expected_D, expected_I)
        ("abc", "abc", 0.0, 0, 0, 0),          # 完全匹配
        ("abd", "abc", 1/3, 1, 0, 0),          # 1 替换
        ("ab", "abc", 1/3, 0, 0, 1),           # 1 插入 (gt 有但 pred 没有)
        ("abcd", "abc", 1/3, 0, 1, 0),         # 1 删除 (pred 有但 gt 没有)
        ("xyz", "abc", 1.0, 3, 0, 0),          # 全部替换
        ("", "abc", 1.0, 0, 0, 3),             # pred 为空
        ("abc", "", 1.0, 0, 3, 0),             # gt 为空 (特殊处理)
    ]
    
    for pred, gt, expected_cer, exp_s, exp_d, exp_i in cer_cases:
        cer, ops = calculate_cer(pred, gt, return_details=True)
        
        # 注意：空 gt 的情况 CER 计算特殊
        if len(gt) == 0:
            passed = abs(cer - expected_cer) < 0.01
        else:
            passed = (abs(cer - expected_cer) < 0.01 and 
                      ops.substitutions == exp_s and
                      ops.deletions == exp_d and
                      ops.insertions == exp_i)
        
        status = "[PASS]" if passed else "[FAIL]"
        all_passed = all_passed and passed
        
        print(f"  {status} pred='{pred}', gt='{gt}'")
        print(f"       CER={cer:.4f} (S={ops.substitutions}, D={ops.deletions}, I={ops.insertions})")
    
    # ========== Test 2: OCR-R (过度纠错率) ==========
    print("\n[2] OCR-R 测试 (过度纠错检测):")
    print("-" * 65)
    
    ocr_r_cases = [
        # (Agent A, System Output, Ground Truth, Expected OCR-R, Description)
        ("在时间的未尾", "在时间的末尾", "在时间的末尾", 0.0, "正确修正"),
        ("中国科学院", "中国科学院", "中国科学院", 0.0, "无需修正"),
        ("深度学习", "身度学习", "深度学习", 0.25, "过度纠错 (深->身)"),
        ("计算机视觉", "计算机视觉", "计算机视觉", 0.0, "完全正确"),
        ("ABC", "XBC", "ABC", 1/3, "过度纠错 (A->X)"),
        ("ABC", "ABX", "ABC", 1/3, "过度纠错 (C->X)"),
        ("人工智能", "人工智能技", "人工智能", 0.0, "插入不影响 OCR-R"),
        ("ABCD", "ABCD", "ABCD", 0.0, "完全正确"),
    ]
    
    for agent_a, system, gt, expected_ocr_r, desc in ocr_r_cases:
        ocr_r, details = calculate_ocr_r(agent_a, system, gt)
        
        passed = abs(ocr_r - expected_ocr_r) < 0.05
        status = "[PASS]" if passed else "[FAIL]"
        all_passed = all_passed and passed
        
        print(f"  {status} {desc}")
        print(f"       A='{agent_a}' -> S='{system}' | GT='{gt}'")
        print(f"       OCR-R={ocr_r:.4f} (期望: {expected_ocr_r:.4f})")
        print(f"       正确={details.get('total_correct_in_a', 0)}, 过度纠错={details.get('overcorrected', 0)}")
    
    # ========== Test 3: Correction Rate ==========
    print("\n[3] Correction Rate 测试 (纠错能力):")
    print("-" * 65)
    
    cr_cases = [
        # (Agent A, System, GT, Expected CR, Description)
        ("在时间的未尾", "在时间的末尾", "在时间的末尾", 1.0, "完全纠正"),
        ("中国科学院", "中国科学院", "中国科学院", 1.0, "无需纠正"),
        ("深度学习", "深度学习", "深度学习", 1.0, "保持正确"),
        ("ABC", "ABD", "ABD", 1.0, "成功纠正 C->D"),
        ("XYZ", "ABC", "ABC", 1.0, "完全重写正确"),
    ]
    
    for agent_a, system, gt, expected_cr, desc in cr_cases:
        cr, details = calculate_correction_rate(agent_a, system, gt)
        
        passed = abs(cr - expected_cr) < 0.05
        status = "[PASS]" if passed else "[FAIL]"
        all_passed = all_passed and passed
        
        print(f"  {status} {desc}")
        print(f"       CR={cr:.4f} (期望: {expected_cr:.4f})")
    
    # ========== Test 4: 批量评估 ==========
    print("\n[4] 批量评估测试:")
    print("-" * 65)
    
    # 构造测试数据
    predictions = ["在时间的末尾", "中国科学院", "身度学习", "计算机视觉"]
    references = ["在时间的末尾", "中国科学院", "深度学习", "计算机视觉"]
    agent_a_texts = ["在时间的未尾", "中国科学院", "深度学习", "计算机视觉"]
    
    result = evaluate_batch(predictions, references, agent_a_texts, verbose=False)
    
    # 打印完整报告
    result.print_summary()
    
    # 验证关键指标
    print("\n  关键指标验证:")
    
    # CER 应该较低
    cer_pass = result.cer < 0.1
    print(f"    CER < 0.1: {'[PASS]' if cer_pass else '[FAIL]'} (实际: {result.cer:.4f})")
    all_passed = all_passed and cer_pass
    
    # OCR-R 应该存在（因为有过度纠错样本）
    ocr_r_pass = result.ocr_r > 0
    print(f"    OCR-R > 0: {'[PASS]' if ocr_r_pass else '[FAIL]'} (实际: {result.ocr_r:.4f})")
    all_passed = all_passed and ocr_r_pass
    
    # 过度纠错样本数
    overcorr_pass = result.overcorrected_samples == 1
    print(f"    过度纠错样本 == 1: {'[PASS]' if overcorr_pass else '[FAIL]'} (实际: {result.overcorrected_samples})")
    all_passed = all_passed and overcorr_pass
    
    # ========== Test 5: 边界条件 ==========
    print("\n[5] 边界条件测试:")
    print("-" * 65)
    
    edge_cases = [
        ("", "", "", "两个空字符串"),
        ("a", "a", "a", "单字符完全匹配"),
        ("", "a", "a", "pred 为空"),
        ("a" * 100, "a" * 100, "a" * 100, "100字符匹配"),
    ]
    
    for agent_a, system, gt, desc in edge_cases:
        try:
            ocr_r, _ = calculate_ocr_r(agent_a, system, gt)
            cer = calculate_cer(system, gt)
            print(f"  [PASS] {desc}: CER={cer:.4f}, OCR-R={ocr_r:.4f}")
        except Exception as e:
            print(f"  [FAIL] {desc}: {e}")
            all_passed = False
    
    # ========== 总结 ==========
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
        description="L2W1 核心评价指标计算脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估预测文件
    python evaluate.py --predictions ./outputs/predictions.json
    
    # 运行测试
    python evaluate.py --test
    
    # 详细输出
    python evaluate.py --predictions ./outputs/predictions.json --verbose
        """
    )
    
    parser.add_argument("--predictions", type=str, default="",
                        help="预测结果文件路径 (JSON/JSONL)")
    parser.add_argument("--references", type=str, default="",
                        help="真值文件路径（如果不在预测文件中）")
    parser.add_argument("--output", type=str, default="",
                        help="输出评估结果到文件")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细信息")
    parser.add_argument("--test", action="store_true",
                        help="运行测试")
    
    return parser.parse_args()


def generate_evaluation_report(result: EvaluationResult, output_path: str):
    """
    生成评估报告文件 (evaluation_report.json)
    
    按照 module5_evaluate_spec.md 规范输出
    """
    report = result.to_dict()
    
    # 添加时间戳
    from datetime import datetime
    report['timestamp'] = datetime.now().isoformat()
    report['version'] = 'L2W1 v5.0'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估报告已保存至: {output_path}")


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
        print("  python evaluate.py --predictions ./outputs/inference_results.jsonl")
        print("  python evaluate.py --test")
        return
    
    # 加载数据
    try:
        final_texts, gt_texts, agent_a_texts, is_hard_samples, metadata_list = load_predictions(args.predictions)
    except Exception as e:
        print(f"加载文件失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 如果 gt_texts 为空，尝试从单独文件加载
    if not gt_texts and args.references:
        with open(args.references, 'r', encoding='utf-8') as f:
            gt_texts = [line.strip() for line in f]
    
    if not gt_texts:
        print("错误: 未找到真值数据 (gt_text)")
        print("请确保预测文件包含 'gt_text' 字段，或使用 --references 指定真值文件")
        return
    
    if len(final_texts) != len(gt_texts):
        print(f"错误: 预测数量 ({len(final_texts)}) 与真值数量 ({len(gt_texts)}) 不匹配")
        return
    
    # 执行评估
    print("\n正在评估...")
    
    result = evaluate_batch(
        predictions=final_texts,
        references=gt_texts,
        agent_a_texts=agent_a_texts,
        is_hard_samples=is_hard_samples,
        metadata_list=metadata_list,
        verbose=args.verbose
    )
    
    # 打印评估报告
    result.print_summary()
    
    # 保存评估报告
    if args.output:
        generate_evaluation_report(result, args.output)
    else:
        # 默认保存到与输入同目录
        default_output = Path(args.predictions).parent / "evaluation_report.json"
        generate_evaluation_report(result, str(default_output))


if __name__ == "__main__":
    main()

"""
L2W1 自动化数据处理流水线

核心功能:
1. 加载 HCTR 数据集（SCUT-HCCDoc, VisCGEC, CASIA 等）
2. 使用 Agent A (TextRecognizerWithLogits) 进行全量推理
3. 基于 difflib 进行难例挖掘和错误索引定位
4. 生成 Agent B SFT 训练数据（JSONL 格式）

技术要点:
- 零切割原则：直接输入原始行图像，不做字符级切割
- 完美 Router 模拟：使用 difflib 计算绝对正确的错误位置
- EIP 数据格式：符合显式索引提示规范

Usage:
    python data_pipeline.py --data_dir ./data/raw/scut_hccdoc \
                            --output_dir ./data/sft \
                            --batch_size 16 \
                            --max_cer 0.3
"""

import os
import sys
import json
import argparse
import difflib
from pathlib import Path

# 尝试导入 editdistance，如果失败则使用内置实现
try:
    import editdistance

    def levenshtein_distance(s1: str, s2: str) -> int:
        return editdistance.eval(s1, s2)
except ImportError:

    def levenshtein_distance(s1: str, s2: str) -> int:
        """计算 Levenshtein 编辑距离（内置实现）"""
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


from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

try:
    from tqdm import tqdm
except ImportError:
    # 简单的 tqdm 替代
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# 数据结构定义
# =============================================================================


@dataclass
class DataSample:
    """
    数据样本

    符合 L2W1 Master Data Protocol v2.0
    支持 Train/Val/Test 物理隔离
    """

    id: str  # 样本唯一 ID
    image_path: str  # 图像路径 (相对或绝对)
    ground_truth: str  # 真值文本 (gt_text)
    prediction: str = ""  # Agent A 预测文本
    confidence: float = 0.0  # Agent A 置信度
    cer: float = 0.0  # 字符错误率
    error_index: int = -1  # 第一个错误位置 (0-indexed)
    error_char_pred: str = ""  # 预测的错误字符
    error_char_gt: str = ""  # 真值的错误字符

    # 扩展字段 (来自 Data Protocol v2.0)
    source: str = ""  # 数据来源 (viscgec, scut, casia 等)
    split: str = ""  # 数据集划分 (train, val, test) [v2.0 新增]
    error_type: str = ""  # 错误类型 (grammar_omission, similar_char 等)
    difficulty: str = "normal"  # 难度评估


@dataclass
class SFTConversation:
    """
    SFT 对话格式 (符合 Data Protocol v2.0)

    输出结构:
    {
        "id": str,
        "image": str,
        "gt_text": str,
        "agent_a": { "text", "suspicious_index", "suspicious_char" },
        "conversations": [{"from": "user", "value": ...}, {"from": "assistant", "value": ...}],
        "metadata": { "source", "split", "error_type", "difficulty" }
    }
    """

    id: str
    image: str
    gt_text: str
    conversations: List[Dict[str, str]]
    agent_a_text: str = ""
    suspicious_index: int = -1  # 0-indexed
    suspicious_char: str = ""
    source: str = ""
    split: str = ""  # train, val, test [v2.0 新增]
    error_type: str = ""
    difficulty: str = "normal"

    def to_dict(self) -> Dict:
        """转换为符合 Data Protocol v2.0 的嵌套结构"""
        return {
            "id": self.id,
            "image": self.image,
            "gt_text": self.gt_text,
            "agent_a": {
                "text": self.agent_a_text,
                "suspicious_index": self.suspicious_index,  # 0-indexed
                "suspicious_char": self.suspicious_char,
            },
            "conversations": self.conversations,
            "metadata": {
                "source": self.source,
                "split": self.split,  # [v2.0 新增]
                "error_type": self.error_type,
                "difficulty": self.difficulty,
                "gt_char_len": len(self.gt_text),
            },
        }


@dataclass
class PipelineConfig:
    """
    流水线配置 (Data Protocol v2.0)

    支持 Train/Val/Test 分割
    """

    # 数据路径
    data_dir: str = "./data/raw"
    output_dir: str = "./data/sft"

    # 数据集分割 (v2.0 新增)
    split: str = "train"  # 目标分割 (train, val, test, all)
    strict_validation: bool = True  # 严格验证图像存在

    # 推理配置
    batch_size: int = 16
    use_gpu: bool = True

    # 过滤条件
    max_cer: float = 0.3  # 最大 CER 阈值，过滤识别过差的样本
    min_text_length: int = 2  # 最小文本长度
    max_text_length: int = 100  # 最大文本长度

    # 输出配置
    output_filename: str = ""  # 留空则自动根据 split 生成

    # Agent A 模型配置
    rec_model_dir: str = "./models/agent_a_ppocr"  # PP-OCRv5 模型目录
    rec_image_shape: str = "3,48,320"
    rec_algorithm: str = "SVTR_LCNet"
    rec_char_dict_path: str = "./ppocr/utils/ppocrv5_dict.txt"  # PP-OCRv5 官方字典

    def get_output_filename(self) -> str:
        """获取输出文件名（根据 split 自动生成）"""
        if self.output_filename:
            return self.output_filename

        # 根据 split 生成默认文件名
        split_filenames = {
            "train": "agent_b_train.jsonl",
            "val": "agent_b_val.jsonl",
            "test": "agent_b_test.jsonl",
            "all": "agent_b_all.jsonl",
        }
        return split_filenames.get(self.split, "agent_b_train.jsonl")


# =============================================================================
# 数据加载器
# =============================================================================


class HCTRDatasetLoader:
    """
    HCTR 数据集加载器 (Data Protocol v2.0)

    支持 Train/Val/Test 物理隔离的数据集结构:

    目录结构:
        data/raw/[dataset_name]/
        ├── images/              # 图像文件 (可有子目录)
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── train.jsonl          # 训练集元数据
        ├── val.jsonl            # 验证集元数据
        └── test.jsonl           # 测试集元数据

    JSONL 格式 (Data Protocol v2.0):
        {"id": "viscgec_001", "image": "images/train/001.png", "gt_text": "...", "source": "viscgec"}

    兼容模式:
        - metadata.jsonl: 旧版单文件格式
        - labels.txt: 通用格式
        - SCUT/CASIA: 图像同名 txt 格式
    """

    SUPPORTED_FORMATS = ["protocol_v2", "protocol", "scut", "casia", "generic", "auto"]
    SUPPORTED_SPLITS = ["train", "val", "test", "all"]

    def __init__(
        self,
        data_dir: str,
        format: str = "auto",
        split: str = "all",
        strict_validation: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录路径
            format: 数据格式类型 ('protocol_v2', 'protocol', 'scut', 'casia', 'generic', 'auto')
            split: 数据集划分 ('train', 'val', 'test', 'all')
            strict_validation: 是否严格验证图像文件存在
        """
        self.data_dir = Path(data_dir)
        self.format = format
        self.split = split.lower()
        self.strict_validation = strict_validation

        # 统计信息
        self.stats = {
            "total_lines": 0,
            "loaded": 0,
            "skipped_no_image": 0,
            "skipped_invalid_json": 0,
            "skipped_missing_fields": 0,
        }

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        if self.split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split: {split}. Use one of {self.SUPPORTED_SPLITS}"
            )

    def _detect_format(self) -> str:
        """自动检测数据格式"""
        # 优先检测 Data Protocol v2.0 (train.jsonl, val.jsonl, test.jsonl)
        split_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
        if any((self.data_dir / f).exists() for f in split_files):
            return "protocol_v2"

        # 检测 Data Protocol v1.0 (metadata.jsonl)
        for jsonl_name in ["metadata.jsonl", "data.jsonl", "samples.jsonl"]:
            if (self.data_dir / jsonl_name).exists():
                return "protocol"

        # 检查 labels.txt (通用格式)
        if (self.data_dir / "labels.txt").exists() or (
            self.data_dir / "label.txt"
        ).exists():
            return "generic"

        # 检查 SCUT 格式 (图像同名 txt)
        image_files = list(self.data_dir.glob("*.jpg")) + list(
            self.data_dir.glob("*.png")
        )
        if image_files:
            txt_file = image_files[0].with_suffix(".txt")
            if txt_file.exists():
                return "scut"

        # 默认通用格式
        return "generic"

    def _get_split_files(self) -> List[Tuple[Path, str]]:
        """
        获取要加载的 JSONL 文件列表

        Returns:
            List[(file_path, split_name)]
        """
        files = []

        if self.split == "all":
            # 加载所有存在的分割文件
            for split_name in ["train", "val", "test"]:
                jsonl_path = self.data_dir / f"{split_name}.jsonl"
                if jsonl_path.exists():
                    files.append((jsonl_path, split_name))

            # 如果没有找到分割文件，尝试旧版格式
            if not files:
                for name in ["metadata.jsonl", "data.jsonl", "samples.jsonl"]:
                    if (self.data_dir / name).exists():
                        files.append((self.data_dir / name, "unknown"))
                        break
        else:
            # 加载指定的分割
            jsonl_path = self.data_dir / f"{self.split}.jsonl"
            if jsonl_path.exists():
                files.append((jsonl_path, self.split))
            else:
                print(f"[Warning] Split file not found: {jsonl_path}")
                # 尝试从旧版格式加载
                for name in ["metadata.jsonl", "data.jsonl", "samples.jsonl"]:
                    if (self.data_dir / name).exists():
                        print(f"[INFO] Falling back to legacy format: {name}")
                        files.append((self.data_dir / name, self.split))
                        break

        return files

    def load(self) -> Generator[DataSample, None, None]:
        """
        加载数据集

        Yields:
            DataSample: 数据样本 (包含 split 字段)
        """
        if self.format == "auto":
            self.format = self._detect_format()

        print(f"[INFO] Detected format: {self.format}")
        print(f"[INFO] Loading split: {self.split}")

        if self.format in ["protocol_v2", "protocol"]:
            yield from self._load_protocol_v2_format()
        elif self.format == "scut":
            yield from self._load_scut_format()
        elif self.format == "casia":
            yield from self._load_casia_format()
        else:
            yield from self._load_generic_format()

        # 打印统计信息
        self._print_stats()

    def _load_protocol_v2_format(self) -> Generator[DataSample, None, None]:
        """
        加载 L2W1 Data Protocol v2.0 格式

        支持:
        - train.jsonl, val.jsonl, test.jsonl (v2.0 推荐)
        - metadata.jsonl (v1.0 兼容)

        JSONL 格式:
        {"id": "viscgec_001", "image": "images/001.png", "gt_text": "文本", "source": "viscgec"}
        或嵌套格式 (推理结果):
        {"id": "...", "image": "...", "gt_text": "...", "agent_a": {...}, "metadata": {...}}
        """
        split_files = self._get_split_files()

        if not split_files:
            print(f"[ERROR] No JSONL files found in {self.data_dir}")
            return

        for jsonl_path, split_name in split_files:
            print(f"[INFO] Loading: {jsonl_path.name} (split={split_name})")

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    self.stats["total_lines"] += 1
                    line = line.strip()
                    if not line:
                        continue

                    # 解析 JSON
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.stats["skipped_invalid_json"] += 1
                        if self.strict_validation:
                            print(
                                f"[Warning] {jsonl_path.name}:{line_num}: Invalid JSON - {e}"
                            )
                        continue

                    # 解析样本
                    sample = self._parse_protocol_item(
                        item, split_name, jsonl_path, line_num
                    )
                    if sample:
                        self.stats["loaded"] += 1
                        yield sample

    def _parse_protocol_item(
        self, item: Dict, default_split: str, jsonl_path: Path, line_num: int
    ) -> Optional[DataSample]:
        """
        解析单条 Protocol 格式记录

        支持扁平格式和嵌套格式
        """
        # 提取基本字段 (支持嵌套和扁平)
        sample_id = item.get("id", f"sample_{line_num:06d}")
        image_rel_path = item.get("image", "")
        gt_text = item.get("gt_text", "")

        # 从 metadata 提取扩展字段 (嵌套格式)
        metadata = item.get("metadata", {})
        source = metadata.get("source", item.get("source", ""))
        split = metadata.get("split", item.get("split", default_split))
        error_type = metadata.get("error_type", item.get("error_type", ""))
        difficulty = metadata.get("difficulty", item.get("difficulty", "normal"))

        # 验证必需字段
        if not image_rel_path or not gt_text:
            self.stats["skipped_missing_fields"] += 1
            if self.strict_validation:
                print(
                    f"[Warning] {jsonl_path.name}:{line_num}: Missing 'image' or 'gt_text'"
                )
            return None

        # 解析图像路径 (支持多种相对路径基准)
        image_path = self._resolve_image_path(image_rel_path, jsonl_path.parent)

        if image_path is None:
            self.stats["skipped_no_image"] += 1
            if self.strict_validation:
                print(
                    f"[Warning] {jsonl_path.name}:{line_num}: Image not found - {image_rel_path}"
                )
            return None

        return DataSample(
            id=sample_id,
            image_path=str(image_path),
            ground_truth=gt_text,
            source=source,
            split=split,
            error_type=error_type,
            difficulty=difficulty,
        )

    def _resolve_image_path(
        self, image_rel_path: str, jsonl_dir: Path = None
    ) -> Optional[Path]:
        """
        解析图像路径，支持多种格式

        解析顺序 (智能路径解析，避免重复拼接):
        1. 绝对路径
        2. 直接相对于项目根目录 (CWD) - 处理 JSONL 中已包含完整相对路径的情况
        3. 相对于 data_dir
        4. 相对于 jsonl 文件所在目录
        5. 仅文件名 (在常见位置搜索)

        Args:
            image_rel_path: 相对路径或绝对路径
            jsonl_dir: JSONL 文件所在目录 (用于相对路径解析)

        Returns:
            解析后的完整路径，如果不存在则返回 None
        """
        if jsonl_dir is None:
            jsonl_dir = self.data_dir

        # 标准化路径分隔符 (跨平台兼容)
        image_rel_path = os.path.normpath(image_rel_path)

        # 处理绝对路径
        if Path(image_rel_path).is_absolute():
            abs_path = Path(image_rel_path)
            if abs_path.exists():
                return abs_path
            self._log_path_attempt(image_rel_path, [abs_path], "绝对路径不存在")
            return None

        # ========== 智能路径解析 ==========
        # 核心思路: 先尝试直接路径，再尝试拼接路径

        candidates = []

        # 优先级 1: 直接相对于 CWD (项目根目录)
        # 处理 JSONL 中 image 字段已包含完整路径的情况
        # 例如: "data/raw/viscgec/images/train/001.png"
        direct_path = Path(image_rel_path)
        candidates.append(direct_path)

        # 优先级 2: 检测并避免路径重复拼接
        # 如果 image_rel_path 已包含 data_dir 的部分路径，则智能提取
        data_dir_str = str(self.data_dir).replace("\\", "/").strip("./")
        image_path_str = image_rel_path.replace("\\", "/").strip("./")

        if image_path_str.startswith(data_dir_str):
            # 路径已包含 data_dir，直接使用
            candidates.append(Path(image_rel_path))
        else:
            # 路径不包含 data_dir，尝试拼接
            candidates.append(self.data_dir / image_rel_path)

        # 优先级 3: 相对于 JSONL 文件所在目录
        candidates.append(jsonl_dir / image_rel_path)

        # 优先级 4: 相对于 data_dir/images/
        candidates.append(self.data_dir / "images" / image_rel_path)

        # 优先级 5: 仅使用文件名，在常见位置搜索
        filename_only = Path(image_rel_path).name
        candidates.extend(
            [
                self.data_dir / filename_only,
                self.data_dir / "images" / filename_only,
                jsonl_dir / filename_only,
            ]
        )

        # 去重并保持顺序
        seen = set()
        unique_candidates = []
        for c in candidates:
            c_normalized = os.path.normpath(str(c))
            if c_normalized not in seen:
                seen.add(c_normalized)
                unique_candidates.append(Path(c_normalized))

        # 尝试各种扩展名
        extensions = ["", ".jpg", ".png", ".jpeg", ".bmp", ".JPG", ".PNG", ".JPEG"]

        for candidate in unique_candidates:
            for ext in extensions:
                full_path = Path(str(candidate) + ext) if ext else candidate
                if full_path.exists():
                    return full_path

        # 失败时记录详细调试信息
        self._log_path_attempt(
            image_rel_path, unique_candidates, "所有候选路径均不存在"
        )
        return None

    def _log_path_attempt(self, original_path: str, candidates: list, reason: str):
        """
        记录路径解析失败的调试信息

        仅在 strict_validation=True 且首次失败时打印详细日志
        """
        if not self.strict_validation:
            return

        # 限制日志输出，避免刷屏
        if not hasattr(self, "_path_log_count"):
            self._path_log_count = 0

        if self._path_log_count < 5:
            print(f"\n[DEBUG] 路径解析失败详情:")
            print(f"  原始路径: {original_path}")
            print(f"  data_dir: {self.data_dir.absolute()}")
            print(f"  尝试过的路径:")
            for i, c in enumerate(candidates[:5], 1):
                abs_c = Path(c).absolute()
                print(f"    {i}. {abs_c}")
            print(f"  失败原因: {reason}")
            self._path_log_count += 1

            if self._path_log_count == 5:
                print("  [后续相同错误将被静默，避免刷屏]")

    def _print_stats(self):
        """打印加载统计信息"""
        print(f"\n[STATS] Data Loading Summary:")
        print(f"  Total lines:       {self.stats['total_lines']}")
        print(f"  Successfully loaded: {self.stats['loaded']}")
        print(f"  Skipped (no image):  {self.stats['skipped_no_image']}")
        print(f"  Skipped (invalid JSON): {self.stats['skipped_invalid_json']}")
        print(f"  Skipped (missing fields): {self.stats['skipped_missing_fields']}")

    def _load_generic_format(self) -> Generator[DataSample, None, None]:
        """
        加载通用格式

        格式: labels.txt 每行为 "image_name\ttext" 或 "image_name text"
        """
        # 查找标签文件
        label_file = None
        for name in ["labels.txt", "label.txt", "gt.txt", "annotation.txt"]:
            if (self.data_dir / name).exists():
                label_file = self.data_dir / name
                break

        if label_file is None:
            # 尝试查找图像目录
            image_dir = (
                self.data_dir / "images"
                if (self.data_dir / "images").exists()
                else self.data_dir
            )
            for name in ["labels.txt", "label.txt", "gt.txt"]:
                if (self.data_dir / name).exists():
                    label_file = self.data_dir / name
                    break

        if label_file is None:
            print(f"[Warning] No label file found in {self.data_dir}")
            return

        # 确定图像目录
        image_dir = (
            self.data_dir / "images"
            if (self.data_dir / "images").exists()
            else self.data_dir
        )

        # 读取标签文件
        with open(label_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # 解析行：支持 tab 分隔或空格分隔
                if "\t" in line:
                    parts = line.split("\t", 1)
                else:
                    parts = line.split(" ", 1)

                if len(parts) < 2:
                    continue

                image_name, text = parts[0], parts[1]

                # 查找图像文件
                image_path = None
                for ext in [".jpg", ".png", ".jpeg", ".bmp", ""]:
                    candidate = image_dir / f"{image_name}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                    # 尝试原始名称
                    candidate = image_dir / image_name
                    if candidate.exists():
                        image_path = candidate
                        break

                if image_path is None:
                    continue

                yield DataSample(
                    id=f"sample_{idx:06d}",
                    image_path=str(image_path),
                    ground_truth=text,
                )

    def _load_scut_format(self) -> Generator[DataSample, None, None]:
        """
        加载 SCUT-HCCDoc 格式

        每个图像有对应的同名 txt 文件
        """
        image_files = sorted(
            list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        )

        for idx, image_path in enumerate(image_files):
            txt_path = image_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if text:
                yield DataSample(
                    id=f"scut_{idx:06d}", image_path=str(image_path), ground_truth=text
                )

    def _load_casia_format(self) -> Generator[DataSample, None, None]:
        """加载 CASIA 格式（与 SCUT 类似）"""
        yield from self._load_scut_format()


# =============================================================================
# 错误分析工具
# =============================================================================


class ErrorAnalyzer:
    """
    错误分析器

    使用 difflib 进行预测文本和真值文本的对齐分析
    """

    @staticmethod
    def calculate_cer(pred: str, gt: str) -> float:
        """
        计算字符错误率 (CER)

        Args:
            pred: 预测文本
            gt: 真值文本

        Returns:
            CER 值 (0-1)
        """
        if len(gt) == 0:
            return 1.0 if len(pred) > 0 else 0.0

        distance = levenshtein_distance(pred, gt)
        return min(distance / len(gt), 1.0)

    @staticmethod
    def find_first_error_index(pred: str, gt: str) -> Tuple[int, str, str]:
        """
        找到第一个错误位置

        使用 difflib.SequenceMatcher 进行对齐，找到第一个不匹配的位置

        Args:
            pred: 预测文本
            gt: 真值文本

        Returns:
            Tuple[error_index, pred_char, gt_char]:
                - error_index: 错误位置 (基于预测文本的 0-indexed)
                - pred_char: 预测的字符（如果是插入错误则为空）
                - gt_char: 真值的字符（如果是删除错误则为空）
        """
        if pred == gt:
            return -1, "", ""

        # 使用 SequenceMatcher 进行对齐
        matcher = difflib.SequenceMatcher(None, pred, gt)

        # 获取操作码
        opcodes = matcher.get_opcodes()

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            elif tag == "replace":
                # 替换错误：预测和真值在该位置不同
                return (
                    i1,
                    pred[i1] if i1 < len(pred) else "",
                    gt[j1] if j1 < len(gt) else "",
                )
            elif tag == "delete":
                # 删除错误：预测中多了字符
                return i1, pred[i1] if i1 < len(pred) else "", ""
            elif tag == "insert":
                # 插入错误：预测中少了字符
                return i1, "", gt[j1] if j1 < len(gt) else ""

        # 如果没有找到，返回第一个不同的位置
        for i, (p, g) in enumerate(zip(pred, gt)):
            if p != g:
                return i, p, g

        # 长度不同的情况
        if len(pred) != len(gt):
            return min(len(pred), len(gt)), "", ""

        return -1, "", ""

    @staticmethod
    def get_all_error_indices(pred: str, gt: str) -> List[Tuple[int, str, str, str]]:
        """
        获取所有错误位置

        Returns:
            List[Tuple[index, tag, pred_char, gt_char]]
        """
        if pred == gt:
            return []

        errors = []
        matcher = difflib.SequenceMatcher(None, pred, gt)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            if tag == "replace":
                for offset in range(min(i2 - i1, j2 - j1)):
                    errors.append(
                        (
                            i1 + offset,
                            "replace",
                            pred[i1 + offset] if i1 + offset < len(pred) else "",
                            gt[j1 + offset] if j1 + offset < len(gt) else "",
                        )
                    )
            elif tag == "delete":
                for offset in range(i2 - i1):
                    errors.append(
                        (
                            i1 + offset,
                            "delete",
                            pred[i1 + offset] if i1 + offset < len(pred) else "",
                            "",
                        )
                    )
            elif tag == "insert":
                errors.append((i1, "insert", "", gt[j1:j2]))

        return errors


# =============================================================================
# SFT 数据生成器
# =============================================================================


class SFTGenerator:
    """
    SFT 数据生成器

    生成符合 Agent B 微调格式的对话数据
    """

    # 提示模板
    PROMPT_TEMPLATE = (
        "OCR识别结果为：'{ocr_text}'。\n"
        "系统检测到第 {idx} 个字（即'{char}'）置信度存疑。\n"
        "请结合行级视觉上下文，确认该位置的字符，并输出修正后的整行文本。"
    )

    # 备选模板（当无法定位具体错误时）
    FALLBACK_TEMPLATE = (
        "OCR识别结果为：'{ocr_text}'。\n"
        "系统检测到该行文本可能存在识别错误。\n"
        "请结合行级视觉上下文，输出修正后的整行文本。"
    )

    def __init__(self, output_dir: str, filename: str = "agent_b_train.jsonl"):
        """
        Args:
            output_dir: 输出目录
            filename: 输出文件名
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / filename
        self.samples_written = 0

    def format_prompt(self, sample: DataSample) -> str:
        """
        格式化提示文本

        Args:
            sample: 数据样本

        Returns:
            格式化后的提示文本

        Note:
            索引转换规则:
            - 内部逻辑使用 0-indexed (sample.error_index)
            - Prompt 显示使用 1-indexed (人类可读)
        """
        # 导入统一索引管理工具
        try:
            from modules.utils.indexing import to_display_index
        except ImportError:
            # 回退：手动转换
            def to_display_index(idx):
                return idx + 1

        if sample.error_index >= 0 and sample.error_char_pred:
            # 转换为 1-indexed (人类可读)
            display_index = to_display_index(sample.error_index)

            return self.PROMPT_TEMPLATE.format(
                ocr_text=sample.prediction,
                idx=display_index,
                char=sample.error_char_pred,
            )
        else:
            return self.FALLBACK_TEMPLATE.format(ocr_text=sample.prediction)

    def generate_conversation(self, sample: DataSample) -> SFTConversation:
        """
        生成对话格式数据 (符合 Data Protocol v2.0)

        Args:
            sample: 数据样本

        Returns:
            SFTConversation 对象 (包含嵌套结构和 split 字段)
        """
        prompt = self.format_prompt(sample)

        conversations = [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": sample.ground_truth},
        ]

        # 确定错误类型
        error_type = sample.error_type
        if not error_type and sample.error_index >= 0:
            # 根据错误特征自动推断类型
            if sample.error_char_pred and sample.error_char_gt:
                error_type = "similar_char"  # 形近字替换
            elif sample.error_char_pred and not sample.error_char_gt:
                error_type = "extra_char"  # 多余字符
            elif not sample.error_char_pred and sample.error_char_gt:
                error_type = "missing_char"  # 缺失字符

        return SFTConversation(
            id=sample.id,
            image=sample.image_path,
            gt_text=sample.ground_truth,
            conversations=conversations,
            agent_a_text=sample.prediction,
            suspicious_index=sample.error_index,  # 0-indexed
            suspicious_char=sample.error_char_pred,
            source=sample.source,
            split=sample.split,  # [v2.0 新增]
            error_type=error_type,
            difficulty=sample.difficulty,
        )

    def write_sample(self, sample: DataSample):
        """写入单个样本到 JSONL"""
        conversation = self.generate_conversation(sample)

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")

        self.samples_written += 1

    def write_batch(self, samples: List[DataSample]):
        """批量写入样本"""
        with open(self.output_path, "a", encoding="utf-8") as f:
            for sample in samples:
                conversation = self.generate_conversation(sample)
                f.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
                self.samples_written += 1

    def clear_output(self):
        """清空输出文件"""
        if self.output_path.exists():
            self.output_path.unlink()
        self.samples_written = 0


# =============================================================================
# 主流水线
# =============================================================================


class DataPipeline:
    """
    L2W1 自动化数据处理流水线 (Data Protocol v2.0)

    整合数据加载、推理、错误分析和 SFT 生成
    支持 Train/Val/Test 物理隔离
    """

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: 流水线配置
        """
        self.config = config
        self.error_analyzer = ErrorAnalyzer()

        # 使用自动生成的输出文件名
        output_filename = config.get_output_filename()
        self.sft_generator = SFTGenerator(
            output_dir=config.output_dir, filename=output_filename
        )

        # 统计信息 (按 split 分组)
        self.stats = {
            "total_samples": 0,
            "processed_samples": 0,
            "error_samples": 0,
            "filtered_by_cer": 0,
            "filtered_by_length": 0,
            "sft_samples": 0,
            "split": config.split,
            "by_split": {},  # 按 split 统计
        }

        # Agent A 推理器（延迟初始化）
        self.recognizer = None

    def _init_recognizer(self):
        """初始化 Agent A 推理器"""
        if self.recognizer is not None:
            return

        try:
            # 尝试导入 TextRecognizerWithLogits
            from modules.paddle_engine.predict_rec_modified import (
                TextRecognizerWithLogits,
            )
            import tools.infer.utility as utility

            # 构建参数 - 包含所有 PaddleOCR 推理器需要的参数
            class Args:
                pass

            args = Args()

            # ==================== 识别器核心参数 ====================
            args.rec_model_dir = self.config.rec_model_dir
            args.rec_image_shape = self.config.rec_image_shape
            args.rec_algorithm = self.config.rec_algorithm
            args.rec_char_dict_path = self.config.rec_char_dict_path
            args.rec_batch_num = self.config.batch_size
            args.use_space_char = True
            args.max_text_length = 25
            args.rec_image_inverse = False
            args.drop_score = 0.5
            args.return_word_box = False

            # ==================== GPU 配置 ====================
            args.use_gpu = self.config.use_gpu
            args.gpu_mem = 500  # GPU 内存限制 (MB)
            args.gpu_id = 0  # GPU 设备 ID

            # ==================== 其他硬件加速 ====================
            args.use_xpu = False  # 昆仑 XPU
            args.use_npu = False  # 华为 NPU
            args.use_mlu = False  # 寒武纪 MLU
            args.use_metax_gpu = False  # MetaX GPU
            args.use_gcu = False  # GCU

            # ==================== TensorRT 优化 ====================
            args.use_tensorrt = False
            args.min_subgraph_size = 15
            args.precision = "fp32"
            args.max_batch_size = 10

            # ==================== CPU 优化 ====================
            args.enable_mkldnn = None  # 自动检测
            args.cpu_threads = 10
            args.ir_optim = True

            # ==================== 推理选项 ====================
            args.use_onnx = False
            args.benchmark = False
            args.warmup = False
            args.show_log = True
            args.save_log_path = "./output/"

            self.recognizer = TextRecognizerWithLogits(args)
            print("[INFO] Agent A (TextRecognizerWithLogits) 初始化成功")

        except Exception as e:
            print(f"[WARNING] 无法初始化 Agent A: {e}")
            print("[INFO] 将使用模拟推理模式")
            self.recognizer = None

    def _simulate_inference(self, samples: List[DataSample]) -> List[DataSample]:
        """
        模拟推理（当无法使用真实模型时）

        用于测试流水线逻辑
        """
        import random

        for sample in samples:
            gt = sample.ground_truth

            # 模拟识别错误：随机替换/删除/插入字符
            pred = list(gt)
            if len(pred) > 0 and random.random() < 0.3:  # 30% 概率出错
                error_type = random.choice(["replace", "delete", "insert"])
                pos = random.randint(0, len(pred) - 1)

                if error_type == "replace":
                    # 模拟形近字替换
                    similar_chars = {
                        "未": "末",
                        "末": "未",
                        "己": "已",
                        "已": "己",
                        "土": "士",
                        "士": "土",
                        "日": "曰",
                        "曰": "日",
                        "天": "夭",
                        "夭": "天",
                    }
                    if pred[pos] in similar_chars:
                        pred[pos] = similar_chars[pred[pos]]
                    else:
                        # 随机字符
                        pred[pos] = chr(ord("一") + random.randint(0, 1000))
                elif error_type == "delete" and len(pred) > 1:
                    pred.pop(pos)
                elif error_type == "insert":
                    pred.insert(pos, chr(ord("一") + random.randint(0, 1000)))

            sample.prediction = "".join(pred)
            sample.confidence = random.uniform(0.7, 0.99)

        return samples

    def _run_inference(self, samples: List[DataSample]) -> List[DataSample]:
        """
        运行 Agent A 推理

        Args:
            samples: 待推理的样本列表

        Returns:
            更新了预测结果的样本列表
        """
        if self.recognizer is None:
            return self._simulate_inference(samples)

        try:
            import cv2

            # 加载图像
            images = []
            valid_indices = []
            for i, sample in enumerate(samples):
                img = cv2.imread(sample.image_path)
                if img is not None:
                    images.append(img)
                    valid_indices.append(i)

            if not images:
                return samples

            # 批量推理
            output = self.recognizer(images)
            results = output["results"]

            # 更新样本
            for idx, result in zip(valid_indices, results):
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    samples[idx].prediction = result[0]
                    samples[idx].confidence = result[1]
                elif isinstance(result, dict):
                    samples[idx].prediction = result.get("text", "")
                    samples[idx].confidence = result.get("conf", 0.0)

            return samples

        except Exception as e:
            print(f"[WARNING] 推理失败: {e}")
            return self._simulate_inference(samples)

    def _analyze_errors(self, samples: List[DataSample]) -> List[DataSample]:
        """
        分析错误

        计算 CER 并定位第一个错误位置
        """
        for sample in samples:
            # 计算 CER
            sample.cer = self.error_analyzer.calculate_cer(
                sample.prediction, sample.ground_truth
            )

            # 定位第一个错误
            error_idx, pred_char, gt_char = self.error_analyzer.find_first_error_index(
                sample.prediction, sample.ground_truth
            )
            sample.error_index = error_idx
            sample.error_char_pred = pred_char
            sample.error_char_gt = gt_char

        return samples

    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """
        过滤样本

        - 只保留有错误的样本 (pred != gt)
        - 过滤 CER 过高的样本
        - 过滤长度不合适的样本
        """
        filtered = []

        for sample in samples:
            # 跳过完全正确的样本
            if sample.prediction == sample.ground_truth:
                continue

            self.stats["error_samples"] += 1

            # 过滤 CER 过高的样本
            if sample.cer > self.config.max_cer:
                self.stats["filtered_by_cer"] += 1
                continue

            # 过滤长度不合适的样本
            if (
                len(sample.ground_truth) < self.config.min_text_length
                or len(sample.ground_truth) > self.config.max_text_length
            ):
                self.stats["filtered_by_length"] += 1
                continue

            filtered.append(sample)

        return filtered

    def run(self, data_dir: str = None, data_format: str = "auto", split: str = None):
        """
        运行流水线 (Data Protocol v2.0)

        Args:
            data_dir: 数据目录（如果为 None，使用配置中的路径）
            data_format: 数据格式
            split: 数据集划分（如果为 None，使用配置中的划分）
        """
        data_dir = data_dir or self.config.data_dir
        split = split or self.config.split

        print("=" * 60)
        print("L2W1 自动化数据处理流水线 (Data Protocol v2.0)")
        print("=" * 60)
        print(f"数据目录: {data_dir}")
        print(f"输出目录: {self.config.output_dir}")
        print(f"数据集划分: {split}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"最大 CER: {self.config.max_cer}")
        print()

        # 初始化推理器
        self._init_recognizer()

        # 清空输出文件
        self.sft_generator.clear_output()

        # 加载数据 (使用 split 参数)
        print("[1/4] 加载数据集...")
        loader = HCTRDatasetLoader(
            data_dir,
            format=data_format,
            split=split,
            strict_validation=self.config.strict_validation,
        )

        # 收集所有样本
        all_samples = list(loader.load())
        self.stats["total_samples"] = len(all_samples)

        # 按 split 统计
        split_counts = {}
        for sample in all_samples:
            s = sample.split or "unknown"
            split_counts[s] = split_counts.get(s, 0) + 1
        self.stats["by_split"] = split_counts

        print(f"      共加载 {len(all_samples)} 个样本")
        if split_counts:
            for s, count in split_counts.items():
                print(f"        - {s}: {count}")

        if len(all_samples) == 0:
            print("[ERROR] 未找到有效样本，请检查数据目录和格式")
            return

        # 批量处理
        print("[2/4] 运行 Agent A 推理...")
        batch_size = self.config.batch_size

        for i in tqdm(range(0, len(all_samples), batch_size), desc="推理进度", miniters=100):
            batch = all_samples[i : i + batch_size]

            # 推理
            batch = self._run_inference(batch)
            self.stats["processed_samples"] += len(batch)

            # 错误分析
            batch = self._analyze_errors(batch)

            # 过滤
            filtered_batch = self._filter_samples(batch)

            # 生成 SFT 数据
            if filtered_batch:
                self.sft_generator.write_batch(filtered_batch)
                self.stats["sft_samples"] += len(filtered_batch)

        # 输出统计
        print()
        print("[3/4] 生成 SFT 数据...")
        print(f"      输出文件: {self.sft_generator.output_path}")

        print()
        print("[4/4] 统计信息:")
        print(f"      目标划分: {split}")
        print(f"      总样本数: {self.stats['total_samples']}")
        print(f"      处理样本数: {self.stats['processed_samples']}")
        print(f"      错误样本数: {self.stats['error_samples']}")
        print(f"      CER 过滤: {self.stats['filtered_by_cer']}")
        print(f"      长度过滤: {self.stats['filtered_by_length']}")
        print(f"      SFT 样本数: {self.stats['sft_samples']}")

        # 保存统计信息
        stats_filename = (
            f"pipeline_stats_{split}.json" if split != "all" else "pipeline_stats.json"
        )
        stats_path = self.sft_generator.output_dir / stats_filename
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **self.stats,
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "data_dir": str(data_dir),
                        "split": split,
                        "batch_size": self.config.batch_size,
                        "max_cer": self.config.max_cer,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("=" * 60)
        print("流水线执行完成!")
        print("=" * 60)

    def run_test(self, num_samples: int = 10):
        """
        测试模式：使用模拟数据验证流水线逻辑

        Args:
            num_samples: 测试样本数量
        """
        print("=" * 60)
        print("L2W1 数据流水线测试模式")
        print("=" * 60)

        # 生成模拟样本
        test_samples = []
        test_texts = [
            "中国科学院计算技术研究所",
            "在时间的未尾",
            "人工智能技术发展",
            "深度学习模型训练",
            "自然语言处理应用",
            "计算机视觉识别",
            "机器学习算法优化",
            "神经网络架构设计",
            "数据挖掘与分析",
            "知识图谱构建方法",
        ]

        for i, text in enumerate(test_texts[:num_samples]):
            test_samples.append(
                DataSample(
                    id=f"test_{i:04d}",
                    image_path=f"./test_images/img_{i:04d}.jpg",
                    ground_truth=text,
                )
            )

        print(f"生成 {len(test_samples)} 个测试样本")
        print()

        # 模拟推理
        print("[1/3] 模拟推理...")
        test_samples = self._simulate_inference(test_samples)

        # 错误分析
        print("[2/3] 错误分析...")
        test_samples = self._analyze_errors(test_samples)

        # 显示结果
        print("[3/3] 分析结果:")
        print("-" * 60)

        for sample in test_samples:
            print(f"ID: {sample.id}")
            print(f"  真值: '{sample.ground_truth}'")
            print(f"  预测: '{sample.prediction}'")
            print(f"  CER: {sample.cer:.2%}")

            if sample.error_index >= 0:
                print(f"  错误位置: 第 {sample.error_index + 1} 个字符")
                print(
                    f"  预测字符: '{sample.error_char_pred}' -> 真值: '{sample.error_char_gt}'"
                )

                # 生成提示
                prompt = self.sft_generator.format_prompt(sample)
                print(f"  提示: {prompt[:80]}...")
            else:
                print("  无错误")
            print()

        # 过滤并生成 SFT
        filtered = self._filter_samples(test_samples)
        print(f"过滤后样本数: {len(filtered)}")

        if filtered:
            self.sft_generator.clear_output()
            self.sft_generator.write_batch(filtered)
            print(f"SFT 文件已生成: {self.sft_generator.output_path}")

            # 显示第一条
            with open(self.sft_generator.output_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                print(f"\n第一条 SFT 数据:")
                print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))

        print()
        print("=" * 60)
        print("测试完成!")
        print("=" * 60)


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="L2W1 自动化数据处理流水线 (Data Protocol v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 处理训练集
    python data_pipeline.py --data_dir ./data/raw/viscgec --split train
    
    # 处理测试集
    python data_pipeline.py --data_dir ./data/raw/viscgec --split test
    
    # 处理所有划分
    python data_pipeline.py --data_dir ./data/raw/viscgec --split all
    
    # 测试模式
    python data_pipeline.py --test
    
    # 指定输出目录和 CER 阈值
    python data_pipeline.py --data_dir ./data/raw --output_dir ./data/sft --max_cer 0.2

目录结构 (Data Protocol v2.0):
    data/raw/[dataset_name]/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
        """,
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data/raw", help="数据目录路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/sft", help="输出目录路径"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="数据集划分 (train, val, test, all)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="推理批次大小")
    parser.add_argument(
        "--max_cer", type=float, default=0.3, help="最大 CER 阈值 (0-1)"
    )
    parser.add_argument("--min_length", type=int, default=2, help="最小文本长度")
    parser.add_argument("--max_length", type=int, default=100, help="最大文本长度")
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "protocol_v2", "protocol", "scut", "casia", "generic"],
        help="数据格式",
    )
    parser.add_argument(
        "--rec_model_dir", type=str, default="", help="Agent A 模型目录"
    )
    parser.add_argument("--use_gpu", action="store_true", default=True, help="使用 GPU")
    parser.add_argument("--no_strict", action="store_true", help="禁用严格图像验证")
    parser.add_argument("--test", action="store_true", help="运行测试模式")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 构建配置
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        strict_validation=not args.no_strict,
        batch_size=args.batch_size,
        max_cer=args.max_cer,
        min_text_length=args.min_length,
        max_text_length=args.max_length,
        use_gpu=args.use_gpu,
        rec_model_dir=args.rec_model_dir,
    )

    # 创建流水线
    pipeline = DataPipeline(config)

    if args.test:
        # 测试模式
        pipeline.run_test(num_samples=10)
    else:
        # 正式运行
        pipeline.run(
            data_dir=args.data_dir, data_format=args.data_format, split=args.split
        )


if __name__ == "__main__":
    main()

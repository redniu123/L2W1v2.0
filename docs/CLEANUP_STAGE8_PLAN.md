# Stage 8 计划 — VLM / OCR 抽象层 `src/l2w1/vlm/` `src/l2w1/ocr/`

**日期：** 2026-06-26
**负责：** Claude（接口规格 + review）/ Codex（实现）
**目标：** 为 Agent B（VLM）和 Agent A（OCR）建立**干净的接口抽象**，隔离模型/Provider 依赖。本阶段是**接口设计**（新代码，非数值抽取，无需 parity），实现用 **mock** 和 **cache_only**（只读缓存，供 Paper1 复现）。

---

## 一、设计依据（真实契约）

- **Agent B 调用契约**（`build_agent_b_callable`）：输入 `dict{T_A, image_path, sample_id, min_conf_idx, user_prompt}` → 输出 `dict{corrected_text, latency_ms, token_usage, error_type}`；callable 带元数据 `_supports_parallel / _max_concurrency / _key_count`；`skip=true` 时走 mock（从 `user_prompt` 的【…】提取，否则回退 `T_A`）。
- **Agent A 结果 schema**：缓存行含 `T_A, mean_conf, min_conf, drop, conf, r_d, sample_id, image_path, domain, ...`。
- **Paper1 全调用缓存**（full_call_cache）行：`sample_id, vlm_raw_output, final_text_if_upgraded, latency_ms, token_usage, error_type`。

## 二、交付物

```
src/l2w1/vlm/
├── __init__.py
├── types.py        # VLMRequest, VLMResponse (frozen dataclasses)
├── base.py         # BaseVLMExpert(ABC) + capability metadata
├── mock.py         # MockVLMExpert（确定性，无依赖）
└── cache_only.py   # CacheOnlyVLMExpert（只读 full_call_cache，Paper1 复现）

src/l2w1/ocr/
├── __init__.py
├── types.py        # OCRRequest, OCRResult (frozen dataclasses)
├── base.py         # BaseOCREngine(ABC)
├── mock.py         # MockOCREngine（确定性）
└── cache_only.py   # CacheOnlyOCREngine（只读 agent_a cache）

tests/
├── test_vlm_mock.py
├── test_vlm_cache_only.py
├── test_ocr_mock.py
└── test_ocr_cache_only.py
docs/CLEANUP_STAGE8_REPORT.md
```

## 三、接口契约

```python
# vlm/types.py
@dataclass(frozen=True)
class VLMRequest:
    image_path: str          # 路径字符串（不读取文件内容）
    t_a: str                 # Agent A 文本
    sample_id: str = ""
    min_conf_idx: int = -1
    user_prompt: str = ""    # 可选，构造好的提示词
@dataclass(frozen=True)
class VLMResponse:
    corrected_text: str
    latency_ms: float | None = None
    token_usage: int | None = None
    error_type: str = "none"
    raw_output: str = ""     # 原始 vlm 输出（默认等于 corrected_text）
    provider: str = ""       # backend/model label
    def to_dict(self) -> dict: ...   # 与 Agent B 调用契约 dict 对齐（corrected_text/latency_ms/token_usage/error_type）

# vlm/base.py
class BaseVLMExpert(ABC):
    backend: str = "base"
    model_label: str = "base"
    supports_parallel: bool = False
    max_concurrency: int = 1
    key_count: int = 1
    @abstractmethod
    def query(self, request: VLMRequest) -> VLMResponse: ...
    def query_dict(self, prompt: dict) -> dict:   # 便捷：dict in/out，桥接旧调用契约
        ...

# vlm/mock.py
class MockVLMExpert(BaseVLMExpert):
    # 复刻 build_agent_b_callable 的 mock 语义：从 user_prompt 的【…】提取，否则回退 t_a；latency_ms 为确定性值（不用 time/random，便于测试）；error_type="none"

# vlm/cache_only.py
class CacheOnlyVLMExpert(BaseVLMExpert):
    def __init__(self, cache_rows: Iterable[dict] | None = None, *, lookup: Mapping[str, dict] | None = None): ...
        # 复用 l2w1.replay.paper1_cache.build_cached_result_lookup 按 sample_id 索引
    # query: 命中 → 用 vlm_raw_output/final_text_if_upgraded/latency_ms/token_usage/error_type 构造 VLMResponse；
    #        未命中 → VLMResponse(corrected_text=request.t_a, error_type="cached_result_missing")（与 run_online_routeronly 语义一致）
    # 绝不调用任何 API / 不读 key.txt
```

```python
# ocr/types.py
@dataclass(frozen=True)
class OCRRequest:
    image_path: str
    sample_id: str = ""
@dataclass(frozen=True)
class OCRResult:
    text: str                 # = T_A
    mean_conf: float = 0.0
    min_conf: float = 0.0
    drop: float = 0.0
    conf: float = 0.0
    r_d: float = 0.0
    sample_id: str = ""
    def to_dict(self) -> dict: ...   # 含 T_A 键，便于与 replay/agent_a 缓存衔接

# ocr/base.py
class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, request: OCRRequest) -> OCRResult: ...

# ocr/mock.py
class MockOCREngine(BaseOCREngine):
    # 确定性：text 由构造参数或一个 sample_id->text 映射给出；conf 字段取确定性默认或映射

# ocr/cache_only.py
class CacheOnlyOCREngine(BaseOCREngine):
    def __init__(self, cache_rows: Iterable[dict] | None = None, *, lookup=None): ...
    # 命中 → 用缓存行字段构造 OCRResult（T_A/mean_conf/min_conf/drop/conf/r_d）；未命中 → KeyError 或带标记的空结果（明确文档化所选语义）
```

## 四、依赖与铁律

- `src/l2w1/vlm` 和 `src/l2w1/ocr` **只可**依赖：stdlib、`l2w1.io`、`l2w1.replay.paper1_cache`（缓存索引）。**不得** import `modules.*` / `scripts.*` / torch / transformers / paddle / google / openai / requests。
- ❌ cache_only / mock **绝不**调用 API、加载模型、读取 `key.txt`、写实验结果。
- ❌ 不触碰受保护路径。
- ✅ 测试只用合成内存数据；不依赖真实模型/数据/网络。
- ✅ 现代类型注解；frozen dataclass；强制关键字参数；英文代码注释。

## 五、测试要求

- **vlm_mock**：`MockVLMExpert.query` 对带【X】的 user_prompt 返回 X；无则返回 t_a；`query_dict` 与 `query` 一致；`to_dict` 键齐全；latency 确定性。
- **vlm_cache_only**：合成 full_call_cache → 命中返回正确字段；未命中返回 `error_type="cached_result_missing"` 且 corrected_text=t_a；不触网（可断言无 import requests 等——通过纯逻辑即可）。
- **ocr_mock**：确定性结果；`to_dict` 含 `T_A`。
- **ocr_cache_only**：合成 agent_a cache → 命中构造 OCRResult；未命中按文档语义。

## 六、验收命令（Claude 独立执行）

```bash
python -m pytest tests/ -q
python -m ruff check src/ tests/
python -m mypy src/l2w1
```
全绿后分模块提交。

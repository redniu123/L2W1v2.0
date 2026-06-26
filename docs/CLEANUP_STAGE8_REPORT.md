# Stage 8 报告 — VLM / OCR 抽象层

**日期：** 2026-06-26
**范围：** 新增 `src/l2w1/vlm/` 与 `src/l2w1/ocr/`，仅实现接口、mock、cache_only；不调用 API、不加载模型、不读取密钥。

## 交付内容

### VLM Agent B 抽象

- `src/l2w1/vlm/types.py`
  - `VLMRequest`：冻结、keyword-only dataclass，承载 `image_path`、`t_a`、`sample_id`、`min_conf_idx`、`user_prompt`。
  - `VLMRequest.to_dict()` 输出旧 Agent B 输入契约键，包括 `T_A`。
  - `VLMResponse`：冻结、keyword-only dataclass，承载 `corrected_text`、`latency_ms`、`token_usage`、`error_type`、`raw_output`、`provider`。
  - `VLMResponse.to_dict()` 对齐旧 Agent B dict 契约：`corrected_text`、`latency_ms`、`token_usage`、`error_type`。
- `src/l2w1/vlm/base.py`
  - `BaseVLMExpert` ABC。
  - 类属性能力元数据：`backend`、`model_label`、`supports_parallel`、`max_concurrency`、`key_count`。
  - `query_dict(prompt)` 桥接旧 `dict{T_A,image_path,sample_id,min_conf_idx,user_prompt}` 输入契约。
- `src/l2w1/vlm/mock.py`
  - `MockVLMExpert` 复刻 `build_agent_b_callable(skip=true)` 的 mock 语义。
  - 从 `user_prompt` 的全角 lenticular brackets `【...】` 中提取文本；无匹配时回退 `t_a`。
  - `latency_ms` 为构造参数控制的确定性值，不使用 `time` 或 `random`。
- `src/l2w1/vlm/cache_only.py`
  - `CacheOnlyVLMExpert` 用 `l2w1.replay.paper1_cache.build_cached_result_lookup` 按 `sample_id` 建索引。
  - 命中缓存时从 `vlm_raw_output`、`final_text_if_upgraded`、`latency_ms`、`token_usage`、`error_type` 构造 `VLMResponse`。
  - 未命中时返回 `corrected_text=request.t_a` 且 `error_type="cached_result_missing"`。

### OCR Agent A 抽象

- `src/l2w1/ocr/types.py`
  - `OCRRequest`：冻结、keyword-only dataclass，承载 `image_path`、`sample_id`。
  - `OCRResult`：冻结、keyword-only dataclass，承载 `text` 与 Agent A 置信度字段。
  - `OCRResult.to_dict()` 包含 `T_A` 键，便于衔接 Agent A cache/replay schema。
- `src/l2w1/ocr/base.py`
  - `BaseOCREngine` ABC，定义 `recognize(request) -> OCRResult`。
- `src/l2w1/ocr/mock.py`
  - `MockOCREngine` 支持 `sample_id -> text` 映射或固定文本。
  - 所有置信度字段由构造参数确定，结果可重复。
- `src/l2w1/ocr/cache_only.py`
  - `CacheOnlyOCREngine` 用 `build_cached_result_lookup` 按 `sample_id` 建索引。
  - 命中缓存时从 `T_A`、`mean_conf`、`min_conf`、`drop`、`conf`、`r_d` 构造 `OCRResult`。
  - 未命中时抛出 `KeyError`；docstring 已明确该只读语义。

## 测试

新增 4 个合成内存数据测试文件：

- `tests/test_vlm_mock.py`
- `tests/test_vlm_cache_only.py`
- `tests/test_ocr_mock.py`
- `tests/test_ocr_cache_only.py`

覆盖项：

- VLM mock bracket extraction、fallback、`query_dict == query().to_dict()`、`to_dict` 键、确定性 latency。
- VLM cache_only hit/miss 字段构造。
- OCR mock 确定性输出与 `T_A` 字段。
- OCR cache_only hit 构造与 miss `KeyError`。

## 依赖与安全检查

- 新增 `vlm/` 与 `ocr/` 模块只使用 stdlib、包内相对导入、`l2w1.replay.paper1_cache`。
- 未 import `modules.*`、`scripts.*`、`torch`、`transformers`、`paddle`、`google`、`openai`、`requests`。
- 未读取 `key.txt`。
- 未调用 API、未加载模型、未访问网络。
- 未修改 `modules/` 或 `scripts/`。
- 未触碰受保护实验数据/结果目录。

## 最终验证输出

执行方式：通过 `/home/coder/anaconda3/etc/profile.d/conda.sh` 初始化 conda 后激活 `l2w1` 环境。

```bash
source /home/coder/anaconda3/etc/profile.d/conda.sh && conda activate l2w1 && python -m pytest tests/ -q
```

```text
........................................................................ [ 54%]
............................................................             [100%]
132 passed in 0.88s
```

```bash
source /home/coder/anaconda3/etc/profile.d/conda.sh && conda activate l2w1 && python -m ruff check src/ tests/
```

```text
All checks passed!
```

```bash
source /home/coder/anaconda3/etc/profile.d/conda.sh && conda activate l2w1 && python -m mypy src/l2w1
```

```text
Success: no issues found in 34 source files
```

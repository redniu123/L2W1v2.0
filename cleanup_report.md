# L2W1 SH-DA++ v4.0 清理与初始化报告

**生成时间**: 2024-12-19  
**任务目标**: 为 SH-DA++ v4.0 路由器开发建立纯净环境

---

## ✅ 任务 1: 文件清理与归档

### 1.1 Archive 目录创建
- ✅ 已创建 `archive/` 目录

### 1.2 过时文档归档
以下文档已移动到 `archive/` 目录：
- ✅ `docs/module3_pipeline_spec.md`
- ✅ `docs/module3_sft_spec.md`
- ✅ `docs/module4_agent_b_spec.md`
- ✅ `docs/module4_spec.md`
- ✅ `docs/module5_evaluate_spec.md`

**注意**: `L2W1_v3.0.docx` 未在根目录找到（可能已被移动或不存在）

### 1.3 Python 缓存清理
- ✅ 已递归删除所有 `__pycache__/` 目录

### 1.4 旧实验结果清理
- ✅ 检查完成，`results/l2w1_final_results.jsonl` 未在当前工作目录找到

---

## ✅ 任务 2: 核心代码重构预处理

### 2.1 `modules/router/uncertainty_router.py`

#### 已删除的方法：
- ✅ `check_boundary_sensitivity()` (原 738-854 行)
  - **原因**: SH-DA++ v4.0 将使用新的 Emission 矩阵路由逻辑
  - **影响范围**: `route()` 方法中相关调用已移除

- ✅ `_calculate_simple_ppl()` (原 626-658 行)
  - **原因**: SH-DA++ v4.0 不再使用简化 PPL 估计
  - **影响范围**: `SemanticPPLCalculator.calculate()` 中的回退逻辑

#### 已移除的配置参数 (`RouterConfig`):
- ✅ `boundary_confidence_threshold`
- ✅ `boundary_check_window`
- ✅ `aspect_ratio_warning`
- ✅ `aspect_ratio_critical`
- ✅ `char_density_min`

#### 已移除的字段 (`RoutingResult`):
- ✅ `boundary_risk`
- ✅ `boundary_reason`
- ✅ `left_boundary_confidence`
- ✅ `right_boundary_confidence`
- ✅ `aspect_ratio`
- ✅ `char_density`

#### 方法签名变更：
- ✅ `route()` 方法参数简化：
  - **移除**: `image_size: Tuple[int, int] = None`
  - **移除**: `char_confidences: List[Dict] = None`
  - **保留**: `logits`, `text`, `confidence`

#### 保留的基类：
- ✅ `CTCAligner` 类结构保留（后续将针对 Emission 矩阵进行适配改造）

### 2.2 `modules/paddle_engine/predict_rec_modified.py`

#### 已删除的逻辑：
- ✅ `__call__` 方法中的 `all_logits` 数组初始化 (原 638-640 行)
- ✅ `batch_raw_logits` 变量声明和拦截逻辑 (原 751-754 行)
- ✅ ONNX/非ONNX 路径中的 `deepcopy(outputs[0])` logits 拦截 (原 891, 904, 912 行)
- ✅ 按原始顺序存储 logits 的逻辑 (原 930-935 行)
- ✅ 返回字典中的 `"logits"` 字段 (原 946 行)

#### 备注：
- 📝 该文件后续将接入新的 `E ∈ [0,1]^{T×C}` Softmax 矩阵导出协议
- 📝 当前返回格式已简化为 `{"results": ..., "elapsed_time": ...}`

---

## ✅ 任务 3: 新目录结构初始化

### 3.1 SH-DA++ v4.0 规范目录
- ✅ 已创建 `docs/SH-DA_v4.0_Specs/` 目录

### 3.2 文档移动状态
以下 `.docx` 文件未在根目录找到（可能已被移动或不存在）：
- ⚠️ `01router设计.docx`
- ⚠️ `02工程实施路线.docx`
- ⚠️ `03消融实验方案.docx`
- ⚠️ `04对比实验.docx`

**建议**: 如这些文件存在于其他位置，请手动移动到 `docs/SH-DA_v4.0_Specs/`

---

## ⚠️ 任务 4: 环境审计报告

### 4.1 旧参数定义残留检查

#### ❌ 发现残留文档/代码：

| 文件路径 | 旧参数类型 | 行数 | 状态 |
|---------|-----------|------|------|
| `modules/router/uncertainty_router.py` | `entropy_threshold_low/high`, `ppl_threshold_low/high` | 39-44, 668-676, 724 | ⚠️ **需要保留** (RouterConfig 基类仍需) |
| `docs/ROUTER_DESIGN.md` | `entropy_threshold_*`, `ppl_threshold_*` | 多处 | ⚠️ **文档需要更新** |
| `modules/pipeline.py` | `entropy_threshold_*`, `ppl_threshold_*` | 59-62, 320-323 | ⚠️ **需要重构** |
| `PROJECT_STRUCTURE.md` | `entropy_threshold_*`, `ppl_threshold_*` | 40-44 | ⚠️ **需要更新** |
| `configs/router_config.yaml` | `entropy_threshold_*`, `ppl_threshold_*` | 13-17, 33-37 | ⚠️ **需要迁移至 SH-DA++ 参数** |
| `docs/PROJECT_CONTEXT.md` | 提及旧参数 | 50 | ⚠️ **需要更新** |

#### 📋 分析：
- **`uncertainty_router.py`**: `RouterConfig` 中的阈值参数暂时保留是**合理的**，因为：
  1. SH-DA++ v4.0 的路由器可能仍需要这些阈值作为回退或兼容层
  2. 新路由逻辑开发完成后可以逐步废弃

- **其他文件**: 需要根据 SH-DA++ v4.0 架构更新文档和配置

### 4.2 配置文件状态

#### `configs/router_config.yaml` 当前内容：
```yaml
# 当前仍包含旧版参数：
entropy_threshold_low: 2.0
entropy_threshold_high: 4.0
ppl_threshold_low: 50.0
ppl_threshold_high: 200.0
```

#### ✅ 配置就绪状态：
- ✅ 文件存在：`configs/router_config.yaml` 已找到
- ⚠️ **需要更新**: 应准备接收新的 SH-DA++ v4.0 参数，例如：
  - `ρ = 0.1` (Emission 阈值)
  - `K = 2` (Top-K 候选数)
  - `η = 0.5` (不确定性权重)

#### 📝 建议配置结构：
```yaml
# SH-DA++ v4.0 Router Configuration
router:
  sh_da_v4:
    # Emission 矩阵阈值
    emission_threshold: 0.1  # ρ
    top_k_candidates: 2      # K
    uncertainty_weight: 0.5  # η
    
  # 兼容层（可选，逐步废弃）
  legacy:
    entropy_threshold_low: 2.0
    entropy_threshold_high: 4.0
    ppl_threshold_low: 50.0
    ppl_threshold_high: 200.0
```

---

## 📊 清理统计

### 文件操作统计
- **移动文件**: 5 个 (模块规范文档)
- **删除方法**: 2 个 (`check_boundary_sensitivity`, `_calculate_simple_ppl`)
- **删除代码行**: ~150 行 (包含边界检测逻辑)
- **修改方法签名**: 1 个 (`route()`)
- **清理缓存目录**: 全部 `__pycache__/`

### 代码影响范围
- ✅ **核心路由器**: `uncertainty_router.py` - 已清理旧逻辑
- ✅ **推理引擎**: `predict_rec_modified.py` - 已移除旧 logits 导出
- ⚠️ **管道层**: `pipeline.py` - 需要后续重构
- ⚠️ **配置文件**: `router_config.yaml` - 需要更新为 SH-DA++ v4.0 格式
- ⚠️ **文档**: 多处文档需要更新

---

## ✅ 完成状态总结

| 任务项 | 状态 | 备注 |
|--------|------|------|
| 任务1: 文件清理与归档 | ✅ 完成 | 5 个文档已归档 |
| 任务2: 核心代码重构预处理 | ✅ 完成 | 旧方法已删除，方法签名已简化 |
| 任务3: 新目录结构初始化 | ✅ 完成 | `SH-DA_v4.0_Specs/` 已创建 |
| 任务4: 环境审计 | ⚠️ 部分完成 | 发现残留，需要后续更新 |

---

## 🎯 后续行动建议

### 优先级 P0 (立即执行)
1. ✅ **已完成**: 核心路由器代码清理
2. ⚠️ **待执行**: 更新 `configs/router_config.yaml` 以支持 SH-DA++ v4.0 参数 (`ρ`, `K`, `η`)

### 优先级 P1 (近期完成)
3. ⚠️ **待执行**: 重构 `modules/pipeline.py`，移除对旧阈值参数的依赖
4. ⚠️ **待执行**: 更新文档 (`ROUTER_DESIGN.md`, `PROJECT_STRUCTURE.md`, `PROJECT_CONTEXT.md`)

### 优先级 P2 (后续优化)
5. 📝 开发 SH-DA++ v4.0 路由器实现（基于 Emission 矩阵）
6. 📝 实现新的 `E ∈ [0,1]^{T×C}` Softmax 矩阵导出协议
7. 📝 更新 `CTCAligner` 以支持 Emission 矩阵对齐

---

## 📝 备注

- **文档无歧义**: ⚠️ 项目中仍存在包含 `PPL_threshold` 或 `entropy_threshold` 的文档，但这些主要作为**历史参考**或**兼容层**。建议在 SH-DA++ v4.0 开发完成后统一清理。

- **配置就绪**: ⚠️ `configs/router_config.yaml` 需要更新以接收新的 SH-DA++ 参数。建议采用**向后兼容**的方式（保留旧参数，新增新参数），便于渐进式迁移。

---

**报告生成完成** ✅  
**环境准备状态**: 🟡 **部分就绪** (核心清理完成，配置和文档待更新)


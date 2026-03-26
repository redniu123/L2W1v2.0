# SH-DA++ v5.1 项目技术报告

**版本**：v5.1  
**日期**：2026-03-26  
**状态**：实验进行中（Gemini 100% 上限实验）

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [Agent A：PaddleOCR 识别引擎](#3-agent-a-paddleocr-识别引擎)
4. [5维特征提取](#4-5维特征提取)
5. [路由器：RuleOnlyScorer](#5-路由器-ruleonlyscorer)
6. [在线预算控制器](#6-在线预算控制器)
7. [提示词生成器](#7-提示词生成器)
8. [Agent B：Gemini API 专家](#8-agent-b-gemini-api-专家)
9. [严格回填控制器](#9-严格回填控制器)
10. [领域知识引擎](#10-领域知识引擎)
11. [评测框架](#11-评测框架)
12. [Agent A 缓存机制](#12-agent-a-缓存机制)
13. [Gemini 上限实验](#13-gemini-上限实验)
14. [v4.0 → v5.1 战略转向说明](#14-v40--v51-战略转向说明)
15. [已知问题与待办](#15-已知问题与待办)

---

## 1. 项目概述

**SH-DA++**（Staged Hybrid Document Analysis++）是一套面向地质勘探文档 OCR 的**预算感知型分层协同推理系统**。

### 核心问题

地质勘探文档（扫描件/手写）中存在大量专业术语，PaddleOCR（Agent A）对这类文本的识别错误率（CER）约为 **11.88%**，直接影响后续数据分析的准确性。然而，调用大型视觉语言模型（VLM）对每一行文本纠错的 API 成本极高，不可能 100% 调用。

### 解决方案

用廉价的 Agent A 做初步识别，再用智能路由器筛选出「困难样本」，只对这部分样本花钱调用 Agent B（Gemini/本地VLM），从而在**严格控制预算**的前提下最大化纠错收益。

### 核心指标

| 指标 | 含义 |
|---|---|
| **CER**（字符错误率） | `Levenshtein(T_out, T_GT) / len(T_GT)`，越低越好 |
| **Budget**（调用预算） | Agent B 实际调用次数 / 总样本数，严格控制 |
| **AER**（接受纠错率） | 被 Backfill 接受的修改比例 |
| **CVR**（违规率） | 被 Backfill 拒绝的修改比例 |

---

## 2. 整体架构

```
输入图像
    │
    ▼
┌─────────────────────────────┐
│  Agent A：PaddleOCR          │  PP-OCRv5 Server Rec
│  输出：T_A + CTC Logits      │  模型：SVTR_LCNet
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  5维特征提取                  │
│  Mean_Conf, Min_Conf,        │
│  b_edge, drop, r_d           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  RuleOnlyScorer              │  计算 s_b, s_a, q
│  OnlineBudgetController      │  动态调整阈值 λ
└─────────────────────────────┘
    │
    ├──── q < λ ──────────────→ 直接输出 T_A（不调用 Agent B）
    │
    ▼ q ≥ λ
┌─────────────────────────────┐
│  ConstrainedPrompter         │  生成 Targeted Correction 提示词
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Agent B：Gemini API         │  gemini-3-flash-preview
│  或本地 VLM                  │  30个 API Key 并发
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  StrictBackfillController    │  ED≤3 且 长度变化≤20%
└─────────────────────────────┘
    │
    ▼
最终输出 T_final
```

---

## 3. Agent A：PaddleOCR 识别引擎

**文件**：`modules/paddle_engine/predict_rec_modified.py`

### 模型配置

| 参数 | 值 |
|---|---|
| 模型 | PP-OCRv5 Server Rec |
| 算法 | SVTR_LCNet |
| 输入形状 | `[3, 48, 320]` |
| 字典 | `ppocr/utils/ppocrv5_dict.txt`（约6625字符）|
| Batch Size | 6 |
| CPU 线程数 | 10 |

### 输出格式

Agent A 不仅输出识别文本，还输出用于特征提取的原始信号：

```python
output = {
    "results": [(T_A, conf), ...],         # 识别文本 + 总体置信度
    "top2_info": [{
        "top1_probs": [...],               # 每字符 Top-1 概率
        "top2_probs": [...],               # 每字符 Top-2 概率
        "top1_chars": [...],               # Top-1 字符序列
        "top2_chars": [...],               # Top-2 候选字符
        "top2_status": "available",
    }],
    "boundary_stats": [{
        "blank_mean_L": float,             # 左边界 blank 均值
        "blank_mean_R": float,             # 右边界 blank 均值
        "blank_peak_L": float,             # 左边界 blank 峰值
        "blank_peak_R": float,             # 右边界 blank 峰值
        "valid": True,
    }],
}
```

---

## 4. 5维特征提取

**文件**：`scripts/run_efficiency_frontier.py`（`infer_all_samples` 函数）

从 Agent A 的输出中提取 5 个特征，是整个系统的信息基础。

### 特征定义

#### ① Mean_Confidence（全局置信度均值）

```python
mean_conf = mean(top1_probs)  # 所有字符 Top-1 概率的均值
```

- 范围：[0, 1]，越低说明整行识别越不可信
- 作用：粗粒度全局风险信号

#### ② Min_Confidence（局部最低置信度）

```python
min_conf = min(top1_probs)       # 全行最低字符置信度
min_conf_idx = argmin(top1_probs)  # 对应字符位置（告知 Agent B 重点关注哪里）
```

- 范围：[0, 1]，越低说明存在具体的「危险字符」
- 作用：细粒度局部风险信号，同时定位存疑字符

#### ③ b_edge（CTC 边界 blank 强度）

```python
b_edge_L = 0.6 * blank_mean_L + 0.4 * blank_peak_L
b_edge_R = 0.6 * blank_mean_R + 0.4 * blank_peak_R
b_edge = max(b_edge_L, b_edge_R)
```

- 物理含义：CTC 解码时边界区域 blank 符号的强度
- 越高说明图像边缘字符可能被裁剪/遮挡，存在漏字风险
- 取左右中较大值，因为任意一侧出问题都需要处理

#### ④ drop（边界不对称性）

```python
drop = clip(|blank_mean_L - blank_mean_R|, 0, 1)
```

- 物理含义：左右边界 blank 强度之差
- 越大说明字符从某一侧"掉落"（漏字）

#### ⑤ r_d（领域语义风险分）

```python
r_d = min(n_match / 3.0, 1.0)
# n_match = 命中地质词典的总字符数（ahocorasick 多模式匹配）
```

- 物理含义：文本中地质专业术语的密集程度
- 越高说明这行文本越专业，Agent A 识别错一个字的代价越大
- 词典：`data/dicts/Geology.txt`，使用 ahocorasick 自动机实现 O(n) 匹配

---

## 5. 路由器：RuleOnlyScorer

**文件**：`modules/router/uncertainty_router.py`（`RuleOnlyScorer` 类）  
**文件**：`modules/router/sh_da_router.py`（`SHDARouter` 类）

### 为什么用规则而非训练模型？

v5.1 实验表明，训练得到的 `CalibratedScorer`（逻辑回归）中 `v_edge` 特征权重为 **-2.85**，方向与预期相反，说明训练数据量不足以可靠拟合权重。`RuleOnlyScorer` 是无需训练的纯公式方法，稳定性更高。

### 核心评分公式

#### 边界风险评分 s_b

```
s_b = clip(1/3×(v_edge_norm×b_edge) + 1/3×b_edge + 1/3×drop, 0, 1)
```

| 项 | 含义 |
|---|---|
| `b_edge` | CTC 边界 blank 强度，取左右最大值 |
| `drop` | 左右边界不对称性 |
| `v_edge_norm` | 边界视觉熵归一化，无数据时用 blank_peak 代理 |

#### 识别歧义评分 s_a

```
# Top-2 可用：
m_i = p_i^(1) - p_i^(2)
s_a = clip(1 - min(m_i), 0, 1)

# Top-2 不可用（降级）：
s_a = clip(1 - min(p_i^(1)), 0, 1)
```

#### 综合优先级 q

```
q = max(s_b, s_a) + η × r_d     # η = 0.5
```

### 分诊类型

| 条件 | RouteType |
|---|---|
| s_b < λ 且 s_a < λ | NONE（不升级）|
| s_b ≥ λ 且 s_a < λ | BOUNDARY |
| s_b < λ 且 s_a ≥ λ | AMBIGUITY |
| s_b ≥ λ 且 s_a ≥ λ | BOTH |

> v5.1 提示词统一为 TARGETED_CORRECTION，分诊类型仅用于日志分析。

---

## 6. 在线预算控制器

**文件**：`modules/router/uncertainty_router.py`（`OnlineBudgetController`）

```
λ_{t+1} = clip(λ_t + k×(B̄ - B), λ_min, λ_max)
```

| 参数 | 值 | 含义 |
|---|---|---|
| B | 目标预算 5%/10%/20%/30% | 期望调用率 |
| B̄ | 滑动窗口 W=200 实际调用率 | 实测调用率 |
| k | 0.05 | 步长 |
| λ_init | 0.5 | 初始阈值 |
| λ_min/max | 0.0 / 2.0 | 阈值范围 |

**Warmup**：前 200 个样本固定用 λ_init，不更新。之后 P 控制器接管，实测误差 ±0.01%。

- B̄ > B → λ 上升 → 更难触发升级 → 调用率下降
- B̄ < B → λ 下降 → 更容易触发升级 → 调用率上升

---

## 7. 提示词生成器

**文件**：`modules/vlm_expert/constrained_prompter.py`

### v5.1 统一模式：Targeted Correction（软提示）

```
你是一个严格的 OCR 纠错专家。以下是初步的单行文本识别结果：
【 {T_A} 】

系统检测到该文本可能存在识别错误。
其中第 N 个字符的机器置信度极低，请重点关注。
本文本属于【地质勘探】领域，请留意专业术语的准确性。

请结合提供的图像，修正上述文本中的错别字或漏字。
最高约束红线：
1. 尽可能保持原句原貌，禁止润色、改写或大幅增删。
2. 如果认为没有错误，请直接原样输出。

请直接输出修正后的完整文本，不要任何解释：
```

### 为何从硬约束改软提示？

v4.0 的 BOUNDARY/AMBIGUITY 硬约束（只允许改首尾/只允许改指定位置）导致 CVR 过高，模型频繁违规。v5.1 改为软提示 + Backfill 把关，给模型更大纠错空间。

---

## 8. Agent B：Gemini API 专家

**文件**：`modules/vlm_expert/gemini_expert.py`

| 参数 | 值 |
|---|---|
| 模型 | gemini-3-flash-preview |
| 端点 | https://new.lemonapi.site/v1（OpenAI兼容）|
| API Keys | 30 个，Round-robin 轮询，threading.Lock 线程安全 |
| Temperature | 0.1 |
| Max Tokens | 256 |
| 超时 | 60s |
| 最大重试 | 3 次（自动换 Key）|

### 调用流程

```
图像 → base64编码 → 构建multimodal payload
→ POST /v1/chat/completions
→ 失败(429/timeout) → 换Key重试
→ _parse_output() 提取修正文本（去引号/取第一行）
→ 全部重试失败 → 静默降级返回 T_A
```

---

## 9. 严格回填控制器

**文件**：`modules/router/backfill.py`（`StrictBackfillController`）

防止 VLM 幻觉的最后防线。

### v5.1 全局红线

```
Reject = I[ED(T_A, T_cand) > 3  ∨  |len(T_cand)-len(T_A)|/len(T_A) > 0.2]
```

| 红线 | 阈值 | 场景 |
|---|---|---|
| 编辑距离超限 | ED > 3 | VLM 改动过多，疑似幻觉 |
| 长度变化超限 | > 20% | VLM 大幅增删字符 |

触发任一条件 → 强制回退到 T_A。

---

## 10. 领域知识引擎

**文件**：`modules/router/domain_knowledge.py`（`DomainKnowledgeEngine`）

```python
r_d = min(n_match / 3.0, 1.0)
# n_match = ahocorasick 匹配到的地质词条总字符数
```

- 词典：`data/dicts/Geology.txt`
- 使用 Aho-Corasick 自动机实现 O(n) 多模式匹配
- 命中字符数越多 → r_d 越高 → 综合风险 q 越高 → 更容易升级到 Agent B
- 饱和点：命中 3 个字符即达到 r_d=1.0 上限

---

## 11. 评测框架

**文件**：`scripts/run_efficiency_frontier.py` / `scripts/run_all_frontiers.py`

### 四条评测策略

| 策略 | 说明 | 作用 |
|---|---|---|
| AgentA_Only | 0% 预算，纯 OCR | 性能下限 |
| Random@B | 随机选 B% 样本调用 Agent B | 验证提升来自特征而非运气 |
| ConfOnly@B | 只用 Min_Confidence 路由 | 消融：去掉边界/领域特征 |
| SH-DA++@B | 完整系统 5维特征+预算控制+回填 | 我们的方法 |

### 评测预算点

B ∈ {5%, 10%, 20%, 30%}，横向对比画出 **CER vs. Budget Pareto 效率前沿曲线**。

### 输出指标

| 字段 | 含义 |
|---|---|
| Overall_CER | 整体字符错误率 |
| AER | 接受纠错率（Backfill 接受的修改比例）|
| CVR | 违规率（Backfill 拒绝的修改比例）|
| Actual_Call_Rate | 实际 Agent B 调用率 |

---

## 12. Agent A 缓存机制

**文件**：`scripts/run_efficiency_frontier.py`、`scripts/run_all_frontiers.py`

Agent A 推理（PP-OCRv5 前向传播）是最耗时的步骤。缓存机制将推理结果保存为 JSON，后续实验直接加载，避免重复推理。

```bash
# 第一次跑（建立缓存）
python scripts/run_all_frontiers.py --use_gpu

# 之后调参复跑（加载缓存，秒级加载）
python scripts/run_all_frontiers.py --use_gpu --use_cache

# 数据集更新后重建缓存
python scripts/run_all_frontiers.py --use_gpu 
--rebuild_cache
`

缓存文件：
esults/stage2_v51/agent_a_cache.json，包含 5维特征、img_path、T_A、T_GT 等完整字段。

---

## 13. Gemini 上限实验

**文件**：scripts/run_gemini_ceiling.py

| 模式 | 说明 | 目的 |
|---|---|---|
| correction | Agent A 先识别，Gemini 100% 纠错 | 纠错上限 |
| ocr | 跳过 Agent A，Gemini 直接看图识字 | 纯识别上限 |
| both | 两个都跑 | 完整上限基准 |

**并发机制**：5路并发，失败最多重试10次（2s/4s/6s递增等待），30个Key轮换。

**输出**：
esults/stage2_v51/gemini_ceiling.csv

| 字段 | 含义 |
|---|---|
| Overall_CER | 最终 CER |
| AgentA_CER | Agent A baseline CER |
| Improve_Rate | Gemini 改对的比例 |
| Worsen_Rate | Gemini 改错的比例 |

---

## 14. v4.0 → v5.1 战略转向

| 模块 | v4.0 | v5.1 |
|---|---|---|
| 特征 | v_edge + b_edge + drop | Mean_Conf + Min_Conf + b_edge + drop + r_d |
| 评分器 | CalibratedScorer（需训练）| RuleOnlyScorer（纯规则）|
| 提示词 | 路径专属硬约束 | 统一软提示（Targeted Correction）|
| 回填 | 路径专属约束 + 全局红线 | 仅全局红线（ED≤3, 长度变化≤20%）|

**转向原因**：v_edge 训练权重为 -2.85（方向相反），CalibratedScorer 数据量不足泛化性差；路径专属硬约束导致 CVR 过高。

---

## 15. 已知问题与待办

| 问题 | 状态 |
|---|---|
| Gemini 30路并发触发限流 | 已修复：降为5路+10次重试 |
| 多进程同时运行互抢 Key | 已处理：pkill后重启单进程 |
| drop 特征部分样本为0 | 待优化 |

**待完成实验**
- [ ] Gemini 100% Correction 上限（进行中）
- [ ] Gemini 100% Pure OCR 上限
- [ ] SH-DA++ 完整 Pareto 前沿（B=5%/10%/20%/30%）
- [ ] 多模型对比（SmolVLM / Qwen-VL / InternVL）
- [ ] CER vs Budget 可视化曲线

---

## 附录：关键文件索引

| 文件 | 说明 |
|---|---|
| modules/paddle_engine/predict_rec_modified.py | Agent A 推理引擎 |
| modules/router/uncertainty_router.py | RuleOnlyScorer + OnlineBudgetController |
| modules/router/sh_da_router.py | SHDARouter 顶层封装 |
| modules/router/domain_knowledge.py | DomainKnowledgeEngine（r_d）|
| modules/router/backfill.py | StrictBackfillController |
| modules/vlm_expert/constrained_prompter.py | 提示词生成器 |
| modules/vlm_expert/gemini_expert.py | Gemini API Agent B |
| scripts/run_efficiency_frontier.py | 单模型效率前沿评测 |
| scripts/run_all_frontiers.py | 多模型全景跑批 |
| scripts/run_gemini_ceiling.py | Gemini 上限实验 |
| 
esults/stage2_v51/agent_a_cache.json | Agent A 推理缓存（977条）|
| key.txt | 30个 Gemini API Keys |

---

*本报告基于代码生成，反映 2026-03-26 最新状态。*

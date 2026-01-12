# A) SH-DA++ Router 设计方案（Method）

> 面向：后续开发与实验复现的详细设计文档。强调接口、变量、默认参数、可落地算法、日志与约束。[file:2]

## A1. 问题定义与输入输出规范

对于给定的单行文本图像 \(I\)，系统首先由 Agent A（PP‑OCRv5）生成初步识别结果。[file:2]

### A1.1 输入参数

**Agent A 输出**：[file:2]

- 识别文本 \(T_A\)。[file:2]
- 字符级置信度序列 \(p=\{p_1,p_2,\ldots,p_N\}\)。[file:2]
- CTC Emission 序列 \(E\)。[file:2]

**Emission 定义**：[file:2]

- \(E\in[0,1]^{T\times C}\) 为 softmax 概率矩阵，满足 \(\sum*{c=1}^{C}E*{t,c}=1\)。[file:2]
- Blank 类索引 \(c=blank\) 通常取 0 或来自 PaddleOCR 配置中的 blank_id。[file:2]

**语义输入（Semantic Sentinel）**：[file:2]

- 领域标签 \(D\)。[file:2]
- 风险分数 \(r_d\in[0,1]\)。[file:2]
- 关键片段索引集合 \(C\_{span}=\{(l_j,r_j)\}\_j\)，\((l_j,r_j)\) 为片段在 \(T_A\) 中起止索引。[file:2]

### A1.2 输出结果

- 升级决策 \(upgrade\in\{0,1\}\)。[file:2]
- 路径 \(route_type\)。[file:2]
- 感兴趣区域 \(RoI\)。[file:2]
- 动态提示词 Prompt。[file:2]
- 最终转录文本 \(T\_{final}\)。[file:2]

---

## A2. 三层路由架构设计

### A2.1 Layer 1：多源风险特征探测（Risk Scorers）

#### (1) 边界漏字风险 \(s_b\)

目标：通过“视觉-序列冲突”拦截高置信 Deletion 错误。[file:2]

**CTC 边界证据 \(b\_{edge}\)**：[file:2]  
设时间窗比例 \(\rho\)（默认 0.1），左边界窗 \(L=[1,\lfloor\rho T\rfloor]\)，右边界窗 \(R=[\lceil(1-\rho)T\rceil,T]\)。[file:2]

\[
b*{edge}=\max\left(\frac{1}{|L|}\sum*{t\in L}E*{t,blank},\ \frac{1}{|R|}\sum*{t\in R}E\_{t,blank}\right)
\]
[file:2]

**视觉边缘探针 \(v\_{edge}\)**（Sobel 梯度强度）：[file:2]  
将图像 \(I\) 归一化至 \([0,1]\)，提取边缘切片 \(I_L=I[:,: \lfloor\rho W_I\rfloor]\)，\(I_R=I[:,\lceil(1-\rho)W_I\rceil:]\)。[file:2]  
Sobel 梯度强度 \(G=\sqrt{G_x^2+G_y^2}\)，定义：[file:2]

\[
v\_{edge}=\text{norm}\left(\frac{1}{2}(mean(G(I_L))+mean(G(I_R)))\right)
\]
[file:2]

其中 \(\text{norm}(\cdot)\) 的 Min-max 参数通过验证集统计获得。[file:2]

**置信度陡降 \(drop\)**：[file:2]  
设 \(K\)（默认 2）为边界窗口大小：[file:2]

\[
\bar{p}_{left}=\frac{1}{K}\sum_{i=1}^{K}p*i,\quad
\bar{p}*{right}=\frac{1}{K}\sum\_{i=N-K+1}^{N}p_i
\]
[file:2]

\[
\bar{p}_{mid}=mean(\{p_i\}_{i=K+1}^{N-K})\ \ (\text{若 }N\le2K,\ \bar{p}\_{mid}=0)
\]
[file:2]

\[
drop=\max(0,\bar{p}_{mid}-\min(\bar{p}_{left},\bar{p}\_{right}))
\]
[file:2]

**合成逻辑（Rule-only）**：[file:2]

\[
s*b=clip(a_1\cdot(v*{edge}\cdot b*{edge})+a_2\cdot b*{edge}+a_3\cdot drop,0,1)
\]
[file:2]

默认系数 \(a_1=a_2=a_3=1/3\)。[file:2]

#### (2) 识别歧义风险 \(s_a\)

Margin 定义：位置 \(i\) 上 top‑1/top‑2 概率为 \(p*i^{(1)},p_i^{(2)}\)，则 \(m_i=p_i^{(1)}-p_i^{(2)}\)。[file:2]  
得分：\(s_a=clip(1-\min_i m_i,0,1)\)。[file:2]  
存疑位置：\(idx*{susp}=argmin\ m_i\)。[file:2]

---

### A2.2 Layer 2：战略分诊与参数定义（Strategic Triage）

基于得分强度与动态阈值 \(\lambda\) 判定路径：[file:2]

- 若 \(q<\lambda\)：\(route_type=NONE\)。[file:2]
- 否则若 \(s_b\ge\lambda\) 且 \(s_a\ge\lambda\)：\(route_type=BOTH\)。[file:2]
- 否则若 \(s_b\ge s_a\)：\(route_type=BOUNDARY\)；反之为 \(AMBIGUITY\)。[file:2]

**RoI 物理实现（等分近似法）**：[file:2]

- BOUNDARY：裁取图像左右边缘各 15% 宽度区域。[file:2]
- AMBIGUITY：以 \(idx*{susp}\) 为中心，裁取宽度范围
  \[
  \left[\frac{W_I}{N}(idx*{susp}-1),\ \frac{W*I}{N}(idx*{susp}+1)\right]
  \]
  [file:2]

---

### A2.3 Layer 3：在线预算控制（Online Budget Control）

**综合优先级评分**：[file:2]

\[
q=\max(s_b,s_a)+\eta r_d
\]
[file:2]

其中 \(\eta\) 在验证集上通过网格搜索确定。[file:2]

**在线控制器**：[file:2]  
设窗口 \(W=200\)，\(\widehat{B}\) 为窗口内实际升级比例。[file:2]

\[
\lambda \leftarrow clip(\lambda+k(\widehat{B}-B),\lambda*{min},\lambda*{max})
\]
[file:2]

初始值 \(\lambda_0\) 取验证集上 \(q\) 的 \((1-B)\) 分位数。[file:2]

---

## A3. 确定性回填与拒改规则

为抑制 VLM 幻觉，系统实施以下约束：[file:2]

- **BOUNDARY 约束**：仅允许在 \(T_A\) 的首字符前插入/替换，或在尾字符后插入/替换。[file:2]
- **AMBIGUITY 约束**：仅允许修改 \(idx\_{susp}\) 处字符，且修正字符必须属于 Top‑2 候选集。[file:2]
- **全局降级（Global Rejection）**：若编辑距离 \(ED(T*A,T*{修正})>2\) 或长度变化超过 20%，则判定为幻觉并回退 \(T\_{final}=T_A\)。[file:2]
- **关键指标**：CVR（Constraint Violation Rate）定义为触发全局降级或专家输出违反回填约束样本比例，用于衡量纠错可靠性。[file:2]

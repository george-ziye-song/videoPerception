---
type: brainstorm-options
field: RL/OPD post-training for video temporal understanding — method novelty
created: 2026-07-05
updated: 2026-07-05
language: zh-CN
status: 候选方案 + 数学 + 落地校正,非最终 spec。2026-07-05 讨论后发现两个阻塞性前提未验证(见 §0.6),plan.md 的"训teacher+蒸student"两段式设计**已推翻**,待验证结果定最终故事。
setting: **teacher 现成、不额外训练**,能力上限=我们已实测量级(TOMATO ~30%,不是"强teacher");RLVR(合成 GT 可验证奖励)为主信号,teacher 只提供过程性结构信号(辅助,非答案)。可取 hidden states / attention / logits。评测走 lmms-eval(uniform 抽帧,已用 vLLM 后端跑通)。
related:
  - 02-社区时序理解与推理方法-cn.md            # 本文的 idea 明确要"和这六类不一样"
  - ../video-任务全景-思维导图-cn.md            # OPD = RL,reward 换 teacher per-token KL(§5)
  - ../opd-vs-kd-vs-rlvr-与-ground-truth-结合-cn.md
---

# 时序理解 RL/OPD 的 novelty brainstorm(现成 teacher,非"白盒强teacher"——见 §0.6)

> 目标:提升视频**时序理解**的 RL/OPD 后训练,要**意料之外又情理之中**、借**经典时序思想**、用**数学 formulate**、且**和 doc 02 的六类机制都不一样**。

## 0. TL;DR

- **先批你的两个 idea**:OPRD(per-token 表示蒸馏)和 attention-RL 踩**同一个命门**——都作用在**一阶/逐 token**量上,而**时序结构是二阶(关系)/动力学(状态)/因果(Granger)的**。per-token 表示匹配对帧**排列不变**,梯度里没有"顺序"项 → 教不了 order。
- **统一修法**:把目标从"匹配一阶表示"抬到"匹配**跨时间结构**"。三个正交的经典时序视角 → 三个新 idea:
  1. **TRD**(关系):蒸馏 teacher 的**帧-帧依赖图** $A^T_{ij}$,不是逐帧表示。← 这是你 #1 的"二阶正确版",最合你口味。
  2. **PSD**(动力学):蒸馏 teacher 的**递归状态 + 未来预测 + 时间箭头**(HMM/Kalman/CPC)。← 最"意料之外"。
  3. **GCR**(因果):用 teacher 当 oracle 给**逐帧 Granger 因果必要性** $\varphi^T_t$ 作 reward。← RL 的那一半,T-GRPO 的因果化泛化。
- **推荐(2026-07-05 已更新)**:teacher 不是白盒强 teacher,是现成、不额外训练的模型,能力上限就是我们实测的量级——**先验证 §5①(合成难度 vs 真实是否相关)、§5②③(teacher 的结构信号是否哪怕答不对也看得到关键帧)这两组更上游的假设**,通过了再定投路线 A = TRD + GCR + AoT;§5④AoT 可以顺手先跑但不是决定性的。

---

## 0.5 ★ 落地前提(先厘清,否则后面全是空中楼阁)

**(1) eval 与 train 是两套东西**:
`lmms-eval` 是**评测框架**,只做推理算分:`抽帧 → encode → 生成答案 → 打分`。**本文的 TRD / PSD / GCR 全是训练时目标**,发生在你**另一套训练代码**(OPD trainer / verl / trl)里,那里才有对 attention、hidden state 的白盒访问。**训练完的 student 拿去 lmms-eval 正常跑,所有辅助头丢弃,评测侧一行不改**。

**(2) 帧怎么进模型**(Qwen-VL / LLaVA-Video 这类):

$$\text{帧}_i \xrightarrow{\text{vision encoder (ViT/SigLIP)}} \text{一}\textbf{块}\text{ visual token}(64\text{~}256\text{ 个}) \xrightarrow{\text{projector}} \text{拼进 LLM 输入序列}.$$

于是 LLM 输入 = `[帧1的token块][帧2的token块]…[帧N的token块][文本token]`,self-attention 在**所有 token** 上算。**下文的 $A_{ij}$ / $\varphi_t$ / 状态 $b_t$,都是训练时从这条前向里取/算的量,不是 lmms-eval 里的东西。**

**(3) 抽帧约定**:student 与 teacher 用**同一 uniform 抽帧**(同视频、同帧数 $N$、同网格)。这是 §2.3 GCR 的硬前提,后面详述。

---

## 0.6 ★★ 两个阻塞性前提(2026-07-05 讨论后追加——下面§1-§4的数学在这两条验证之前都只是假设)

> 这一节是从"用户觉得展开太抽象"这个反馈来的:不再用 RL 文献类比打比方,直接说清楚哪里可能塌、怎么现在就能查。

### (1) "白盒强 teacher"这个 setting 本身很可能不成立,而且"造teacher"和"蒸student"是两个不同的贡献,不该绑一起

原 setting 写"白盒强 teacher(如 Qwen3-VL-72B/32B)"——隐含 teacher **已经**很会时序推理,TRD/PSD/GCR 做的只是"把 teacher 已经会的东西教给 student"。

**这和实测冲突**:我们已经跑完的 P1/P2 评测(四个模型——锚 Qwen3-VL-8B、Qwen3.5-9B、Qwen3.5-35B-A3B、Qwen3.6-35B-A3B;三个 benchmark;含 thinking-on)显示,**没有一个现成模型是强时序 teacher**——最强的 Qwen3.6-35B 在 TOMATO(5 选 1,随机基线 ~20%)上也只有 ~31%。三个候选 teacher 在"时序推理"这个具体维度上,和 8B student 的差距很小,谈不上"强"。

若要按 plan.md 原方案"造"一个强 teacher(在合成 grounded CoT 上 SFT + 可选 GRPO,炼出真感知),**这件事本身的技术难度和价值,已经够格单独成文**("怎么用合成数据训出强时序感知模型")。把"造 teacher"和"蒸 student"绑进同一个故事有两个坏处:
- **审稿人分不清你在证明哪个论点**——teacher 的训练配方新颖?还是蒸馏/RL 方法新颖?两个都讲等于都讲不透。
- **单点故障**:teacher 没炼成("强感知"没坐实),下游 TRD/PSD/GCR 全部失去意义——把两个独立风险硬串成一条线,风险不必要地集中。

**推论(接下来的设计必须遵守这条约束)**:teacher **只能是现成的、不额外训练的模型**,能力上限就是我们已经测出来的量级(TOMATO ~30%)。TRD/PSD/GCR 的设计**必须假设 teacher 在时序推理/因果判断上并不可靠**,不能默认它给的信号是对的——这也是为什么 §0"TL;DR"里"RLVR 当主信号、teacher 只当过程性辅助信号"这个重新表述是必要的,不是可选的润色:**只有 GT-verifiable reward 才能保证正确性下限,teacher 靠不住的部分不能指望它兜底。**

### (2) 合成 → 真实的迁移 gap,没人验证过,不能假设成立

SynRL 论文声称合成训练能迁移到真实 benchmark,但:
- 论文协议本身模糊(harness、帧数、teacher 强度均未写明——baseline.md 已经吐槽过这套自写脚本和官方对不齐,本文档 §0.5 也提过 SynRL 论文的表述前后矛盾);
- 就算 SynRL 自己的 setting 下成立,不代表**我们的 setting**(不同 teacher、不同训练配方、可能不同的合成生成器参数)下 gap 同样小;
- 这条和"teacher 到底多强"是**同等级的阻塞性假设**:如果合成任务测的其实是另一种东西(比如渲染风格/分布,而不是时序推理本身的难度结构),那不管 TRD/PSD/GCR 设计得多精巧,练出来的东西可能压根不迁移到真实视频。

**这两条不先验证,§1-§4 的全部数学设计都是建在两个未证实假设之上。** 对应的验证实验见 §5(已按这两条重新排过优先级)。

---

## 1. ★ 先批你的两个 idea:共同命门 = 一阶 / 排列不变

记号:student/teacher 在层 $\ell$ 的**逐帧池化隐状态** $h_i^S, h_i^T \in \mathbb{R}^d$($i$ 索引帧或帧组,共 $N$ 帧);student on-policy 生成推理/答案 $y$。

### 1.1 #1 OPRD(per-token 表示蒸馏)——dense 但**教不了时序**

你的形式(大意):

$$L_{\text{OPRD}} = \sum_{i} d\!\left(h_i^S,\, h_i^T\right)$$

**命门(数学)**:对任意帧序排列 $\sigma$,若同时作用于 student 与 teacher,则 $L_{\text{OPRD}}$ 不变;且它**没有任何耦合 $i\neq j$ 的项**,于是

$$\frac{\partial L_{\text{OPRD}}}{\partial(\text{帧序})} \equiv 0.$$

也就是说,它优化的是"每帧长什么样"(appearance),对"帧之间怎么依赖/谁在前"零梯度。而 teacher 的 video-token 表示是 **appearance 主导**的 → 朴素 OPRD 主要蒸的是**表观**,正中 doc 03 的"长上下文≈表观、非时序"陷阱。
> 你的直觉"表示更有效"**对一半**:表示确实 dense、信号强;**错的一半**是——要做成**二阶关系**(§2.1)才带时序梯度。

### 1.2 #2 Reinforcement attention learning——proxy 不可靠

用 RL 优化 student 对历史输入的 attention。三个问题:
1. **attention 不 faithful**:attention 权重 $\neq$ 信息被因果使用(信息可走 value/MLP 路径)。奖励 attention 图 = 优化一个与"真时序推理"松耦合的 proxy。
2. **无 GT attention**:"正确的注意力"没有监督标签;若真要目标,只能来自 teacher → 其实变成**注意力蒸馏(白盒)**,不是 RL reward。
3. **末端 reward 高方差**:用最终答案 reward 反传到 attention,attention 会塌到"与 reward 相关但非因果"的伪模式。

### 1.3 统一洞察(三个新 idea 的种子)

> **时序理解活在"跨时间结构"里**——帧 $i$ 如何依赖/被帧 $j<i$ 修改(顺序、因果、状态转移)。任何作用在**边际逐帧表示(OPRD)或注意力幅值(#2)**上的目标都会漏掉它,因为时序是**二阶/关系、动力学、因果**属性,不是一阶属性。
> 三个经典刻画时序过程的方式 → 三个 idea:**依赖结构(关系)、生成动力学(状态/预测)、因果影响(Granger)**。

---

## 2. 三个新 idea(白盒、OPD 味、各借一个经典时序思想)

### 2.1 ★ Idea 1 — TRD:Temporal Relation Distillation(你 #1 的二阶正确版)

**借的经典思想**:relational KD + "self-attention 即时间依赖图" + 风格迁移里的 **Gram/协方差匹配**(把它搬到**时间轴**)。

**$A_{ij}$ 是什么(回答你的问题:是的,就是那个意思,但要说清它怎么来)**:
- 语义上,$A_{ij}$ = 把**帧 $i$ 当 query、帧 $j$ 当 key** 的注意力——"模型形成对帧 $i$ 的理解时,回看/取用了帧 $j$ 多少"。限定 $j\le i$,则每行 $A_{i,:}$ 是**帧 $i$ 对所有过去帧的依赖分布**(整行和为 1)。举例:判"球在帧 $i$ **反弹**",模型须回看帧 $i{-}k..i{-}1$ 的运动 → 那些 $A_{ij}$ 大。整张 $A$ = 模型的**时间依赖图**。
- **它不是模型里现成的一个量,要你从 token 级注意力聚合出来**:回忆 §0.5,帧 $i$ 是**一整块** token。把"帧 $i$ 的 token 块 → 帧 $j$ 的 token 块"的注意力权重,按 head、按 token **平均**成一个帧级标量 → 得到 $N\times N$ 的帧-帧矩阵:

$$A_{ij} \;=\; \frac{1}{|H|\,|B_i|\,|B_j|}\sum_{h\in H}\sum_{p\in B_i}\sum_{q\in B_j} \text{Attn}^{(h)}_{p\to q},\qquad B_i=\text{帧 }i\text{ 的 token 块}.$$

- **工程**:训练前向时 `output_attentions=True`(或 forward hook)抓注意力,按帧块 pool。**代价**:注意力图占显存($\sim (N\!\cdot\!|B|)^2$/层/头,如 $2048^2$),故通常**只抓 1–2 层、pool 完即丢**。**这全在训练 loop,不在 lmms-eval。**
- **备选**:若不想抓真注意力,学一个 relation 探针 $A^{}_{ij}=\operatorname{softmax}_j(\langle W_q h_i, W_k h_j\rangle/\sqrt d)$ 作用在池化帧表示上,更省显存、更可控。

**做法**:teacher、student 各得 $A^T,A^S$(on-policy,在 student 自己 rollout 的帧上),蒸馏其**依赖图**:

$$L_{\text{TRD}} = \sum_{i} D_{\mathrm{KL}}\!\left(A^T_{i,:}\;\big\|\;A^S_{i,:}\right).$$

**为什么能提升时序(数学直觉)**:
- **排列敏感**:$A_{ij}$ 耦合 $(i,j)$;排列帧序会**同时置换行与列** → 矩阵改变 → $L_{\text{TRD}}$ 对帧序有**非零梯度**(与 OPRD 的零梯度形成鲜明对比)。这是"能不能教 order"的分水岭。
- **基不变**:匹配**关系**而非向量,7B 的特征基与 72B 不同也能对齐同一张**依赖图** → 对 teacher$\leftrightarrow$student 容量/基失配鲁棒(per-token 蒸馏做不到)。
- **语义**:强 teacher 已学会"读懂帧 $t$ 需依赖帧 $t{-}k..t{-}1$";蒸 $A^T$ = 直接把这张**时间依赖图**教给 student,而非指望它从答案 reward 里涌现。

**pros**:dense、白盒、**排列敏感(真时序)**、基不变。**直接 rescue 你的 OPRD**。
**cons**:关系是"因果使用"的 proxy(但用 **teacher 的**关系当**目标**,比盲奖 student attention 原则得多);需 attention hook + 选层/头 + 显存;梯度须进主干(见 §2.2)。
**Δ vs doc 02**:六类里没有"关系/二阶"蒸馏——表示的**二阶统计**,而非 ①-⑥ 的 reward/输出/token-credit。

---

### 2.2 ★ Idea 2 — PSD:Predictive State Distillation(最"意料之外")

**借的经典思想**:**HMM / Kalman filter 的递归状态** + **Contrastive Predictive Coding(CPC)的未来预测** + **arrow-of-time 不对称**。

**做法**:在主干上加一个轻量**递归信念头** $\psi$,得 student 的因果状态(递归 = "到 $t$ 为止"的充分统计):

$$b_t^S = \psi\!\left(b_{t-1}^S,\; \operatorname{pool}(h_t^S)\right).$$

三个目标:

$$
\begin{aligned}
\text{(a) 状态蒸馏:}\quad & L_{\text{state}} = \sum_{t} d\!\left(b_t^S,\, b_t^T\right) \\[4pt]
\text{(b) 预测未来 (CPC):}\quad & L_{\text{pred}} = -\sum_{t}\sum_{k} \log \frac{\exp\langle g_k(b_t^S),\, h_{t+k}^T\rangle}{\sum_{n\in\text{Neg}} \exp\langle g_k(b_t^S),\, h_n^T\rangle} \\[4pt]
\text{(c) 时间箭头:}\quad & L_{\text{AoT}}^{\text{rep}} = \sum_{t} \max\!\left(0,\; m + L_{\text{pred}}^{\rightarrow}(t) - L_{\text{pred}}^{\leftarrow}(t)\right)
\end{aligned}
$$

(b) 里负例取其它时刻/其它视频的帧;(c) 令"顺放比倒放更可预测"($L_{\text{pred}}^{\rightarrow}$ 更低)至少差 margin $m$。(AoT 还有更简单的**答案侧**版本,见 §3,推荐先用那个。)

**★ 训练方式(直接回答你的问题:主模型不能冻结)**:$\psi$、$g_k$ 确实是**训练期外挂的小网络**,但**重点不是训它们**——而是让 $\lambda(L_{\text{state}}+L_{\text{pred}}+L_{\text{AoT}})$ 的梯度**反传进(不冻结的)主干**,重塑主干隐状态 $h_t^S$ 使其更具预测性/时序性。小网络只是**脚手架,推理时丢弃**(和 SSL 辅助头一样),与主 OPD/RL 目标联合优化。
> 辩证:若**冻结主干、只训小网络** → 那只是个 **probe**(读出"表示里有没有时序结构"),**根本不改进 student**。**这条对三个 idea 都成立**:TRD 的 $L_{\text{TRD}}$、GCR 的 reward 同理都必须让梯度进主干,才谈得上"提升时序理解"。

**为什么能提升时序**:
- **(b) 预测=逼近动力学**:从"过去状态"预测"teacher 未来表示",必须内化运动/状态**动力学**(速度、方向、隐状态转移)——正是 SynRL 的"temporal primitives",但作为**蒸馏目标**。
- **(c) 打破时间反演对称**:表示时间对称则 $L_{\text{pred}}^{\rightarrow}=L_{\text{pred}}^{\leftarrow}$,margin 恒被激活 → 逼**方向敏感**,治 TVBench/TempCompass 的"分不清方向"。
- **(a)+递归**:状态是**递归充分统计**(不能回看,须逐步更新)→ 教 **state-tracking**(SynRL 长时原语)。

**pros**:直击**动力学/方向/状态**;最贴 SynRL 原语。
**cons**:加预测头 + InfoNCE 负例设计;最难调。
**Δ vs doc 02**:六类里没有"预测编码/状态空间";②T-GRPO 是"奖励-答案-在-乱序下变差",这里是**生成式训练表示去预测未来+编码方向**,机制不同。

---

### 2.3 ★ Idea 3 — GCR:Granger Causal-necessity Reward(RL 的那一半)

**★ 先讲直觉(这个 idea 到底在干嘛)**
一句话:**不只看 student 答得对不对,而是看它"答对时靠的是不是对的那几帧"——用强 teacher 当"哪些时刻重要"的答案钥匙。**

1. **一道时序题里,只有少数帧是关键的**。如"人先拿杯子还是先坐下?"——关键帧只有"拿杯子"和"坐下"两个瞬间,其余无关。
2. **怎么量一帧多关键?把它挡住,看答案掉多少(留一帧 / ablation)**。teacher 看**全部帧**答对概率 $p_T(y^*\mid\text{全部})$;**去掉帧 $t$** 后 $p_T(y^*\mid\text{去掉 }t)$;两者**对数差 $\varphi_t$** = 帧 $t$ 的必要性。**这正是 Granger 因果 / 条件互信息的算子化**。
3. **对所有帧算一遍 → teacher 的"时序重要性地图" $\varphi^T$**:答案因果依赖了哪些时刻、各多少。
4. **奖励 student 让其 $\varphi^S$ 对齐 $\varphi^T$**:靠对的关键帧答对 → 高 reward;**靠某张表观帧蒙** → 低 reward。

**为什么这治时序**:**直接惩罚"靠单帧表观蒙对"的捷径**,且**逐帧、分级**,远比 T-GRPO 的"全局乱序有没有变差"这一个 bit 丰富。**与你 #2 的根本区别**:#2 奖励 attention(不 faithful);GCR 用 **ablation 实测因果**(挡住帧看答案真的掉不掉)。

**★★ 关键澄清 + 硬前提 + 待验证假设(你两个质疑都对)**:
- **我们优化的是"模型",不是"选帧算法"**:帧集**固定**(uniform),GCR 奖励的是"在这固定 $N$ 帧里,答案有没有因果依赖对的帧"——这是**模型的证据使用/推理**,不是 which-frame-to-pick。
- **硬前提:student 与 teacher 必须同一抽帧**(§0.5(3))。否则 $\varphi^S$ 在 student 帧上、$\varphi^T$ 在 teacher 帧上,$\cos(\varphi^S,\varphi^T)$ 是在**不同时间点**对齐,无意义。uniform 抽帧天然满足;**一旦谁用自适应/不同抽帧,GCR 就崩**。
- **待验证假设(GCR 的生死):** 它假设 *student 答错是因为依赖了**错的帧**,而非"盯着对的帧却推不出"*。**若 gap 其实在推理能力,GCR 无用**。→ **必须先做 §5-② 的验证再投入**。

**做法(形式化)**:用**白盒强 teacher 当 oracle**,对 query $q$、正确答案 $y^*$:

$$\varphi_t^T = \underbrace{\log p_T\!\left(y^* \mid x_{1:N}\right)}_{\text{看全部帧}} - \underbrace{\log p_T\!\left(y^* \mid x_{1:N}\setminus x_t\right)}_{\text{去掉帧 }t}.$$

student 同法得 $\varphi_t^S$(挡自己的帧看自己答案掉多少)。剖面对齐作 reward,进 GRPO/OPD:

$$r_{\text{temporal}} = \cos\!\left(\varphi^S,\, \varphi^T\right) \qquad\text{或}\qquad -\,D\!\left(\operatorname{softmax}(\varphi^S)\;\big\|\;\operatorname{softmax}(\varphi^T)\right).$$

**与 T-GRPO 的关系**:T-GRPO 是**一个 bit**(全局乱序是否变差);GCR 是其**因果化 + 逐帧化泛化**。
**pros**:与"真时序必要性"绑定最紧;泛化 T-GRPO;可解释;**天生 on-policy RL reward**(补 TRD/PSD 的 RL 味)。
**cons**:$\varphi^T$ 需 $O(N)$ 次 teacher 前向(留一帧)→ 可**只挡少数候选关键帧**或用 attention-rollout 近似;归因有噪声;**teacher 不是强teacher(见 §0.6),$\varphi^T$ 本身可靠不可靠必须先用 §5② 验证,不能默认它对**。
**Δ vs doc 02**:②的严格泛化,但逐帧 CMI 向量 vs 单 bit 机制显著不同,首次把 **Granger 因果**引入视频 RL reward。

---

## 3. Arrow-of-time(可叠加、最便宜)+ DTW(已砍)

### 3.1 ★ AoT arrow-of-time —— 其实非常简单,建议第一个试

**思想**:时间有方向;真懂时序的模型,对**顺放**和**倒放**同段视频**应表现不同**。顺倒不分 = 没编码方向。

**最简做法(答案侧,~10 行,不需要预测头)**:
1. 一道**方向/顺序**类题,顺放 → 记正确答案 logprob $s_{\text{fwd}}$;
2. 帧**倒序** `frames[::-1]` 再跑 → $s_{\text{rev}}$(倒放后"A 先于 B"应变假);
3. margin 项逼"顺放比倒放明显更对":

$$L_{\text{AoT}} = \max\!\big(0,\; m - (s_{\text{fwd}} - s_{\text{rev}})\big).$$

顺倒不分($s_{\text{fwd}}\approx s_{\text{rev}}$)就罚 → 逼方向敏感。
- **只对"倒放会改答案"的题用**(方向/顺序题);"弹了几次"这种计数题倒放不变,**不能用**。
- **与 T-GRPO 区别**:T-GRPO 随机**打乱**测"用没用时序";AoT 专门**倒放**测**方向**。
- 表示侧还有个进阶版($L_{\text{AoT}}^{\text{rep}}$,见 §2.2c),但**先用这个答案侧的**。

### 3.2 DTW / 最优传输 —— 解释完,已砍

**DTW(动态时间规整)**是语音识别老算法:比较两条**节奏可能不同**的时间序列——一个人念得慢时逐点对齐会错位,DTW 弹性拉伸/压缩时间轴找最佳对齐再比。我原想用它对齐 student/teacher 的**注意力轨迹**(怕推理节奏不同)。**但这是过度设计**:前提(节奏差异大)未必成立、复杂度高、收益不明。**→ 暂砍,需要再拣回。** (若真遇到"student/teacher 关注同一帧但发生在不同推理步"再考虑。)

---

## 4. ★ 推荐组合 + practicality 排序

三个正交视角(**关系 / 动力学 / 因果**)可组合。两条路线:

| 路线 | 组合 | 定位 |
|---|---|---|
| **A(稳,合你直觉)** | **TRD 骨架 + GCR reward + AoT 正则** | 二阶关系 dense 学 + on-policy 因果 reward + 方向正则 |
| **B(激进,最 novel)** | **PSD 骨架 + GCR reward** | 递归状态/预测动力学 + 因果 reward |

**落地成本 / 前提(诚实排序)**:

| idea | 实现成本 | 前提 / 风险 |
|---|---|---|
| **AoT** | 极低(倒放+margin,~10 行)| 只适用方向/顺序题 → **先拿它试水** |
| **GCR** | 中($O(N)$ teacher 前向 + ablation)| **必须同抽帧 + 先验证"gap 是帧依赖"假设** |
| **TRD** | 中(训练时 attention hook + 分块池化 + 显存)| 关系是因果的 proxy;梯度须进主干 |
| **PSD** | 高(预测头 + InfoNCE 负例)| 最 novel 也最难调 |
| ~~DTW~~ | — | 过度设计,**已砍** |

**推荐节奏(2026-07-05 更新)**:先跑 §5**①合成↔真实难度相关性**(定"合成数据靠不靠谱"),再跑 **②③teacher 结构信号 vs 合成GT**(定"teacher 至少看得到关键帧"这个假设成不成立),都过了再决定投 **路线 A**;§5④AoT 便宜可以顺手跑。novelty 一句话:*"不蒸 teacher 的时序表示,而蒸它的时序**依赖图**;不奖励'用了时序',而奖励'因果依赖了对的帧';teacher 不需要会做题,只需要看得到关键帧,答案的正确性由合成 GT 的 verifiable reward 保证。"*

**与 OPD 主线接口**:GCR 的 $r_{\text{temporal}}$ 是**辅助/过程信号**,和 GT verifiable reward(outcome 信号)并列相加,不是替代它;TRD/PSD 同理是辅助头。**风险提示**:$\varphi^T,A^T$ 是否可信,**不能假设**(teacher 不是强teacher)——这正是 §5②③ 要先查的东西,也是本文档相比 2026-07-05 之前版本最大的修正。

---

## 5. 最便宜的证伪实验(2026-07-05 按 §0.6 重新排优先级——①②③不需要训练/RL基建,几小时到半天;④⑤半天;⑥不是"验证实验",是主实验必做的对照组)

1. **①合成任务难度 vs 真实benchmark同概念子任务难度,是否相关(优先级最高,直接查§0.6(2)的迁移gap)**
   - 做法:用 `repos/Synthetic-Video` 的生成器,对每个"原语类型"(01/02脚本的 direction/rotation/count/speed/accel-decel 等 8 类)各生成一小批(~50条)**直答QA**(`sft.jsonl`格式,不需要CoT、不需要训练)。
   - 用我们已经测过的四个模型(锚8B/9B/3.5-35B/3.6-35B),在这批合成QA上零样本跑一遍——复用已经搭好的评测脚本,几分钟量级,不需要等P2全部跑完。
   - 对照:这四个模型在真实benchmark里概念对应的子任务上的已有分数——TOMATO的reason_type细分(direction/rotation/count,P1/P2数据已有)、MVBench的moving_direction/action_count/moving_count等子任务。
   - 画"模型在合成原语X上的acc" vs "模型在真实benchmark同概念子任务上的acc"的散点(4模型×~5-8个可对应类型≈20-30个点),算相关系数。
   - **解读**:强正相关 → 合成任务的难度结构和真实一致,至少"哪类任务难"是可迁移信号,支持整条"合成训练"路线;无相关/弱相关 → 合成任务在测另一种东西(可能是渲染分布/风格问题),需要重新考虑合成数据的构造方式(甚至改用真实视频+程序化标注)。
   - **诚实说明**:这只验证"难度结构"是否迁移,不直接证明"训练收益"会迁移——是必要非充分条件,但比什么都不查强得多,而且几乎零成本(复用现有infra和已有P1/P2数据)。

2. **②teacher的结构信号 vs 合成GT时间线是否相关(独立于teacher答没答对)——GCR生死验证的升级版**
   - 旧版做法(比较teacher和student谁的重要性profile更对)依赖"student答错的原因"这个间接推断。**升级**:合成数据的`metadata.jsonl`里有逐事件的精确毫秒时间线,是**真实GT**,不用猜。
   - 做法:挑一批合成样本,teacher对每帧做ablation(留一帧)或直接取attention,得到帧重要性profile $\varphi^T$/$A^T$;直接和`metadata.jsonl`里真实关键事件时刻比较,**不看teacher最终答案对不对**。
   - **相关**(teacher的重要性profile和真实事件时刻明显相关)→ teacher虽然"答不对"但"看得到关键帧",GCR/TRD这类"教它怎么看"的方法有戏;
   - **不相关**(teacher连"往哪看"都不可靠)→ GCR/TRD失去信号来源,**必须放弃教师结构信号这条路**,转向纯RLVR(见⑥)或别的方向。

3. **③零模型baseline(②的对照下限,不需要GPU/模型,几乎零成本)**
   - 同一批样本,算**纯像素运动能量**(逐帧光流幅度/帧间像素差)和`metadata.jsonl`真实事件时刻的相关性。
   - 如果这个零模型启发式已经能大部分解释"哪帧关键",teacher在这件事上的边际价值就很有限——直接用像素启发式当结构信号更便宜,也规避了"teacher不可靠"的问题。

4. **④AoT试水(半天,最便宜的训练类检验)**:只加AoT项训几十步,看方向类问题(合成的rotation-dir/direction类,或真实benchmark的direction维)是否单独涨。涨 → 白捡,直接进方案。

5. **⑤排列敏感性probe(半天,验证§1.1命门论)**:小集上OPRD vs TRD各训几十步,测"帧乱序后loss/acc变化"。TRD时序增益显著、OPRD不显著 → 命门论成立,TRD值得投。

6. **⑥RLVR-only ablation——不是"验证实验",是主实验设计里必须有的对照组**:不管最后选哪个teacher结构信号方法,都要有一组"只用GRPO+合成GT verifiable reward,完全不碰teacher"跑在合成数据上。这组结果决定故事重心:
   - 涨得好 → 结构信号是"锦上添花",故事重心变成"可验证奖励RL在时序任务上有效"(仍是贡献,只是没那么novel);
   - 涨得很少/学不动(时序任务多步,credit assignment可能很差,完全可能发生)→ 结构信号是"让RL学得动"的关键,这才是最贴合"用自己的方法规避teacher短板、student还能超过teacher"的故事。
   - 这组需要RL训练基建(GRPO/verl之类),不是几小时能出的东西,但**必须提前规划进主实验设计**,不能等①-⑤都做完才想起来。

**建议现在(不需要训练基建)就能做的顺序:① → ②+③(同一批样本,一起算相关性) → ④。⑤和⑥需要RL训练环境,留到定下teacher和训练框架之后再做。**

---

## 6. 待你拍

- **teacher 具体选哪个**:目前有三个候选的评测数据在手(Qwen3.5-9B / Qwen3.5-35B-A3B / Qwen3.6-35B-A3B),§5②③需要选定一个才能跑——建议先选**综合分数最高的 Qwen3.6-35B-A3B**(P1: MVBench 74.3、Video-MME 70.9 四模型最强;TOMATO thinking-off 有偏置退化,等 P2 thinking-on 数据出来后可以换)。
- **§5①②③的顺序**:建议①最先跑——它决定"合成数据到底靠不靠谱"这个更上游的问题,②③依赖①成立才有意义深挖(如果合成任务本身测的是另一种东西,再精细的教师信号验证也没意义)。
- 如果①出来发现合成和真实难度不相关——是转向"真实视频+程序化标注",还是调整合成生成器让它更贴近真实分布?这个决策留到①出结果后再定,不用现在纠结。
- plan.md 的"训teacher+蒸student"两段式设计已经推翻(§0.6),但 plan.md 里其他仍然有效的部分(SynRL生成器怎么用、数据规模估算、显存预算)要不要单独抽出来重写,还是等新故事定型后一起重写?

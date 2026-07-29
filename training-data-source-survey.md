---
title: 训练数据来源调研 — SynRL 合成数据 vs 真实视频
date: 2026-07-19
status: 完成一轮，待决策
---

# 训练数据来源调研

> 触发问题：目前 SynRL 合成数据已经被证明和真实 benchmark 有 gap（`distillation-readiness-report.md`：合成 Bouncing_Counting 上 teacher 更差，真实 MVBench 计数任务上 teacher 更好）。
> 用户要求：**不能自建真实视频数据集，只能复用别人现成的**；调研 2026 年后（最好顶会）这个领域训练用什么数据；至少读 10 篇论文再下结论。
>
> 本文是这一轮调研的结论。共完整或重点阅读 **11 篇** 2026 年前后的论文（专门针对"训练数据用什么"这个问题；如果算上更早阶段为 novelty.md 精读的 OPRD、RAL，则总计 13 篇）。全部下载了 HTML 全文并转成 txt 逐篇通读，不是只看摘要。原文都在 `papers/*_full.txt`，可自查。

## 0. 阅读清单（可自查）

| # | 论文 | arXiv ID | 会议（已在 arXiv 元数据核实） | 和本问题的关系 |
|---|------|----------|------|------|
| 1 | OPRD | 2606.06021 | 未核实 | 本项目 OPSD-R 的理论基础（早前阶段精读） |
| 2 | RAL (Reinforced Attention Learning) | 2602.04884 | 未核实 | CAD/PGA 的理论基础（早前阶段精读） |
| 3 | ReWatch-R1 | 2509.23652 | **未在 arXiv 找到 Comments 字段**，此前认为 ICLR 2026，本轮未能重新验证，暂不下断言 | Video-R1 具体被批评了什么；数据构造范式 |
| 4 | TimeLens | 2512.14698 | **CVPR 2026**（已核实：arXiv Comments 字段） | 数据重标注范式；thinking-free RLVR 更优 |
| 5 | Factum（原文标题 *Structured Causal Video Reasoning via Multi-Objective Alignment*，我此前笔记误称"STEER"） | 2604.04415 | 未在 arXiv 找到 Comments 字段 | CausalFacts-60K 数据构造；RL 任务配比 |
| 6 | Video-OPD | 2602.02994 | 论文文中自称投 ICML，未在 arXiv 找到 Comments 字段确认接收 | **本项目 OPSD-R 方向的直接先例**（见 §2） |
| 7 | Video-R1（被批评的原始论文） | 2503.21776 | 未标注 | 亲自核实它到底做错了什么 |
| 8 | VTG-R1 / "Datasets and Recipes for VTG via RL" | 2507.18100 | **EMNLP 2025 Industry Track**（由 Video-OPD 的参考文献列表核实） | 标题直接回答本问题；quality > quantity 的证据 |
| 9 | VideoKR | 2606.05259 | **ICML 2026 Spotlight**（已核实） | 反例：新采集真实视频而非复用现有数据集 |
| 10 | OmniVTG | 2604.25276 | **CVPR 2026**（已核实） | 反例：新采集真实视频；MLLM 擅长 dense caption 甚于直接 grounding 的洞察 |
| 11 | VideoChat-R1 | 2504.06958 | 未标注（早期有影响力工作，2025-04） | 最极端的"直接复用现有数据集标签，不用 CoT 重标注"案例 |

术语说明：凡"未标注/未核实"，是指我在 arXiv 摘要页的 `Comments:` 字段没有查到会议信息，**不代表论文没被接收**，只是我不愿意在没核实的情况下断言会议名称（TimeLens 一开始我记错成"CVPR 2026"到底对不对,这次专门用 curl 抓了 arXiv 元数据核实——确实是 CVPR 2026,但也顺带发现 ReWatch-R1/Factum/Video-OPD 三篇目前查不到,所以如实标注,不是我偷懒）。

---

## 1. 术语澄清：你说的"OPD"，字面意思是"On-Policy"，不是"Off-Policy"

这件事之前可能一直没挑明，借这次机会说清楚：OPRD = "**On**-Policy Representation Distillation"，Video-OPD 论文里的 "OPD" = "**On**-Policy Distillation"（Thinking Machines Lab 提出的概念，Qwen3 技术报告也用它）。这两篇论文（以及 Qwen3 系列本身）里的 "OPD"，都是指：

> 让 student 用自己当前策略采样出轨迹（on-policy），teacher 只负责给这条轨迹的每个 token 打分（reverse KL），不参与生成。

这跟"off-policy distillation"（直接在 teacher 生成的、或 GT 标注的固定轨迹上做蒸馏）是**相反**的概念。Video-OPD 论文里专门设了两个反例 baseline 叫 **OP-RKD / OP-FKD**（Off-Policy Reverse/Forward KL Distillation），结果是：这两个 off-policy 版本虽然在 TVG 指标上还行，但会**明显拖累模型在 general video understanding 上的表现**（Fig 4），而 on-policy 的 Video-OPD 和 GRPO 都不会。

这个区分对我们很重要：本项目目前设计的 OPSD-R（在 SynRL 固定帧上做 hidden-state 蒸馏）严格来说更接近"off-policy"（没有 student 自己的 rollout 采样），而不是真正的"on-policy"。这不是说 OPSD-R 设计错了——`oprd-ral-informed-redesign.md` 里已经诚实指出过这一点（"novelty.md 对固定视频帧的应用丢失了'on-policy'的本质"）——但看完 Video-OPD 之后，这个警示应该被更认真地对待：真正的"on-policy"版本（让 student 自己生成答案再让 teacher 打分）是有真实证据支撑会更好的，而不只是理论推导。

## 2. 最重磅的发现：Video-OPD 几乎就是本项目 OPSD-R 的"抢先实现"

这是本轮读到的最直接相关的一篇。它做的事：

- **模型对**：Teacher = Qwen3-VL-32B（先用 GRPO 后训练过），Student = Qwen3-VL-8B-Instruct —— 和本项目大概率会用的 teacher/student 规模几乎一致。
- **任务**：Temporal Video Grounding（不是我们的 4 个 primitive，但方法论完全对应）。
- **方法**：严格 on-policy —— student 自己采样轨迹，teacher 只对这条轨迹的每个 token 算 reverse-KL 当奖励，不需要 teacher 生成、不需要多轮 rollout。
- **结果**：Video-OPD 全面超过 GRPO（平均 +17% vs GRPO 的 +12%），且训练成本只有 GRPO 的 ~20%；**三轮迭代后 student 反超 teacher**（Table 3, Fig 5）。
- **数据**：真实视频，来自 HiREST + QuerYD + HowTo-Interlink7M + VTimeLLM + DiDeMo 五个**现成公开数据集**，加上 TimeLens-100K 的重标注，一共 96,586 条，但真正参与 OPD 训练的只精选了 **2,500 条**（用他们的 TVDF 课程筛选法）。

这几乎是把本项目 OPSD-R 计划里"β=0 plain-SFT 对照组"之外，又验证了一版"真正 on-policy、teacher 只打分不生成"的路线在真实视频 + Qwen3-VL 家族上是可行且更优的。如果要让 OPSD-R 更接近这篇论文的强证据版本，核心改动方向是：**student 自己生成答案（哪怕只有 1-2 个 token），teacher 只对这个生成结果打分**，而不是像现在计划里那样直接比较 teacher/student 在同一份人工 prompt 上的 hidden state。这个改动和"数据用什么"是两件事，但既然读到了就一并记在这里，供后续修订 `opsd-r-implementation-plan.md` 时考虑。

## 3. 数据构造的普遍模式：不是"重新拍视频"，是"重新标注别人的视频"

除了 VideoKR 和 OmniVTG 两个例外（见 §5），其余全部论文的数据构造都是同一个套路：

```
现成公开视频数据集（原始标注质量参差）
        ↓ 用 Gemini-2.5-Pro / GPT-4o / Qwen2.5-VL-72B 重新生成 caption/QA/CoT
        ↓ 用 IoU / 二次验证 / 文本捷径过滤 等规则做质量筛选
        ↓ 只保留一小撮高质量样本
        ↓ SFT 冷启动（可选）+ GRPO / on-policy distillation
```

具体到"复用了哪些现成数据集"，跨论文高度重合：

| 数据集 | 内容 | 被谁复用 |
|---|---|---|
| HiREST | 分层视频时刻检索 | ReWatch-R1, Video-OPD, VTG-R1 |
| QuerYD | 高质量文本+音频叙述的开放域视频 | ReWatch-R1, Video-OPD, VTG-R1, OmniVTG(反例对比) |
| DiDeMo | 开放域视频片段定位 | Video-OPD, VTG-R1, OmniVTG(反例对比) |
| ActivityNet(-Captions) | 人类活动，72K 条 query | Factum, OmniVTG(反例对比), VideoKR评测集来源之一(反例) |
| Charades-STA | 室内日常活动 | VideoChat-R1, Factum, OmniVTG(反例对比) |
| QVHighlights | vlog/新闻高光时刻 | Factum, OmniVTG(反例对比) |
| COIN | 大规模教学视频 | Factum |
| InternVid-VTime / HowTo-Interlink7M / HowTo100M | 网络叙述视频 | VTG-R1, Video-OPD |
| GoT-10k | 目标跟踪 | VideoChat-R1 |
| NExTGQA | 带定位的问答 | VideoChat-R1, VTG-R1 |
| TACoS | 烹饪领域 | VTG-R1, OmniVTG(反例对比) |

**没有一篇是自己重新去拍视频**（VideoKR/OmniVTG 除外，且理由与我们不同，见 §5）。这直接支持了你说的"我们不能自己做，只能用别人的"这个前提——这是这个领域现在的**主流做法**，不是权宜之计。

第二个一致的发现：**精选后的小样本集，规模都不大**：

| 论文 | 原始规模 | 实际训练规模 |
|---|---|---|
| Video-OPD | 96,586 | **2,500**（OPD）|
| VTG-R1 | 56,000 | 13,000（冷启动）+ 18,000（RL）|
| VideoChat-R1 | — | **18,031**（5 个任务合计，直接用原数据集标签，无需 CoT 重标注）|
| Factum | 现有 VTG 训练集切分 | 32,049（RL 阶段任务配比：53% 时序定位 / 21% 空间VQA / 20% 推理VQA / 6% 其他）|

VideoKR 论文自己还专门做了个"数据难度分析"（§6.5）：把 Video-R1/VideoRFT/OneThinker 等旧语料随机抽 3000 条测 Qwen3-VL-8B 零样本准确率，发现全都在 49–57% 之间，"已经被现有前沿模型打饱和了，学习信号很弱"——**这从另一个角度印证了"小而难"比"大而饱和"更有价值**，这和 VTG-R1 论文里"TVG-R1 用精选 13K 打赢未过滤的 56K"是同一个结论的两次独立验证。

## 4. Thinking / CoT 对纯感知类任务没用——第 5 次独立验证

这是本项目自己在 `distillation-readiness-report.md` 里最早发现的（thinking 模式让 4 个 gap primitive 掉 27–50pp）。这次调研又独立确认了 3 次：

1. **RAL 论文自己的 RAL-zero 消融**：不需要显式 thinking，纯粹优化 attention policy 依然有效。
2. **TimeLens Finding 5**："thinking-free RLVR" 在 VTG 这种"以感知为主导的任务"上，同时打赢 SFT 和 thinking-based RLVR。
3. **VideoChat-R1 §4.2 "Chain-of-thought vs. Direct Output"**（原文直接引用）："the output of the chain of thought has not demonstrated obvious advantages. In some cases, it is even inferior to the direct output"——这是针对 temporal grounding / object tracking 这类纯时空感知任务说的；同一篇论文里，对 QA 这种复杂推理任务，thinking 反而"发挥了显著作用"。

**但 ReWatch-R1 提供了一个重要的补充视角**：thinking 对"未训练/未针对性训练过的模型"有害，但在恰当的 SFT+RL 训练之后，thinking 是有帮助的。换句话说，我们自己实验里"thinking 模式下降"这个结论，很可能是**"没训练过"这个特定状态下的现象**，而不是"这类任务本质上和 thinking 不兼容"的永久结论——这提醒我们在后续训练完成后要重新测一次 thinking on/off 的对比，不能想当然认为训练后依然成立。

## 5. 关键 Gap：没有一篇论文真正在做我们的 4 个 primitive

必须诚实说明：以上 9 篇"训练数据怎么构造"的论文，全部聚焦在 **Temporal Video Grounding（给一句话找时间区间）** 或 **通用/知识密集型视频推理**，没有一篇是在做：

- Rotation_Direction（旋转方向：顺时针/逆时针）
- Rotation_Count（旋转圈数）
- Bouncing_Counting（反弹次数）
- Acceleration_Identification（加速/减速判断）

这些是**周期性运动计数 + 运动学状态判断**，和"给一句话定位时间区间"是完全不同的技能——TVG 数据集里的 query 通常是"the person picks up the cup"这种一次性事件，不是"这个球弹了几次"。所以，直接把 HiREST/QuerYD/DiDeMo 这些数据集拿来用，**对 causal-verification-report.md 里确认有因果落点的 Bouncing_Counting / Acceleration_Identification 这两个 primitive 帮助有限**——它们能帮的，是"时序定位"这个相邻但不同的能力,如果项目未来要往"通用时序grounding"这个方向扩展会有用,但不能替代4个primitive本身的训练数据。

VideoKR 和 OmniVTG 这两篇之所以**没有**走"复用现有数据集"这条路、而是新采集真实视频，理由也值得注意，因为和我们的处境不一样：
- VideoKR：现有数据集覆盖不了"专业领域知识密集型"内容（化学、医学、工程实验视频），所以去 YouTube 按学科大纲采集 145K 条 CC 许可视频。
- OmniVTG：现有数据集的**词汇覆盖率**不够（比如"meticulous"这种抽象词/生僻概念），所以设计了"语义覆盖迭代扩展"去主动补词。

这两个理由都是"现有*真实*数据集本身有覆盖盲区"，跟我们"synthetic 和 real 之间有 gap"是不同性质的问题——他们的解法（去采集更多真实视频）解决不了我们的问题（synthetic 数据本身系统性地不像 real）。

## 6. Gap 的部分解决：真实的"重复计数"数据集确实存在

针对 Bouncing_Counting，专门查证了一下，**周期性重复动作计数这个子领域是有真实数据集的**，而且是独立于本次 11 篇主线之外另外确认的：

| 数据集 | 规模 | 来源 | 备注 |
|---|---|---|---|
| **OVR**（Google DeepMind, arXiv:2407.17085） | 72K+ 视频标注 | Kinetics + Ego4D | 每条标注含：重复次数、起止时间、"重复的是什么"的自由文本描述；开放词汇（不限定动作类别）；官方 GitHub 有 Colab 可直接探索 |
| Countix（Google） | 8,757 视频，45 类 | Kinetics 子集 | 视频"在野外"的重复动作计数，经典 baseline 数据集（RepNet 论文） |
| RepCount | 1,451 视频（A: 1,041 / B 待查） | 自建，聚焦运动/健身 | 标注了每个动作周期的起止时间，"细粒度"程度比 Countix 高 |
| UCFRep | 526 视频，23 类 | UCF101 子集 | 规模较小 |

**OVR 是最值得优先看的**：规模最大（72K）、开放词汇、Ego + Exo 视角都有、且明确给了"重复次数"这个标量标签（正好对应 Bouncing_Counting 需要的监督信号形式），来源可靠（DeepMind 官方维护，2024 年发布，GitHub 仍活跃）。它甚至有可能天然包含一部分"旋转类"重复动作（比如陀螺、转盘、体操转体等——因为是开放词汇、自由文本描述"重复的是什么"，可以用关键词筛出"spin/rotate/turn"相关的子集），这样 **Rotation_Count 也可能间接从 OVR 里筛出一个真实子集**，虽然这一点需要实际下载数据后用关键词过滤验证，现在只能说"有可能"，不能断言。

## 7. 仍未解决的两个 primitive：Rotation_Direction 和 Acceleration_Identification

老实说，这两个目前**没有找到直接对应的现成真实数据集**：

- **Rotation_Direction**（顺时针 vs 逆时针的二分类）：这是一个"方向"标签而非"计数"标签，OVR/Countix/RepCount 这类计数数据集不天然提供这个标注（它们关心"几次"，不关心"哪个方向"）。可能需要在 OVR 筛出的旋转类子集上，额外人工/模型标注方向——工作量不算大（因为视频已经有了，只是加一个二分类标签），但目前没有现成的"已标注方向"数据集。
- **Acceleration_Identification**（加速/减速判断）：这是运动学状态变化，比"某个东西在动"更细粒度。本次调研没有找到专门的公开数据集覆盖这个概念；比较接近的真实世界代理场景可能是体育分析（短跑加速度）、交通监控（车辆加减速事件），但都需要专门验证，本次没有查证到可直接复用的具体数据集。

## 8. 结论与建议（分 primitive 给判断，不笼统说"用/不用真实数据"）

不是"real vs synthetic 二选一"的问题，四个 primitive 现状不同，建议也不同：

| Primitive | 因果验证结论 | 是否有真实数据候选 | 建议 |
|---|---|---|---|
| Bouncing_Counting | 有清晰因果落点（layer 3/7/11） | **有：OVR**（可能还有 Countix/RepCount 做交叉验证） | 优先下载 OVR，筛出弹跳/周期性反弹相关子集，替代或补充 SynRL 做这个 primitive 的训练/评测 |
| Rotation_Count | 因果验证在测的 8 层里无信号（24 层 linear attention 未测) | **可能有：OVR 的旋转类子集**（需关键词筛选验证） | 先用 OVR 关键词筛一遍看数量是否够，同时不要忘记 24 层 linear attention 还没测，也许该先把因果验证补测完再决定要不要花力气找真实数据 |
| Acceleration_Identification | 有清晰因果落点（layer 3/7/11/15/19 中的一部分） | 暂无 | 短期内继续用 SynRL，但**务必在真实 benchmark 的子集上做验证**（比如 MVBench/VSI-Bench 里凡是涉及加减速的题目），不要只在 synthetic held-out set 上报数字——这是目前能做的最低成本的"gap 监控" |
| Rotation_Direction | 8 层里无信号 | 暂无 | 同上：继续用 SynRL 但用真实 benchmark 子集做验证；由于因果验证都没找到落点，这个 primitive 目前的优先级本来就该往后排 |

**总体建议**：不必因为"synthetic 有 gap"就整体放弃 SynRL——真实数据目前只能部分覆盖（Bouncing_Counting，可能加 Rotation_Count），另外两个 primitive 现实里没有直接可用的现成数据集。合理的下一步是**混合策略**：

1. 立即可做：下载 OVR，看它的开放词汇标注里能筛出多少"弹跳/周期反弹"和"旋转"相关的真实视频样本，数量是否够支撑 OPSD-R 需要的规模（参照 Video-OPD 只用了 2,500 条精选样本就见效——不需要几万条）。这是本次调研能给出的最具体、最可执行的下一步。
2. 中期：如果 OVR 筛选后样本充足，Bouncing_Counting（可能加 Rotation_Count）的训练/评测切换到真实数据；Acceleration_Identification / Rotation_Direction 继续用 SynRL，但增加"真实 benchmark 子集验证"这一环,把这一条也补进 `opsd-r-implementation-plan.md` 的 §5 评估计划里。
3. 如果未来要扩展到"通用时序 grounding"作为辅助能力（不是 4 个 primitive 本身），HiREST + QuerYD + DiDeMo + Charades-STA + ActivityNet 这一组现成数据集 + Video-OPD 的 on-policy distillation 配方,是目前领域里验证过的最强组合,且和本项目的 Qwen3-VL teacher/student 设置几乎直接对应。

## 9. 对已有文档的影响

- `opsd-r-implementation-plan.md` 目前默认用 SynRL 的 `Atomic`/`Atomic2` 数据。如果采纳上面的建议，§2（复用清单）和 §3（设计选择）需要为 Bouncing_Counting 补一条"真实数据分支"，且 §3.2 的工程简化（用同一次 prefill 的 hidden state 代替真正 generate）可能需要重新评估——因为 Video-OPD 的强证据版本恰恰依赖"student 真的生成、teacher 只打分"这个更严格的 on-policy 设置，而不是复用同一次 forward 的技巧。这两件事要不要一起改，需要你决定优先级。
- `oprd-ral-informed-redesign.md` 里 CAD/PGA/AP-RLVR 目前已经根据 causal-verification-report.md 限定在 Acceleration_Identification / Bouncing_Counting 两个 primitive 上；如果 Bouncing_Counting 改用真实数据（OVR），这两个 idea 的"数据来源"这一栏也要同步改。

下一步如果你要我做，最具体可执行的是：**去下载 OVR 的标注文件（2 个 JSON，google-deepmind/ovr），跑一遍关键词筛选，看看"弹跳类"和"旋转类"能筛出多少条样本，值不值得基于它重新设计 Bouncing_Counting 的训练数据**。

---

## 10. 2026-07-19 动手核查更新：字节级获取现状 + 三个额外数据集挖掘

上面 §9 的下一步已经做了，并且顺带把可以直接查的都查了。这一节是动手核查后的**实测结果**，不是推测。

### 10.1 网络 / 权限现状

- **OVR 的 92 条弹跳视频，字节级（真正的视频画面，不是标注文字）获取受阻**：YouTube 直连在这台服务器上 DNS 都解析不出来（`www.youtube.com` 直接 resolve 失败，比 GitHub 那种"能解析但连不上"更彻底），不是能重试解决的问题；按 CLAUDE.md 规则不该为了下视频去占代理这条给 Claude-Anthropic 用的线。Kinetics 有 CVDF 维护的 S3 镜像可直连（已验证 `s3.amazonaws.com` 直连 200，拉过 21MB 的标注 CSV），但 700 个 train 分片、50 个 test 分片**没有 video_id → 分片的索引**，逐个探测不现实（train 一个分片就 1.76GB，全部约 1.2TB）。
- **Ego4D**：需要签署许可协议才能拿 AWS credentials，你已决定以个人身份去签（预计 48 小时批下来）。本机已经提前装好 `ego4d` CLI 和 `awscli`，批下来后可以用 `--video_uids` 精确只拉我们要的视频，不用下载整个数据集。

### 10.2 本地共享数据集实测（关键：目录存在 ≠ 文件能读）

`/remote-home/share/datasets/` 上有一个规模很大的共享数据集缓存（另外还有其他研究者的个人目录，没有深入翻动）。逐个用 `dd if=... of=/dev/null count=1` 实测读取，而不是只看 `ls`：

| 数据集 | 大小 | 实测可读性 | 备注 |
|---|---|---|---|
| COIN | 286GB | ✅ 可读 | 单个 tar.gz |
| Charades-Ego | 227GB | ✅ 可读 | |
| Epic-Kitchens-100 | 1.5TB | ✅ 可读 | 标注 + 实际 .MP4 视频都验证过 |
| ActivityNet（视频） | — | ❌ **I/O error** | 目录/文件存在，但读不出来，疑似存储层问题，不是我能修的 |
| HowTo100M（视频） | — | ❌ **I/O error** | 同上 |
| Kinetics-400（本地 OpenMMLab 副本） | — | ❌ **I/O error** | 同上；远程 CVDF/S3 镜像倒是能连，但没索引（见 10.1） |

这三个 I/O error 值得让你或服务器管理员看一眼——可能是这几个大文件所在的存储节点/层当前离线，重新挂载或者重传一下也许就好了，这个我在这台机器上修不了。

### 10.3 三个数据集的关键词挖掘结果（补充 §6-7 的 OVR 结果）

除了 OVR，又挖了 **ActivityNet Captions**（标注从 Stanford 官方直接下载，`cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip`，和视频文件是否能读无关）、**COIN**、**Epic-Kitchens-100** 的自带标注。汇总：

| Primitive | 数据源 | 命中数 / 视频数 | 视频字节能读吗 | 备注 |
|---|---|---|---|---|
| Bouncing_Counting | OVR | 175 / 92 | ❌（阻塞见10.1） | count 2-22，标注干净 |
| Bouncing_Counting | ActivityNet Captions | 168 / 134 | ❌（本地损坏） | 场景更丰富（乒乓、蹦床、跳水板） |
| Rotation_Count | OVR | 193 / ~？（"刚体旋转"筛选后） | ❌ | 几乎全是"手转轮子/方向盘"，弱匹配，上次已指出 |
| **Rotation_Count** | **ActivityNet Captions** | **827 / 652** | ❌（本地损坏） | **质量远好于 OVR**——真的是体操单杠转体、花样滑冰旋转、棍棒 twirl，不是手动转轮子 |
| Rotation_Count（弱替代） | Epic-Kitchens verb_class=23(turn) | 298 / 97 | ✅ 可读 | 但"turn"有歧义——大部分是"翻动食物"（turn meat/sausage/chicken），真正旋转的（knob/tap）只有约27条 |
| Acceleration_Identification | COIN（ThrowHammer 类） | 137 / ~40 | ✅ 可读 | "rotate body **and accelerate** the hammer"，链球投掷技术分解教学，真实但极窄（单一任务类别） |
| **Acceleration_Identification** | **ActivityNet Captions** | **58 / 54** | ❌（本地损坏） | 质量好于COIN（"accelerates out of a turn on his motorcycle"），但混有"视频剪辑加速/慢放"的噪音需要人工二次过滤 |
| Rotation_Direction | OVR / ActivityNet / Epic-Kitchens | **0** | — | **三个数据集查了个遍，没有一个有方向标签**——这个结论现在比上次更确定了 |
| 周期性动作计数（潜在可扩展池） | COIN（stir/pump/knead/screw等） | 4,282 / 2,425 | ✅ 可读 | 没有精确 count，只有起止时间，需要额外补标注 |
| 周期性动作计数（潜在可扩展池） | Epic-Kitchens（shake + mix/stir） | 2,519 / ~400+ | ✅ 可读 | 同上，没有精确 count |

### 10.4 这一轮更新的结论

1. **Rotation_Direction 依然彻底空白**，且现在是查了 OVR+ActivityNet+Epic-Kitchens 三家之后的结论，不是只查一家的印象。
2. **ActivityNet Captions 的标注质量全面超过 OVR**（尤其 Rotation：827 vs 193，且是真旋转不是转轮子），但偏偏本地视频文件是坏的——这是纯粹的基础设施问题，不是数据本身的问题，值得找人修一下存储再回来捡这部分。
3. **Acceleration_Identification 不再是彻底死局**：COIN 的 ThrowHammer（视频可读，但极窄）+ ActivityNet 的 58 条（质量更好，但视频访问受阻）——两个真实来源都比"完全没有"进了一步。
4. **COIN 和 Epic-Kitchens 的视频现在就能读**，可以不等 Ego4D 批下来、不等 ActivityNet 存储修好，先基于这两家（尤其 Epic-Kitchens 的 turn/shake/mix verb class，标注是结构化的受控词表，比自由文本干净）动手搭一个小规模真实 pilot 集。
5. 周期性动作那两个大池子（COIN 4,282 条 + Epic-Kitchens 2,519 条，加起来数千个视频）目前只有起止时间、没有精确计数标签——要用的话，需要走文献里那套"MLLM 自动标注 + 过滤"流程去生成 count 标签，这是文献里 11 篇论文的标准做法，不是新问题。

### 10.5 下一步

- 建议找人看一下 ActivityNet / HowTo100M / Kinetics-400 本地副本的 I/O error（存储层问题，我修不了）——修好后 ActivityNet 的 827 条真旋转标注是目前质量最好的 Rotation_Count 真实数据源，值得优先捞回来。
- Ego4D 那边继续等你签署 + 48 小时批复；批下来后应该专门查一次 Ego4D 自己的叙事标注里有没有方向/加速度相关的描述（目前三家都没有，Ego4D 因为标注更细致，值得单独确认一次，不能默认它也没有）。
- 在等待期间，Epic-Kitchens + COIN 已经是可以直接动手的真实数据（视频+标注都验证可读），足够先把 pilot 搭起来，不必等前两项。

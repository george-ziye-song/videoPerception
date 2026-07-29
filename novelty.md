---
type: brainstorm-options
field: "RL/OPD post-training for video temporal understanding -- method novelty"
created: 2026-07-05
updated: 2026-07-06
language: zh-CN
status: "候选方案 + 数学 + 落地校正,非最终 spec。2026-07-05 讨论后发现两个阻塞性前提未验证(见 0.6 节),plan.md 的'训teacher+蒸student'两段式设计已推翻。2026-07-06 对照 02 号文档排重、逐条查引用缺口后,删除 GCR(命名与机制均有问题)和教师委员会(依附于GCR,一并撤),换成 PGR(物理一致性奖励,附严格数学论证);AoT 降级为'借用现成任务的廉价诊断',补齐引用。"
setting: "teacher 现成、不额外训练,能力上限=我们已实测量级(TOMATO 约 30%,不是强teacher);RLVR(合成 GT 可验证奖励)为主信号,teacher 只提供过程性结构信号(辅助,非答案)。可取 hidden states / attention / logits。评测走 lmms-eval(uniform 抽帧,已用 vLLM 后端跑通)。"
related:
  - "02-社区时序理解与推理方法-cn.md (本文的 idea 要和这六类机制不一样,已逐条排重)"
  - "../video-任务全景-思维导图-cn.md (OPD = RL,reward 换 teacher per-token KL)"
  - "../opd-vs-kd-vs-rlvr-与-ground-truth-结合-cn.md"
---

# 时序理解 RL/OPD 的 novelty brainstorm(现成 teacher,非"白盒强teacher"——见 §0.6)

> 目标:提升视频**时序理解**的 RL/OPD 后训练,要**意料之外又情理之中**、借**经典时序思想**、用**数学 formulate**、且**和 doc 02 的六类机制都不一样**。

## 0. TL;DR

- **先批你的两个 idea**:OPRD(per-token 表示蒸馏)和 attention-RL 踩**同一个命门**——都作用在**一阶/逐 token**量上,而**时序结构是二阶(关系)/动力学(状态)的**。per-token 表示匹配对帧**排列不变**,梯度里没有"顺序"项 → 教不了 order。
- **统一修法**:把目标从"匹配一阶表示"抬到"匹配**跨时间结构**"。三个 idea,两条走"蒸馏"(需要teacher表示),一条走"可验证奖励"(不需要teacher):
  1. **TRD**(关系):蒸馏 teacher 的**帧-帧依赖图** $A^T_{ij}$,不是逐帧表示。← 这是你 #1 的"二阶正确版"。
  2. **PSD**(动力学·蒸馏):蒸馏 teacher 的**递归状态 + 未来预测 + 时间箭头**(HMM/Kalman/CPC)。目标来自 teacher 的表示——继承"teacher 靠不靠谱"的风险。
  3. **PGR**(动力学·可验证,2026-07-06 替换旧 GCR):目标不来自 teacher,来自我们**自己写的物理模拟器**——SynRL 生成器是显式物理方程渲染的,任意时刻的真实物理状态(位置/速度/角速度)**精确已知**。用这个当训练信号,不需要teacher可靠,有严格的数学保证(§2.3)。
- **推荐(2026-07-06 已更新)**:teacher 不是白盒强 teacher,是现成、不额外训练的模型,能力上限就是我们实测的量级——**先验证 §5①(合成难度 vs 真实是否相关)、§5②(teacher 的结构信号是否哪怕答不对也看得到关键帧,这决定 TRD/PSD 值不值)这两组更上游的假设**,PGR 不依赖这个验证(它不需要teacher),可以独立推进。

---

## 0.5 ★ 落地前提(先厘清,否则后面全是空中楼阁)

**(1) eval 与 train 是两套东西**:
`lmms-eval` 是**评测框架**,只做推理算分:`抽帧 → encode → 生成答案 → 打分`。**本文的 TRD / PSD / PGR 全是训练时目标**,发生在你**另一套训练代码**(OPD trainer / verl / trl)里,那里才有对 attention、hidden state 的白盒访问。**训练完的 student 拿去 lmms-eval 正常跑,所有辅助头丢弃,评测侧一行不改**。

**(2) 帧怎么进模型**(Qwen-VL / LLaVA-Video 这类):

$$\text{帧}_i \xrightarrow{\text{vision encoder (ViT/SigLIP)}} \text{一}\textbf{块}\text{ visual token}(64\text{~}256\text{ 个}) \xrightarrow{\text{projector}} \text{拼进 LLM 输入序列}.$$

于是 LLM 输入 = `[帧1的token块][帧2的token块]…[帧N的token块][文本token]`,self-attention 在**所有 token** 上算(标准 decoder-only 因果 mask,帧 $i$ 只能看到 $j\le i$ 的帧,因为帧本身按时间顺序拼进了序列)。**下文的 $A_{ij}$ / 状态 $b_t$ / 物理状态 $s_t$,都是训练时从这条前向里取/算的量,不是 lmms-eval 里的东西。**

**(3) 抽帧约定**:student 与 teacher 用**同一 uniform 抽帧**(同视频、同帧数 $N$、同网格)。这是 §2.1 TRD 和 §2.2 PSD 的硬前提(两者都要把 teacher 和 student 的帧级信号按帧索引对齐)——**PGR 不需要这条**,它的监督来自模拟器,不需要和 teacher 对齐帧。

---

## 0.6 ★★ 两个阻塞性前提(2026-07-05 讨论后追加——下面§1-§4的数学在这两条验证之前都只是假设)

> 这一节是从"用户觉得展开太抽象"这个反馈来的:不再用 RL 文献类比打比方,直接说清楚哪里可能塌、怎么现在就能查。

### (1) "白盒强 teacher"这个 setting 本身很可能不成立,而且"造teacher"和"蒸student"是两个不同的贡献,不该绑一起

原 setting 写"白盒强 teacher(如 Qwen3-VL-72B/32B)"——隐含 teacher **已经**很会时序推理,TRD/PSD 做的只是"把 teacher 已经会的东西教给 student"。

**这和实测冲突**:我们已经跑完的 P1/P2 评测(四个模型——锚 Qwen3-VL-8B、Qwen3.5-9B、Qwen3.5-35B-A3B、Qwen3.6-35B-A3B;三个 benchmark;含 thinking-on)显示,**没有一个现成模型是强时序 teacher**——最强的 Qwen3.6-35B 在 TOMATO(5 选 1,随机基线 ~20%)上也只有 ~31%。三个候选 teacher 在"时序推理"这个具体维度上,和 8B student 的差距很小,谈不上"强"。

若要按 plan.md 原方案"造"一个强 teacher(在合成 grounded CoT 上 SFT + 可选 GRPO,炼出真感知),**这件事本身的技术难度和价值,已经够格单独成文**("怎么用合成数据训出强时序感知模型")。把"造 teacher"和"蒸 student"绑进同一个故事有两个坏处:
- **审稿人分不清你在证明哪个论点**——teacher 的训练配方新颖?还是蒸馏/RL 方法新颖?两个都讲等于都讲不透。
- **单点故障**:teacher 没炼成("强感知"没坐实),下游 TRD/PSD 全部失去意义——把两个独立风险硬串成一条线,风险不必要地集中。

**推论(接下来的设计必须遵守这条约束)**:teacher **只能是现成的、不额外训练的模型**,能力上限就是我们已经测出来的量级(TOMATO ~30%)。TRD/PSD 的设计**必须假设 teacher 在时序推理上并不可靠**,不能默认它给的信号是对的。这也正是 **PGR 存在的意义**——彻底绕开"teacher 靠不靠谱"这个问题,监督来自我们自己的模拟器,不来自 teacher。

### (2) 合成 → 真实的迁移 gap,没人验证过,不能假设成立

SynRL 论文声称合成训练能迁移到真实 benchmark,但:
- 论文协议本身模糊(harness、帧数、teacher 强度均未写明——baseline.md 已经吐槽过这套自写脚本和官方对不齐,本文档 §0.5 也提过 SynRL 论文的表述前后矛盾);
- 就算 SynRL 自己的 setting 下成立,不代表**我们的 setting**(不同 teacher、不同训练配方、可能不同的合成生成器参数)下 gap 同样小;
- 这条和"teacher 到底多强"是**同等级的阻塞性假设**:如果合成任务测的其实是另一种东西(比如渲染风格/分布,而不是时序推理本身的难度结构),那不管 TRD/PSD/PGR 设计得多精巧,练出来的东西可能压根不迁移到真实视频。

**这两条不先验证,§1-§4 的全部数学设计都是建在两个未证实假设之上。** 对应的验证实验见 §5(已按这两条重新排过优先级)。

---

## 1. ★ 先批你的两个 idea:共同命门 = 一阶 / 排列不变

记号:student/teacher 在层 $\ell$ 的**逐帧池化隐状态** $h_i^S, h_i^T \in \mathbb{R}^d$($i$ 索引帧或帧组,共 $N$ 帧);student on-policy 生成推理/答案 $y$。

### 1.1 #1 OPRD(per-token 表示蒸馏)——dense 但**教不了时序**

你的形式(大意),和真实存在的论文对应上了。**2026-07-18更新:已下载论文全文(含附录证明)通读,不是只看摘要,补充精确细节**:**Yang, Zhu, Song, Wang, Xia, Zheng, Ma, Chen, Wang & Chen,*OPRD: On-Policy Representation Distillation*,arXiv:2606.06021(2026年6月,浙大+蚂蚁集团)**。

**真实的损失函数(原文Eq 6,比我们这里写的$\sum_i d(h_i^S,h_i^T)$精确得多)**:

$$L_{\text{OPRD}}=\mathbb E_{x,\hat y\sim\pi_\theta(\cdot|x)}\Big[\frac{1}{|\mathcal L_{\text{layer}}|}\sum_{l\in\mathcal L_{\text{layer}}}\frac{1}{\sum_t m_t}\sum_t m_t\,\frac{1}{d}\big\|h_{\theta,t}^{(l)}-\text{sg}(h_{T,t}^{(l)})\big\|_2^2\Big]$$

几个原文有、我们原来简化掉的关键细节:①$\text{sg}(\cdot)$是stop-gradient,只让梯度走student侧;②$1/d$对隐藏维度归一化,architecture无关;③$\mathcal L_{\text{layer}}$、位置mask $m_t$是可调的两个旋钮(论文默认:**全部层 + 只监督最后$k$个token**,不是所有位置一视同仁);④$\hat y\sim\pi_\theta(\cdot|x)$——**这个"on-policy"精确指的是:$\hat y$是student自己采样生成出来的rollout,teacher是在这个student自己生成的序列上被重新跑一遍forward去"打分"的,不是在一份固定的ground-truth序列上**。论文本身有完整的形式化证明(附录A):Theorem 1(OPRD梯度方差严格为零,OPD的REINFORCE式梯度方差不随训练收敛而消失,而是$O(\delta)$保持不变,导致信噪比$\text{SNR}(g_{OPD})=O(\delta)\to0$而$\text{SNR}(g_{OPRD})=+\infty$)、Theorem 2(LM head把隐藏空间压缩成一个"有效零空间"$\mathcal N_W$,沿最小奇异值方向的隐藏状态偏差,可以比沿最大奇异值方向大$(\sigma_1/\sigma_d)^2$倍——实测生产级LLM这个比值达到$10^6\sim10^8$倍——而任何输出空间loss都看不见这个差异)。实测在AIME24/25/AIMO上,OPRD把student-teacher的gap基本关掉(AIMO只差0.4分,在16-sample评测的噪声范围内),而输出空间蒸馏(OPD top-1/top-16)会在teacher水平以下几个点处**明显停滞**。

**关键澄清:novelty.md原来把这个机制搬到"视频帧位置"上,其实丢掉了"on-policy"这个最核心的设定**——真实论文的"on-policy"是"teacher给**student自己生成的token**打分",而我们这里原来的公式$\sum_i d(h_i^S,h_i^T)$里,$i$是**视频帧的位置**,视频是固定输入、不是student生成出来的——这严格来说不是"on-policy"的,是"在固定输入位置上做off-policy的表示匹配",更接近论文§6明确对比、明确说"不是我们方法"的FitNets/TinyBERT/MiniLM那条**离线特征蒸馏**路线(论文原文:"FitNets、TinyBERT、MiniLM在固定输入上算feature loss,student从不在自己的rollout分布上被训练,这是encoder-style表示蒸馏,不是我们这里说的on-policy"),不是OPRD本身。这个区分很重要,新文件里会给出一个**真正忠于原论文"on-policy"这个核心设定**的版本——把它用在student自己生成的推理/CoT token上,不是视频帧上。

**命门(数学,这一条本身依然成立,只是要澄清"是谁的命门")**:对任意帧序排列 $\sigma$,若同时作用于 student 与 teacher,则 $L_{\text{OPRD}}$ (按$\sum_i d(h_i^S,h_i^T)$这个边际比较的写法)不变;且它**没有任何耦合 $i\neq j$ 的项**,于是

$$\frac{\partial L_{\text{OPRD}}}{\partial(\text{帧序})} \equiv 0.$$

也就是说,它优化的是"每帧长什么样"(appearance),对"帧之间怎么依赖/谁在前"零梯度。而 teacher 的 video-token 表示是 **appearance 主导**的 → 朴素 OPRD 主要蒸的是**表观**,正中 doc 03 的"长上下文≈表观、非时序"陷阱。**这条批评是把机制搬到"视频帧位置"这个新场景之后才成立的,原论文自己完全不需要面对这个问题(通读全文确认,包括附录都没有讨论排列/顺序敏感性)**:原论文的$\hat y$是decoder因果生成出来的文本rollout,每个位置的隐藏状态$h_{\theta,t}^{(l)}$本来就是在"这个token是在看过全部前缀之后才生成的"这个因果约束下算出来的,压根不存在"把rollout同步打乱、看loss变不变"这种操作的合理性(打乱一个已经生成的因果序列,不再是一个模型会生成的合法rollout);视频帧则不同,帧可以被独立打乱成一个"合法但物理上错误"的输入(这正是probe-experiment-report.md§1实际做过的乱序实验)。这个批评是**视频这个新场景、且是"把机制用在固定帧位置而非student自己生成的token位置"这个具体错位适用之后**才成立的,不是原方法本身的漏洞。
> 你的直觉"表示更有效"**对一半**:表示确实 dense、信号强,原论文在文本推理场景下也确实靠这个机制把on-policy蒸馏的gap关掉了;**错的一半**是——搬到视频帧场景后,要做成**二阶关系**(§2.1)才带时序梯度,原样照搬会掉进排列不变性这个坑,原论文的场景不需要担心这一点,我们的场景需要。

**顺带一提,原论文自己列的"future work"第4条,直接预言了我们的TRD/CAD**:原文§5"Future Work"明确写"OPRD目前只对齐隐藏状态向量,不监督产生这些向量的attention pattern……用on-policy的attention匹配去扩展OPRD,可以更直接地迁移teacher的路由/组合行为"——这正是TRD(§2.1)和CAD(§2.5)在做的事,不是我们凭空想出来、和现有工作八竿子打不着的方向,是这篇论文自己指出的、还没人做的下一步。这条值得在写作时明确引用,增加"这个方向不是拍脑袋"的说服力。

### 1.2 #2 Reinforcement attention learning——**2026-07-18重大更正:这是一篇真实存在、已发表、实测有效的论文,原批评需要大幅修正**

**先说更正的严重性**:这里原来的批评("proxy不可靠"三条)是在没有查过真实文献的情况下,凭直觉写的。用户2026-07-18指出这一点后,直接下载论文全文(不是摘要)通读了一遍——**这三条批评里,第2条是事实错误,第1/3条理论上仍然成立但被真实实验数据反驳了"会导致不可行"这个推论。这是本文档一个需要严肃对待的准确性问题,不是小修小补。**

**真实论文**:Li, Ni, Qu, Miao, Yang, Fu, Chen & Cheng,*Reinforced Attention Learning*,arXiv:2602.04884(2026年2月,UC Davis + Google DeepMind + Google + Princeton)。**这篇论文的场景就是视频/图像MLLM的VQA任务(Qwen2.5-VL系列),不是文本推理,比OPRD离我们的场景更近。**

**真实机制(和原批评设想的"奖励一个attention标量"完全不是一回事)**:
- **Aggregated causal Attention Distribution Policy**(§3.1):把生成阶段第$t$个输出token对**所有**前面位置$i<t$(包括prompt/视觉token**和模型自己已经生成的token**,不只是视觉token)的attention,在**最后一层、所有head平均**之后,重新归一化成一个概率分布$p_\theta^t(i)=\alpha_{t,i}/\sum_j\alpha_{t,j}$——这是一整个**分布**,不是单个标量权重。
- **训练目标**(§3.2,Eq 3):$L_{\text{AttnRL}}=\mathbb E_t[A_t\cdot D(p_\theta^t\|p_{old}^t)]$——$D$用**Jensen-Shannon散度**(对称、有界,专门为训练稳定性选的,不是随手用KL),$A_t$是GRPO式的组内相对优势(标准的、基于可验证规则奖励算出来的优势,不是凭空发明的信号)。**$A_t>0$时把当前attention策略拉向"取得高奖励时的attention分布";$A_t<0$时推开"取得低奖励时的attention分布"**——这是把PPO/GRPO"新旧策略信任域"这套机制原样搬到attention分布上,不是"奖励一个具体的注意力值"。
- **On-Policy Attention Distillation**(§3.5,Eq 5):$L_{\text{AttnDistill}}=\mathbb E_{\tau\sim\pi_\theta}[\sum_t \text{JSD}(p_\theta^t\|p_\phi^t)]$——在student自己生成的rollout上,让student的attention分布去逼近teacher的,不带advantage项,纯结构模仿。最终目标(Eq 6)是$L_{RL}+\mu L_{GKD}+\gamma_{attn}L_{AttnDistill}$——**RL reward、token级蒸馏(GKD,即§2.1.2 ARD在这篇论文里对应的东西)、attention蒸馏三项加在一起**,不是三选一。

**逐条重新审视原来的三条批评**:
1. *"attention不faithful"*——**理论上这条批评依然成立,论文没有反驳这一点,但实测数据表明它不妨碍方法有效**:RAL在7个长视频benchmark里6个跑赢GRPO(MVBench 65.5 vs 64.0、NExTQA 74.1 vs 70.7、LongVideoBench 60.1 vs 57.9),8个图像benchmark全部跑赢GRPO;attention蒸馏加在标准token蒸馏之上,在7/8图像benchmark、多数视频benchmark上进一步提升(比如VideoMME 61.3→63.9,NExTQA 70.9→75.3)。**理论上的不faithful和实践上的有效,是两件可以同时成立的事**,不能拿理论顾虑直接推出"这个方法不可行"这个结论,这是原批评的推理漏洞。
2. *"无GT attention,只能退化成蒸teacher的attention"*——**这条是事实错误,不是"部分不准",是错的**。RAL的核心机制(§3.1-3.4)完全不需要teacher、不需要"正确的attention"这种标签:它用的是PPO/GRPO式的"新策略 vs 旧策略"自比较(JSD到**自己之前的**attention分布),由**可验证的规则奖励**(答案对不对、格式对不对)算出的advantage来决定"强化还是推开"——这是一个自洽的、不需要外部"正确答案"、也不需要teacher的强化学习机制,原批评设想的"要么有GT要么退化成蒸馏"这个二选一本身就是不完整的,漏掉了这第三种、真实存在且验证有效的选项。
3. *"末端reward高方差,attention会塌到伪模式"*——**理论顾虑依然合理,论文也没有从数学上证明不会塌陷,但工程设计+实测结果都不支持"一定会塌"这个悲观预期**:论文特意选JSD(有界、对称,比无界KL更抗extreme value)、逐token粒度(避免长序列梯度消失)来管理这个风险,**消融实验(RAL-zero,完全去掉thinking过程、只留直接答案)显示纯粹的attention策略优化,在没有CoT辅助的情况下,依然在5/7视频benchmark上跑赢base model、5/7跑赢完整GRPO**,这说明"reward driven attention"没有像原批评担心的那样系统性塌陷成噪声。

**这个更正对我们项目的直接意义,极其重要,必须写清楚**:RAL-zero这个消融——"去掉思考过程,只留直接答案,纯靠attention策略优化"依然有效——和我们自己在`distillation-readiness-report.md`实验0里发现的"thinking-on让4类gap原语明显变差"**方向完全一致、互相印证**:两边独立得出同一个结论——**在这类任务上,让模型"多想几步"不是正确的杠杆,直接在没有显式CoT的情况下优化attention/读出机制,可能才是正确的杠杆**。更关键的是:**RAL的纯RL变体完全不需要teacher**,只需要可验证的规则奖励(我们的7类合成原语本来就是MCQ,天然可验证)——这正好绕开了TRD/CAD/ARD这三个idea一直卡住的"teacher在这几类原语上没有验证出优势"这个共同瓶颈,因为RAL根本不需要teacher有优势,它只需要outcome reward。这个具体的新方向,详细设计见新文件`oprd-ral-informed-ideas.md`。

### 1.3 统一洞察(三个 idea 的种子)——**2026-07-18更正范围**

> **时序理解活在"跨时间结构"里**——帧 $i$ 如何依赖/被帧 $j<i$ 修改(顺序、状态转移、物理动力学)。任何作用在**边际逐帧表示**上的目标都会漏掉它,因为时序是**二阶/关系、动力学**属性,不是一阶属性。
> 两个刻画时序过程的经典视角 → 两条蒸馏线(TRD 关系、PSD 动力学-表示);**第三条不蒸teacher,直接对齐我们自己的模拟器物理真值(PGR,动力学-可验证)**。

**这条"共同命门"论证,读完真实的RAL论文后需要缩小适用范围,只对OPRD成立,不能笼统覆盖idea #2**:上面这段原文说"边际逐帧表示(OPRD)**或注意力幅值(#2)**"共享同一个一阶/排列不变的命门——这对OPRD是精确成立的(§1.1、§2.1.1都有严格证明);但**对idea #2不成立**,原因是真实的RAL机制($p_\theta^t$)本身就是一个**归一化到全部历史位置上的完整分布**,不是"某一个attention标量幅值",JSD比较的是两个分布的整体形状,这在结构上已经是"关系型"的(和TRD的$A_{i,:}$整行比较是同一类数学对象),不是OPRD那种逐位置边际比较。真正对idea #2成立、且被论文自己承认、也被我们自己的§2.5.1重复过的软肋,是"attention不faithful"这一条(信息可能走value/MLP路径),不是"排列不敏感"——这两条是不同的批评,不能混为一谈,原文把它们并列在同一句话里是不准确的。

---

## 2. 三个 idea(白盒/可验证、OPD 味、各借一个经典时序思想)

**三个idea共享的一条信息论上限(先讲一次,后面不重复推导)**:TRD 的 $A$、PSD 的 $b_t$、PGR 的 $\hat s$,都是 student 隐状态 $h_t^S$ 的**确定性函数**(分别是 $A=f_{\text{attn}}(h)$、$b_t=\psi(b_{t-1},h_t)$、$\hat s=g_\phi(h_t)$)。由数据处理不等式(Cover & Thomas),对任意这样的确定性摘要 $Z=f(h_t^S)$ 和任意目标 $y$:

$$I(Z; y) \le I(h_t^S; y).$$

**这条对三个idea统一成立,不需要在每一节重新证一遍**:不管选哪个方法去蒸馏/验证,能教给 student 的时序信息,上限都是"原始表示(或对 PGR 而言,模拟器给的真值本身)"里实际含有多少信息——这正是 probe-experiment.md 存在的理由,也是为什么 §5②(查 teacher 的原始表示里到底有没有时序信息)必须走在投入 TRD/PSD 训练之前(PGR 不需要这一步,它的目标本身就是真值,不经过 teacher 这一环再打折扣)。

### 2.1 ★ Idea 1 — TRD:Temporal Relation Distillation(你 #1 的二阶正确版)

**大白话先讲做法**:不去逼 student"每一帧看起来"和 teacher 一样(那只是让它学会画面细节长什么样),而是逼 student"心里认为哪几帧互相有关系"和 teacher 一样。比如判断"球有没有反弹",人真正靠的是回头看前面几帧的运动轨迹——TRD 就是把 teacher"回头看了哪几帧、每帧看的比重多少"这张关系图,直接教给 student,而不是让 student 自己去猜该往哪看。

**借的经典思想**:relational KD(Park, Kim, Lu & Cho, CVPR 2019, *Relational Knowledge Distillation*)+ attention transfer(Zagoruyko & Komodakis, ICLR 2017, *Paying More Attention to Attention*)——这两支 KD 经典工作分别做"蒸关系"和"蒸attention",TRD 是把它们**专门用到帧间时序关系**上,不是发明"attention可以蒸"这件事本身,这两篇需要明确引用,避免显得核心机制是凭空发明的。另外借了风格迁移里的 **Gram/协方差匹配**(把它搬到**时间轴**)。

**$A_{ij}$ 是什么**:
- 语义上,$A_{ij}$ = 把**帧 $i$ 当 query、帧 $j$ 当 key** 的注意力——"模型形成对帧 $i$ 的理解时,回看/取用了帧 $j$ 多少"。限定 $j\le i$,则每行 $A_{i,:}$ 是**帧 $i$ 对所有过去帧的依赖分布**(整行和为 1)。举例:判"球在帧 $i$ **反弹**",模型须回看帧 $i{-}k..i{-}1$ 的运动 → 那些 $A_{ij}$ 大。整张 $A$ = 模型的**时间依赖图**。
- **它不是模型里现成的一个量,要你从 token 级注意力聚合出来**:回忆 §0.5,帧 $i$ 是**一整块** token。把"帧 $i$ 的 token 块 → 帧 $j$ 的 token 块"的注意力权重,按 head、按 token **平均**成一个帧级标量 → 得到 $N\times N$ 的帧-帧矩阵:

$$A_{ij} \;=\; \frac{1}{|H|\,|B_i|\,|B_j|}\sum_{h\in H}\sum_{p\in B_i}\sum_{q\in B_j} \text{Attn}^{(h)}_{p\to q},\qquad B_i=\text{帧 }i\text{ 的 token 块}.$$

- **工程**:训练前向时 `output_attentions=True`(或 forward hook)抓注意力,按帧块 pool。**代价**:注意力图占显存($\sim (N\!\cdot\!|B|)^2$/层/头,如 $2048^2$),故通常**只抓 1–2 层、pool 完即丢**。**这全在训练 loop,不在 lmms-eval。**
- **备选**:若不想抓真注意力,学一个 relation 探针 $A^{}_{ij}=\operatorname{softmax}_j(\langle W_q h_i, W_k h_j\rangle/\sqrt d)$ 作用在池化帧表示上,更省显存、更可控。

**做法**:teacher、student 各得 $A^T,A^S$(on-policy,在 student 自己 rollout 的帧上),蒸馏其**依赖图**:

$$L_{\text{TRD}} = \sum_{i} D_{\mathrm{KL}}\!\left(A^T_{i,:}\;\big\|\;A^S_{i,:}\right).$$

**硬前提(必须明确写出,不能只在 §0.5 提一句就默认读者记得)**:$A^T$ 和 $A^S$ 要逐行做 KL,**必须是同一个 $N$、同一组物理时刻对应同一个帧索引**——teacher 和 student 抽帧不一致,这个 KL 就是在对齐不同时间点的分布,没有意义(同 §0.5(3))。

**为什么能提升时序(数学直觉)**:
- **排列敏感**:$A_{ij}$ 耦合 $(i,j)$;排列帧序会**同时置换行与列** → 矩阵改变 → $L_{\text{TRD}}$ 对帧序有**非零梯度**(与 OPRD 的零梯度形成鲜明对比)。这是"能不能教 order"的分水岭。
- **基不变**:匹配**关系**而非向量,7B 的特征基与 72B 不同也能对齐同一张**依赖图** → 对 teacher$\leftrightarrow$student 容量/基失配鲁棒(per-token 蒸馏做不到)。
- **语义**:teacher 学会"读懂帧 $t$ 需依赖帧 $t{-}k..t{-}1$"(如果它真学会了,这一点本身待 §5② 验证);蒸 $A^T$ = 直接把这张**时间依赖图**教给 student,而非指望它从答案 reward 里涌现。

**pros**:dense、白盒、**排列敏感(真时序)**、基不变。**直接 rescue 你的 OPRD**。
**cons**:关系是"因果使用"的 proxy,attention 本身不 faithful 这个 §1.2 批评 #2 时用的论点,同等力度适用于 teacher 的 attention(只是这里蒸的是 teacher 的关系模式当**目标**,而不是奖励 student 自己的 attention,原则上更站得住,但不代表 teacher 的 attention 模式本身就是"真时序依赖");需 attention hook + 选层/头 + 显存;梯度须进主干(见 §2.2)。
**Δ vs doc 02**:六类里没有"关系/二阶"蒸馏——表示的**二阶统计**,而非 ①-⑥ 的 reward/输出/token-credit。

#### 2.1.1 数学论证(为什么 TRD 真的比 OPRD 强,不只是"数学直觉",三条各自独立)

**结论一(功能形式论证——无条件成立,是"零梯度"这句话的严格版本)**

先把"零梯度"说精确。$L_{\text{OPRD}}=\sum_i d(h_i^S,h_i^T)$ 作为 $\{h_i^S,h_i^T\}_{i=1}^N$ 的函数,**只通过逐个 $i$ 的边际项依赖输入**——用统计学的话说,$L_{\text{OPRD}}$ 依赖联合 $(h^S,h^T)$,只通过一个**对跨帧关系视而不见的统计量**(逐帧边际距离构成的向量)。推论:**存在两种不同的 $\{h_i^S,h_i^T\}$ 配置——一种保留了正确的跨帧关系结构、一种打乱了这个结构——只要每个位置的边际距离 $d(h_i^S,h_i^T)$ 不变,$L_{\text{OPRD}}$ 数值完全相同,无法区分**。这是一个关于损失函数**依赖了哪个统计量**的精确陈述,不是"经验上学不到"这种要跑实验才知道的猜测(经验验证见 §5⑤)。

相比之下,$L_{\text{TRD}}=\sum_i D_{KL}(A_{i,:}^T\|A_{i,:}^S)$ 依赖的是**联合**关系矩阵 $A_{ij}$(对所有 $i,j$ 对),同一个"每帧边际距离不变、关系结构不同"的反例会让 $L_{\text{TRD}}$ 数值**不同**——因为 $A_{ij}$ 本身就是跨帧的量,不是任何边际统计量的函数。**这不是训出来才发现TRD更好,是两个损失函数在写下公式的那一刻,依赖了不同的统计量,这一点本身就已经确定了。**

**结论二(基不变性,严格化——条件在"教师学生表示只差一个正交变换"这个理想化假设,但这正是这类比较方法本就针对的场景)**

假设 teacher 和 student 编码的时序关系信息本质相同,但表示空间之间差一个(未知的)正交变换:$h_i^T = Q\,h_i^{S*} + \epsilon_i$,$h^{S*}$ 是"完美对齐"时 student 该有的表示,$Q$ 是任意正交矩阵。这正是表示相似度比较文献的标准理想化(Kornblith, Norouzi, Lee & Hinton, ICML 2019,*Similarity of Neural Network Representations Revisited*——CKA 那篇,该文正是为解决"两个网络的表示可能只差一个基变换,直接比较原始向量没有意义"这个问题,才提出比较 Gram/关系矩阵)。在这个设定下:
- OPRD 的 $d(h_i^S,h_i^T)$(若用 L2/cosine 这类依赖原始坐标的距离)——**即使 student 学得完美**($h_i^S=h_i^{S*}$),只要 $Q\neq I$,$d(h_i^{S*},Qh_i^{S*})$ 一般不会小,这是在惩罚一个不该惩罚的坐标系差异。
- TRD 的 $A_{ij}=\operatorname{softmax}_j\langle W_q h_i,W_k h_j\rangle$——因为 $W_q,W_k$ 是**分别针对 teacher/student 学出来的**投影,teacher 用 $(W_q^T,W_k^T)$ 作用在 $Qh^{S*}$ 上,和 student 用等价的 $(W_q^TQ,\,W_k^TQ)$ 作用在 $h^{S*}$ 上,得到的注意力模式**完全相同**——即使原始向量 $h^T\neq h^{S*}$。这就是"对基失配鲁棒"的精确含义:比较的是**投影后的相对关系**,投影矩阵能吸收掉这个正交变换,原始向量的比较不能。

**诚实的边界**:这个论证的前提是"teacher/student 表示差异可以刻画成一个全局正交变换"——真实的 teacher/student(不同架构、不同训练)之间的差异不太可能是一个干净的全局正交变换。这证明的是"TRD 对**这一类**已被充分研究的基失配鲁棒",不是"对任何形式的表示差异都鲁棒"。

**结论三(和 PGR/PSD 共享的信息论上限,已在 §2 开头统一给出)**:$A^T$ 是 $h^T$ 的确定性函数,$I(A^T;\text{真实时序结构})\le I(h^T;\text{真实时序结构})$——TRD 能教给 student 的东西,上限是 teacher **原始表示**里有多少时序信息,这正是 §5② 必须先查的原因。

---

#### 2.1.2 ★ 从"TRD 为什么不该做"这个批评里浮现的新变体——ARD:Answer-level Response Distillation(读出层蒸馏,不碰 attention)

**背景(2026-07-14 组会追加讨论)**:probe-experiment 发现"4类原语 probe 读得出、直答答不对"之后,§7 的结论是"没有一个更聪明的 teacher 可以学,TRD 没有用武之地"。这条结论被追问了一句很尖锐的话:*"teacher 总分明明更强,表示层又没 gap,那真正的 gap 就在读出层——让 student 学 teacher 在读出层的分布,不就行了?"* 这一问把 TRD(蒸 attention/关系图)之外的**另一条独立机制**逼了出来,值得单独立项、单独判定,而不是被 TRD 的判死刑连坐。

**大白话说明**:TRD 蒸的是 teacher"心里怎么形成判断"(哪几帧互相有关系);ARD 蒸的是 teacher"最后给出的答案分布本身"——不管 teacher 内部怎么算出来的,只要它算出来的**答案**比 student 准,就把这个答案分布(softmax 之后、不是 one-hot 的硬标签)当目标,让 student 的输出去逼近它。这是最经典的 Hinton et al. 2015《Distilling the Knowledge in a Neural Network》response-based KD,不需要 TRD 那套"表示只差一个正交变换"的基不变性论证撑腰——**唯一需要成立的前提,是 teacher 在这个具体任务上的读出准确率确实比 student 高**。

**数学上要多简单就多简单(和 TRD 对比,这是它的优势不是劣势)**:设 $g^T,g^S$ 是 teacher/student 各自的"隐状态 → 答案分布"读出函数。ARD 蒸的是 $L_{\text{ARD}} = D_{\mathrm{KL}}\big(g^T(h_t^T)\,\|\,g^S(h_t^S)\big)$。这个损失有没有意义,不取决于 $h^T$ 和 $h^S$ 是不是同一个基(ARD 根本不碰 $h$,只碰 $g(h)$ 这个最终分布),**只取决于一件可以直接实测的事**:$\text{acc}(g^T(h^T)) > \text{acc}(g^S(h^S))$ 在这个具体任务上是否成立。这条不成立,ARD 就是在教 student 模仿一个不比它准（甚至更差）的过程,不但没用,可能还有害——这正是 §3 判定表要查的东西。

**判定表(套用 probe-experiment-report.md §3 已经测过的直答准确率,不需要新实验就能判)**:

| 原语类型 | teacher直答 vs student直答 | ARD 在这个类型上适不适用 |
|---|---|---|
| Rotation_Direction | 52.0% vs 52.0%(打平) | ❌ 不适用——没有优势可蒸 |
| Rotation_Count | 69.0% vs 67.0%(几乎打平) | ❌ 不适用——差距在噪声量级 |
| Acceleration_Identification | 64.0% vs 60.0%(小幅领先) | ⚠️ 边界情况,优势小,不确定是不是噪声 |
| Bouncing_Counting(合成) | 29.0% vs 40.5%(**teacher更差**) | ❌ 不适用,蒸了反而有害 |
| **(对照)MVBench action_count/moving_count(真实)** | **+9.0~+14.5pp(teacher明显更强)** | ✅ **矛盾未解决,最该优先验证的战场** |

**当前判断**:旋转/方向类——两种独立数据(合成 Rotation_Direction + 真实 TOMATO rotation/MVBench moving_direction,见 `distillation-readiness-report.md` 实验1)都确认没有 teacher 优势,ARD 在这一类上**没有信号来源,不建议投入**,和 TRD 的判死刑是同一个结论、同一条证据。但**计数类是一个真正悬而未决的矛盾**:合成的 Bouncing_Counting 显示 teacher 更差,但真实的 MVBench 计数任务显示 teacher 明显更强——这不是"要不要投入"的问题,是"我们现在还不知道真相"的问题。**在投入 ARD 之前,必须先把 probe-experiment 那套方法论(§4.1-4.3 的 probe vs 直答对比)直接搬到真实 MVBench action_count/moving_count 数据上重跑一次**,而不是继续依赖已经被证明不可靠的合成 Bouncing_Counting 代理——如果真实数据上 teacher 的 probe 和 student 打平、但 teacher 直答明显更准,那 ARD 在计数类上就有了干净的立项依据;如果 teacher 的 probe 本身就比 student 高,那问题退回到表示层,不是 ARD 该管的范围。

**数学论证(补齐,和 TRD/CAD 同等级别,2026-07-14)**

**结论一(ARD 不需要、也没有声称修复 OPRD 的排列不变性问题——这是诚实的范围声明,不是缺陷)**:ARD 和 OPRD 一样,是对**单个输出分布**的边际比较($g^T(h^T)$ vs $g^S(h^S)$,不涉及任何跨帧/跨位置的联合结构),所以 ARD **不能独立证明**它对帧序排列敏感——它的排列敏感性是**继承**来的,不是自证的:如果 teacher 自己生成答案的过程本来就依赖帧序(这一点已经实测过,不是假设——`probe-experiment-report.md`§1 的乱序实验显示 35B-A3B 在"真的需要时序推理"的子任务上乱序后掉分 9.20pp/10.71pp,说明 teacher 自己的输出确实依赖帧序),那么让 student 去逼近 teacher 的输出分布,会把这个依赖**连带**教给 student,不需要 ARD 的 loss 函数本身有特殊的数学结构去保证这件事——这和 TRD/CAD"loss 函数自己对排列有非零梯度"是两种不同类型的保证,ARD 这条是"借来的"、条件在"teacher 自己确实是时序敏感的"这个前提上,必须写清楚,不能含糊说 ARD 也"解决了"OPRD 的问题。

**结论二(核心子问题——在 logit 空间是无约束凸优化,可证明收敛到全局最优,和 CAD §2.5.1 结论二同构)**:设 $u\in\mathbb R^{K}$ 是 student 在 $K$ 个候选答案(比如 MCQ 的字母选项,$K$ 通常只有 4–6)上的 logit,$p^T=g^T(h_t^T)$ 是 teacher 算出来的、固定不变的目标分布(teacher 冻结,off-policy)。

$$L_{\text{ARD}}(u) \;=\; D_{\mathrm{KL}}(p^T\,\|\,\mathrm{softmax}(u)) \;=\; -\sum_{k=1}^K p^T_k\,u_k \;+\;\log\sum_{k=1}^K e^{u_k} \;+\;\text{const}.$$

这和 §2.5.1 结论二的形式完全一样(线性项+log-sum-exp),是无约束光滑凸优化(Boyd & Vandenberghe, *Convex Optimization*, 2004, §3.1.5),标准梯度下降有 $O(1/T)$ 收敛到全局最优的保证,固定"加常数到所有 logit"这个规范自由度后可以拿到线性收敛率。**这一条比 CAD 还要干净**:$K$(候选答案数,4–6)通常比 $N$(视觉 token/帧组数,这里是 8)还小,而且不需要像 CAD 那样操心"选哪一层的 attention""是不是 full_attention 层"这些架构细节——ARD 只依赖模型最终吐出来的 logits,这是**任何** decoder-only 模型都有的、结构最简单的量。

**结论三(整体训练——block coordinate descent,收敛到驻点,和 TRD/CAD 共享同一套引用)**:把训练拆成"固定主干优化最后的读出线性层(通往 $u$ 的那一层)"(结论二的凸子问题)和"固定读出层、对主干做一步 SGD"两块交替,论证结构、引用(Tseng 2001;Bottou, Curtis & Nocedal 2018)和 §2.5.1 结论三完全一致,不重复展开。**诚实边界同样适用**:结论二的全局收敛只在"$u$ 直接优化"的理想化子问题成立,联合训练主干参数后只有驻点保证。

**结论四(和 §2 共享的 DPI 论证——ARD 教的是"读出函数",不是"信息量")**:$g^S_\theta$ 训练后逼近 $g^T$,由 DPI,$I(g^S(h^S);y)\le I(h^S;y)$——ARD **不能**让 student 学到任何超出它自己表示 $h^S$ 本身已有信息量的东西。但这恰恰不是 ARD 要解决的问题:probe-experiment.md 已经证明,这几类原语上 $I(h^S;y)$(用 85–100% 的 probe 准确率衡量)本来就**远高于**直答准确率(35–69%)反映出来的 $I(g^S(h^S);y)$——瓶颈根本不在信息量上限,在读出函数 $g^S$ 本身没把这个上限用够。ARD 要做的,恰恰是把 $g^S$ 换成一个更接近 $g^T$(如果 $g^T$ 确实更好)的读出函数,把已经存在于 $h^S$ 里、但没被 $g^S$ 有效利用的信息挖出来——这条和 §2 开头统一给出的 DPI 上限论证完全相容,不冲突。

**pros**:数学上是这三个 idea(TRD/CAD/ARD)里最简单、最干净的——不需要单纯形约束(比 RAD 简单),不需要操心 attention 该选哪一层/哪种 attention 机制(比 CAD/TRD 简单,§2.5 已发现 Qwen3.5 只有 8/32 层是 full_attention);完全绕开"attention 不 faithful"这整个未解决的争议(§2.5.1 已详细讨论),因为它锚定在生成过程真正的终点(最终答案分布)上,没有更下游的路径可以绕过它;结论二的收敛证明是这几个 idea 里约束最少、最"干净"的一版。
**cons**:唯一、也是最致命的前提——teacher 在这个具体任务上直答准确率必须真的更高——目前 4 类原语里 3 类不成立、1 类反而更差,和 TRD/CAD 是同一个致命前提、同一条证据链;不像 CAD/TRD 那样能从"中间过程"里挖出点信号(比如哪怕最终答案打平,attention 模式本身可能有点参考价值),ARD 完全依赖最终准确率这一个数字,没有更细粒度的信号来源可以退而求其次。**2026-07-14核实补充**:Yang et al. 2026(*OPRD*,arXiv:2606.06021,§1.1已引用)明确指出"只匹配最终输出分布"这类做法有一个**架构层面的LM-head信息瓶颈**——teacher中间层更丰富的结构信息,全部会被"压缩成一个vocab大小的分布"这一步给挤掉,这条批评不分on/off-policy,对ARD同样成立,是ARD相对TRD/CAD/RAD(都在中间层做文章,不经过这层瓶颈)一个结构性的、诚实的劣势,不能因为ARD数学干净就忽略这一层信息损失。该论文同时也报告,在文本推理场景下纯输出空间蒸馏会"plateau在teacher水平以下",这是否在我们的视频场景下同样成立,目前没有直接证据,但这条独立文献的发现,和"ARD只有在teacher读出确实更准时才有用"这条我们自己测出来的前提,方向上是互相印证的——都在说"只看最终输出"这条路天花板更低。
**Δ vs doc 02**:六类机制里没有"专门在‘教表示 vs 教读出’这条分界线上,把读出函数单独拎出来当蒸馏目标,且用 probe vs 直答这套诊断去判定该不该做"的机制——response-based KD(Hinton 2015)本身当然不是新东西,这里的 delta 是**把它和 TRD/CAD 放在同一个判定框架下,明确划出"教表示"和"教读出"是两件独立的事、要用独立的证据分别判定要不要投入**。

---

### 2.2 ★ Idea 2 — PSD:Predictive State Distillation(动力学·蒸馏版,依赖 teacher 表示)

**大白话先讲做法**:只看一帧一帧的画面,没法总结出"这个物体现在的状态是什么(在哪、往哪走、多快)"。PSD 给 student 加一个额外的小脑子,专门负责"把到目前为止看到的画面,总结成一个内部信念状态,再用这个信念状态去猜接下来 teacher 会看到什么样的画面"。如果 student 真的理解了运动/状态怎么演变,它应该能比较准地猜中接下来发生什么;这个额外的小脑子只在训练时存在,用来倒逼 student 的内部表示变得更懂"事情是怎么随时间演变的",推理的时候直接丢掉,不留痕迹。

**借的经典思想**:**HMM / Kalman filter 的递归状态** + **Contrastive Predictive Coding(CPC,van den Oord, Li & Vinyals, 2018)的未来预测** + **arrow-of-time 不对称**(Wei, Lim, Zisserman & Freeman, CVPR 2018, *Learning and Using the Arrow of Time*——这篇是"顺放/倒放判别"这个自监督任务的出处,§2.2(c) 和 §3.1 的 AoT 是同一个概念的两个应用位置,下面统一引用这一篇,不重复展开)。

**做法**:在主干上加一个轻量**递归信念头** $\psi$,得 student 的因果状态(递归 = "到 $t$ 为止"的充分统计):

$$b_t^S = \psi\!\left(b_{t-1}^S,\; \operatorname{pool}(h_t^S)\right).$$

三个目标:

$$
\begin{aligned}
\text{(a) 状态蒸馏:}\quad & L_{\text{state}} = \sum_{t} d\!\left(b_t^S,\, b_t^T\right) \\[4pt]
\text{(b) 预测未来 (CPC):}\quad & L_{\text{pred}} = -\sum_{t}\sum_{k} \log \frac{\exp\langle g_k(b_t^S),\, h_{t+k}^T\rangle}{\sum_{n\in\text{Neg}} \exp\langle g_k(b_t^S),\, h_n^T\rangle} \\[4pt]
\text{(c) 时间箭头(见 Wei et al. 2018):}\quad & L_{\text{AoT}}^{\text{rep}} = \sum_{t} \max\!\left(0,\; m + L_{\text{pred}}^{\rightarrow}(t) - L_{\text{pred}}^{\leftarrow}(t)\right)
\end{aligned}
$$

(b) 里负例取其它时刻/其它视频的帧;(c) 令"顺放比倒放更可预测"($L_{\text{pred}}^{\rightarrow}$ 更低)至少差 margin $m$。

**★ 训练方式(主模型不能冻结)**:$\psi$、$g_k$ 确实是**训练期外挂的小网络**,但**重点不是训它们**——而是让 $\lambda(L_{\text{state}}+L_{\text{pred}}+L_{\text{AoT}})$ 的梯度**反传进(不冻结的)主干**,重塑主干隐状态 $h_t^S$ 使其更具预测性/时序性。小网络只是**脚手架,推理时丢弃**(和 SSL 辅助头一样),与主 OPD/RL 目标联合优化。
> 辩证:若**冻结主干、只训小网络** → 那只是个 **probe**(读出"表示里有没有时序结构"),**根本不改进 student**。**这条对 TRD/PSD/PGR 都成立**:只有梯度进主干,才谈得上"提升时序理解",而不只是"读出"。

**"训练期外挂小网络、梯度回传主干、推理期丢弃"这个模式眼熟吗?——查过了,这是深度学习里一个很成熟、且有多篇高影响力论文在用的标准范式,不是本文档发明的架构**:
- **BERT(Devlin et al., NAACL 2019)**:预训练阶段主干上外挂一个 Next Sentence Prediction 分类头,和主任务(MLM)一起训、梯度回传进主干,fine-tune/推理阶段直接丢弃这个头——BERT 是被引用最多的 NLP 论文之一,这个"外挂头+丢弃"的操作本身完全是标准配置,没人觉得奇怪。
- **UNREAL(Jaderberg et al., ICLR 2017,DeepMind,*Reinforcement Learning with Unsupervised Auxiliary Tasks*)**:给 RL agent 的主干外挂好几个辅助头(像素控制、reward 预测),纯粹为了让主干学到更好的表示,真正要用的是策略/价值头——这篇是"RL 辅助任务"这条研究线的开创性论文,引用量很大,机制和 PSD"外挂小网络、梯度进主干、丢弃"完全一样。
- **Mirowski et al.(ICLR 2017,DeepMind,*Learning to Navigate in Complex Environments*)**:给导航 RL agent 的视觉主干外挂"深度预测"和"回环检测"两个辅助头,专门用来让主干学到更好的空间表示,和最终导航策略头分开、训练完就不再需要——这篇和 PGR(§2.3)的相似度最高:都是"预测一个物理/几何真值(这里是深度,PGR 是物理状态)当辅助任务,逼主干学到更懂空间/物理规律的表示"。
- **自监督对比学习(SimCLR、BYOL、MoCo)的 projection/predictor head**:训练时接在 encoder 后面参与对比 loss,下游任务时统一丢弃、只用 encoder 本身——这已经是这个领域的标准范式。
- **Multi-task learning 本身(Caruana, *Machine Learning*, 1997,*Multitask Learning*)**是这整条思路的理论源头:证明了共享主干 + 多个(甚至部署时用不上的)辅助任务头,能让主干学到更好的表示——这是有近三十年历史、教科书级别的机器学习范式。

**结论**:PSD/PGR 用的这个训练模式不是发明出来的、也不奇怪。**真正需要讲清楚 novelty 的地方,是这个辅助任务具体预测什么(PSD:teacher 未来的隐状态;PGR:模拟器给出的精确物理真值)、为什么要预测这个(时序理解 + 我们的合成数据场景),不是"外挂小网络"这个机制本身——这部分可以放心引用上面几篇,不用担心 reviewer 觉得 weird。

**为什么能提升时序**:
- **(b) 预测=逼近动力学**:从"过去状态"预测"teacher 未来表示",必须内化运动/状态**动力学**(速度、方向、隐状态转移)——正是 SynRL 的"temporal primitives",但作为**蒸馏目标**。
- **(c) 打破时间反演对称**:表示时间对称则 $L_{\text{pred}}^{\rightarrow}=L_{\text{pred}}^{\leftarrow}$,margin 恒被激活 → 逼**方向敏感**,治 TVBench/TempCompass 的"分不清方向"。
- **(a)+递归**:状态是**递归充分统计**(不能回看,须逐步更新)→ 教 **state-tracking**(SynRL 长时原语)。

**pros**:直击**动力学/方向/状态**;最贴 SynRL 原语。
**cons**:加预测头 + InfoNCE 负例设计;最难调;**(b) 的目标是 teacher 未来的表示,继承"teacher 表示本身准不准"的风险**——这也是 PGR(§2.3)存在的原因:同样是"动力学预测",PGR 把预测目标换成模拟器的精确物理真值,不需要相信 teacher。
**Δ vs doc 02**:六类里没有"预测编码/状态空间";②T-GRPO 是"奖励-答案-在-乱序下变差",这里是**生成式训练表示去预测未来+编码方向**,机制不同。

#### 2.2.1 数学论证(PSD 的三个子项理论依据不同,分开看,不能笼统一句"数学直觉"带过)

**(b) CPC 预测项——下界不是本文档推出来的,是 CPC 原论文自己的定理,应该引用而不是当直觉写**

van den Oord, Li & Vinyals(2018,*Representation Learning with Contrastive Predictive Coding*)在提出 InfoNCE 损失时,自己证明了:

$$I\big(b_t^S\,;\,h_{t+k}^T\big) \;\ge\; \log(|\text{Neg}|+1) - L_{\text{pred}}.$$

**最小化 $L_{\text{pred}}$ 拉高这个互信息下界,这个证明是 CPC 论文本身的结论**,PSD 只是把它用在"student 预测 teacher 未来表示"这个具体位置上——应该明确引用,不能让读者以为这是本文档推导出来的新结果。

**和 PGR(§2.3.1结论二)的关系**:两者是**同一族**变分MI下界(Barber & Agakov 框架),只是变分分布族选择不同——PGR 用高斯回归族(无负例数量限制,代价是单峰高斯假设),PSD 用范畴/softmax分布族(McAllester & Stratos 2020 证明的负例数量天花板在这里适用,PSD 继承这个限制,真实MI较大时需要指数级多的负例才能估准)。这是同一个数学工具用在两个目标上得到的两种下界,不是两套独立理论,值得明说,方便读者判断该投哪个。

**(a) 状态蒸馏项——它不重蹈 OPRD 的覆辙,原因是架构选择,不是损失函数选择**

$L_{\text{state}}=\sum_t d(b_t^S,b_t^T)$ 表面上和 OPRD 一样是逐位置的边际距离,按 §2.1.1 结论一的论证逻辑,似乎也该"看不见跨帧关系"。**但这里不一样,原因在 $b_t$ 的递归定义本身**:$b_t=\psi(b_{t-1},\operatorname{pool}(h_t))$。只要 $\psi$ 不退化成"和顺序无关的聚合"(比如简单平均——那样 $b_t$ 确实会退化成边际统计量,重蹈 OPRD 覆辙),$b_t$ 作为 $h_1,...,h_t$ 全部历史的**递归充分统计量**,对这段历史的重新排列**一般不是不变的**(标准 RNN/GRU 式更新对输入顺序天然敏感,这是这类架构的通常性质)。**所以 $L_{\text{state}}$"看得见关系"这个性质,来自网络架构的选择(用递归 $\psi$),不是来自损失函数的形式**——这和 TRD(损失函数本身耦合跨帧项)是两种不同的机制来源,不能笼统归为一类。**前提**:这依赖 $\psi$ 确实是非对称/非置换不变的设计,如果谁把 $\psi$ 设计成对称聚合,这条论证不成立,这是个设计约束,不是自动满足的。

**(c) 时间箭头 margin——这是损失在对称点取值的精确推论,不是经验观察**

定义"时间对称表示"为:正放和倒放预测同样容易,即 $\mathbb{E}[L_{\text{pred}}^{\rightarrow}]=\mathbb{E}[L_{\text{pred}}^{\leftarrow}]$。**按这个定义**,margin 项 $\max(0,m+L_{\text{pred}}^{\rightarrow}-L_{\text{pred}}^{\leftarrow})$ 在对称点上取值恰好是 $\max(0,m)=m>0$(两个 $L_{\text{pred}}$ 相等,差为零)——**只要 $m>0$,损失在"时间对称"这一点上严格大于零**,梯度下降会推着表示离开这个对称点。这是损失函数在定义点上取值的直接推论,精确刻画了这个 margin 项**为什么会**惩罚时间对称的表示,而不是"经验上发现有效"这种需要跑实验才能确认的说法(这一支的出处见 §3.1 对 Wei et al. 2018 的引用,这里不重复展开)。

---

### 2.3 ★★ Idea 3 — PGR:Physics-Grounded Reward(动力学·可验证版,2026-07-06 替换旧 GCR,不需要 teacher)

**大白话先讲做法**:PSD(§2.2)让 student 去预测"teacher 会看到什么",但 teacher 自己都不一定看得准(§0.6)。PGR 换一个不会犯错的老师——我们自己写的物理模拟器。因为视频是我们自己用物理公式生成的,任意时刻球的真实速度、位置,我们是精确知道的,不用猜、不用问 teacher。PGR 就是也给 student 装一个小脑子,让它根据当前画面猜"这个物体现在的真实物理状态",直接拿模拟器算出来的精确答案对答案——猜得准,说明它内部表示真的抓住了物理规律;猜不准,就调整参数直到能猜准。这个小脑子和 §2.2 的用法一样,只在训练时存在,训完丢掉。

**"外挂小网络训练"这个模式的先例,见 §2.2 已经列的那几篇(BERT/UNREAL/Mirowski et al./对比学习 predictor head)——PGR 用的是同一套已经很成熟的训练模式,不再重复列一遍,差异只在"这个小网络具体要预测什么"。**

**为什么删掉旧 GCR、换成这个**:旧版 GCR 有两个硬伤——① 起名"Granger causal",但实际算的是留一帧消融重要性(occlusion-based attribution,更准确的说法参见 Zeiler & Fergus 2014 的"遮挡敏感度"、或 Shapley 值归因这一支),和 Granger causality(时间序列econometrics里"过去的X能不能帮助预测未来的Y"的具体统计检验)不是一回事,审稿人里但凡懂因果推断会直接指出这个命名错误;② 核心信号 $\varphi^T$ 完全依赖 teacher 答对题目的 log 概率,而我们已经证实 teacher 在这些任务上基本等于随机猜(§0.6),在噪声上做差分大概率还是噪声。**PGR 从根上避开这两个问题**:不叫因果,叫"物理一致性";监督不来自 teacher,来自我们自己写的模拟器。

**核心资源(这个项目独有,别人复现不了)**:SynRL 生成器不是随便渲染的,是**显式物理方程模拟**——匀速直线运动、碰撞反弹、匀角速度旋转、加减速——任意帧 $t$ 的真实物理状态 $s_t$(位置、速度、角位置、角速度等低维连续量)对我们来说是**精确已知的**,因为生成器代码本身就是这个物理过程的实现。真实视频没有人知道镜头里的球到底什么速度,**这条路径只有在物理模拟合成数据的设定下才成立**。

**做法(形式化)**:
让 $h_t^S = \text{Encoder}_\theta(x_{\le t}) \in \mathbb{R}^d$ 是 student 在帧 $t$ 位置的隐状态(因果,只依赖 $\le t$ 的帧)。训一个轻量读出头 $g_\phi: \mathbb{R}^d \to \mathbb{R}^m$,预测 $k$ 步之后的真实物理状态:

$$\hat s_{t+k} = g_\phi(h_t^S), \qquad L_{\text{PGR}} = \mathbb{E}_t\big[\,\|\hat s_{t+k} - s_{t+k}^{\text{GT}}\|_2^2\,\big].$$

$s_{t+k}^{\text{GT}}$ 直接从生成器的模拟状态日志里读(工程上需要在生成器里加一点逐帧物理状态 dump,目前 `metadata.jsonl` 只存了事件时间线,不存逐帧速度/角速度,这是个需要补的轻量生成器改动,不难)。

**用法**:①当训练损失(梯度进主干,同 §2.2 的辩证);②折成 reward 形式 $r_{\text{PGR}}=-L_{\text{PGR}}/(2\sigma^2)$ 并入 RLVR;③当 probe-experiment.md 的一个新增探针维度(回归物理量,而不只是分类方向/计数标签)——这一步不需要额外训练机制,复用已有 infra。

#### 2.3.1 数学论证(为什么这不只是直觉,三条各自独立、各自有边界的严格结论)

> 这里刻意把"能证明什么"和"不能证明什么"分开写清楚,不做超出结论本身的宣称。

**结论一(信息论必要性——无条件成立,直接给 probe-experiment.md 一个正式的理论基础)**

对任意下游函数 $\hat y = \Psi(h^S_{1:N})$(student 自己最终生成答案所依赖的整个计算,不管这个计算多复杂、CoT 多长),数据处理不等式(Cover & Thomas, *Elements of Information Theory*)给出:

$$I\big(\hat y \,;\, s_{t+k}^{\text{GT}}\big) \;\le\; I\big(h_t^S \,;\, s_{t+k}^{\text{GT}}\big).$$

这是**无条件成立**的信息论恒等式,不依赖任何关于模型结构或训练过程的假设。它的含义:**答案里包含的关于真实物理状态的信息,永远不会超过产生这个答案所依据的表示里已经有的信息。** 这正是 probe-experiment.md 整套诊断逻辑的形式化版本:如果 probe 在 $h_t^S$ 上都读不出 $s_{t+k}^{\text{GT}}$(即 $I(h_t^S;s_{t+k}^{\text{GT}})$ 本身就低),那么无论后续用多复杂的读出机制/多长的CoT/多强的RLVR压力,答案对真实物理状态的信息量都**不可能超过这个上限**——这从数学上说明了"为什么必须先做 probe 实验,再决定投 TRD/PSD/PGR":这三个方法都在试图**提高** $I(h_t;s_{t+k})$,只有当这个量本身是瓶颈时,投资才有意义;如果瓶颈在读出层(概率上,probe 能读出但答案还是错),这三个方法都不对症。

**结论二(PGR 降低 $L_{\text{PGR}}$ 等价于拉高一个 $I(h_t^S;s_{t+k}^{\text{GT}})$ 的合法下界——条件在于选定的变分分布族,但下界本身对任何选择都合法)**

Barber & Agakov(2003)的变分互信息下界:对任意变分分布 $q_\phi(s\mid h)$(不要求等于真实后验 $p(s\mid h)$):

$$I(h_t^S; s_{t+k}^{\text{GT}}) \;\ge\; H(s_{t+k}^{\text{GT}}) + \mathbb{E}\big[\log q_\phi(s_{t+k}^{\text{GT}} \mid h_t^S)\big].$$

这个不等式对**任意** $q_\phi$ 都成立(只在 $q_\phi$ 等于真实后验时取等号,其余情况是有效但可能不紧的下界)——这是变分推断里一个标准、久经考验的结论,不是本文档发明的。取 $q_\phi(s\mid h) = \mathcal N(g_\phi(h), \sigma^2 I)$(高斯变分族,均值就是我们的读出头),代入:

$$\mathbb{E}[\log q_\phi(s_{t+k}^{\text{GT}}\mid h_t^S)] = -\frac{1}{2\sigma^2} L_{\text{PGR}} - \frac{m}{2}\log(2\pi\sigma^2).$$

于是

$$I(h_t^S; s_{t+k}^{\text{GT}}) \;\ge\; H(s_{t+k}^{\text{GT}}) - \frac{1}{2\sigma^2} L_{\text{PGR}} - \frac{m}{2}\log(2\pi\sigma^2).$$

$H(s_{t+k}^{\text{GT}})$ 是个**只由物理生成过程决定的常数**,和 $\theta,\phi$ 无关。所以**最小化 $L_{\text{PGR}}$ 直接拉高这个互信息下界**——这是训练 PGR 能提升表示的物理信息含量的正式依据,不只是"预测误差小了应该更懂物理"这种直觉。

**和 PSD 的 CPC 目标对比(诚实的优劣两面)**:CPC 的 InfoNCE 目标也是同一族的变分 MI 下界,只是变分族换成"有限负例上的 softmax 分类"而不是这里的高斯回归。McAllester & Stratos(2020,*Formal Limitations on the Measurement of Mutual Information*)证明 InfoNCE 这类基于有限对比样本的 MI 估计,下界数值被负例数量的对数**天花板**卡住($\lesssim \log|\text{Neg}|$),真实 MI 较大时需要指数级多的负例才能估准。PGR 用**精确连续回归目标**,没有这个负例数量瓶颈。**但代价是**:这个高斯变分族假设后验是单峰的——如果真实物理状态在给定帧下有多解(比如对称遮挡场景),高斯假设会让下界变松(仍然合法,只是不紧),这点要老实承认,不能只讲优点不讲局限。

**结论三(纳入 RL 训练时的安全性保证——需要用势函数形式,且保证的是"理想化 MDP 最优解不变",不是"具体训练一定收敛到它")**

如果把 PGR 折成 potential-based reward shaping(Ng, Harada & Russell, ICML 1999,*Policy Invariance Under Reward Transformations*)的形式,而不是直接把 $-L_{\text{PGR}}$ 加到 reward 上:

$$F_{\text{PGR}}(s_t, s_{t+1}) = \gamma\,\Phi(s_{t+1}) - \Phi(s_t), \qquad \Phi(s_t) := -\|g_\phi(h_t^S) - s_t^{\text{GT}}\|^2,$$

该定理保证:**在总 reward $r_{\text{outcome}}+F_{\text{PGR}}$ 下的最优策略,和只用 $r_{\text{outcome}}$(纯 RLVR 正确性奖励)时的最优策略完全相同**——PGR 只能改变学到这个最优策略的**速度/路径**,不能改变**最优策略本身是什么**。

**这比当年给 GCR 设计的同款补丁更站得住**:定理对"势函数写成什么"没有要求,永远保证策略不变;但**势函数选得好不好,决定这个 shaping 在实践中有没有用**。GCR 的势函数如果建立在 teacher 不可靠的 $\varphi^T$ 上,虽然定理仍然成立(策略不变),但这个 shaping 本身可能是**没有信息量的噪声**,加了白加。PGR 的势函数建立在**精确物理真值**上,不存在"势函数本身就是错的"这个问题——同一个定理,套在一个可信得多的势函数上。

**诚实的边界(不能证明的部分)**:Ng et al. 的定理是关于**理想化 MDP/收敛到的不动点**的陈述,不是关于"某个具体深度RL算法(GRPO,函数逼近,有限样本,非平稳策略更新)在有限训练步数内一定能找到这个最优策略"的陈述——这是所有深度RL共享的、目前没有通解的问题,PGR 不能免俗。定理保证的是"目标没被偷换",不是"一定能到达目标"。

**结论四(概率/probe 阶段的完整收敛保证——只在"主干冻结+线性读出头"这个子问题成立)**

如果 $\theta$(主干)冻结、$g_\phi$ 限制成线性($g_\phi(h)=Wh+b$),$L_{\text{PGR}}(\phi)$ 是 $\phi=(W,b)$ 的**凸(二次)函数**——这是标准最小二乘回归,梯度下降(步长选得当)可证明收敛到全局最优,甚至有闭式解 $\phi^*=(H^\top H)^{-1}H^\top S$。**这个干净的收敛保证只对 probe-experiment.md 里"冻结主干做诊断"这一步成立**;一旦把 PGR 用作端到端训练(主干本身也在更新),问题变成非凸,和深度学习里几乎所有目标一样,不再有通用收敛保证——这不是 PGR 特有的缺陷,是诚实地承认深度学习优化本身没有普适收敛证明。

**小结这四条**:结论一告诉我们**什么时候** PGR(或TRD/PSD)值得投——只有 $I(h_t;s_{t+k})$ 本身是瓶颈时(probe-experiment.md 负责查这件事);结论二告诉我们 PGR 的训练目标**确实**在拉高这个信息量,且指出它相对 PSD 的 CPC 目标在哪方面更强、哪方面要打折扣;结论三告诉我们**把它接入 RL 训练不会偷偷改变"正确性"这个终极目标**,只是可能加速;结论四告诉我们**诊断阶段**(probe)有干净的收敛证明,训练阶段没有——四条各自的适用范围都写明白,不越界宣称。

**pros**:不需要teacher(彻底规避 §0.6 的核心风险);监督**精确**(不是"更可信",是**已知为真**);数学论证完整(见上);可以直接扩展 probe-experiment.md,复用已有 infra。
**cons**:需要给生成器加逐帧物理状态 dump(轻量工程,目前没有);只对"物理量可以显式定义"的原语类型有效(比如 03-10 号那些符号操作/记忆类生成器,如洗牌/滑块,可能没有干净的连续物理量可回归,需要针对每个生成器单独看有没有自然的 $s_t$ 定义);高斯假设在多解场景下会让 §2.3.1 结论二的下界变松。
**Δ vs doc 02**:六类机制里没有任何一个能拿到"精确物理真值"这种东西——因为它们全部基于真实视频,而真实视频没有已知的生成物理方程。这条**只有在合成物理模拟数据下才成立**,是这个项目相对任何真实视频方法的结构性优势,不是重新包装已有机制。

---

### 2.4 ★ 补充 Idea — ESD:Event Segmentation Distillation(借认知科学,不需要 teacher)

**大白话先讲做法**:人脑怎么知道"一件事结束了、新事件开始了"?认知科学的解释是靠"预测失误"——你脑子一直在下意识预测接下来会发生什么,当预测突然错得离谱,你就意识到"刚才发生了点意外的事",这就是一个事件边界。ESD 照搬这个机制:给 student 也装一个"猜下一帧大概长什么样"的小脑子,如果猜测误差在某一帧突然变大,就说明那一帧发生了关键事件——这样不需要 teacher、不需要标注,就能自动找出"哪帧关键"。这个小脑子同样只在需要的时候用,机制上和 §2.2 列的那几篇先例是同一类做法。

**借的经典思想**:认知科学的**事件切分理论**(Event Segmentation Theory,Zacks & Swallow, *Current Directions in Psychological Science*, 2007)。核心机制:人脑把连续体验切成离散事件,靠的是**预测误差在事件边界处突然飙高**——"接下来会发生什么"的预测模型在边界处失效,误差尖峰触发"这是新事件"的判断。**老实说明**:"用预测误差做无监督事件/时序切分"这个机制本身,在计算认知科学和预测编码(predictive coding)相关的计算建模文献里已有探索,不是这篇文档首创——这里的 delta 在于**把它用作 teacher-student 蒸馏信号**,而不是当成独立的无监督表征学习目标。

**和 PSD/PGR 的关系**:§2.2 的 PSD 已经有一个 CPC 式的预测损失 $L_{\text{pred}}$,§2.3 的 PGR 有精确物理回归损失,但那里预测误差只是**训练目标**(逼表示更能预测),没有被当成**信号本身**去用。ESD 把这层关系挑明、且反过来用:

$$e_t = \big\|\, \hat h_t - h_t \,\big\|,\qquad \hat h_t = g\big(\operatorname{pool}(h_{t-1})\big)$$

$e_t$ 高 = 帧 $t$ 处发生了"预测不到的事"(事件边界)。这给出一个**完全不需要 teacher** 的、无监督的"哪帧关键"信号——直接在 student 自己的表示上算,不依赖任何外部 oracle 是否可靠。

**用法(两种,成本递增)**:
1. **当验证工具**:算 student(和 teacher)各自的 $e_t$ 曲线,和合成数据 `metadata.jsonl` 的真实事件时刻对比,看"预测误差尖峰"是否和真实事件对齐——这是比 §5② 更便宜的一种"表示有没有编码事件结构"的检验,不需要 ablation,几个前向就出结果。
2. **当训练信号**:把 $e_t$ 归一化后当 TRD 式的辅助监督目标(逼 student 的 $e_t$ 尖峰和 teacher 的 $e_t$ 尖峰对齐,或者干脆和 GT 事件时刻对齐——后者甚至不需要 teacher,和 PGR 一样彻底绕开 teacher 可靠性问题)。

**pros**:不需要 teacher(规避 §0.6 的核心风险);比消融式方法便宜得多(一次前向);有认知科学理论背书,不是拍脑袋的启发式。
**cons**:预测误差是"意外性"的代理,不是"重要性"本身(纯噪声也会有高预测误差,需要和"合理性"做区分,比如结合 §2.1 的 attention 权重做联合判据);$g$ 这个小预测头需要额外训练(但比 PSD 的完整递归信念头+InfoNCE 轻量很多);机制本身(预测误差当事件边界)不是本文档原创,这一点要老实承认。
**Δ vs doc 02**:六类里没有"无监督预测误差当事件边界"这个机制——五类(VideoSSR)是"扰动位置已知"的自监督,六类是"归因已有 reward 怎么摊",ESD 是**在没有任何外部信号时,从预测误差本身诞生一个"哪帧关键"信号**,机制来源不同。

---

### 2.5 ★★ 补充 Idea — CAD:Cross-attention Attention Distillation(2026-07-14,读出层的TRD——query从"帧"换成"输出token")

**背景/和上一版RAD的关系(如实记录一次返工)**:这一节最初写成了RAD(逼student用自己的帧重建teacher的帧表示),但这不是最初的想法——原话是"**输出的token会有注意力在已经编码进去的视觉token里面,让student学习teacher这个已经编码进去的视觉token的分布**"。这句话说的其实是TRD的$A_{ij}$的一个自然推广:TRD的query是"帧$i$"(编码阶段,帧看帧);这里的query是"**生成答案时的输出token**"(读出阶段,答案token看视觉token)。数学骨架和TRD完全一样(都是softmax行的KL匹配),只是query从"另一帧"换成了"正在生成的这个词"。下面按这个更贴近原话的版本重写,RAD作为一个提过的、更间接的备选方案保留在§2.5.2,不再作为主线。

**大白话先讲做法**:模型在吐出答案的每一个词时,会去看上下文里已经编码好的那些视觉token——看哪几个、每个看多少,是一个attention分布。CAD要做的事很直接:**把teacher在生成答案时,每个词对视觉token的这个"看的分布",直接当成目标,让student在生成同样的词时,也去逼近这个分布。** 不管teacher内部怎么算出这个分布的,只要它看的地方是对的(比如数到第3次反弹这个词的时候,确实主要在看第3次反弹发生的那几帧),就把这个"看的地方"直接教给student,不需要student自己去试错发现该往哪看。

**借的经典思想**:①**Attention Transfer**(Zagoruyko & Komodakis, ICLR 2017,已在§2.1引用)——CAD和TRD共享同一个"蒸attention分布"的机制内核,区别只在query轴选的是编码阶段的帧、还是读出阶段的输出token,这一点必须明确写出来,不能显得是重新发明;②**序列到序列蒸馏里对cross-attention做监督**这条路子,在机器翻译/摘要的encoder-decoder蒸馏里有过(用teacher的cross-attention/对齐矩阵指导student,这类"attention alignment as supervision"的做法不是本文档首创,是把它专门用在"教一个视频模型该往回看哪几帧"这个问题上);③**Hinton et al. 2015 KD**的"用连续分布而不是one-hot硬标签当监督目标"这个基本思想,这里被用在attention分布上而不是最终类别分布上。

**记号**:视觉token/帧组表示 $h_1,\dots,h_N$(已经编码进上下文,teacher/student各自的$h_j^T,h_j^S$,和TRD一样要求同一个$N$、同一组物理时刻对应同一帧索引,§0.5(3)硬前提原样适用)。生成答案(或teacher-force进同一段GT答案文本/CoT,保证teacher/student在**同一个**答案序列上逐位置可比,这点和TRD"必须同一批帧"是同一类对齐要求)在第$t$步的输出token表示记$z_t^T,z_t^S$。定义输出token对视觉token的attention分布:

$$B_{tj} \;=\; \frac{\exp\!\big(\langle W_q z_t,\,W_k h_j\rangle/\sqrt d\big)}{\sum_{j'=1}^{N}\exp\!\big(\langle W_q z_t,\,W_k h_{j'}\rangle/\sqrt d\big)},\qquad j=1,\dots,N$$

(不需要像TRD的$j\le i$那样加因果窗口——生成到第$t$步时,全部$N$个视觉token早就已经在上下文里了,可以看全部)。teacher/student各自算出$B^T,B^S$(**这一步天然不需要$d^S=d^T$**——$B$是"分配在$N$个视觉位置上的概率",维度只取决于视觉token数$N$,和teacher/student各自hidden_dim无关,不需要RAD里那个投影矩阵去凑维度,这是这版相对RAD的一个真实的工程简化)。蒸馏损失,和TRD形式完全一样(把query换成$t$):

$$L_{\text{CAD}} \;=\; \sum_{t} D_{\mathrm{KL}}\!\big(B^T_{t,:}\;\big\|\;B^S_{t,:}\big).$$

**2026-07-14修正:$t$在我们真实要用的协议下基本只有1个,不是一长串,这是实测出来的,不是假设**——`direct_answer_9b.json`里4类gap原语的真实原始输出(thinking关闭协议下)清一色是"字母+换行"(比如Rotation_Direction的真实response是`'C\n'`,2个字符),不是一长段CoT文本。所以上面这个对$t$求和的公式,在我们要用的场景下**退化成单个$t$**(生成那个答案字母时刻的那一个位置),不是真的有很多个输出token要逐个处理。**这不是砍掉了CAD,是把它的scope修正到和我们已有证据(thinking-on重测已实测证明明确变差,`distillation-readiness-report.md`实验0)一致的地方**——真正要优化的行为本来就是"直接吐一个字母",不是一长串推理,所以CAD盯着这一个决策点上的attention分布,而不是假设一整段CoT,才是对的scope。**但"只有一个token"不等于"只有一个数"**:这一个决策点在模型内部要经过几十层,每层(每个head)各自算一次对视觉token的attention分布,层与层之间通常不同——$L_{\text{CAD}}$真正要蒸的是"这一个位置,在选定的层/head上"的分布,求和/平均的维度是**层和head**,不是输出位置$t$。下面的数学论证在$t$退化成单个位置时全部照样成立(结论一/二/三都没有依赖"$t$要有很多个"这个前提,退化成单点只是让求和变成单项)。

作为辅助loss叠加在主任务上训练。

#### 2.5.1 数学论证

**先老实过一遍§1.2对"reinforcement attention learning"的三条批评,CAD修了哪几条、没修哪条(不要选择性地只讲修好的部分)**

1. *"attention不faithful,权重≠信息被因果使用"*——**这条CAD没有修,和TRD共享同一个软肋,这里把"为什么"说具体(2026-07-14组会追加)**。一层attention里,权重($W_q,W_k$算出来)和实际取到的内容(独立的$W_v$算出来)是两套完全分开训练的参数,权重高不等于取到的内容真的被用上;算完这一层还有**残差连接**(token自己原来的表示原样往下传,不会因为这层权重低就被冲掉)和**MLP层**(完全不看attention、独立处理,已有可解释性研究——Geva et al. 2021,*Transformer Feed-Forward Layers Are Key-Value Memories*——发现transformer相当一部分"知道什么"其实存在MLP里);几十层叠起来后,还会出现"多跳"问题:某个视觉token的信息可能在早期层被另一个token"听走"、再往后该token只需要看那个中转token就间接拿到信息,只看某一层的attention图会误判成"原始token不重要"。这不是本文档的担忧,是有实证研究的争议:Jain & Wallace(EMNLP 2019,*Attention is not Explanation*)找到过对同一输入、差别很大的attention分布却给出几乎相同输出的反例(Wiegreffe & Pinter 2019有部分反驳,这场争论目前无定论)。直接匹配attention分布,不保证student真的按这个分布去"用"视觉信息,这一点在TRD自己的cons里已经写过,对CAD同等力度适用,不能因为换了query轴就假装这个问题消失了。
2. *"没有GT attention,只能退化成蒸teacher的attention"*——**CAD不是在回避这条批评,是坦然接受它**:CAD从一开始就没打算装成RL,它就是蒸馏,目标就是teacher的分布,不装"探索"。这条原本是冲着"打着RL旗号却偷偷变成蒸馏"去的,CAD不存在这个自我矛盾。
3. *"末端reward高方差,attention会塌到和reward相关但非因果的伪模式"*——**这条被修掉了**:$L_{\text{CAD}}$是逐token、逐位置的稠密KL,不经过任何"生成完整答案→算对错→反传”这条稀疏、高方差的链路,方差结构上和TRD一样干净。

**结论一(排列敏感性——直接复用TRD§2.1.1结论一的论证,不需要重新证)**:$L_{\text{CAD}}$和$L_{\text{TRD}}$在数学形式上是同一个东西(某个softmax行对固定目标行的KL),TRD证过的"这类loss依赖整行联合分布、不是任何边际统计量的函数,因此对(视觉token的)排列有非零依赖"这条论证,把"帧$i$"替换成"输出token$t$"后逐字成立,不需要另开一套证明。**这一条现在还有实证支持,不是纯理论**:§1.5的乱序实验已经实测确认,把视觉输入打乱后,两个模型的真实准确率(因而,合理推断,它们各自的attention分布)确实会变,这不是假设,是`probe-experiment-report.md`已经量出来的数字。

**结论二(核心子问题在logit空间是凸优化,可证明收敛到全局最优——比RAD的版本更干净,不需要单纯形约束)**

固定视觉表示$\{h_j\}$和query表示$z_t$,把attention的**logit** $u_j:=\langle W_q z_t,W_k h_j\rangle/\sqrt d$当自由变量(不经过$(W_q,W_k)$参数化这层),要解的问题是:

$$\min_{u\in\mathbb R^N}\ D_{\mathrm{KL}}\big(B^T_{t,:}\,\big\|\,\mathrm{softmax}(u)\big) \;=\; -\sum_j B^T_{tj}\,u_j \;+\;\log\sum_{j}e^{u_j} \;+\;\text{const}.$$

**这是无约束凸优化**:$-\sum_j B^T_{tj}u_j$是线性(凸);$\log\sum_j e^{u_j}$(log-sum-exp)是标准的凸函数,这是凸分析教科书级别的事实(Boyd & Vandenberghe,*Convex Optimization*,2004,§3.1.5)。**比RAD那版更干净的地方**:不需要在单纯形上做约束优化(不需要Frank-Wolfe/投影),这就是普通的、无约束的光滑凸优化,标准梯度下降就有$O(1/T)$收敛到全局最优的保证(Nesterov经典结果);log-sum-exp的Hessian是$\mathrm{diag}(\mathrm{softmax}(u))-\mathrm{softmax}(u)\mathrm{softmax}(u)^\top$,半正定但沿全1方向奇异(加常数到所有logit不改变softmax,这是这个问题内在的规范自由度,不是缺陷)——**固定这个规范**(比如约束$\sum_j u_j=0$)后,问题在商空间上是严格凸的,可以拿到更强的线性收敛率。这条结论只对"$u$本身被当成自由变量"这个理想化子问题严格成立,真实训练里$u$由$(W_q,W_k)$通过内积产生,这层参数化本身不保证凸——**这是和RAD结论二完全同类型的诚实边界**,不能回避。

**结论三(整体训练——block coordinate descent,收敛到驻点,论证结构和RAD一致,照抄同一套引用)**:把训练拆成"固定主干优化$(W_q,W_k)$对应的$u$"和"固定attention、对主干做一步SGD"两块交替。前者在$u$-空间是结论二的凸问题;后者是标准非凸SGD的下降引理(学习率$\le 2/L$,$L$-光滑时单步不增,Bottou, Curtis & Nocedal 2018)。两块拼起来是标准block coordinate descent,套用Tseng(2001)的定理:**损失序列单调不增、收敛到某极限;梯度Lipschitz的额外假设下,迭代点的极限点是联合(非凸)目标的驻点**。**诚实边界和RAD一致**:这不是全局最优的证明,深度非线性主干上不存在这种通用证明,结论二的$O(1/T)$全局收敛才是"干净"的那一半,结论三只能到驻点,这是深度学习优化理论目前的天花板,不是CAD的特有短板。

**结论四(诊断意义,呼应§2共享的DPI论证,也呼应probe-experiment.md的方法论)**:$B^S$收敛后如果还是学不像$B^T$(KL降不下去),说明student的query-key几何结构本身表达不出这个attention模式(是表示/架构容量问题,不是CAD训练本身的问题);如果学得像但答案还是不对,说明"看对地方"本身不足以答对(读出的其它环节,比如从attention输出到最终token的映射,还有别的毛病)——这条诊断路径和probe-experiment.md"先查有没有、再查读不读得出来"的整体方法论一致,不是另起一套。

**pros**:数学骨架直接复用TRD、无需另起炉灶;不要求hidden_dim对齐(和我们9B/35B-A3B维度不同的真实情况天然兼容,比RAD更简单);逐token稠密梯度,不是稀疏末端reward(修掉了reinforcement attention learning的第③条硬伤);logit空间的收敛证明比RAD的单纯形约束版本更干净(无约束凸优化,标准梯度下降$O(1/T)$)。
**cons(不要回避)**:**attention不faithful这条,和TRD一模一样,没有被修掉**——这是CAD最大的软肋,不能靠换个query轴假装解决;需要teacher-force同一段答案文本让$t$可比,工程上比TRD多一层对齐要求;和RAD一样,收敛证明只在"logit/权重直接优化"这个理想化子问题严格成立,真实的深度参数化训练只有驻点保证。

**最关键的、决定这个idea到底有没有用武之地的一条,和TRD/ARD(§2.1.2/§7.1)是同一条判据,必须重复一遍不能省**:**CAD能不能帮上忙,只取决于teacher在生成时"看的地方"是不是真的比student更对——这等价于teacher的直答准确率是不是真的比student高**。这一条已经在4类原语上实测过(`probe-experiment-report.md`§3):Rotation_Direction/Rotation_Count/Acceleration_Identification三类teacher和student基本打平,Bouncing_Counting上teacher反而更差——**这4类原语上没有信号可蒸,CAD和TRD、ARD一样没有用武之地**。唯一还悬而未决、值得优先验证的是**计数类在真实MVBench数据上的discrepancy**(§7.1已经写过):合成的反弹计数说没优势,真实的action_count/moving_count说teacher明显更强——如果这条差异被证实,CAD(或ARD)该优先在**真实计数任务**上试,而不是继续在已经证明没有信号的合成代理上打转。
**Δ vs doc 02**:六类机制里没有"把TRD的query从编码阶段的帧换成读出阶段的输出token、专门蒸cross-attention"这个做法。

#### 2.5.2 备选:2026-07-14最初写的RAD版本(重建式,间接,现在降级为备选)

用student自己的帧表示,按一个学出来的attention加权组合去重建teacher某一帧的表示(而不是直接匹配attention分布本身),损失$L_{\text{RAD}}=\sum_i\|P_\theta(\sum_{j\le i}\alpha_{ij}h_j^S)-h_i^T\|^2$。这个版本的好处是完全不需要teacher的attention分布本身有意义(只需要teacher的表示可信),坏处是多了一个投影矩阵$P_\theta$去处理$d^S\neq d^T$,机制上也比直接匹配attention绕了一层。**两版共享同一条"要不要投入"的判据(teacher读出准不准更高)**,工程上CAD(§2.5.1)更直接、更贴近最初的想法,优先选它;RAD留作"如果CAD训练发现teacher的attention分布本身很不稳定/噪声很大,不适合直接当KL目标"时的备选方案。

---

## 3. Arrow-of-time(借用现成任务当廉价诊断,不是novelty claim)+ DTW(已砍)

### 3.1 AoT arrow-of-time —— 便宜、能顺手验证,但要讲清楚它不是我们的贡献

**思想**:时间有方向;真懂时序的模型,对**顺放**和**倒放**同段视频**应表现不同**。顺倒不分 = 没编码方向。**这个任务本身来自 Wei, Lim, Zisserman & Freeman(CVPR 2018,*Learning and Using the Arrow of Time*)**——"arrow of time"这个名字就是这篇论文定的,训练网络判定视频顺放/倒放。本文档用它当 RL margin loss 是**新的应用位置**(自监督分类任务 → RL/OPD 的辅助 margin),但"顺放/倒放这件事本身能反映时序理解"这个洞察不是我们的,必须引用,不能让读者以为是本文档发明的概念。

**和 T-GRPO(doc02 机制二)的真实区别**:表面都是"原始帧序 vs 扰动帧序"的对照,容易被审稿人认为是同一件事的重新包装,但这两者继承的是自监督视频学习里**两个不同、各自命名的任务**——T-GRPO 的随机打乱对应 **Shuffle-and-Learn**(Misra, Zitnick & Hebert, ECCV 2016,*Unsupervised Learning using Temporal Order Verification*),测的是"用没用到顺序"(粗粒度、一个bit);AoT 的确定性反转对应上面这篇 **Arrow of Time**,测的是"分不分得清方向"——模型完全可能对打乱敏感(察觉"这段不对劲")却依然分不清哪头在前哪头在后,这是比"order-sensitive"更强的一个要求。这个区分在文献里是成立的(两个任务确实分别独立被研究),不是本文档现造的说法。

**最简做法(答案侧,~10 行,不需要预测头)**:
1. 一道**方向/顺序**类题,顺放 → 记正确答案 logprob $s_{\text{fwd}}$;
2. 帧**倒序** `frames[::-1]` 再跑 → $s_{\text{rev}}$(倒放后"A 先于 B"应变假);
3. margin 项逼"顺放比倒放明显更对":

$$L_{\text{AoT}} = \max\!\big(0,\; m - (s_{\text{fwd}} - s_{\text{rev}})\big).$$

顺倒不分($s_{\text{fwd}}\approx s_{\text{rev}}$)就罚 → 逼方向敏感。
- **只对"倒放会改答案"的题用**(方向/顺序题);"弹了几次"这种计数题倒放不变,**不能用**。
- 表示侧还有个进阶版($L_{\text{AoT}}^{\text{rep}}$,见 §2.2c,同一篇 Wei et al. 2018 的应用),但**先用这个答案侧的**。
- **定位澄清**:这条不是我们的novelty贡献,是借了2018年就有的经典自监督任务;它在本文档里的价值是"便宜、能立刻跑、可以顺手验证",不建议在故事里当成"contribution"去卖。

#### 3.1.1 两个更细粒度的变体(借自监督视频表示学习经典课题——已对照 doc02 排重)

doc02 的机制二(T-GRPO)、机制五(TPO/Synthetic-Pref/VideoSSR)都已经在用"打乱/倒放/扰动"当时序信号来源了。下面两个变体的价值**不在于"又发现了打乱有用",而在于比 doc02 现有做法更细的粒度**:

- **排序预测(借 Misra 2016 Shuffle-and-Learn / Lee, Huang, Singh & Yang, ICCV 2017,*Unsupervised Representation Learning by Sorting Sequences*)**:doc02 机制二只是"正序 acc 是否高于乱序 acc"(一个标量对比);机制五的 TPO/Synthetic-Pref 是"偏好正序 vs 乱序"(一个二元偏好)。**排序预测**给一组打乱的帧,要求模型(或表示上的辅助头)**恢复完整排列**——信息量远大于"正序更好"或"偏好正序",是同一大类思想里**更 dense 的监督**,而且**不需要 teacher**(自监督,直接在 student 自己的表示上训)。
- **最小对局部交换(借语言学 minimal pairs + CV hard negative mining,和 Temporal Cycle-Consistency 这支细粒度时序对齐工作相关——Dwibedi, Aytar, Tompson, Sermanet & Zisserman, CVPR 2019,*Temporal Cycle-Consistency Learning*)**:doc02 机制五的扰动都是**粗粒度**的(整段打乱、整体倒放、扰动一整段)。SynRL 生成器给了我们**逐事件的精确控制权**——可以只交换**两个相邻事件的先后顺序**,其余像素级完全不变,生成"最小对"。这比全局打乱/倒放精确得多:模型必须真的分辨"事件A在事件B之前"这种局部时序,而不是"整体顺序对不对"这种粗糙信号。**这个粒度只有在合成数据完全可控时才做得到,真实视频做不出精确的最小对**——这是这条相对 doc02 五类里现有方法、以及 TCC 这类真实视频细粒度对齐工作的真正 delta。

### 3.2 DTW / 最优传输 —— 解释完,已砍

**DTW(动态时间规整)**是语音识别老算法:比较两条**节奏可能不同**的时间序列——一个人念得慢时逐点对齐会错位,DTW 弹性拉伸/压缩时间轴找最佳对齐再比。我原想用它对齐 student/teacher 的**注意力轨迹**(怕推理节奏不同)。**但这是过度设计**:前提(节奏差异大)未必成立、复杂度高、收益不明。**→ 暂砍,需要再拣回。** (若真遇到"student/teacher 关注同一帧但发生在不同推理步"再考虑。)

### 3.3 ★ 补充:评测侧新维度 —— grounded CoT 的"时间戳忠实度"(借 video moment retrieval,不是训练机制,是诊断/评测工具)

**借的经典思想**:video moment retrieval / temporal grounding 文献(Moment-DETR 一类:给一句文本查询,预测视频里对应的时间片段)。

**动机**:plan.md 最初的野心是"grounded CoT"——推理链条里提到具体时间点(比如"00:03 秒球开始反弹")。但目前不管是 §2 的 TRD/PSD/PGR 还是 §5 的验证实验,**都没有任何机制去检查 CoT 里提到的时间戳是不是编的**——一条 CoT 完全可能读起来头头是道、时间戳却是幻觉出来的。

**做法**:这不是新的训练信号,是一个**评测/诊断指标**——对 student(或 teacher)生成的 CoT,用正则/简单 NLP 抽出所有"在 XX 秒/第 X 帧发生了 YY"这类时间戳声明,和合成数据 `metadata.jsonl` 的 `video_events_timeline_ms` 真值核对,定义:

$$\text{Grounding Fidelity} = \frac{\text{CoT 里时间戳声明和真实事件时刻吻合的条数}}{\text{CoT 里时间戳声明总数}}$$

**用法**:①作为 P2/probe 实验之外的**额外诊断维度**,顺手就能算,不需要额外训练;②如果后面真做 TRD/PSD/PGR 的训练,可以拿这个当**训练中途的健康检查**(观察 grounding fidelity 是否随训练提升,而不是只看最终 QA 准确率);③如果想让它变成训练信号,可以直接改造成 reward(和 doc02 机制三/四同源——Time-R1 的 tIoU reward、VideoRFT 的语义一致性 reward——**这条如果拿去做训练信号,和 doc02 有重叠,不是这条本身的价值所在,价值在于"作为诊断工具几乎零成本、能让'grounded'这个词名副其实"**)。
**Δ vs doc 02**:doc02 六类都是**训练时的 reward 机制**;这条的核心定位是**评测/诊断工具**,回答"我们的 CoT 到底有没有 grounded"这个目前完全没人查的问题——只有把它当训练 reward 用时才会和机制三/四同源,当诊断工具用时是独立的。

---

## 4. ★ 推荐组合 + practicality 排序

两条蒸馏线(**关系 TRD / 动力学-表示 PSD**)+ 一条可验证奖励线(**动力学-可验证 PGR,不需要teacher**)。两条路线:

| 路线 | 组合 | 定位 |
|---|---|---|
| **A(稳)** | **TRD 骨架 + PGR reward + AoT 正则** | 二阶关系 dense 学 + 物理可验证 reward(不依赖teacher) + 方向正则(借用现成任务) |
| **B(激进,最 novel,但继承teacher风险)** | **PSD 骨架 + PGR reward** | 递归状态/预测动力学(依赖teacher表示)+ 物理可验证 reward 兜底 |

**落地成本 / 前提(诚实排序,2026-07-06 更新)**:

| idea | 实现成本 | 前提 / 风险 |
|---|---|---|
| **AoT** | 极低(倒放+margin,~10 行)| 借用现成任务,不是贡献,只适用方向/顺序题 |
| **ESD**(§2.4) | 低(一次前向 + 小预测头,不需要teacher)| 预测误差是"意外性"代理,不是"重要性"本身;机制非原创 |
| **PGR**(§2.3) | 中(需要给生成器加逐帧物理状态dump + 训读出头)| 只对"物理量可显式定义"的原语有效;不需要teacher,数学论证见§2.3.1 |
| **TRD** | 中(训练时 attention hook + 分块池化 + 显存)| 关系是因果的 proxy;梯度须进主干;需同抽帧前提 |
| **PSD** | 高(预测头 + InfoNCE 负例)| 最 novel 也最难调;依赖teacher表示质量 |
| **grounding fidelity 诊断**(§3.3) | 极低(正则抽取+比对,不训练)| 只是诊断工具,当训练reward用才和doc02机制三/四同源 |
| ~~DTW~~ | — | 过度设计,**已砍** |

**推荐节奏(2026-07-06 更新)**:**PGR 不依赖 teacher,可以独立推进**——先把生成器的逐帧物理状态 dump 补上(轻量工程),跑 probe-experiment.md 的物理量回归探针,直接看 §2.3.1 结论一(DPI)划的那条线在哪。TRD/PSD 依赖 teacher 表示质量,需要先跑 §5**①合成↔真实难度相关性**(定"合成数据靠不靠谱")、**②teacher 结构信号 vs 合成GT**(定"teacher 至少看得到关键帧"这个假设成不成立),都过了再投。§5④AoT 和 **ESD(§2.4)** 都便宜、都不需要teacher,可以顺手一起跑。novelty 一句话:*"不蒸 teacher 的时序表示,而蒸它的时序**依赖图**(TRD);teacher 靠不住的地方,不勉强蒸,换成我们自己模拟器给出的**精确物理真值**当可验证奖励(PGR)——正确性由 GT 保证,不需要相信任何teacher。"*

**与 OPD 主线接口**:PGR 的 $r_{\text{PGR}}$(势函数形式)是**辅助/过程信号**,和 GT verifiable reward(outcome 信号)并列相加,不是替代它,且有 §2.3.1 结论三的策略不变性保证;TRD/PSD 是辅助头,继承 teacher 可靠性风险,这正是 §5①②要先查的东西。

---

## 5. 最便宜的证伪实验(2026-07-05 按 §0.6 重新排优先级——①②③不需要训练/RL基建,几小时到半天;④⑤半天;⑥不是"验证实验",是主实验必做的对照组)

1. **①合成任务难度 vs 真实benchmark同概念子任务难度,是否相关(优先级最高,直接查§0.6(2)的迁移gap)**
   - 做法:用 `repos/Synthetic-Video` 的生成器,对每个"原语类型"(01/02脚本的 direction/rotation/count/speed/accel-decel 等 8 类)各生成一小批(~50条)**直答QA**(`sft.jsonl`格式,不需要CoT、不需要训练)。
   - 用我们已经测过的四个模型(锚8B/9B/3.5-35B/3.6-35B),在这批合成QA上零样本跑一遍——复用已经搭好的评测脚本,几分钟量级,不需要等P2全部跑完。
   - 对照:这四个模型在真实benchmark里概念对应的子任务上的已有分数——TOMATO的reason_type细分(direction/rotation/count,P1/P2数据已有)、MVBench的moving_direction/action_count/moving_count等子任务。
   - 画"模型在合成原语X上的acc" vs "模型在真实benchmark同概念子任务上的acc"的散点(4模型×~5-8个可对应类型≈20-30个点),算相关系数。
   - **解读**:强正相关 → 合成任务的难度结构和真实一致,至少"哪类任务难"是可迁移信号,支持整条"合成训练"路线;无相关/弱相关 → 合成任务在测另一种东西(可能是渲染分布/风格问题),需要重新考虑合成数据的构造方式(甚至改用真实视频+程序化标注)。
   - **诚实说明**:这只验证"难度结构"是否迁移,不直接证明"训练收益"会迁移——是必要非充分条件,但比什么都不查强得多,而且几乎零成本(复用现有infra和已有P1/P2数据)。

2. **②teacher的结构信号 vs 合成GT时间线是否相关(独立于teacher答没答对)——决定 TRD/PSD 值不值得投**
   - 合成数据的`metadata.jsonl`里有逐事件的精确毫秒时间线,是**真实GT**,不用猜。
   - 做法:挑一批合成样本,teacher对每帧做ablation(留一帧)或直接取attention,得到帧重要性profile;直接和`metadata.jsonl`里真实关键事件时刻比较,**不看teacher最终答案对不对**。
   - **相关**(teacher的重要性profile和真实事件时刻明显相关)→ teacher虽然"答不对"但"看得到关键帧",TRD 这类"教它怎么看"的方法有戏;
   - **不相关**(teacher连"往哪看"都不可靠)→ TRD 失去信号来源,**必须放弃教师结构信号这条路**,转向 PGR(不需要teacher)或纯RLVR(见⑥)。
   - **这条不再需要验证 PGR**——PGR 的监督来自模拟器,不依赖这个假设。

3. **③零模型baseline(②的对照下限,不需要GPU/模型,几乎零成本)**
   - 同一批样本,算**纯像素运动能量**(逐帧光流幅度/帧间像素差)和`metadata.jsonl`真实事件时刻的相关性。
   - 如果这个零模型启发式已经能大部分解释"哪帧关键",teacher在这件事上的边际价值就很有限——直接用像素启发式当结构信号更便宜,也规避了"teacher不可靠"的问题。

4. **④AoT试水(半天,最便宜的训练类检验)**:只加AoT项训几十步,看方向类问题(合成的rotation-dir/direction类,或真实benchmark的direction维)是否单独涨。涨 → 白捡,直接进方案(但记住 §3.1 的定位澄清:这不是贡献,只是便宜的验证)。

5. **⑤排列敏感性probe(半天,验证§1.1命门论)**:小集上OPRD vs TRD各训几十步,测"帧乱序后loss/acc变化"。TRD时序增益显著、OPRD不显著 → 命门论成立,TRD值得投。

6. **⑥RLVR-only ablation——不是"验证实验",是主实验设计里必须有的对照组**:不管最后选哪个辅助信号方法(TRD/PSD/PGR),都要有一组"只用GRPO+合成GT verifiable reward,完全不碰teacher、不碰PGR"跑在合成数据上。这组结果决定故事重心:
   - 涨得好 → 辅助信号是"锦上添花",故事重心变成"可验证奖励RL在时序任务上有效"(仍是贡献,只是没那么novel);
   - 涨得很少/学不动(时序任务多步,credit assignment可能很差,完全可能发生)→ 辅助信号是"让RL学得动"的关键,这才是最贴合"用自己的方法规避teacher短板、student还能超过teacher"的故事,而且这种情况下 **PGR 的势函数形式(§2.3.1结论三)恰好是最合适的补法**——它不依赖teacher、有理论保证不偷换目标。
   - 这组需要RL训练基建(GRPO/verl之类),不是几小时能出的东西,但**必须提前规划进主实验设计**,不能等①-⑤都做完才想起来。

**建议现在(不需要训练基建)就能做的顺序:① → ②+③(同一批样本,一起算相关性,只影响 TRD/PSD 要不要投) → ④(AoT,顺手)。同时可以独立推进 PGR 的生成器物理状态 dump + probe-experiment.md 扩展(不依赖①②③的结果)。⑤和⑥需要RL训练环境,留到定下训练框架之后再做。**

---

## 6. 待你拍

- **teacher 具体选哪个(只影响 TRD/PSD,不影响 PGR)**:student 已定 **Qwen3.5-9B**(同源约束,见 §0.6 setting);teacher 候选是 Qwen3.5-35B-A3B / Qwen3.6-35B-A3B(都已下载评测过),可选再加 Qwen3.6-27B(dense,和9B架构更match,但要新下载)。§5②需要选定一个才能跑——建议先选**综合分数最高的 Qwen3.6-35B-A3B**(P1: MVBench 74.3、Video-MME 70.9 四模型最强;TOMATO thinking-off 有偏置退化,等 P2 thinking-on 数据出来后可以换)。
- **§5①②③的顺序**:建议①最先跑——它决定"合成数据到底靠不靠谱"这个更上游的问题,②③依赖①成立才有意义深挖(如果合成任务本身测的是另一种东西,再精细的教师信号验证也没意义)。这条只影响 TRD/PSD,**PGR 可以并行独立推进,不用等**。
- 如果①出来发现合成和真实难度不相关——是转向"真实视频+程序化标注",还是调整合成生成器让它更贴近真实分布?这个决策留到①出结果后再定,不用现在纠结。
- plan.md 的"训teacher+蒸student"两段式设计已经推翻(§0.6),但 plan.md 里其他仍然有效的部分(SynRL生成器怎么用、数据规模估算、显存预算)要不要单独抽出来重写,还是等新故事定型后一起重写?
- **PGR 需要给生成器加逐帧物理状态 dump**(目前 `metadata.jsonl` 只存事件时间线,不存逐帧速度/角速度这类连续量)——这个工程改动谁来做、优先级多高,需要你定。
- **ESD(§2.4)和 grounding fidelity(§3.3)要不要也塞进 §5 的验证清单**?我个人倾向 ESD 值得和 AoT 一起顺手跑(反正都不需要teacher、都便宜)。

---

## 7. ★★ 2026-07-13 更新:probe-experiment 跑完了,现在谁还活着?(大白话总结,不再堆数学)

> 前面 §1-§6 是"讨论阶段"写的,当时四个idea都还只是假设。这一节是 `probe-experiment-report.md` 出结果之后的真实情况,用大白话说清楚现在的判断,细节和数字见那份报告,这里不重复。

### 一句话总结

**之前以为 student 的问题是"看不懂时序、理不清哪几帧有关系"(TRD 的前提),查完发现大概率不是这样——至少在已经测过的这几类任务上,student(甚至连teacher也一样)脑子里其实已经有正确答案了,只是嘴上说错了。** 这把整个故事的重心,从"教它更好的表示"往"逼它把已经知道的东西正确说出来"上搬。

### 具体怎么查出来的(不用公式讲一遍)

拿几类"数一数、看方向、看快慢"这种简单的合成视频任务测了一下:先冻结模型、看它脑子里的内部状态(不管它嘴上答什么)能不能被一个很简单的小工具**猜出正确答案**——结果是能,而且猜得很准(85-100%)。但是**让模型自己开口回答同一道题,却只对了35-69%**。这说明它不是"没看懂",是"看懂了,但说出来的时候说错了"——有点像人心算会数错1个,但心里其实是有数的。

更关键的是:本来想着"如果student自己不会,那就找一个更聪明的teacher模型,把它的本事教给student"——结果一查,**teacher(参数量是student的好几倍)在这几类具体任务上,并不比student更会"说对答案"**,甚至有一类还更差。也就是说:**这不是"student比teacher差",是"teacher和student共享同一个盲点"**——这几个具体的小任务(精确计数、判断细微的转动方向、分辨快慢)可能是这一整代模型不管多大都搞不定的东西,单纯换个更大的模型解决不了。

这解释了一个之前看起来很怪的现象:**为什么teacher在这几个具体小任务上不比student强,但teacher在总分/大盘子上明显比student强?** 因为总分是二十几种五花八门的任务混在一起的平均分,里面大部分考的是"认不认识东西、懂不懂常识、听不听得懂指令"这类"模型越大通常越强"的能力,teacher在这些上确实更强,拉高了总分。但"精确计数""判断转动方向"这几个很窄、很具体的"精细定量判断"能力,和模型总体聪明不聪明是两件不完全相关的事——这类窄能力不随着模型变大自动变好,这在别的大模型研究里也是有名的现象(比如大模型算数经常翻车,不是因为它笨,是这类能力本来就不太随规模涨)。

### 现在的判断:谁还活着,谁基本判死刑

- **TRD(教它"哪几帧有关系")—— 目前的证据不支持,但要按原语类型拆开看,不能一句话说死**。旋转/方向类:合成数据和真实数据(`distillation-readiness-report.md`实验1)交叉验证一致,teacher都没有优势,这一类TRD确实没有用武之地。计数类:合成的反弹计数显示teacher更差,但真实的MVBench计数任务显示teacher明显更强——这个矛盾还没解决,不能直接说计数类的TRD/蒸馏思路也已经排除,细节和后续验证方案见下面7.1和§2.1.2。
- **PSD(教它"记住状态、预测接下来发生什么")—— 已经补测,证据支持,而且是三个里最干净的**。用最简单的"三杯扣球"任务试点:遮挡前probe 100%读得出,遮挡后probe准确率暴跌到接近或低于随机瞎猜的水平——这不是"说错了",是内部状态真的丢了,比TRD那4类严重得多。teacher在这个任务上也没有优势(甚至更差)。细节见`distillation-readiness-report.md`实验2。
- **PGR(拿我们自己模拟器算出来的精确物理真值当训练目标,不需要teacher)—— 目前证据支持,是目前最有戏的一个**。查出来模型的内部状态确实带着不少关于"这个东西现在速度多少、转了多少度"这类物理信息,足够拿来当一个可靠的训练目标——而且它天生不需要teacher,刚好绕开"teacher和student共享同一个盲点"这个新发现的坑。
- **RLVR(用最终答案对不对去纠正模型)—— 已经用"打开thinking"这个免费代理测过,结果是负面的,不能再当"几乎不花钱的办法"这条捷径**,细节见下面7.1。真正的RLVR(训练、不是推理时打开thinking)还没测,但"免费版本不work"这一条要如实计入判断。
- **ARD(2026-07-14新增,§2.1.2):从TRD批评里浮现的独立机制——蒸teacher的答案分布,不碰attention**。旋转/方向类和TRD一样没有信号(teacher读出没有优势);计数类是这个idea最该优先验证的地方,细节见§2.1.2的判定表。

### 7.1 组会追加讨论(2026-07-14):"免费捷径"已经测过、失败了;读出层的gap是不是也能蒸馏,答案要分原语类型看

**thinking-on这条"几乎不花钱的办法"已经跑过,不是"接下来最可能"了,是已经排除的选项**:`distillation-readiness-report.md`实验0补测了4类gap原语在打开thinking模式下的表现,结果是**明确变差**(1024预算下全部4类退步27-50pp,4096预算下依然明显低于直答且撞满预算比例仍高达90-96.7%)——根因是模型长CoT反复推翻重来、撞预算不收敛,不是"想清楚了但没说出来"。这条路已经关闭,不是待验证的假设。

**"让student学teacher读出层的分布,能不能解决问题"——这个问题被组会追问后,发现之前"TRD没有用武之地"这条结论说得太满了**。核心区分:TRD蒸的是teacher内部"哪几帧有关系"的attention图,这是**表示层**的东西;而"学teacher的读出分布"蒸的是teacher最终给出的答案本身,是**读出层**的东西,两者是独立的机制,不能因为TRD不行就连坐判定读出层蒸馏也不行。读出层蒸馏(§2.1.2定义为ARD)能不能用,只取决于一件事——teacher在这个具体任务上的直答准确率是不是真的比student高。已经测过的4类原语里,3类teacher和student打平甚至更差,没有信号;但真实benchmark的计数类任务上teacher明显更强,和合成Bouncing_Counting的结果方向相反——这是一个真实存在、还没解决的矛盾,不是"没戏",是"还不知道"。**下一步要做的不是训练,是先把probe-experiment的方法论(probe vs直答对比)直接搬到真实MVBench计数数据上重跑一次**,才能知道计数类到底是ARD能救,还是这条也会像旋转/方向类一样打平。

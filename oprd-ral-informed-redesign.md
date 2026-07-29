---
title: "基于OPRD和RAL两篇真实论文重新设计的idea(2026-07-18,2026-07-19按因果验证实验更新)"
status: "新方案,基于对两篇论文全文(含附录证明)的通读,不是基于摘要或推测。已补齐用户明确感兴趣的CAD/ARD,已按causal-verification-report.md的实测结果更新CAD/PGA/AP-RLVR的适用范围"
date: "2026-07-19"
related:
  - "novelty.md §1.1(OPRD引用修正)、§1.2(RAL引用修正)、§2.1(TRD)、§2.5(CAD)、§2.1.2(ARD)"
  - "papers/oprd_full.txt、papers/ral_full.txt(两篇论文全文,html转txt)"
  - "causal-verification-plan.md(实验计划)、causal-verification-report.md(2026-07-19实验结果,本次更新的依据)"
---

# 基于OPRD和RAL重新设计的idea

## 0. 两篇论文,一句话讲清楚各自的核心贡献

- **OPRD**(Yang et al. 2026,arXiv:2606.06021,浙大+蚂蚁):在**文本推理**场景下,把on-policy蒸馏从"只匹配输出token分布"升级成"匹配teacher/student在student自己生成的rollout上的中间隐藏状态"——理由有二:①输出空间的REINFORCE式梯度方差不随训练收敛而消失,信噪比会崩溃(有完整的形式化证明,附录A);②LM head把隐藏空间压成一个"有效零空间",隐藏状态的巨大偏差可能在输出空间完全看不见(实测量级$10^6\sim10^8$倍)。**核心设定是"on-policy"——teacher在student自己生成的token上打分,不是在固定数据上。**
- **RAL**(Li et al. 2026,arXiv:2602.04884,UC Davis+Google DeepMind等):在**视频/图像VQA**场景下(和我们的场景一样,Qwen系列MLLM),把attention分布本身当成一个policy,用GRPO式的advantage-weighted JSD散度去强化"高奖励时的attention模式"、推开"低奖励时的attention模式"——**完全不需要teacher,也不需要"正确的attention"这种标签**,只需要可验证的outcome reward。同时提出On-Policy Attention Distillation(需要teacher时,把这套机制从RL奖励换成蒸馏目标)。**关键消融(RAL-zero)证明:去掉显式思考过程、只留直接答案,纯靠attention策略优化依然有效**——这和我们自己测出的"thinking-on让4类gap原语变差"是两个独立来源、方向一致的证据。

## 1. 我们已经诊断出的具体约束(下面所有idea都要满足)

- 4类"读出层gap"原语(Rotation_Direction/Count、Bouncing_Counting、Acceleration_Identification):**probe显示信息在(85-100%),直答显示读不出来(35-69%)**,这是问题本身。
- **teacher在这4类原语上没有验证出优势**(3类打平、1类更差)——任何需要"teacher更强"这个前提的方案(TRD原始设计、CAD、ARD),在这4类原语上目前都没有立项依据。
- **计数类在真实MVBench数据上显示teacher确实更强**,矛盾未解决,是唯一还可能支持"蒸teacher"这条路线的开口,需要另外验证(见novelty.md §7.1)。
- 用户要求:**优先OPD,不去挤RLVR赛道**,除非有充分理由。
- **thinking-on(不训练,只是推理时打开)已经证明无效**——这排除了"免费换个推理模式"这条路,但不代表"训练attention机制"这条路也无效,两者是不同的操作。
- **2026-07-19新增约束(`causal-verification-report.md`已完成,消融实验结果):4类gap原语在attention层面的因果落点完全不一样,不能当成一个整体看待**:
  - **Acceleration_Identification**:干净、跨层一致的信号——层7/15/19(整层,mean+zero消融方向一致,幅度20-32pp),是唯一一个"多个full_attention层都关键"的原语。
  - **Bouncing_Counting**:部分支持,但只在**整层**级别(层3/7/11两种消融方法方向一致),head级别信号经常翻转,不可信。
  - **Rotation_Count**:**没有信号**——8层full_attention层里6层是负delta或接近0,mean/zero两种方法经常方向对不上,消融这些层甚至有时候准确率反而涨了。
  - **Rotation_Direction**:**没有信号**——所有层/head的delta都很小且方向不稳定。
  - **头(head)级别的结果普遍不可靠**(n=50下mean/zero交叉验证只有约60%方向一致率),现阶段不建议按单head设计监督目标,只能按"整层pooled attention"这个粒度。
  - **24层linear_attention(Gated DeltaNet)完全没测**,如果Rotation_Count/Rotation_Direction真有读出层gap,机制大概率在这24层里,是一个尚未验证的开放问题。
  **这条新证据直接推翻了下面2.2/2.2.1/2.4"直接用RAL默认的最后一层"这个假设,需要按原语分别处理,不能一刀切**——已经在对应小节里更新。

## 2. 五个idea,按优先级排序(**用户明确感兴趣的CAD、ARD在2.2/2.3,完整篇幅,不是配角**)

### 2.1 ★★★ Idea A(优先级最高)—— OPSD-R:On-Policy Self-Distillation for Readout(自蒸馏,彻底不需要teacher)

**这个idea从哪来**:OPRD论文自己在§5"High-value application 2"提出了"On-policy self-distillation(OPSD)"——teacher不是另一个模型,是**注入了特权信息的student自己**。我们有一个现成的、比论文原场景更干净的OPSD实例:**student自己在"冻结前向、看视频编码阶段"产生的hidden state,已经被probe-experiment.md证明85-100%线性可读出正确答案**——这就是一个天然的"更可靠的自己"。真正读不出来的,是**生成答案这一步**用的表示。OPSD-R要做的,就是把"student编码阶段那个更可靠的自己"蒸给"student生成答案时那个不太可靠的自己",全程不需要另一个模型。

**大白话**:student心里(看完视频那一刻)其实是知道答案的,只是在"组织语言说出来"的时候把这个认知弄丢了/用歪了。OPSD-R做的事,就是让"组织语言说出来"这一步的内部状态,去像"心里刚知道答案那一刻"的内部状态看齐——用的是同一个模型自己两个不同时刻的状态,不需要问一个更聪明的模型该怎么答。

**具体机制(照搬OPRD原文Eq 6的精确形式,只换监督目标)**:

记student冻结前向(§4.1协议,`extract_hidden_states.py`已有)在层$\ell_{enc}$、时间组$i$的表示为$h_i^{S,enc}$(已验证:线性探针可读出正确答案)。记student**真实生成答案**时,答案token所在位置对视觉token的attention加权表示(或者更简单:生成答案token那个位置本身在层$\ell_{gen}$的hidden state)为$h_t^{S,gen}$。损失:

$$L_{\text{OPSD-R}}=\frac{1}{d}\Big\|P_\theta\big(h_t^{S,gen}\big)-\text{sg}\big(h_{i^*}^{S,enc}\big)\Big\|_2^2$$

其中$\text{sg}(\cdot)$是stop-gradient(只让梯度走生成侧,不倒流回"编码阶段"那次forward——这一点和OPRD原文一模一样,原文明确说这是为了让目标保持稳定,不被"追逐一个会动的靶子"这个问题干扰);$P_\theta$是当$\ell_{enc}\neq\ell_{gen}$维度不同时的线性投影(FitNets式regressor,同一个模型不同层的hidden_dim通常一致,大概率不需要);$i^*$是和当前问题相关的时间组(如果只有一个整体判断,用全部$T$个时间组的mean pooling;如果因果验证实验——见`causal-verification-plan.md`——查出来某几个时间组更关键,可以只用那几个)。

**这版比字面照搬OPRD更简单、更该优先做的三个理由**:
1. **不需要on-policy采样**:OPRD原文的"on-policy"要求teacher在student自己采样出来的rollout上打分,需要采样+跑两次forward(student rollout + teacher重新forward)。OPSD-R可以做成纯**teacher-forcing**版本:直接把真值答案字母/一段模板CoT喂给student当"生成目标"(不采样),对应位置算出$h_t^{S,gen}$,和同一条样本的$h_{i^*}^{S,enc}$对齐——这是一次前向就能算完的、标准的辅助SFT loss,不需要任何RL/policy-gradient机制,是这几个idea里工程上最简单的一个,完全符合"优先OPD"的要求。
2. **完全不需要teacher**:teacher-student之间"读出准不准更高"这个卡住TRD/CAD/ARD的前提,在这里根本不出现——比较的是同一个student的两个内部状态,不涉及第二个模型。
3. **直接对准已经诊断出的具体机制**:probe-experiment.md已经证明$h_i^{S,enc}$里有答案,直答baseline已经证明生成阶段用歪了——OPSD-R的监督目标不是凭空构造的,是**这次探针实验本身已经验证过存在、且可靠**的那部分表示。

**数学论证(直接复用OPRD附录A的Theorem 4,几乎不需要改)**:$L_{\text{OPSD-R}}$对student参数$\theta$的梯度是

$$g_{\text{OPSD-R}}=\frac{2}{d}\big(\nabla_\theta h_t^{S,gen}\big)^\top\big(P_\theta(h_t^{S,gen})-h_{i^*}^{S,enc}\big)$$

如果用teacher-forcing(非采样)版本,这个梯度**条件方差严格为零**——和OPRD原文Theorem 4逐字同构(证明也一样:$h_{i^*}^{S,enc}$在停梯度后是常数,$h_t^{S,gen}$是$\theta$的确定性函数,整个表达式没有任何随机采样步骤)。这是这几个idea里**唯一**能拿到"零方差、确定性梯度"这个最干净保证的,因为OPRD的Theorem 4本来就是针对"非采样token的MSE loss"证的,而OPSD-R的teacher-forcing版本刚好完整满足这个前提,不需要打折扣。

**风险/诚实边界**:①这本质是自蒸馏,如果"编码阶段的表示"和"生成阶段该有的表示"之间存在**架构性**差异(比如生成阶段的位置天然需要额外做"决策/压缩"这一步计算,不是单纯复制编码结果),强行拉近两者可能会挤压掉生成阶段必要的计算,需要小心设计$\ell_{gen}$选哪一层(建议:不选最后一层输出前那一刻,选生成第一个答案token**之前**的最后一个prompt位置,即将要做决策但还没输出的那个点,这样有空间保留决策计算)。②这是全新的机制,没有像TRD/CAD那样的直接文献先例专门做"encoding-vs-generation自对齐",最接近的先例是OPRD自己提的OPSD概念和DINO(自监督表示对齐),但都不是完全同构的场景,需要小规模先跑通再决定要不要扩大投入。

#### 2.1.1 ★★★ 2026-07-22:OPSD-R + 表示变化接地的读出正则(Representation-Change Grounded Readout Regularization)

**问题精确定位在哪**(`opsd-r-implementation-plan.md`§3.2的协议):$h^{S,enc}$和$h^{S,gen}$来自**同一次forward、同一层$\ell$**,只是token位置不同——$\bar h^{S,enc}$是视觉token们在层$\ell$的均值(已验证probe读出85-100%,这个表示本身没问题),$h^{S,gen}$是最后一个prompt位置在**同一层$\ell$**的表示(直答只有35-69%)。最后这个位置明明可以通过因果self-attention回看全部视觉token,但读出的表示准确率远低于视觉token自己——**问题在生成位置"往回看"这个读出机制,不在视觉token自己的表示**,所以监督目标必须完全保留$\bar h^{S,enc}$的均值池化,不能碰,只针对读出机制本身设计新的监督。

**机制:$\bar h^{S,enc}$不动,新增一项直接监督"生成位置该往哪看"**

原来的表示匹配项保留、不改:

$$L_{\text{OPSD-R}}=\frac1d\big\|P_\theta(h^{gen}_\ell)-\text{sg}(\bar h^{enc}_\ell)\big\|_2^2,\qquad \bar h^{enc}_\ell=\frac1T\sum_i h^{enc}_{\ell,i}$$

新增一项,直接监督"生成位置在层$\ell$往回看视觉token"这个**真实存在、可以直接从这次forward里取出来**的attention分布$a^{gen}_\ell\in\mathbb R^T$(不是新建模块,是模型自己在算的东西,取出来即可):

$$m_i=\big\|h_i^{enc}-h_{i-1}^{enc}\big\|\qquad\tilde r_i=\text{softmax}(m_i/\tau)\qquad L_{\text{rep-change-attn}}=D\big(a^{gen}_\ell\,\|\,\tilde r\big)$$

$$L_{\text{total}}=L_{CE}+\beta_1 L_{\text{OPSD-R}}+\beta_2 L_{\text{rep-change-attn}}$$

**$m_i$不按primitive分别定义,用一条公式覆盖所有任务**:$m_i$直接是相邻时间组在层$\ell$的hidden state变化量(或余弦距离版本),不经过光流、不经过物理量。任何"视觉上有意义的变化"——不管是转得快、反弹、加速、新物体出现、还是画面文字变了——只要这一刻对任务有信息量,模型在这一层的表示大概率会跟着变,而这一层已经验证过是"读得出答案"的好表示,不是随便一层。这样不需要为每个primitive手写公式,也不局限于运动学类任务。

**零额外工程**:$h_i^{enc}$就是`extract_hidden_states.py`已经抽出来、用于算$\bar h^{enc}$的那批向量,不需要额外算光流或读物理量,合成/真实数据统一处理。

**收敛性**:$\tilde r$是训练前算好的固定目标(不依赖$\theta$),$D(\cdot\|\text{softmax}(u))$在attention logits空间$u$是无约束凸优化,标准梯度下降有收敛保证。

**工程前提,和已有选层对得上**:$a^{gen}_\ell$必须是真实的softmax attention(不能是Gated DeltaNet线性递归层)。`opsd-r-implementation-plan.md`§3.1的选层——Rotation_Direction用shallow(层7),其余3类用mid(层15)——这两层都在8层full_attention里([3,7,11,15,19,23,27,31]),4个原语现有层选择不用改。deep(层28,linear_attention)不适用这一项。

**诚实的边界**:"任务相关的变化会体现在这一层的表示变化里"是合理推断,不是已经验证过的事实,和$\bar h^{enc}$本身85-100%那个证据不是同一件事;如果某个任务的答案是"全程恒定的静态属性"(没有哪个时刻更关键),这个信号会退化成没有明显peak,帮助有限——这类任务本来也不适合"该往哪个时刻看"这整套框架。

**CAD那边的PGA不动,这次改动只针对OPSD-R。**

#### 2.1.2 设计过程记录(对话讨论留痕,方便回看怎么走到2.1.1这一版的)

这一节不是方案本身,是把2026-07-21到07-22这段设计讨论的关键转折点留下来,方便以后回看"为什么最后长这样",不是要在正文里重复论证。

1. **第一次尝试(已否决)**:把$\bar h^{S,enc}$的池化方式从"均值池化"换成PGA的probe归因加权($\tilde h^{enc}=\sum_i r_i h_i^{enc}$,$r_i$来自probe权重$w_{c^*}$)。**否决原因**:`opsd-r-implementation-plan.md`§3.1明确写死"85-100%可靠"这个证据是在均值池化上测出来的,换池化方式就不能再引用这个证据;而且PGA依赖probe,probe需要独立于MCQ字母的精确真值,这种真值只有SynRL合成数据才有,这个方案天然用不了真实数据。用户明确表示不喜欢PGA。
2. **第二次尝试(已否决)**:把加权方式从probe归因换成运动能量(光流/物理量),但仍然是在改$\bar h^{S,enc}$怎么算。**否决原因**:方向和第一次一样错——问题诊断结论是"表示层没问题,问题在输出/读出层",不管用什么方式重新加权$\bar h^{S,enc}$,都是在修一个已经验证没坏的东西,不是在修真正的读出层gap。
3. **第三次(定下主体结构)**:$\bar h^{S,enc}$完全不动,新增一项监督"生成位置的真实attention分布该长什么样",目标$\tilde r$由运动能量(光流/物理量)定义。这一版第一次把改动范围收窄到读出侧,方向对了。
4. **第四次(按primitive细化,后来发现细化过头)**:把$m_i$按每个primitive具体在问什么分别定义(角速度/速度翻转/速度变化量)。用户指出两个问题:①这套手工公式只服务于运动学类任务,项目里做得不好的primitive有很多根本不是运动学问题,这个机制撑不起更大范围的问题;②即便只看这4个任务,这张"primitive对应哪个物理量"的表本身也在不断膨胀,感觉是打补丁攒出来的,但完全去掉"往哪看"这个监督又确实少了东西(纯MSE不保证模型是靠正确关注视觉证据做到的,这个顾虑本身是对的)。
5. **第五次(当前版,2.1.1)**:把$m_i$的定义从"手写物理量"换成"相邻时间组在同一层的hidden state变化量"——一条公式覆盖所有任务,不用查表,也不需要额外算光流,直接复用已经抽出来的$h_i^{enc}$。同时保留了第三次版本"只监督读出侧、不碰表示层"这个正确的结构。

---

### 2.2 ★★ Idea B —— CAD:Cross-attention Distillation(你感兴趣的第一个idea,用RAL的真实机制重新校准)

**和之前novelty.md §2.5版本的区别**:之前CAD的设计(query从"帧"换成"输出token",KL散度)是我自己推演出来的,现在读完RAL原文,发现**这个机制已经被人真实做过、发表过、且给出了具体的、可以直接照抄的工程参数**——不再是我们自己凭空设计,是有真实benchmark数据背书的方案。下面按RAL原文的口径重新校准。

**大白话不变**:模型吐出答案的每一个词时,会去看上下文里已经编码好的信息(视觉token,也包括它自己已经吐出来的前面几个词)——看哪几个、每个看多少,是一个attention分布。CAD把teacher这个"看的分布"直接当目标,让student去逼近它。

**精确机制,三处按RAL原文校准(不是我们自己猜的选择)**:

1. **用JSD不用KL**——原来的CAD设计用$D_{KL}(B^T\|B^S)$,RAL原文明确说选**Jensen-Shannon散度**(对称、有界)是为了训练稳定性,不是随手选的。JSD定义:$\text{JSD}(P\|Q)=\frac12 D_{KL}(P\|M)+\frac12 D_{KL}(Q\|M)$,$M=(P+Q)/2$。**这是一个真实的取舍,要老实说清楚**:JSD对称有界更稳,但JSD复合softmax重参数化之后,不像纯KL那样能借助log-sum-exp的凸性给出logit空间的无约束凸优化证明(纯KL复合softmax是凸的,这是标准结论;JSD虽然本身对$(P,Q)$联合凸——Lin, *Divergence measures based on the Shannon entropy*, IEEE Trans. Info. Theory, 1991——但复合$Q=\text{softmax}(u)$这个非线性重参数化之后,凸性不能照搬过来,需要单独证明,这里没有现成结论)。**建议**:如果要保留§2.5.1那条干净的logit空间凸优化证明,继续用KL;如果更看重RAL实测验证过的稳定性,换成JSD,但要接受收敛保证退回到"block coordinate descent收敛到驻点"这一档,不能两者都要。
2. **不能照抄RAL"只用最后一层"这个默认设定,**`causal-verification-report.md`**(2026-07-19,消融实验已完成)证明这条对我们不成立**——原来这里写的是"可以先用最后一层跑一版",现在有实测数据,直接更正:
   - **Acceleration_Identification**:因果落点在**第7/15/19层(整层,不是最后一层32)**,而且是**多层同时关键**(4个full_attention层里3个有清晰、跨消融方法一致的信号),不是单一层就够——监督目标应该是"这几层pooled attention的加权组合",不是照搬RAL"只用最后一层"这个默认值。
   - **Bouncing_Counting**:因果落点在**第3/7/11层(整层)**,同样不是最后一层,而且**只有整层级别可信,head级别信号在n=50下经常翻转**——监督粒度必须是"整层16个head全部平均"(呼应TRD原始设计"按head、按token平均"这个做法),不能挑单个head。
   - **Rotation_Count / Rotation_Direction**:**8层full_attention层(含最后一层)都没有找到消融后掉分的信号**,有几层消融后准确率反而涨了——这两类原语上,CAD(不管用哪一层)大概率无效,建议改投Idea A(OPSD-R)这类不依赖具体attention层的方案,或者等以后有人设计出针对24层linear_attention(Gated DeltaNet)的消融方法再重新评估。
   
   **结论:CAD只建议在Acceleration_Identification和Bouncing_Counting这两类原语上投入,且必须先看`causal-verification-report.md`挑出的具体层号,不能像RAL论文那样图省事默认用最后一层——这是我们比RAL多做的一步验证工作,理应用上,不能白做。**
3. **key集合是全部前面位置,不只是视觉token**——RAL的$\alpha_{t,i}$里$i$跑遍**全部**$i<t$(prompt、视觉token、模型自己已经生成的token都算在内),不是只筛视觉token这个子集。这一点上,我们的诊断目的(想知道"答案token有没有看对视觉证据")决定了我们大概率还是想筛出"对视觉token的注意力占比"这个子集来看——**这里建议保留"筛视觉token"这个我们自己的诊断需求,但训练时用RAL的"对全部前序位置"这个更宽的版本**(不人为限制,让模型自己决定要不要看之前生成的词),诊断和训练目标可以不完全绑死同一个集合。

**损失(RAL§3.5 Eq 5原文,没有advantage项,纯结构模仿,直接照抄)**:

$$L_{\text{CAD}}=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_t \text{JSD}\big(p_\theta^t\|p_\phi^t\big)\Big]$$

**真实证据(不是理论推演,是RAL原文Table 2/3的数字)**:attention蒸馏加在标准token蒸馏(即Idea C/ARD)之上,在7/8图像benchmark、多数视频benchmark上进一步提升——VideoMME从61.3到63.9(+2.6)、NExTQA从70.9到75.3(+4.4)、MuirBench从39.9到43.4(+3.5)。**这不是我们自己的猜测,是这套机制在和我们同一个模型家族(Qwen-VL)、同一类任务(video/image QA)上真实测出来的正向结果**——比novelty.md原来只能引用抽象的"attention transfer"这类CNN年代的先例,证据力度强很多。

**排列敏感性论证不变**:$L_{\text{CAD}}$(不管用KL还是JSD)依赖的是整行分布,不是边际统计量,这条论证在§2.5.1已经证过,换散度不影响这个结论。

**没有解决的软肋,还是老实说清楚,不因为有真实benchmark撑腰就回避**:"attention不faithful"(信息可能走value/MLP路径)这条,RAL论文也没有反驳或证明不存在——它只是提供了"即使有这个理论顾虑,实测依然work"的证据,不是"这个顾虑不成立"的证据,这两者要分清楚。

**门槛不变,这是最关键的一条,必须重复**:CAD能不能帮上忙,还是取决于teacher在这个具体任务上读出是不是真的更准。RAL自己的teacher(Qwen2.5-VL-32B)相对student(7B)是**验证过的、真实更强**的模型(这是RAL整个实验成立的前提)。我们的4类gap原语上,teacher(35B-A3B)**没有**这个验证过的优势——3类打平、1类更差。**这意味着CAD这个机制本身是对的、有真实数据背书,但目前在我们的4类gap原语上没有可蒸的信号,不是机制不行,是我们这里暂时没有满足它生效的前提**。计数类(真实数据显示teacher更强)是唯一值得先跑CAD的地方,细节见novelty.md §7.1的待办。

**现在有两道独立的门槛,CAD要同时过**:①teacher读出准不准更高(上面这条,4类里目前都没过,计数类待验证)；②这个原语在我们能监督的attention机制(8层full_attention)里有没有真实的因果落点(`causal-verification-report.md`刚测出来的)。**即使①以后被计数类验证通过了,②依然只对Acceleration_Identification和Bouncing_Counting成立**——Rotation_Count/Rotation_Direction这两类,就算某天发现teacher读出更准,CAD在我们能碰到的这8层full_attention层上也没有下手的地方,这是两件独立的事,不能只查①就动手,②现在已经查完了,直接说明CAD的实际可投入范围比"4类gap原语"这个笼统说法要窄。

#### 2.2.1 ★★★ 真正的创新点:PGA(Probe-Guided Attention)——不是RAL的移植,是利用我们独有的资产

**RAL做不到、我们能做到的事**:RAL的attention监督信号只有两种来源——outcome reward(纯RL,不知道"该往哪看",只知道"这次结果好不好")、或者teacher的attention(需要teacher验证有优势,我们这里没有)。**我们有第三种、RAL的论文场景里根本不存在的信号来源:一个已经训练好、已经验证过85-100%准确率的线性probe。** 这个probe本身,就已经精确地知道"student的hidden state里,答案信息编码在哪",不用再靠"强化成功的经验"去间接摸索。

**关键的数学观察(这是可以严格证明、不是近似的)**:`train_probes.py`里的分类probe,输入是**均值池化**后的向量$\bar h=\frac1T\sum_i h_i$,输出是**线性**的logit $z=W\bar h+b$。均值和线性可以交换顺序:

$$z_{c}=\sum_j W_{cj}\Big(\frac1T\sum_i h_{i,j}\Big)+b_c=\frac1T\sum_i\big(w_c^\top h_i\big)+b_c$$

也就是说,**每个时间组$i$对类别$c$的logit,贡献恰好是$w_c^\top h_i$这一项,不多不少,这是精确的代数恒等式,不是近似估计**——这在合作博弈论里就是**线性可加模型的Shapley值**(Shapley, 1953;显式写在Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017里,称为"Linear SHAP",专门指出线性模型的精确归因不需要采样近似)。

**PGA的做法**:对真值类别$c^*$(我们的合成数据有精确ground truth,直接知道),取probe的对应权重$w_{c^*}$,算出每个时间组的贡献$s_i=w_{c^*}^\top h_i^{S,enc}$,softmax归一化成一个分布:

$$r_i=\frac{\exp(s_i/\tau)}{\sum_j\exp(s_j/\tau)}$$

用这个**由probe精确推导出来、锚定在真值答案上**的分布,去监督student生成答案时的attention分布$p_\theta^t$:

$$L_{\text{PGA}}=D_{\mathrm{KL}}\big(r\,\|\,p_\theta^t\big)$$

**为什么这是"意料之外情理之中"**:意料之外,是因为两篇论文都没有想到用一个**诊断工具**(probe)反过来当**训练目标**——RAL自己不知道"该往哪看",只能靠试错(RL);CAD/TRD需要相信teacher"看得对",但我们的teacher没验证过这一点。情理之中,是因为这个probe本来就是我们自己训出来验证"信息在不在"的,它的权重天然就编码了"信息具体在哪"这个问题的答案,拿它当监督目标只是把这个已有的、免费的知识重新利用了一遍。

**这版相对CAD/RAL的具体优势**:①**彻底不需要teacher**,监督信号来自student自己的probe,不存在teacher优势门槛,4类gap原语现在就能做,不用等计数类验证;②**比RAL更直接**,不需要GRPO/outcome-reward这套RL机制,是纯监督loss(和OPSD-R一样,是标准的OPD);③**收敛证明可以直接照搬§2.5.1/§2.1.2已经建立的logit空间凸优化框架**——$r$是固定目标(不依赖训练中的$\theta$),$p_\theta^t=\text{softmax}(u)$,$D_{KL}(r\|\text{softmax}(u))$在$u$空间是无约束凸优化,标准梯度下降有$O(1/T)$收敛到全局最优的保证,这条证明是干净的,不需要打折扣。

**诚实的边界**:①probe本身是线性的,只能捕捉"线性可分"这部分信息,如果真正决定答案的是一个非线性的时间组合(比如"组1和组5的差值"这种probe学不到的模式),$s_i$这个归因就不准了——这依赖§4.2分类probe本身的假设(线性探针能读出85-100%,已经实测验证过,不是新假设);②$\tau$(softmax温度)是一个新超参数,需要调,过小会让$r$退化成one-hot(过度自信只盯一个时间组),过大会让$r$退化成均匀分布(等于没有监督信号)。

**2026-07-19补充:PGA的"该往哪看"($r$)和CAD一样,依然要挑一层去监督,这条也要按`causal-verification-report.md`的结果来,不能自由选**:$r$本身(probe归因出来的、该看哪个时间组)不需要挑层——它是从`extract_hidden_states.py`已经固定的shallow/mid/deep三层里抽出来的,和消融实验无关。**但PGA要监督的对象$p_\theta^t$(生成答案时,实际的attention分布)是某一层生成阶段的attention,这一层选哪个,和CAD面对的是同一个问题**——消融结果显示,只有Acceleration_Identification(层7/15/19)和Bouncing_Counting(层3/7/11,只能整层监督)在8层full_attention层上有真实因果落点,Rotation_Count/Rotation_Direction没有。**所以PGA现在也只建议先在这两类原语上做,监督层的选择直接复用CAD那条结论,不用重新跑一次消融**——这是PGA相对CAD一个额外的好处:两者可以共享同一份`causal-verification-report.md`的结果,不用为每个新idea单独重新验证一次该看哪层。

---

### 2.3 ★★ Idea C —— ARD:Answer-level Response Distillation(你感兴趣的第二个idea,用OPRD的Theorem 2重新定量)

**和之前novelty.md §2.1.2版本的区别**:之前只是定性地说"ARD只匹配最终输出,可能丢掉中间结构信息"。现在有OPRD的Theorem 2给出了**精确的数学刻画和真实的量级**,不再是定性描述。

**精确的数学论证(直接引用OPRD原文Theorem 6/7,不是重新推导)**:

设$W_{head}\in\mathbb R^{|\mathcal V|\times d}$是LM head,SVD为$W_{head}=U\Sigma V^\top$,奇异值$\sigma_1\ge\cdots\ge\sigma_d>0$。定义**有效零空间**$\mathcal N_W=\{\Delta h:W_{head}\Delta h\in\text{span}\{\mathbf 1\}\}$(隐藏状态的偏差,只要被$W_{head}$映射成"加一个常数"这种softmax不敏感的方向,就完全不会体现在输出分布里)。OPRD的Theorem 6证明:

$$h^S-h^T\in\mathcal N_W\implies \ell_{\text{out}}(h^S,h^T)=0$$

对任意输出空间的loss(ARD的KL loss也算在内)。Theorem 7进一步给出量级:沿最小奇异值方向$v_d$的隐藏状态偏差,在**同样的输出loss预算下**,可以比沿最大奇异值方向$v_1$的偏差大$(\sigma_1/\sigma_d)^2$倍——**论文实测这个比值在生产级LLM上是$10^6\sim10^8$倍**。

**这对ARD意味着什么,说得具体一点**:ARD($L_{ARD}=D_{KL}(g^T(h^T)\|g^S(h^S))$)本质上就是OPRD论文里说的"output-space OPD"这个baseline家族的一个成员——只是我们的场景不是文本token分布,是MCQ答案分布,但数学结构完全一样(都是"隐藏状态经过一个线性投影+softmax,再比较这个投影后的分布")。**只要student的隐藏状态偏差恰好落在(或接近落在)$\mathcal N_W$这个方向上,ARD训练完之后KL loss降到很低,但隐藏状态本身可能仍然和teacher差得很远**——训练看起来收敛了,但学到的东西可能非常有限。这不是我们凭空担心,是这篇论文用严格的线性代数证明+真实量级数字给出的结论。

**真实证据支持"ARD单独用会不够,但和隐藏状态级方法组合会更好",不是我们自己猜的**:OPRD原文的$\mu$ sweep实验(Figure 6):纯output-space OPD(对应我们的ARD)在AIME24上是42.3分,加一点点隐藏状态监督($\mu=1$)跳到47.7分(已经超过更强的output-space baseline top-16的47.1),加更多($\mu=10$)到50.2分,基本追平teacher的50.8。**这是两个独立的真实实验(OPRD的$\mu$ sweep + RAL的KD/KD+Attn对比)得出的同一个结论:纯粹的、只看最终输出的蒸馏(ARD)会遇到一个真实存在、量级很大的信息瓶颈;这个瓶颈的解法不是抛弃ARD,是把它和隐藏状态/attention级别的方法组合起来用,单独任何一个都不如组合起来强。**

**这不改变ARD在我们项目里"要不要投入"这个判断,但改变了"投入之后该怎么用"这个设计**:§2.1.2已经确定的判定表(teacher读出准不准更高)依然是先决条件,4类gap原语目前没通过,计数类待验证——这一点OPRD/RAL都没有、也不能改变,因为这是我们自己的实测数据,不是理论问题。**但如果计数类验证通过、决定投入ARD,现在有了明确依据:不要让ARD单独训练,要按OPRD的$\mu$ sweep思路,和Idea A(OPSD-R,不需要teacher)或者验证通过后的CAD(Idea B)组合起来加,系数从小往大扫,而不是一上来就指望ARD单独把gap关掉。**

**结论(不变,和§2.1.2一致但现在有更硬的证据)**:排列敏感性这条,ARD依然是"借"teacher的(不是自证的),这一点没变;LM-head瓶颈这条,现在有了Theorem 6/7的精确刻画和$10^6$-$10^8$倍这个真实量级,比之前"架构层面的瓶颈"这种定性说法更有说服力。

#### 2.3.1 ★★★ 数学延伸(2026-07-18,在OPRD的Theorem 6/7基础上往下推,推出两条原论文没有的新结论)

OPRD原文的Theorem 6只证明了一件**静态**的事:"如果偏差恰好落在$\mathcal N_W$里,输出loss就是0"——这是一个条件句(if...then...),没有说"训练过程中偏差会不会跑到$\mathcal N_W$里、或者已经在里面的部分会不会被训练动到"。下面把这条**动态化**,并且专门针对我们的MCQ场景(而不是原论文的开放词表生成场景)重新算一遍量级,这是两条原论文没有、我自己往下推的新结论。

**新结论一(动态不变性,严格强于原文Theorem 6)**:在"$h^S$本身被当自由变量直接梯度下降"这个理想化设定下(和§2.1.2/§2.5.1一直沿用的理想化级别一致),把$\mathbb R^d$正交分解成$\mathcal N_W\oplus\mathcal N_W^\perp$,记$h^S=a+b$($a\in\mathcal N_W$,$b\in\mathcal N_W^\perp$)。由OPRD自己的Lemma 2(softmax核:$\sigma(z+c\mathbf1)=\sigma(z)$)可知,对任意$\Delta\in\mathcal N_W$和任意$\epsilon$,$\ell_{out}(h^S+\epsilon\Delta,h^T)=\ell_{out}(h^S,h^T)$**恒成立**(不是近似)——也就是说$\ell_{out}$作为$a$的函数是**严格常数**,偏导数$\partial\ell_{out}/\partial a\equiv 0$。在纯ARD训练(只有$\ell_{out}$这一个loss)下,梯度流$\dot h^S=-\nabla_{h^S}\ell_{out}$的$a$分量满足$\dot a=-\partial\ell_{out}/\partial a=0$,所以

$$a(t)=a(0),\qquad\forall t\ge 0.$$

**这比原文Theorem 6强的地方**:原文只说"这个方向的偏差可能造成看不见的loss"(一种可能性);这里证明的是"只用ARD训练,student在$\mathcal N_W$方向上的分量,从第一步到最后一步,分毫不动,不随训练时长增加而改变"——是一个关于**训练动力学**的确定性结论,不是关于loss函数取值的静态描述。

**新结论二(MCQ场景下,零空间比原论文的例子更极端——具体数字)**:OPRD原文算的$(\sigma_1/\sigma_d)^2$是对**完整词表**($|\mathcal V|\approx150$K)算的。我们的ARD只需要比较$K$个候选答案字母(MCQ通常$K=4$-$6$)的logit,有效的输出投影是$W_K\in\mathbb R^{K\times d}$(LM head里只取这$K$个候选token对应的行),不是完整的$W_{head}$。$W_K$作为线性映射$\mathbb R^d\to\mathbb R^K$,秩最多是$K$(泛型情形下恰好是$K$,因为$K\ll d$,这$K$行线性无关几乎必然成立),由秩-零化度定理,$\ker(W_K)$维度是$d-K$;而"softmax不变零空间"$\mathcal N_{W_K}$(原像多加的那个span{1}方向)维度是$\ker$再加1:

$$\dim(\mathcal N_{W_K})=d-K+1.$$

代入Qwen3.5-9B的$d=4096$,$K=6$:$\dim(\mathcal N_{W_K})=4091$,**占整个4096维隐藏空间的99.88%**。

**这里有一处我自己推导时先算错、后来验证发现不对、需要如实更正的地方**:我一开始类比说"OPRD原论文的全词表场景零空间也是$d-1$维、占比差不多",这个类比**是错的**,已经用真实矩阵形状验证过(见下方,不是空口更正)。原因是两种场景的矩阵形状根本不同:
- **我们的$W_K\in\mathbb R^{K\times d}$是"矮胖"矩阵**($K=6\ll d=4096$,行数远小于列数)——这种矩阵泛型情形下是**满行秩**、**满射**(能盖满整个$K$维输出空间),这时候"全1向量在不在像空间里"这个问题是**平凡成立**的(像空间就是整个$\mathbb R^K$,任何向量当然都在里面),所以$\mathcal N_{W_K}$泛型情形下确实有$d-K+1$维这么大,上面的推导没问题。
- **OPRD原论文的$W_{head}\in\mathbb R^{|\mathcal V|\times d}$是"高瘦"矩阵**($|\mathcal V|\approx151$K$\gg d=1536$,行数远大于列数)——这种矩阵泛型情形下是**满列秩**、**单射**(不满射,像空间只是$|\mathcal V|$维空间里一个$d$维的"薄片子空间")。单射意味着"全1向量在不在像空间里"这件事**不是自动成立的**——泛型情形下,一个固定的向量(全1)几乎不可能恰好落在一个随机的、低维得多的子空间里(我用随机矩阵实测验证过:50维子空间里,全1向量到子空间的投影残差范数远大于0,不在里面)。也就是说,**OPRD原论文场景下$\mathcal N_W$泛型情形下只有0维或最多1维,不是$d-1$维那么大**。

**修正后,两边比的不是"谁的零空间更大",而是"瓶颈的结构完全不同,我们的更硬"**:OPRD原论文的信息瓶颈主要靠Theorem 7的谱比$(\sigma_1/\sigma_d)^2\sim10^6$-$10^8$撑起来——这是一种"软"瓶颈:沿最差方向的偏差,只是被压缩了一个巨大但**有限**的倍数,理论上还是能在输出里看见,只是需要极高精度才能分辨。我们的MCQ场景不一样:$W_K$的定义域里有整整$d-K+1=4091$维方向,对应的奇异值**严格为零**(不是"很小",是"恰好是0")——这是一种"硬"瓶颈:这4091个方向,不管信噪比多高、精度多高,**输出logit压根不含这些方向的任何信息,不是分辨率不够,是根本没有**。**这比OPRD论文自己的场景更极端,但极端在"瓶颈的种类"上(硬零vs软衰减),不是极端在"数字更大"上——这个更正后的、更精确的区分,才是这条延伸真正站得住的新结论**,原来那个"数字看起来差不多"的类比是我推错了,已经改正,不能让错的类比留在文件里。

**这两条结论合起来,给出一个具体、现在就能排的、几乎不花钱的新诊断实验**:对`hidden_states/{9b,35b_35}/*.pt`里已经抽好的hidden state,用$W_K$(每个task对应的候选字母那几行LM head权重)做SVD,把每条样本的$h^{S,enc}$投影到$\mathcal N_{W_K}$和$\mathcal N_{W_K}^\perp$两个子空间,**分别**重跑一遍`train_probes.py`已有的分类probe逻辑,看85-100%这个准确率主要落在哪一半:
- 如果**$\mathcal N_{W_K}^\perp$**(输出空间能看见的部分)就已经能重现大部分probe准确率——说明答案信息本来就活在"ARD理论上够得着"的地方,ARD(在teacher优势验证通过后)有真正生效的数学空间,新结论一/二不构成障碍。
- 如果准确率主要靠**$\mathcal N_{W_K}$**(输出空间看不见的部分)才能达到——新结论一直接告诉我们:**不管怎么调、训多久,纯ARD都不可能触碰到这部分信息,这不是"信号弱"的问题,是"结构上不可达"**,必须走OPSD-R/CAD/PGA/TRD这类直接监督hidden state或attention的路线,ARD在这种情况下即使教师优势通过验证也不值得投入。

这是一个新的、比"计数类真实数据验证"更细粒度的判据,建议排在那个验证之前或同时做,因为它成本更低(不需要重新抽数据,只需要对已有的`.pt`文件做一次投影+重新训一次线性probe)。

---

### 2.4 ★ Idea D —— AP-RLVR:Attention-Policy RLVR(照搬RAL的纯RL机制,不需要teacher,但是RLVR)

**这个idea从哪来**:直接照搬RAL论文§3.1-3.4的机制,这是目前唯一一个**真实发表、有benchmark数据支持**、专门针对"信息在但读不出来"这类问题、且不需要teacher的方案。

**具体机制(几乎是RAL原文Eq 2-4的直接复用,换成我们的奖励)**:

生成答案时,取第$t$个输出token对所有前面位置(prompt+视觉token+已生成的token)的attention,归一化成分布$p_\theta^t(i)$。用GRPO式训练(每个prompt采样$G$条rollout,组内相对奖励算advantage),损失:

**层的选择,2026-07-19按`causal-verification-report.md`更正,不能照抄RAL"只用最后一层"**:和CAD(§2.2)同一个更正——AP-RLVR原来打算照搬RAL"最后一层、全部head平均"这个默认值,现在有实测消融数据,应该按验证过的层来,不是图省事用最后一层:Acceleration_Identification用层7/15/19(多层同时优化,可以对每层各算一个$\text{JSD}$项加起来);Bouncing_Counting用层3/7/11(整层,不挑head)。**这其实是一个我们能做、RAL论文自己做不到的改进**——RAL没有做过这种逐层因果消融,只能凭经验默认用最后一层;我们多做了`causal-verification-plan.md`这一步,应该把这个额外的验证结果用上,不是白验证。Rotation_Count/Rotation_Direction这两类原语,和CAD一样,建议不要投入AP-RLVR(至少不在这8层full_attention层上),优先考虑Idea A(OPSD-R)。

$$L_{\text{AP-RLVR}}=L_{RL}+\lambda_{attn}\,\mathbb E_t\big[A_t\cdot\text{JSD}(p_\theta^t\|p_{old}^t)\big]$$

奖励$r_i$完全复用我们已有的、共享的`extract_mcq_answer`:答案对了给1分,格式对了给0.1分附加分(和RAL原文的$r=0.9r_{acc}+0.1r_{fmt}$一模一样,我们连数字都不用改,因为MCQ场景高度相似)。

**"要不要显式CoT"这一步,RAL-zero这个消融直接回答了我们的顾虑**:RAL论文自己测过"完全去掉thinking、只留直接答案"这个版本(RAL-zero),发现纯靠attention策略优化,在5/7视频benchmark上依然跑赢base model和完整GRPO——**这意味着我们不需要重新趟"thinking-on到底要不要用"这个已经在distillation-readiness-report.md里验证过是坑的路**,可以直接用"直接答案+attention策略优化"这个更轻量的组合,不需要生成一长串CoT再对齐。

**这是RLVR,和用户"优先OPD"的要求有冲突,需要摆在明面上讨论,不能藏着**:AP-RLVR用的是policy-gradient(GRPO)机制,不是纯监督学习,严格说属于RLVR家族,和"不要去挤RLVR赛道"这条指示有张力。但有两点值得摆出来给你决策:①这不是"卷烂了的"那种vanilla token-level GRPO/RLVR,是**RL作用在attention policy这个具体的、2026年2月才发表、目前几乎没有后续工作跟进的窄分支**上,拥挤程度和"再做一个token级GRPO"完全不是一回事;②这个机制**精确对上**我们的诊断("信息在、没读对"),比任何token级方法都更有针对性——是否因为这两点例外对待,决定权在你,这里如实列出来,不是我擅自决定要不要走这条路。

**建议的实际顺序**:先做2.1(Idea A,OPSD-R,纯监督、零方差、不需要RL基建),如果验证后发现效果有限,再考虑这一条作为进阶方案——不是同时上两个。

---

### 2.5 组合recipe(仅在teacher优势被验证后才启用,照搬RAL的Eq 6结构)

RAL论文自己的完整recipe(Eq 6)是$L_{RL}+\mu L_{GKD}+\gamma_{attn}L_{AttnDistill}$——RL奖励、token级蒸馏(Idea C/ARD)、attention蒸馏(Idea B/CAD)**三项一起用**,论文实测这个组合比任何单项都强。**在我们这里,ARD/CAD这两项现在还不能启用**,原因见2.2/2.3——都需要teacher在具体任务上确实更准,4类gap原语目前没有这个前提,计数类矛盾未解决。**这里先把"如果验证通过之后该怎么组合"的方案定下来,不代表现在就投入**:

$$L_{\text{combined}}=\underbrace{L_{RL}+\lambda_{attn}\mathbb E_t[A_t\cdot\text{JSD}(p_\theta^t\|p_{old}^t)]}_{\text{Idea D,永远开启,不需要teacher}}+\underbrace{\mu\, D_{KL}(g^T\|g^S)}_{\text{Idea C/ARD,判定表通过才开}}+\underbrace{\gamma_{attn}\,\text{JSD}(p_\theta^t\|p_\phi^t)}_{\text{Idea B/CAD,同一判定表通过才开}}$$

三个系数$(\lambda_{attn},\mu,\gamma_{attn})$按`causal-verification-plan.md`的消融结果和`novelty.md`§7.1的计数类真实数据验证结果,分场景开关,不是无脑全开。也可以把Idea A(OPSD-R,不需要teacher)一起加进来,四项组合,因为它和ARD/CAD监督的对象完全不同(自己 vs teacher),不冲突。

**待办(启用ARD/CAD前必须做的验证,novelty.md已经列过,这里不重复展开)**:计数类probe直接在真实MVBench action_count/moving_count数据上重跑一次(novelty.md §2.1.2/§7.1已经写清楚具体怎么做)。

## 3. 六个方案的对照表(含2.2.1的创新点PGA,已按`causal-verification-report.md`更新;**2026-07-21再更新:teacher优势门槛用真实benchmark重新核实,不再是"未验证"**)

**这一版更新的关键事实**:之前"teacher优势未验证"这句话,依据的是SynRL合成域内部的direct/probe对比(`distillation-readiness-report.md`§1a/1b)。后来专门重新核实了MVBench全20任务+TOMATO全6类的真实teacher-vs-student准确率(`training-data-decision.md`§7),结果是:
- **Bouncing_Counting** ↔ MVBench action_count/moving_count:真实teacher优势 **+9.0~+14.5pp**(明确、验证过)
- **Acceleration_Identification** ↔ TOMATO velocity&frequency:真实teacher优势 **+18.57pp**(全部26个真实任务/类别里最大的一个)

**这两个数字直接关闭了CAD表格里"且需验证优势"这个待定项**——不是"以后计数类验证通过再说",是**已经验证通过了**,而且通过得很彻底(尤其Acceleration_Identification,是全表最强信号)。

| | A: OPSD-R | B: CAD | **B′: PGA(创新)** | C: ARD | D: AP-RLVR | 组合recipe |
|---|---|---|---|---|---|---|
| 需不需要teacher | 不需要 | 需要,**已验证优势**(见上) | 不需要 | 需要,**已验证优势**(同上) | 不需要 | 需要(部分项) |
| 需不需要attention因果落点(2026-07-19新增门槛) | **不需要**(不依赖具体attention层,对4类原语都适用) | **需要**——只有Acceleration_Identification(层7/15/19)、Bouncing_Counting(层3/7/11,仅整层)通过 | **需要,和CAD共享同一份消融结果**——同上只有这2类通过 | 不需要(只看最终logits,和层无关) | **需要**,同CAD/PGA | 需要(涉及B/B′/D的部分) |
| 是OPD还是RLVR | 纯OPD | OPD(蒸馏) | 纯OPD | OPD(蒸馏) | RLVR | 混合 |
| 工程复杂度 | 最低 | 中(需attention hook) | 中(需attention hook+复用已有probe) | 低(只需最终logits) | 中高(需GRPO基建) | 最高 |
| 收敛证明 | 零方差确定性梯度(照搬OPRD Thm 4) | 用KL可证凸优化;用JSD(RAL验证过的稳定选择)只有驻点保证 | logit空间凸优化(同CAD-KL版) | logit空间凸优化(§2.1.2已证) | 无 | 无 |
| 真实文献证据 | OPRD §5 OPSD概念(未实验,论文自己也没做) | **RAL Table 2/3实测(+2.6~+4.4pt)** | 无直接先例(SHAP的Linear SHAP结论借用到这个场景是新的) | **OPRD μ-sweep实测(42.3→50.2)** | RAL Table 2/3 + RAL-zero消融 | RAL Eq 6(实测组合>单项) |
| **现在能不能立项(2026-07-21更新)** | **能,4类原语都能做,工程最简单** | **能,在Acceleration_Identification/Bouncing_Counting上——两道门槛(teacher优势+因果落点)都已通过** | **能,同上两类原语**,且从来不受teacher门槛限制 | **能,在Acceleration_Identification/Bouncing_Counting上**(teacher优势已过,不受attention层限制) | **能,在Acceleration_Identification/Bouncing_Counting上**(attention门槛过,teacher门槛不适用) | **能,在这两类原语上**(ARD的teacher前提也已满足) |

## 4. 和"不要去挤RLVR赛道"这条要求的最终对齐,以及按原语分类的最终建议(2026-07-21更新)

**结论变了,老实更新,不是小修小补**:之前认为"只有OPSD-R现在能做,CAD/ARD都卡在teacher优势没验证"——这个判断建立在SynRL合成域的对比上,重新核实真实benchmark后不成立了。**Acceleration_Identification和Bouncing_Counting这两个原语上,CAD、PGA、ARD、AP-RLVR全部解锁**,不再只有OPSD-R一条路。

**按原语分类,最终建议**:
- **Acceleration_Identification、Bouncing_Counting**:teacher优势(真实benchmark验证过)+ attention因果落点(`causal-verification-report.md`验证过)两道门槛都过了。这两个原语上,**CAD/PGA/ARD/AP-RLVR/组合recipe理论上都能做**——PGA工程上最直接(不需要teacher、收敛证明最干净);CAD/ARD需要teacher前向(工程量增加但已验证有真实信号可蒸);AP-RLVR是RLVR(按"不挤RLVR赛道"这条要求,除非有特别理由,不作默认选项)。**这两个原语,建议优先级:PGA > CAD/ARD组合 > 单独CAD或ARD > AP-RLVR**。
- **Rotation_Count、Rotation_Direction**:8层full_attention层上没有找到任何因果落点,且真实benchmark上teacher在这两个原语对应的TOMATO rotation类别上是**更差**(-4.54pp)——teacher优势这道门槛在这两个原语上是**反向不通过**(不是"未验证",是"验证过、且方向不支持蒸馏"),CAD/PGA/ARD/AP-RLVR这几个依赖teacher或attention落点的方案在这两个原语上都不成立。**只剩Idea A(OPSD-R)这一条路**,因为它是自蒸馏,不涉及teacher也不依赖具体attention层。

**OPSD-R依然值得作为第一个做的**——不是因为其他方案不能做了,而是它工程最简单、收敛保证最干净、且是唯一对全部4类原语（包括没有真实teacher优势的Rotation两项）都适用的方案,适合先跑通验证流程。但**Acceleration_Identification/Bouncing_Counting这两个原语上,现在有理由把CAD（尤其是PGA这个创新点）作为紧随其后的第二批投入**,不是像之前那样要等一个不确定的"以后"。

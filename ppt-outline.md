---
title: "两轮诊断实验 PPT 提纲"
status: "草稿,供转成slides用"
date: "2026-07-14"
---

# Slide 1:研究问题 & 两轮实验总览

**背景**:目标是验证"off-policy SFT student on 合成grounded CoT能否提升视频时序感知"(项目原始设计,见plan.md)。在真正投入训练前,先做了两轮**诊断实验**,搞清楚student的能力gap到底在哪、值不值得投入蒸馏。

**两轮实验**:
1. **probe实验**(`probe-experiment-report.md`):①先检验真实benchmark子任务本身考不考时序推理,②再判定gap是在"表示层"(没编码)还是"读出层"(编码了但说不出来)。
2. **distillation-readiness实验**(`distillation-readiness-report.md`):追问三件事——thinking模式能不能白捡分、合成结论能不能迁移到真实benchmark、"跨遮挡状态追踪"这个更难的场景表现如何。

**模型**:student=Qwen3.5-9B,teacher=Qwen3.5-35B-A3B(同源MoE,避免OPD分布不一致)。

**现状先说结论**:三个候选idea里,**TRD(教表示)不建议投入,PGR(教物理量回归)和PSD(教跨遮挡状态追踪)建议投入**,细节见后面几页。**目前都还是前提验证阶段,还没有跑过真正的蒸馏/SFT训练**——这点在后面单独说清楚。

---

# Slide 2:两轮诊断实验的起因——9B在真实benchmark上,打开思考反而更差

**在做probe/distillation-readiness这两轮诊断实验之前,先在3个真实benchmark(MVBench/TOMATO/Video-MME)上完整跑过P1(直答,thinking关闭)和P2(CoT,thinking开启,官方推荐采样参数)两轮正式评测——这个"打开思考反而没有变好"的反常结果,就是后面去深挖gap到底在哪一层的直接动机。**

| Benchmark | P1(thinking关闭) | P2(thinking开启,CoT) | 变化 |
|---|---|---|---|
| MVBench | 69.42% | 63.17% | **-6.25pp** |
| TOMATO | 36.19% | 36.12% | -0.07pp |
| Video-MME | 67.07% | 63.37% | **-3.70pp** |

**读法**:9B在3个真实benchmark上,打开原生thinking模式(CoT)之后,MVBench和Video-MME明显掉分,TOMATO持平——**没有一个benchmark变好**。这和"合成grounded CoT蒸馏能提升时序感知"的原始假设(plan.md)方向相反,是促使我们停下来先做两轮诊断实验、搞清楚gap到底出在哪一层(表示层/读出层/感知层)的直接原因,而不是想当然地假设"多推理一定更好"就直接投入训练。

**追查"打开思考为什么反而更差":发现是生成质量问题,不是能力问题**——抽查P2原始输出发现相当一部分样本长篇CoT反复推翻重来、撞满token预算也没收敛(详见`P2_stuck_examples_for_meeting.md`,40条样本里25%属于这种"反复得出正确答案又反复自我怀疑,最后被硬截断"的模式)。查Qwen3.5官方README的"Best Practices"发现根因和修复:

| 问题 | 根因 | 修复 |
|---|---|---|
| 长CoT反复横跳、撞预算不收敛 | 之前全程用`temperature=0`贪婪解码——官方README的4种推荐配置**没有一个用temperature=0**,thinking任务官方推荐`temperature=1.0/top_p=0.95/top_k=20/presence_penalty=1.5/repetition_penalty=1.0`,且明确说`presence_penalty`就是用来"reduce endless repetitions"的 | 换成官方推荐采样参数(transformers原生没有presence_penalty,自己写了个`PresencePenaltyLogitsProcessor`对齐这个值) |
| 答案提取不稳定 | 没有规定输出格式 | 按README建议,提示模型用JSON的`answer`字段作答(如`"answer": "C"`),提取器加了最高优先级的`json_field`模式 |

**换上官方采样参数之后,再单独扫了一遍`max_new_tokens`,确定最终配置**(①-⑤修复全部固定,TOMATO,n=40):

| max_tokens | 2048 | 4096 | 8192 | 32768 |
|---|---|---|---|---|
| tomato_score | 50.0% | **55.0%** | 50.0% | 45.0% |

**读法**:4096是这4个里的局部最优点,不是"越大越好"——32768这一档已经**没有任何样本真正撞满预算**(实测最长31369 token,主动收尾的),但准确率反而更低,说明"更长的思考"本身不解决问题,和Anthropic《Inverse Scaling in Test-Time Compute》、《Don't Overthink It》等文献里"推理越长不等于越准"的发现一致。**最终P2全部benchmark采用`max_new_tokens=4096`官方采样参数**这套配置,即上面P1/P2对比表的协议来源(已核实MVBench/TOMATO/Video-MME、9B/35B-A3B的全部分片,用的都是这套修复后的最终配置,不是修复前的旧配置)。

**⚠️易混淆点,专门澄清**:上面"2048/4096/8192/32768"这张表用的是**40条小样本**(TOMATO数据集最前面40条,offset=0),只是为了选参数、不是完整benchmark,**不能和本页最上面P1/P2那张完整1484条的表混着看**。这40条恰好全部落在完整benchmark分片跑法的"第1片"(前371条)里——而TOMATO这个数据集**不是随机打乱排列的**,分片跑出来的4片(各371条)分数分别是52.02%/27.22%/28.57%/36.66%,第1片本身就明显偏简单。所以"55%"只是"恰好抽到了简单的一段",完整1484条加权平均后是36.12%(=(52.0227.2228.5736.66)/4),和P1的36.19%基本打平——这才是"打开思考没有变好"这个结论真正依据的数字,40条那次纯粹是调参数用的,没有代表性。

---

# Slide 3:时序推理预检验——真实benchmark子任务本身考不考时序推理

**在设计probe实验、选合成任务之前,先做了一个更基础的检验:MVBench/TOMATO这些真实benchmark的各个子任务,是不是真的需要"按时间顺序看懂视频",还是单帧信息就够用了?** 方法很直接:把每个子任务的视频帧序**随机打乱**,用P1同款直答协议(thinking关闭)重新测一遍,和不打乱的baseline比准确率——如果打乱后掉分明显,说明这个子任务真的依赖时序;如果几乎不掉分,说明模型本来就没怎么用时序信息,这类子任务不该作为后面probe实验的目标。

**MVBench(20个子任务,9B/35B-A3B交叉验证,打乱后 vs 不打乱)**:

| 子任务(掉分最明显的5个) | 9B掉分 | 35B掉分 |
|---|---|---|
| action_localization | 22.50pp | 18.00pp |
| moving_direction | 16.00pp | 22.00pp |
| action_count | 14.00pp | 26.50pp |
| moving_count | 14.00pp | 18.00pp |
| moving_attribute | 12.00pp | 10.50pp |
| **20子任务均值** | **6.95pp** | **9.20pp** |
| (对照)scene_transition/unexpected_action/fine_grained_action/egocentric_navigation | ≤2pp,甚至反向 | 同样几乎不掉分 |

**TOMATO(6个reasoning_type)**:

| reasoning_type | 9B掉分 | 35B掉分 |
|---|---|---|
| direction | 15.63pp | 28.04pp |
| shape&trend | 15.25pp | 11.21pp |
| **整体均值** | 8.36pp | 10.71pp |
| rotation | 1.05pp | **-1.75pp**(几乎不掉) |

**结论**:两个模型交叉验证后,一致地"真掉分明显"(>10pp)的是MVBench的**action_localization/moving_direction/action_count/moving_count/moving_attribute**,和TOMATO的**direction/shape&trend**——这7个是两个模型都验证过的"真的需要时序推理"的战场。**这个结论直接决定了后面probe实验该在哪7类合成原语上做**(Slide 5的7类原语就是照着这个映射表选的)。同时也发现一个有意思的反常:TOMATO的**rotation**类几乎不掉分(9B几乎不变,35B打乱后甚至更好一点点)——这和后面Slide 5发现的"旋转类合成原语R²有位置混淆嫌疑"合在一起看,共同指向"两个模型处理旋转类任务时,可能并没有真的紧密依赖逐帧时序信息"这个更大的图景。

---

# Slide 4:实验一怎么做的——probe vs 直答,两条独立路径

**核心方法**:同一批合成视频(7类小任务,比如"判断顺/逆时针转""数球弹了几次",来源见Slide 3的映射),同一个模型,跑两条完全独立的代码路径:

| 路径 | 怎么做 | 衡量什么 |
|---|---|---|
| **A. 探针** | 冻结模型,做1次forward,提取hidden state,训一个线性分类器 | hidden state里**有没有**这个信息 |
| **B. 直答** | 同样的输入,调用generate(),模型自己吐出答案,用官方提取器解析成字母 | 模型自己**说不说得出**这个信息 |

两条路径唯一的差别是"看一眼内部表示"还是"真的生成"——这个对比能区分"信息不在"(该教表示)和"信息在但读不出来"(教表示没用)。

---

# Slide 5:实验一结果——TRD基本判死刑,PGR有戏

**7类原语的probe vs 直答对比**(9B / 35B-A3B,取最优层):

| 子类 | probe准确率 | 直答准确率 | 判读 |
|---|---|---|---|
| Rotation_Direction | 95.0% / 96.7% | 52.0% / 52.0% | 能→不对,**读出层gap** |
| Rotation_Count | 100% / 100% | 67.0% / 69.0% | 能→不对,**读出层gap** |
| Bouncing_Counting | 100% / 100% | 40.5% / 29.0% | 能→不对,**读出层gap,teacher更差** |
| Acceleration_Identification | 100% / 100% | 60.0% / 64.0% | 能→不对,**读出层gap** |
| Complex_Direction_Identification | 96.7% / 90.0% | 99.5% / 100% | 能→对,端到端没问题 |
| Event_Sequence | 100% / 100% | 99.5% / 100% | 能→对,端到端没问题 |
| Directional_Event_Counting | 25.0% / 23.3% | 71.5% / 82.5% | 不能→对,反常(单独存档,有DPI边界条件解释,不是理论矛盾) |

**物理量回归probe**(支撑PGR,6类原语中5类R²=0.59~1.0,只有Directional_Event_Counting表现差)。

**排查过"是不是选项文字混淆,不是真的推理错误"**(把wrong answer按语义值而非随机字母归类统计):
- Rotation_Direction:错误100%集中在"顺逆时针弄反",0%选无关选项——排除格式混淆,是真实的方向判断问题。
- Rotation_Count:67-77%的错误是"差1"(近似正确)——支持"确实在数,只是数不准"。
- Acceleration_Identification:错误集中在"accelerating→误答constant"这一个方向,有清楚物理解释(低速起步和匀速视觉上难分辨)。
- Bouncing_Counting:错误模式不一样,66-85%是"差>1"的离谱错误且系统性偏多——这4类里读出/追踪表现最差的一个。

**位置混淆核查**(旋转类原语R²有嫌疑,专门做了position-only baseline对照):Rotation_Count头条R²=0.997里约**78%其实是纯位置信息就能解释的**(position-only baseline本身R²=0.782),扣除后残差R²仍有0.85-0.98,说明hidden state确实有超出位置的真实视觉信息,但引用头条数字要谨慎;Rotation_Direction经sign-oracle分析确认**是干净的**,不存在这个问题。

**结论**:
- **TRD(教表示)**:4类原语表示已经够好(85-100%线性可读),问题在读出/推理,教表示没用;而且teacher并不比student更会读出(甚至更差);错误模式分析支持"真的在推理只是不精确",不是选项格式问题——**没有可蒸馏的信号,不建议投入**。
- **PGR(教物理量回归)**:hidden state里连续物理状态信息量充足(R² 0.59-1.0,位置混淆已核查排除)——**建议投入**。

---

# Slide 6:实验二怎么做的——三个追问

| 子实验 | 怎么做 |
|---|---|
| **thinking-on重测** | 把4类"读出层gap"原语的直答协议从"关闭思考"换成"打开思考"(1024 token预算——最初按P2同款4096配置试跑,单卡2.5小时不到20条不现实,改用1024),看会不会白捡分 |
| **合成↔真实相关性** | 用Slide 3筛出的、MVBench/TOMATO里"真的需要时序推理"的7个真实子任务,和对应的合成原语结果做交叉对照 |
| **PSD遮挡试点** | 用最简单的"三杯扣球"任务(`basic_shell_game`),把同一条视频切成"揭晓前/遮挡换位后"两段,分别抽hidden state训probe,看"球最后在哪"这个信息在遮挡后还剩多少 |

---

# Slide 7:实验二结果——thinking-on更差、合成结论部分不迁移、PSD问题比TRD更严重

**0. thinking-on重测(9B,120条,补测了1024和4096两档预算)**:

| 原语 | 直答(思考关闭) | thinking-on(1024) | thinking-on(4096) | 撞满预算比例(4096) |
|---|---|---|---|---|
| Rotation_Direction | 52.0% | 10.0% | 20.0% | 96.7% |
| Rotation_Count | 67.0% | 16.7% | 10.0% | 96.7% |
| Bouncing_Counting | 40.5% | 13.3% | 30.0% | 90.0% |
| Acceleration_Identification | 60.0% | 33.3% | 46.7% | 26.7% |

→ **明确变差,不是白捡**——4096比1024好一些(3/4原语有改善),但**全部4类依然明显低于直答**,而且除Acceleration_Identification外撞满预算比例仍高达90-96.7%,不是简单"预算不够"能解释的。thinking-on这条"零成本捷径"排除。

**1. 合成↔真实相关性**(7个真实子任务逐一对照,不是笼统结论):

| 真实子任务 | 9B baseline | 35B baseline | teacher优势 | 对应合成原语teacher优势 | 方向一致? |
|---|---|---|---|---|---|
| MVBench action_localization | 57.50% | 59.00% | 1.50pp | Complex_Direction_Identification: 0.50pp | ✅ |
| MVBench moving_direction | 75.00% | 76.00% | 1.00pp | 同上 | ✅ |
| MVBench action_count | 59.00% | 68.00% | **9.00pp** | Bouncing_Counting: **-11.50pp** | ❌ |
| MVBench moving_count | 66.50% | 81.00% | **14.50pp** | 同上 | ❌ |
| MVBench moving_attribute | 86.00% | 91.50% | 5.50pp | (无直接对应) | — |
| TOMATO direction | 48.39% | 57.32% | 8.93pp | Complex_Direction_Identification | ❌(合成端到端没问题,真实teacher优势不小) |
| TOMATO shape&trend | 39.46% | 39.91% | 0.45pp | — | — |
| (对照)TOMATO rotation | 26.92% | 22.38% | **-4.54pp** | Rotation_Direction: 0.00pp | ✅(都接近0/负) |

→ **旋转/方向类的"没有教师优势"结论有真实数据支持;计数类的合成结论(Bouncing_Counting教师更差)不能直接采信**——真实的MVBench计数任务上teacher明显更强(9~14.5pp),方向完全相反。这是证据链里如实记录的一个缺口,不是好消息也没有藏着:如果要在计数类任务上排除TRD,需要直接在真实MVBench数据上补一次probe,不能依赖合成Bouncing_Counting的结果。

**2. PSD遮挡试点**(`basic_shell_game`,200条。**2026-07-14更新**:直答最初用`max_new_tokens=32`照搬合成原语脚本的值,导致模型自发逐步叙述每次swap时被硬截断,unresolved虚高;查实后改用1024重新跑,数字如下):

| 阶段 | 9B probe | 35B probe | 直答准确率 | 直答unresolved比例 |
|---|---|---|---|---|
| 遮挡前(sanity check) | 100% | 100% | — | — |
| **遮挡后(核心问题)** | **5.0-6.7%** | **3.3-6.7%**(低于随机基线10-12%!) | 15.5% / 8.5% | 22.0% / 9.5% |

→ 修正截断bug后数字上升(9B从4%到15.5%),但**这是4选1MCQ,随机瞎猜期望≈25%,两个模型修正后依然明显低于随机基线**——不是"没答完/被截断答不出",是系统性地选错,信息**比随机猜还差**。不是"读出层"问题,是真正的表示/状态维持层gap,而且比TRD那4类严重得多、干净得多(没有真实数据对不上的问题)。**teacher也没有优势,甚至更差**(8.5% vs 15.5%,约为student的一半)。

---

# Slide 8:目前"蒸馏"进展说明——诚实同步

**目前还没有跑过真正的蒸馏/SFT训练**。以上两轮实验(probe-experiment  distillation-readiness)全部是**前提诊断**——回答"值不值得投入""该往哪个方向投入",不是训练结果本身。这是刻意的顺序:先确认gap真实存在、teacher有没有可蒸馏的优势,再投入训练成本,避免assume错了方向白做。

**三个idea现状**:
- TRD:证据不支持,不建议投入(且"计数类没信号"这条在真实数据上还站不住,需要另测)
- PGR:证据支持,建议投入
- PSD:证据支持,而且是三个里最干净、最值得优先投入的

---

# Slide 9:局限与待办(两份报告如实记录的缺口,汇总)

- **计数类的合成↔真实结论不一致**(Slide 7):需要直接在真实MVBench action_count/moving_count数据上补一次probe才能对TRD在计数类上下最终结论。
- **PSD只测了最简单的1个变体**:`basic_shell_game`(单球追踪)之外,还有11个更难的challenge_type(多球追踪、属性绑定、反向推理等)没测;只测了"遮挡前/遮挡后"两个时间点,没测每次swap后信息是哪一步开始丢的;没测更短遮挡(比如只做1-2次swap)下状态是否还能维持。
- **只测了一个teacher**:按此前拍板只跑了Qwen3.5-35B-A3B,没跑Qwen3.6-35B-A3B交叉验证。
- **分类probe的eval集较小**:n_val=60(70/30切分200条/类),85% vs 25%这种量级差异是稳的,但100%和95%之间不宜过度解读。
- **Event_Sequence的majority baseline很高**(87.3%),100%虽完美但提升幅度不如其他原语大,可能部分是强视觉突变信号,不完全是时序推理。
- **Directional_Event_Counting反常**(probe读不出但直答最好):分类和回归两种方法交叉验证一致,已用DPI边界条件解释(probe是单点hidden state,直答是多步自回归展开,两者不违反同一个理论),不是理论矛盾,但具体机制未深入验证。

---

# Slide 10:下一步——优先OPD,不去挤RLVR赛道

对"读出层gap"(TRD那4类)和"感知层gap"(PSD),报告原文建议的修复方式里提到了RLVR(outcome reward训练)。**按你的要求,优先走OPD(off-policy SFT)路线,只在OPD走不通时才考虑RLVR**:

| 问题 | 报告原建议 | 优先走的OPD替代方案 |
|---|---|---|
| **PSD:跨遮挡状态丢失**(现在最优先) | — | 用生成器自带的真值(球在哪、每次swap后状态)直接模板生成"grounded CoT"(不需要额外训teacher),SFT student学会显式复述/重算状态——这正是plan.md最初设计的"off-policy SFT on合成grounded CoT" |
| **PGR:物理状态回归信号足** | — | 加一个轻量回归头,用生成器真值做监督,SFT阶段联合训练(纯监督,不需要reward/rollout) |
| **TRD 4类的"读出层"问题**(旋转/计数/加减速) | RLVR(reward驱动) | 见下方"重新定性"——不再是teacher蒸馏,是student自我校准 |

**"TRD 4类读出层问题"重新定性(组会讨论后修正)**:既然teacher在这4类原语上并不比student强(Slide 5),那"从teacher蒸馏关系图"这个前提就不成立了——这已经不是"跟teacher学"的问题,而是**student自己内部已经算对了,只是生成答案时没有正确利用自己已经算对的这部分表示**,是纯粹的student自我校准,不需要teacher参与,也就不用再担心teacher/student分布不一致。具体路线(建议顺序):

1. **先低成本诊断,不直接上手写训练**:用`output_attentions=True`抓生成答案时的真实attention权重,对照probe已经确认的"哪几个temporal group编码了正确信息",检查生成阶段的attention有没有把权重放对地方——分清是"往哪看错了"还是"看对了但算错了",两种对应的修法不同。
2. **优先试SFT on grounded CoT**(风险小、工程简单):让student在作答前先显式写出引用具体时刻的推理链(比如"第1次弹跳在~1.2s…第2次…共4次",时刻用`video_events_timeline_ms`真值),间接逼attention落到对的地方,不直接碰attention内部机制,还是标准文本SFT。
3. **如果2效果有限,再考虑直接监督attention权重**:VQA领域"guided attention"是有先例的做法(Qiao et al. 2018 AAAI;Das et al. 2016),但工程更复杂(要选对层/头)、attention权重和准确率的因果关系本身在文献里有争议("Attention is not Explanation" vs "Attention is not not Explanation"),作为更精细但更贵的备选,不作为首选。
4. **RLVR作为最终兜底**,只有1-3都验证无效时才考虑。

**待办/开放问题**:计数类在真实MVBench数据上单独测一次probe(目前证据链缺口),PSD的11个更难变体(多球、属性绑定等)还没测。

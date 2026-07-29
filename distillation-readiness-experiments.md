---
type: experiment-spec
field: probe-experiment 完成后的下一批实验 —— distillation 前还缺什么
created: 2026-07-13
language: zh-CN
status: 待执行。写给负责跑实验的会话执行,讨论本身(novelty分支)不在这份文档里。
depends_on:
  - probe-experiment.md + probe-experiment-report.md(已完成,本文档的全部实验都建立在它的结论之上)
  - novelty.md §2.1(TRD)/§2.2(PSD)/§2.3(PGR)/§5⑥(RLVR-only对照组)
outputs_expected:
  - 实验0:4类"读出层gap"原语在thinking-on下的直答准确率——回答"打开CoT能不能白捡"
  - 实验1:合成原语的read-out gap / 难度 vs 真实benchmark对应子任务的相关性表——回答"合成域的发现能不能迁移到真实benchmark"
  - 实验2:03-10号"长时认知"生成器上的probe结果——回答"PSD的核心场景(跨遮挡状态追踪)到底成不成立"
  - 实验3/4:RLVR-only训练基建 + PGR读出头/势函数reward的具体接入方案(工程checklist,不是一次性出数字的实验)
---

# probe-experiment 之后:distillation 前还缺的实验

## 0. 优先级和为什么这么排

probe-experiment-report.md 已经确认:①4类原语(旋转方向/圈数、反弹计数、加减速)是"表示够用、读出/决策层丢分"，teacher也没有优势可蒸,TRD在这些原语上没有意义;②PGR有信号来源(R²不低,旋转计数类的头条数字要按残差口径打折);③PSD的核心场景(长时状态追踪)完全没测。

**下面按"花多少钱、解决多大的不确定性"排序,不是按 novelty.md 里 idea 的顺序**:

| # | 实验 | 成本 | 解决什么不确定性 |
|---|---|---|---|
| 0 | thinking-on 重测4类gap原语 | 最低(复用P2 infra,几小时) | 会不会是个prompt/协议问题,根本不需要训练 |
| 1 | 合成↔真实相关性 | 低(大部分是已有数据的再分析) | 合成域的发现能不能迁移,不做的话训完了才发现不迁移就晚了 |
| 2 | PSD在03-10号生成器上的probe | 中(需要新造数据+新的多时间点probe方法) | PSD到底还活不活着,现在完全不知道 |
| 3/4 | RLVR-only基建 + PGR代码化 | 高(真正的工程) | 前面3个结果会直接决定这里怎么设计,不该先做 |

**建议顺序:0 → 1 → 2 → (根据0/1/2的结果决定要不要做、怎么做3/4)。**

---

## 1. 实验0(最优先):thinking-on 能不能白捡这4类原语的分数

**做什么**:复用 probe-experiment.md §4.3 的同一批数据(4类原语×200条×2模型=1600条,已经生成好、hidden state也抽好了,不需要重新造数据),把协议从"直答(thinking关闭,max_new_tokens=32)"换成P2同款的"thinking-on"(9B/35B-A3B原生支持,`enable_thinking=True` + `reasoning_prompt`不需要,这两个模型有原生thinking,直接开就行)。`max_new_tokens`按P2协议放宽到1024(这4类原语题目不长,给太多反而慢,1024应该够,如果发现经常撞到上限再加)。答案提取**必须用已经修复的共享`extract_mcq_answer`**(和P1/P2/probe-experiment全程同一把尺子,不要在这一步换回旧逻辑)。

**为什么做这个,而且必须排第一**:probe-experiment-report.md 里这4类原语的错误模式很有意思——Rotation_Count 67-77%的错误是"差1"(近似正确);Acceleration_Identification的错误几乎全部集中在"加速→答成匀速"一个方向,有清楚的物理解释。**这种"心里知道大概、但没写下来核对就容易错1个"的模式,恰好是人类做心算 vs 列竖式计算时的经典差异**——如果让模型"列竖式"(一步步推理、显式数一遍),误差可能自己就消掉了,根本不需要设计reward、不需要训练。这个假设几小时就能验证,如果成立,后面的RLVR/PGR训练投入可以大幅减少范围(只需要处理thinking-on之后依然剩下的那部分gap)。

**怎么解读结果**:
- 4类原语的直答准确率,thinking-on后大幅逼近(或达到)probe准确率(85-100%)→ **这个gap主要是协议问题,不是能力问题**,后续训练只需要针对"thinking-off场景下怎么保持这个能力"(如果最终产品要用thinking-off)或者干脆确认"以后就用thinking-on"即可,RLVR/PGR的训练目标要相应调整(缩小范围)。
- 提升有限(比如只从50%涨到65%,离85-100%还很远)→ **协议不是主要原因**,gap主要还是模型能力问题,原来"投RLVR/PGR"的计划不变,但可以把"thinking-on后剩下的真实gap"当成新的、更准的训练目标基线。
- 完全没提升甚至变差 → 记录下来,可能是这几个模型在这几个任务上有某种thinking-on反而引入噪声的机制(参考本轮之前发现的"thinking-off答案坍缩到A/B"那类现象,不代表thinking-on总是更好),按实际数字说话,不要预设"thinking-on一定更好"。

**产出**:一张"4原语 × 2模型 × {direct(已有)/thinking-on(新跑)/probe(已有)}"三栏对比表,直接续在 probe-experiment-report.md §3 表格后面。

---

## 2. 实验1:合成域的发现,在真实benchmark上站不站得住

**做什么分两部分,大部分是免费的数据复用/再分析,不需要新的GPU实验:**

**1a(完全免费,零GPU,现在就能做)**:回去翻 P1 已经收集的数据——§1.5 筛出的5个MVBench子任务(action_localization、moving_direction、action_count、moving_count、moving_attribute)和2个TOMATO reason_type(direction、shape&trend),**这几个子任务上,teacher(3.6-35B/3.5-35B)比student(9B)强多少**?如果这几个"真的需要时序推理"的真实子任务上,teacher的优势也很小甚至没有(类似probe-experiment在合成4类原语上发现的"teacher不比student强"),那就是**免费拿到的、支持"合成域发现能迁移"的证据**,不需要跑任何新东西,纯粹是把已有的P1 per-subtask数字重新摆一遍对比。

**1b(需要设计但仍然便宜)**:建一张"合成原语 ↔ 真实子任务"对照表(复用 probe-experiment.md §3 已经打的底):

| 合成原语 | 合成direct-acc | 合成probe-acc | 合成gap | 对应真实子任务 | 真实baseline-acc | 真实shuffle掉分 |
|---|---|---|---|---|---|---|
| Rotation_Direction/Count | ... | ... | ... | TOMATO rotation | ... | ... |
| Bouncing/Directional_Event_Counting | ... | ... | ... | MVBench action_count/moving_count | ... | ... |
| Complex_Direction_Identification | ... | ... | ... | MVBench moving_direction / TOMATO direction | ... | ... |

数字全部复用已有的(probe-experiment-report.md + P1数据 + §1.5的shuffle表),不需要新跑模型。填完表后看:**合成gap大的原语,对应的真实子任务是不是也表现出"绝对分低+teacher没有优势"的同款模式**?这是一个描述性的、小样本(~7对)的对照,不指望算出漂亮的相关系数,重点是看方向是否一致,不一致的点要单独指出来。

**1c(可选,更贵,只有1a/1b结果暧昧不清时才做)**:用光流之类经典CV方法给真实视频算一个近似的运动/方向"银标准"(不是精确真值,只是比瞎猜强),测一下"在合成精确真值上训出来的物理量probe,对这个真实视频的近似真值有没有任何预测力"——这是之前讨论PGR的"合成→真实能不能迁移"风险时提过的办法,放在这里当备选,不是必须项。

**为什么做**:probe-experiment的4个gap原语、R²数字全部来自合成数据。如果这个"read-out gap"或者"物理量可读出"的现象在真实视频上根本不出现(比如真实benchmark上这几个子任务teacher其实吊打student,只是我们凑巧在合成数据上没看出来),那整个"针对这个gap设计RLVR/PGR"的investment基础就动摇了。这一步是在投入训练之前最后一道便宜的止损检查。

---

## 3. 实验2:PSD 的核心场景(03-10号生成器,长时状态追踪)到底成不成立

**做什么**:probe-experiment.md 原来的范围完全没有碰 `03_shell_game.py`到`10_grid_movement_tracking.py`这8个生成器(plan.md定义的"长时认知")。这里需要一套**和§4.1/4.2不完全一样的新方法**,原因是这类任务的核心不是"某一时刻的一个数值",而是"状态在被遮挡/操作之后还记不记得"。

**具体设计**:
1. 先挑1-2个最简单、状态最干净的生成器做试点(建议先做 `03_shell_game.py`——"哪个杯子底下有球",状态是离散的、类别少,比滑块/纸牌堆更容易定义probe目标)。同样各生成~200条(复用01/02的生成规模惯例)。
2. **和01/02最大的方法论差异**:探针不能只在"最后一帧"或"生成前一刻"抽一次——要在**遮挡/操作事件前后各抽一次hidden state**(比如"杯子开始被移动前"、"移动结束后、揭晓前"),分别训probe预测"球在哪个杯子下面",看:
   - 遮挡前probe准确率(应该很高,还没被扰乱)
   - 遮挡后probe准确率(核心问题:**还记不记得**)——如果遮挡后掉到接近随机猜测的水平,说明模型确实"没能在内部维持这个状态穿越遮挡",这是PSD要解决的真实场景;如果遮挡后依然很高,说明这个能力已经具备,PSD在这个原语上也没有意义(类似TRD在01/02上的结论)。
3. 同样做§4.3同款的直接作答baseline+teacher对比,同样的判定框架(probe高/答案低→读出层gap;probe本身就低→真感知/状态维持层gap)。
4. 如果第一个试点(shell game)显示probe在遮挡后确实大幅下降,再决定要不要扩展到其余生成器;如果第一个试点显示probe依然很高(说明连最简单的状态追踪当前模型都能维持),这条线大概率可以早点砍掉,不需要把8个生成器都做一遍。

**为什么这么设计(强调多时间点而不是单点)**:PSD 的假设从来不是"模型看不懂某一帧",是"跨越几步之后还能不能维持一个内部信念"——如果只在最后一帧抽一次probe,测的其实还是"感知",不是"记忆/状态追踪",会把PSD和PGR/TRD的探针方法混为一谈,量出来的东西文不对题。

**产出**:先出shell game试点的结果(probe准确率随"遮挡前/遮挡后"变化的表),再决定后续投入。

---

## 4. 实验3/4:训练基建怎么搭(工程checklist,不是"跑一次出数字"的实验)

这两块目前都还只存在于novelty.md的数学里,没有代码。**这里先给出一个搭建顺序的checklist,具体设计要等实验0/1/2的结果出来后再定,不要提前把训练资源投进去。**

**RLVR-only基建(§5⑥,novelty.md原文的强制对照组)**:
1. 需要一套能跑GRPO(或类似verifiable-reward RL算法)的训练循环——项目已有`verl`环境,需要确认它能不能直接用,还是要另起。
2. reward函数最简单版本:直接答案对不对(二元),后续再叠加过程性信号。
3. 先在4类gap原语(如果实验0显示thinking-on没能完全解决)上做小规模训练,看纯outcome reward能不能把direct-answer accuracy从35-69%拉到接近85-100%(probe显示的信息上限)——**这本身就是对"这个gap是不是纯粹的RL credit assignment问题"的直接检验**。

**PGR代码化**:
1. 读出头 $g_\phi$:线性或浅MLP,输入是训练时某一层的pooled hidden state,输出维度按实验1b确认哪些原语要做(优先做实验0/1显示"仍有真实gap、且R²高"的那几个)。
2. 物理真值来源:参考probe-experiment-report.md §4的重建逻辑(能解析重建的不用改生成器,Bouncing_Counting那种需要改生成器的,已经改过了,直接复用)。
3. Loss先用最简单的MSE辅助loss(novelty.md §2.3),验证梯度确实能改变主干表示、且不掉real-benchmark的其他分数(小规模跑通,不要一上来就上势函数形式)。
4. 验证loss确实有效后,再实现potential-based shaping的势函数形式(§2.3.1结论三),接入RLVR的reward,而不是简单相加。

**这两块都建议在实验0/1/2出结果之后再排期,现在只是把"要做什么、大概顺序"记录下来,不占用GPU。**

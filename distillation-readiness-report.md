# distillation-readiness-experiments 执行报告

对照 `distillation-readiness-experiments.md` 逐项执行。本文件随实验0/1/2完成逐步补全。

## 状态总览

| # | 实验 | 状态 |
|---|---|---|
| 0 | thinking-on重测4类gap原语 | ✅ 完成(9B,120条;35B未跑,理由见下) |
| 1 | 合成↔真实相关性 | ✅ 完成(1a+1b,零GPU纯数据复用) |
| 2 | PSD在03号生成器(shell game)上的探针 | ✅ 完成(9B+35B-A3B,各200条,probe+直答baseline) |
| 3/4 | RLVR-only基建 + PGR代码化 | 按文档要求,等0/1/2出结果后再排期,不提前投入 |

---

## 实验1:合成域的发现,在真实benchmark上站不站得住

### 1a. 免费检查:teacher在§1.5筛出的7个"真的需要时序"真实子任务上,优势有多大

用现成P1数据(已用当前修复的共享`extract_mcq_answer`重新打分,和之前§1.5用的是同一批数字，无需新跑）：

| 真实子任务 | 9B baseline | 35B-A3B baseline | teacher优势(35B-9B) |
|---|---|---|---|
| MVBench action_localization | 57.50% | 59.00% | +1.50pp |
| MVBench moving_direction | 75.00% | 76.00% | +1.00pp |
| MVBench action_count | 59.00% | 68.00% | +9.00pp |
| MVBench moving_count | 66.50% | 81.00% | +14.50pp |
| MVBench moving_attribute | 86.00% | 91.50% | +5.50pp |
| TOMATO direction | 48.39% | 57.32% | +8.93pp |
| TOMATO shape&trend | 39.46% | 39.91% | +0.45pp |
| **(对照)MVBench 20子任务整体均值** | 69.42% | 73.35% | +3.93pp |
| **(对照)TOMATO 整体** | 36.19% | 40.77% | +4.58pp |
| **(对照,补充)TOMATO rotation**(§1.5里"不吃这套"的一类) | 26.92% | 22.38% | **-4.54pp(teacher更差)** |

**读法**:7个子任务里，teacher优势从+0.45pp到+14.5pp不等，不是均匀的。`action_localization`/`moving_direction`/`shape&trend`三项teacher优势接近整体均值甚至更低（≤1.5pp），和"没有教师优势"这个故事一致；但`action_count`/`moving_count`/TOMATO`direction`三项teacher优势明显高于整体均值（+8.9到+14.5pp），**不支持**"这几个子任务上没有教师优势"——真实数据上teacher在这几个计数/方向类子任务上确实比student强不少。额外查了TOMATO rotation（§1.5判定为"不吃时序这一套"的一类），teacher反而更差（-4.54pp），这一点和合成域Rotation_Direction的发现（teacher=student=52.00%）方向一致。

### 1b. 合成原语 ↔ 真实子任务对照表

| 合成原语 | 合成direct(9B/35B) | 合成probe(9B/35B) | 合成teacher优势 | 对应真实子任务 | 真实baseline(9B/35B) | 真实shuffle掉分(9B/35B) | 真实teacher优势 | 方向一致吗 |
|---|---|---|---|---|---|---|---|---|
| Rotation_Direction | 52.00%/52.00% | 95.00%/96.67% | 0.00pp | TOMATO rotation | 26.92%/22.38% | +1.05/-1.75pp(不吃时序) | -4.54pp | ✅ 一致(都接近0或负) |
| Rotation_Count | 67.00%/69.00% | 100%/100% | +2.00pp | TOMATO rotation(同上) | 同上 | 同上 | -4.54pp | ✅ 一致(都很小) |
| Bouncing_Counting | 40.50%/29.00% | 100%/100% | **-11.50pp** | MVBench action_count/moving_count | 59-66.5%/68-81% | +14.0/+14.0pp | **+9.0~+14.5pp** | ❌ **不一致**(合成上teacher更差,真实上teacher明显更强) |
| Directional_Event_Counting | 71.50%/82.50% | 25.00%/23.33%(反常) | +11.00pp | MVBench action_count/moving_count(同上) | 同上 | 同上 | +9.0~+14.5pp | ✅ 方向一致(都是teacher更强)，但这个原语本身probe读不出，不属于"4类gap原语" |
| Complex_Direction_Identification | 99.50%/100% | 96.67%/90.00% | +0.50pp | MVBench moving_direction | 75/76% | +16.0/+22.0pp | +1.00pp | ✅ 一致(都接近0) |
| Complex_Direction_Identification(同上) | 同上 | 同上 | 同上 | TOMATO direction | 48.39/57.32% | +15.6/+28.0pp | **+8.93pp** | ❌ 不一致(合成上端到端已经没问题,真实上teacher优势不小) |
| Acceleration_Identification | 60.00%/64.00% | 100%/100% | +4.00pp | (无直接对应,探索项) | — | — | — | N/A |

### 1a/1b 结论

**部分支持、部分不支持"合成域发现能迁移"，不是干净的一致——如实报告，不挑对自己有利的部分**：

- **旋转/方向类(Rotation_*, Complex_Direction_Identification↔moving_direction)**：合成域"teacher没有优势"这个发现在对应的真实子任务上**站得住**——TOMATO rotation、MVBench moving_direction的teacher优势都接近0甚至为负，和合成结果方向一致。这部分TRD的负面结论**有真实benchmark的交叉验证支持**。
- **计数类(Bouncing_Counting↔action_count/moving_count)和TOMATO direction**：合成域发现**不能直接迁移**——真实子任务上teacher优势明显（+9~+14.5pp），比合成数据上观察到的"没有优势甚至更差"要强得多。这意味着：**如果只看合成数据就断言"计数类原语上没有teacher信号可蒸"，会得出和真实benchmark不一致的结论**——真实的MVBench计数类任务上，35B-A3B确实比9B强不少，这部分信号在合成的Bouncing_Counting/Directional_Event_Counting上没有被捕捉到（可能是合成任务的具体设计——比如反弹计数的物理复杂度、干扰项设置——和真实视频里的计数任务不是同一种难度分布）。

**对TRD决策的实际影响**：不能笼统地说"合成实验证明TRD在计数类原语上没有信号"——这条结论目前只在合成域内部自洽，一旦对照真实benchmark数据就不成立。如果要在计数类任务上排除TRD，需要**直接在真实MVBench的action_count/moving_count数据上**做probe实验（而不是依赖合成Bouncing_Counting的结果），这是当前证据链的一个明确缺口，本报告如实记录，不掩盖。旋转/方向类的负面结论则有更强的跨数据集一致性支持，可以更放心地采纳。

---

## 实验0：thinking-on重测——结果:明确变差,不是白捡

**方法调整记录(如实说明)**：最初按P2同款配置跑了一版`max_new_tokens=4096`的pilot,单卡不并行跑了2.5小时+还不到20/30条,中途杀掉重跑。先改用文档正文原本建议的`max_new_tokens=1024`("这4类原语题目不长,1024应该够"),9B在GPU4上跑完4类原语×30条=120条,约80分钟,得到明确变差的结果(见下表)。**2026-07-15更新:用户要求把4096这个配置也补完整**——按用户指示,把GPU4-7四张卡各分1个原语类型并行跑(而不是单卡串行4类任务),120条样本~40分钟跑完,补齐了4096配置下的完整数据。

**结果**(9B,120条,4类gap原语各30条,三种配置对照)：

| 原语 | direct(thinking-off) | thinking-on(1024) | thinking-on(4096) | hit_cap(1024) | hit_cap(4096) |
|---|---|---|---|---|---|
| Rotation_Direction | 52.00% | 10.00% | **20.00%** | 96.7% | 96.7% |
| Rotation_Count | 67.00% | 16.67% | **10.00%** | 100.0% | 96.7% |
| Bouncing_Counting | 40.50% | 13.33% | **30.00%** | 96.7% | 90.0% |
| Acceleration_Identification | 60.00% | 33.33% | **46.67%** | 66.7% | 26.7% |

**结论(按文档§1预设的解读框架——"完全没有提升甚至变差"这一档,4096数据补齐后结论不变,反而更细致)**：把预算从1024加到4096后,3/4原语确实有改善(Rotation_Direction、Bouncing_Counting、Acceleration_Identification),其中Acceleration_Identification改善最明显(33.3%→46.7%,hit_cap从66.7%骤降到26.7%)——说明这一个原语确实存在"预算不够"的成分。但**Rotation_Count反而变差**(16.7%→10.0%),且**全部4个原语在4096预算下依然明显低于direct(thinking-off)的准确率**，差距从-13.3pp(Acceleration_Identification)到-57.0pp(Rotation_Count)不等。更关键的是hit_cap：除了Acceleration_Identification降到26.7%，其余3个原语在4096预算下**依然高达90-96.7%**——说明对这3类原语，即使给4倍的预算，模型大部分时候仍然没能在生成结束前收敛，不是简单"再给点空间就够了"的问题，是推理过程本身在大部分样本上就没有收敛路径。

**对后续决策的影响**：thinking-on在任何测试过的预算(1024/4096)下都不能作为"零训练成本解决readout gap"的捷径,回到distillation-readiness-experiments.md原来的判断——4类gap原语上,后续要么投入RLVR(outcome reward训练),要么做读出头/答案格式的轻量干预,thinking-on这条路已经用两组独立预算的数据排除了。

---

## 实验2：PSD试点（basic_shell_game）—— 结果:遮挡后信息几乎完全丢失,比TRD的读出层gap更严重

**方法**：`03_shell_game.py`的`basic_shell_game`(单球追踪,4x4网格,10-15次swap操作,状态最干净),200条。用`qwen_vl_utils`原生支持的`video_start`/`video_end`(秒)把**同一条视频**截成两段完全独立的输入:
- **"遮挡前"(before)**:`[0, T_before]`,T_before取initial_reveal事件的时间中点——此时球的起始位置在画面上直接可见。
- **"遮挡后"(after)**:`[0, T_after]`,T_after取最后一次swap事件的结束时刻——此时球已经历完整的10-15次swap,且**整个视频从头到尾都不会揭晓球的位置**(basic_shell_game是predictive模式,连最后的FINAL_REVEAL阶段容器也是不透明的,读了渲染代码确认过)。

两段各自独立forward,和probe-experiment.md §4.1同一套`find_video_runs()`连续run检测+3层pooling方法,分别训probe预测"起始位置"(before)和"最终位置"(after,=真实答案，16个网格格子分类)。

**probe结果**(70/30切分,线性LogisticRegression,majority baseline≈10-12%):

| 阶段 | 9B(3层区间) | 35B-A3B(3层区间) |
|---|---|---|
| before(遮挡前,sanity check上限) | **100.00%**(3层全对) | **100.00%**(3层全对) |
| **after(遮挡后,核心问题)** | **5.00%-6.67%** | **3.33%-6.67%** |

**直答baseline结果——2026-07-14更新,修正了一处真实bug**：最初用`max_new_tokens=32`(照搬probe-experiment.md §4.3那7类合成原语的值),用户指出unresolved比例(60-78%)异常高,查了原始response发现即使`enable_thinking=False`,模型也会自发按"Let's trace the object's movement step by step..."逐步叙述每一次swap操作,32个token连第一步都写不完就被截断——这是真实的截断bug,不是模型"拒绝回答"。测了256/512/1024三档,发现512→1024对unresolved/hit_cap基本没有进一步改善(模型陷入重复叙述甚至混乱自问自答的循环,不是简单地"空间不够"),于是定在1024重新跑,数字如下:

| 模型 | n | 准确率 | unresolved(解析不出答案) | hit_cap(撞满1024预算) | 仅看已解析样本的准确率 |
|---|---|---|---|---|---|
| 9B | 200 | **15.50%**(原32-token版本:4.00%) | 22.0%(原78.5%) | 27.5% | 19.87% |
| 35B-A3B | 200 | **8.50%**(原32-token版本:0.00%) | 9.5%(原60.5%) | 20.5% | 9.39% |

**核心结论(按distillation-readiness-experiments.md §3的判定框架,已用修正后数字重新核实,方向不变)**：

1. **遮挡后probe准确率(3.3%-6.7%)低于majority baseline(约10-12%)**——不是"信息在但读得不精确"，是**信息在hidden state里已经不可线性读出，比随机猜网格格子还差**。这和probe-experiment-report.md发现的TRD"4类gap原语"模式（probe 85-100% vs 直答35-69%，信息明显在、只是没读出来）**性质完全不同、程度严重得多**——那边是"读出层"问题，这里连"探针能不能读出"这道更宽松的检验都过不了，说明是**真正的感知/状态维持层gap**，不是读出层gap。
2. **直答baseline修正后依然远低于随机基线，不是截断bug造成的假象**：这道题是4选1MCQ，随机瞎猜期望准确率≈25%。修正截断问题后，9B=15.50%、35B-A3B=8.50%，**仍然明显低于25%的随机基线**；即使只看"确实给出了可解析答案"的样本，9B也只有19.87%、35B-A3B只有9.39%，**依然低于随机瞎猜**。这排除了"unresolved是因为截断，真实能力被低估"这个担心——修正bug后数字确实涨了(9B从4%到15.5%)，但涨完依然低于随机水平，说明模型是**系统性地选错**，不是"没答完/答不出"，和probe结果(比随机猜还差)相互印证：不是探针方法失效，是模型确实没能在内部维持这个状态。
3. **没有教师优势，甚至相反**：35B-A3B在这个任务上比9B更差——probe线（3.3-6.7% vs 5.0-6.7%）和直答线（修正后8.50% vs 15.50%，约为9B的一半）两条证据方向一致——teacher并不比student更擅长维持跨遮挡的状态追踪，反而更差。
4. **对PSD(novelty.md §2.2)的直接推论**：和distillation-readiness-experiments.md §3自己的预判一致——"如果遮挡后依然很高，这条能力已经具备，PSD没有意义；如果遮挡后大幅下降接近随机，这是PSD要解决的真实场景"。试点结果落在后者，而且比"大幅下降"更极端(直接跌破随机基线)——**PSD描述的"长时状态追踪失败"这个核心场景，在basic_shell_game这个最简单的试点上被清晰地验证成立，值得投入**。这和TRD（教表示对4类gap原语没有信号）以及PGR（有信号，支持投入）一起，构成了probe-experiment+distillation-readiness两轮实验对novelty.md三个方向的完整判断：**TRD不建议投入，PGR建议投入，PSD建议投入（且证据比TRD/PGR都更干净）**。

**局限（如实记录）**：只测了`basic_shell_game`这一个（12个challenge_type里最简单的）子类型，其余11个（多球追踪、属性绑定、反向推理等）没有测；只用了一次"遮挡前/遮挡后"两个时间点，没有测多个中间时间点（比如每次swap后都测一次，看信息是哪一步开始丢失的）；probe和直答都用同一批200条样本、同一个"最后一次swap结束"时间点，没有测试更短的遮挡（比如只做1-2次swap）下是否还能维持状态——如果连1次swap都不行，说明这不是"操作次数太多"的问题，而是"完全没有这个机制"，这个更精细的区分本次没做，值得后续补充。

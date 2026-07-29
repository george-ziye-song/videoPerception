# Probe Experiment 执行报告

对照 `probe-experiment.md` 逐项核对。**全部适用的实验已经做完。** `outputs_expected` 列的4项交付物：§1.5、§4.2、§4.3全部完成；§4.4（物理量回归probe）也已完成，7种原语里6种做了，只有Event_Sequence因为物理性质本身（离散事件，非连续运动）如实标注为"留白"（原因见§4）。

**2026-07-13更新**：Bouncing_Counting原本因为初始速度`random.uniform(-4,4)`没有存入metadata而无法解析重建，用户要求补做——已经给生成器加了一行代码把逐帧`(x,y,vx,vy)`真值直接存下来，重新生成了这200条样本（复用原有doc_id 600-799，视频文件/其他6种原语的数据完全没动），重新抽取了这一个原语类型的hidden state（其余6种复用原有checkpoint），现在6/7原语都完成了回归probe。

**2026-07-14更新（数字核实修正）**：`probe-experiment-verification.md`核实时发现§3表格里Bouncing_Counting的直答准确率是补跑前的过期数字（9B曾误写46.50%，35B曾误写34.50%），根因是补跑时用`--task=`只更新了这一个key、其余6项数字保留，但报告正文表格当时没跟着换成补跑后的值。现已从`direct_answer_9b.json`/`direct_answer_35b_35.json`重新核对并改为实际值（9B=40.50%，35B=29.00%），下方§3正文已更新，方向不变（gap更大，teacher更差这个结论更成立），仅数值修正。

## 0. 完成情况核对表（对照 outputs_expected 逐条核对）

| # | 交付物（原文） | 状态 | 证据 |
|---|---|---|---|
| 1 | §1.5：MVBench/TOMATO"打乱帧后掉分幅度"表 | ✅ 完成 | 2模型(9B/35B-A3B) × (20个MVBench子任务+TOMATO 6个reason_type)，见§1 |
| 2a | §4.2：每个(模型,原语类型,层)的probe准确率表 | ✅ 完成 | 2模型 × 7原语类型 × 3层，见§2 |
| 2b | §4.4：物理量回归probe的R²/误差表 | ✅ 完成(6/7原语，1种如实留白) | 2模型 × 6原语类型 × 3层，见§4 |
| 3 | §4.3：每个模型"直接作答"准确率 | ✅ 完成 | 2模型 × 7原语类型，官方`extract_mcq_answer`解析，见§3 |
| 4 | §5：gap在感知/读出/无teacher的判定 | ✅ 完成（仅TRD/PGR） | TRD判定见§3结论；PGR判定见§4结论 |

**TRD和PGR两条判定链条都完整、可以下结论。PSD没有单独判定**——这次测的7类原语全部是plan.md定义的"短时感知"（01/02号生成器），PSD真正要解决的"长时状态追踪"场景（03-10号生成器）完全没有测，见§5/§6说明，这不是漏做，是probe-experiment.md原始范围就没有覆盖到，需要另外补一批实验才能对PSD下结论。

---

## 1. §1.5 前置检验：这些benchmark子任务本身考不考时序推理

**方法**：把MVBench 20个子任task、TOMATO全部1484条样本的帧序随机打乱，用P1同款直答协议（thinking关闭）跑一遍，和未打乱的P1基线比较准确率。两个模型独立跑，交叉验证。

**证据**（用当前已修复的共享`extract_mcq_answer`对两边原始输出重新打分，同一把尺子）：

MVBench 20子任务，两模型均值：9B 69.25%→62.30%（掉6.95pp），35B-A3B 73.35%→64.15%（掉9.20pp）。子任务级别差异巨大（-2.5pp 到 +26.5pp）。**完整20子任务表，两模型并列**（按9B掉分幅度降序）：

| 子任务 | 9B baseline | 9B shuffle | 9B delta | 35B baseline | 35B shuffle | 35B delta |
|---|---|---|---|---|---|---|
| action_localization | 57.50% | 35.00% | **+22.50pp** | 59.00% | 41.00% | +18.00pp |
| moving_direction | 75.00% | 59.00% | **+16.00pp** | 76.00% | 54.00% | +22.00pp |
| action_count | 59.00% | 45.00% | **+14.00pp** | 68.00% | 41.50% | +26.50pp |
| moving_count | 66.50% | 52.50% | **+14.00pp** | 81.00% | 63.00% | +18.00pp |
| moving_attribute | 86.00% | 74.00% | **+12.00pp** | 91.50% | 81.00% | +10.50pp |
| object_existence | 87.50% | 76.50% | +11.00pp | 93.00% | 81.50% | +11.50pp |
| action_sequence | 79.00% | 69.50% | +9.50pp | 84.00% | 74.50% | +9.50pp |
| action_prediction | 70.00% | 62.00% | +8.00pp | 69.00% | 57.50% | +11.50pp |
| state_change | 61.50% | 53.50% | +8.00pp | 65.50% | 56.00% | +9.50pp |
| character_order | 82.00% | 75.00% | +7.00pp | 87.50% | 74.50% | +13.00pp |
| fine_grained_pose | 64.00% | 57.50% | +6.50pp | 73.00% | 66.00% | +7.00pp |
| object_interaction | 84.50% | 78.00% | +6.50pp | 85.00% | 79.50% | +5.50pp |
| action_antonym | 88.00% | 84.50% | +3.50pp | 79.50% | 73.00% | +6.50pp |
| counterfactual_inference | 72.00% | 68.50% | +3.50pp | 73.50% | 72.50% | +1.00pp |
| object_shuffle | 39.00% | 37.00% | +2.00pp | 50.00% | 44.50% | +5.50pp |
| scene_transition | 96.50% | 94.50% | +2.00pp | 97.50% | 98.00% | -0.50pp |
| episodic_reasoning | 55.00% | 56.00% | -1.00pp | 65.50% | 60.00% | +5.50pp |
| egocentric_navigation | 32.00% | 33.50% | -1.50pp | 28.00% | 26.50% | +1.50pp |
| fine_grained_action | 46.50% | 48.50% | -2.00pp | 50.50% | 49.00% | +1.50pp |
| unexpected_action | 83.50% | 86.00% | -2.50pp | 90.00% | 89.50% | +0.50pp |
| **均值** | **69.25%** | **62.30%** | **+6.95pp** | **73.35%** | **64.15%** | **+9.20pp** |

两模型交叉验证后一致地"真掉分明显"（>10pp）的子任务：action_localization、moving_direction、action_count、moving_count、moving_attribute。一致地"几乎不掉分"（可以跳过）：scene_transition、unexpected_action、fine_grained_action、egocentric_navigation。

**TOMATO 6个reason_type，两模型并列**（用同一把已修复的`extract_mcq_answer`重新打分，n=各reason_type的实际样本数，两模型共享同一批1484条样本、同样的doc_id，n相同）：

| reason_type | n | 9B baseline | 9B shuffle | 9B delta | 35B baseline | 35B shuffle | 35B delta |
|---|---|---|---|---|---|---|---|
| direction | 403 | 48.39% | 32.75% | **+15.63pp** | 57.32% | 29.28% | **+28.04pp** |
| shape&trend | 223 | 39.46% | 24.22% | **+15.25pp** | 39.91% | 28.70% | **+11.21pp** |
| count | 292 | 34.93% | 27.74% | +7.19pp | 36.30% | 34.59% | +1.71pp |
| visual cues | 70 | 51.43% | 48.57% | +2.86pp | 52.86% | 44.29% | +8.57pp |
| rotation | 286 | 26.92% | 25.87% | +1.05pp | 22.38% | 24.13% | -1.75pp |
| velocity&frequency | 210 | 18.57% | 18.10% | +0.48pp | 37.14% | 30.00% | +7.14pp |
| **整体** | **1484** | **36.19%** | **27.83%** | **+8.36pp** | **40.77%** | **30.05%** | **+10.71pp** |

两模型一致掉分明显（>10pp）的是**direction**和**shape&trend**；一致几乎不掉分的是**rotation**（9B+1.05pp/35B-1.75pp，反直觉但两模型一致——这一点和§4发现的"旋转类原语高R²有位置混淆嫌疑"合在一起看，共同指向"两个模型处理旋转类任务时可能并没有真的紧密依赖逐帧时序信息"这个更大的图景）。

**结论**：MVBench 5类 + TOMATO 2类（direction、shape&trend）是两个模型都验证过的"真的需要时序推理"的战场，§4 probe应该优先在这些对应的合成原语上做——这也是我选择§3映射表里那7个task type的依据。

---

## 2. §4.2 Probe结果：hidden state里到底编不编码正确信息

**方法**：冻结前向（不生成），抓每条合成样本在3层（浅/中/深，按`num_hidden_layers`的25%/50%/90%位置取）的hidden state，按视频temporal-group（=2个采样帧经`temporal_patch_size=2`合并后的组，不是原始单帧——这是Qwen3.5视觉tokenizer的真实粒度，已用实际forward+token计数验证过，不是假设）做mean pooling，训一个线性probe，监督信号用`metadata.jsonl`里的数值/类别GT（不是字母答案）。

**7种原语类型的标签构造**（都来自`ground_truth_details.other_details`，不受任何文本提取器/选项顺序影响）：
- Complex_Direction_Identification → `path`（2段方向的组合，12类）
- Rotation_Direction → `direction`（顺/逆时针，2类）
- Rotation_Count → `total_rotations`（3类）
- Bouncing_Counting → `target_bounce_count`（5类）
- Directional_Event_Counting → **从`sequence`+`target_direction`现算的真实计数**（不是MCQ答案字母，这是中途踩过的坑：第一版误用了被随机打乱顺序的MCQ答案字母当标签，导致看起来"probe读不出"，后来发现字母是随机分配到选项的、和真实计数无关，改用`sequence`+`target_direction`现算的整数计数重跑后，结论没变——见§4的回归结果交叉验证）
- Acceleration_Identification → `motion_type`（3类）
- Event_Sequence → 逐temporal-group的"是否落在关键事件时间窗口内"二分类（用采样帧均匀时间戳近似还原每组的时间中心）

**结果**（1400条样本，70/30切分，n_val=60，逻辑回归probe。**2026-07-14更新：之前这里只列了3层里的best层，读者看不出层间差异，被指出后改成完整展示shallow/mid/deep三层**——分类probe这里层间差异较小(多数在3-8pp内)，且"读得出/读不出"这个结论性判断在3层上一致，展示全部3层不影响§3的TRD判定）：

| 原语 | 9B shallow/mid/deep | 35B shallow/mid/deep | majority baseline |
|---|---|---|---|
| Complex_Direction_Identification | 88.33% / 96.67% / 91.67% | 85.00% / 88.33% / 90.00% | 12.5% |
| Rotation_Direction | 95.00% / 93.33% / 88.33% | 96.67% / 91.67% / 86.67% | 53.0% |
| Rotation_Count | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 37.0% |
| Bouncing_Counting | 98.33% / 100.00% / 95.00% | 100.00% / 100.00% / 96.67% | 24.0% |
| Directional_Event_Counting | 25.00% / 18.33% / 21.67% | 23.33% / 18.33% / 23.33% | 32.2% |
| Acceleration_Identification | 98.33% / 100.00% / 96.67% | 98.33% / 100.00% / 93.33% | 39.5% |
| Event_Sequence | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 87.3%(逐group) |

**结论**：除了Directional_Event_Counting，其余6类原语的hidden state都能以85-100%的准确率被线性probe读出正确答案，远超随机基线——说明**这些信息在两个模型的表示层里都编码得很清楚**。

---

## 3. §4.3 直接作答baseline：模型自己答得对吗

**方法**：同一批1400条样本，同样的直答协议，这次真正generate（`max_new_tokens=32`，贪婪解码），用共享的官方`extract_mcq_answer`解析（和P1/P2用同一个函数），和GT字母比较。

**结果**（下表"probe"列是shallow/mid/deep三层里的最高值——完整3层数字见§2的表格，这里只取最高值是因为判读逻辑只关心"3层里有没有任何一层能读出"，不影响下面的TRD判定；即使换成shallow层或mid层，"probe远高于直答"这个方向在4类gap原语上都成立，不是靠挑最高层撑出来的结论）：

| 原语 | 9B probe | 9B直答 | 35B probe | 35B直答 | 判读（按§5判定框架） |
|---|---|---|---|---|---|
| Complex_Direction_Identification | 96.67% | 99.50% | 90.00% | 100.00% | 能→对，端到端没问题 |
| Event_Sequence | 100.00% | 99.50% | 100.00% | 100.00% | 能→对，端到端没问题 |
| Rotation_Direction | 95.00% | **52.00%** | 96.67% | **52.00%** | **能→不对，gap在读出层** |
| Rotation_Count | 100.00% | 67.00% | 100.00% | 69.00% | **能→不对，gap在读出层** |
| Bouncing_Counting | 100.00% | 40.50% | 100.00% | **29.00%** | **能→不对，gap在读出层，且teacher更差** |
| Acceleration_Identification | 100.00% | 60.00% | 100.00% | 64.00% | **能→不对，gap在读出层** |
| Directional_Event_Counting | 25.00% | 71.50% | 23.33% | 82.50% | 不能→对，反常 |

**核心结论（TRD判定）**：7类原语里有**4类**（旋转方向/圈数、反弹计数、加减速）呈现完全一致的"probe读得出、答案答不对"模式：
1. Hidden state里线性可读的准确率是85-100%——信息**已经在表示层里**。
2. 但模型自己generate出来的最终答案准确率只有35-69%——**读出/组合推理层丢失了这些信息**。
3. **teacher(35B-A3B)在这4类原语上并不比student(9B)强**（Bouncing_Counting上teacher直答反而更差：29.0% vs 40.5%；其余3类两模型基本持平）。

按`probe-experiment.md` §5原文的判定逻辑："若teacher probe acc和student差不多,但teacher直答acc更高→gap不在表示层...该学的是teacher的读出方式"——但这里连这一条都不完全成立，因为**teacher的直答acc也没有系统性更高**（4类里3类持平、1类反而更差）。这意味着：**不仅gap不在表示层，而且这几类原语上根本没有一个"更会读出"的teacher可以蒸馏**。

**对TRD(novelty.md §2.1)的直接推论**：TRD的前提是"student的gap来自没有正确编码时序/关系信息"。但这4类原语（覆盖了旋转、计数、加减速——相当一部分§1.5筛出的"真的需要时序推理"的原语类型的近亲）显示，**信息已经编码好了，教（更好的）表示不会有帮助，因为连teacher自己都没能把已有的信息正确读出来**。这是对TRD假设的一个直接的反面证据，不是"部分支持"，是"在这4类原语上，representation-teaching这条路线本身缺乏可蒸馏的信号来源"。

**唯一的反常项**：Directional_Event_Counting（数序列里往某方向移动了几次）——probe读不出（~24%，接近或低于基线32.2%），但两个模型的直答反而是相对最好的（71.5%/82.5%，teacher还更强）。这不符合"probe低→答案也该低"的预期。§4的回归probe（用另一套完全不同的方法：连续速度向量回归而不是分类）在这个原语上得到了**同样的负面结果**（9B R²≈0.3-0.4，35B R²为负），两种独立方法交叉验证，说明这不是分类probe方法本身的偶然缺陷，是这类"过滤+计数"复合运算在这两个模型的表示层里确实编码得不好。

**这个反常不违反§2.3.1任何定理，这里说清楚边界条件（不是理论出了问题）**：DPI（结论一）约束的是"生成开始前那一个位置上的frozen hidden state $h_t$能读出多少信息"——probe测的正好是这个量。但模型真正generate答案时，走的不是"直接从$h_t$读出答案"这一步，而是**多步解码**：每生成一个token，都会再做一整轮attention，可以重新扫描视频token、重新聚合信息，累积的计算量远超probe看到的那一份frozen snapshot。所以"生成准确率超过单点probe准确率"，只说明**生成过程本身做了probe没有覆盖到的额外计算**（多步注意力重新提取、可能的隐式复述/计数），不代表$I(h_t;\hat y)>I(h_t;s_{t+k})$——DPI约束的是同一个$h_t$到下游函数的信息流，probe和generate是**从这同一个$h_t$出发的两个不同的下游函数**（probe是"$h_t$直接线性映射"，generate是"$h_t$经过多步自回归展开"），DPI并不要求这两个不同函数的输出准确率有大小关系。这是一个方法论边界，不是矛盾。

**排查"是不是选项语义混淆，而不是真的推理错误"**（抽查了两个模型全部wrong answer，按语义值而不是随机打乱的字母做归类）：
- **Rotation_Direction**：两个模型的错误**100%**集中在"把顺时针答成逆时针（或反过来）"，**0%**误选无关的distractor"没有在转"。说明模型能正确排除不相关选项，错误是在两个真实方向之间发生的——排除了"选项格式让模型选偏"这个解释，但没有完全排除"模型对顺/逆时针这两个概念本身的视觉判断有系统性偏差"（比如坐标系/视角convention问题），这点仍然可能是"读出层"问题的一种具体形式，不是prompt格式问题。
- **Rotation_Count**：67%(9B)/77%(35B)的错误是"差1"（近似正确，比如真值3猜成2或4），只有23-33%是"差>1"的离谱错误——**这个模式支持"模型确实在数，只是数得不够准"，是典型的读出层精度问题，不是猜或者看错题**。
- **Acceleration_Identification**：错误极度集中在"accelerating(真)→误答constant(选)"这一个方向（9B 98%的错误、35B 44-56%），几乎不和decelerating混淆。这有清楚的物理解释：accelerating从speed=0.5起步（很慢），在有限采样帧内看起来和"匀速"很接近；decelerating从speed=12.0起步（很快）再减速，视觉上更容易和"匀速"区分开。**这是一个真实的、可解释的视觉可辨别度问题，不是选项文字混淆**。
- **Bouncing_Counting**：模式和前三个不一样——只有15-34%是"差1"，多数(66-85%)是"差>1"的离谱错误，而且**误差分布明显偏向正方向**（模型系统性地把反弹次数数多了，尤其35B几乎所有错误都是往多了猜）。这和前三类"接近正确"的模式不同，更像是"在较长、较不规则的反弹轨迹上确实追踪不住"，是这4类原语里读出/追踪能力表现最差的一个，值得单独记录。

**综合结论**：4类"读出层gap"原语里，3类（旋转方向、旋转圈数、加减速）的错误模式都支持"真的在推理，只是读出不精确/有系统性视觉偏差"，不是选项文字混淆；Bouncing_Counting的错误模式明显更差、更离谱，可能反映这类较长/较不规则轨迹的追踪本身就更难，这个原语上"gap在读出层"这个判定虽然从probe/直答的对比数字上仍然成立，但错误的性质和其他3类不完全一样，值得在后续设计RLVR reward或诊断实验时单独对待。

---

## 4. §4.4 物理量回归Probe：连续物理量能不能被线性回归出来

**方法（第一轮，2026-07-13上午，未重新生成数据/未重新抽hidden state）**：`probe-experiment.md`原文建议改生成器多吐一份`physical_state_per_frame`。但核实7个生成器的渲染代码后发现：**5/7种原语的连续物理量可以从已经存在的`metadata.jsonl`(`video_events_timeline_ms`时间戳 + `other_details`参数 + CONFIG里固定的fps=30/width=512)精确解析重建**，渲染器的运动公式是确定性的，唯一的自由变量就是这几个已经存下来的参数。因此第一轮**直接复用已经抽好的`hidden_states/{9b,35b_35}/*.pt`**，没有碰视频、没有重新抽取、没有占用GPU4-7。重建逻辑用实际数据做过验证（比如Complex_Direction_Identification的速度向量在path中点精确翻转符号，数值和renderer公式手算吻合）。

**方法（第二轮，2026-07-13下午，补做Bouncing_Counting）**：Bouncing_Counting的初始速度`random.uniform(-4,4)`当时没有存进metadata，无法解析重建。按要求补做：给`generate_bouncing_counting()`加了3行代码，在渲染循环里把已经在算的`pos`/`vel`逐帧append成`physical_state_per_frame`（原来只用来画图、算完就丢），存进`ground_truth_details`。然后：
1. 只重新生成Bouncing_Counting这200条（复用原有doc_id 600-799区间，其余6种原语的`sft.jsonl`/`metadata.jsonl`/视频文件完全没动，改动前后做过逐条diff确认一致）。
2. 只重新抽取这一个原语类型的hidden state（其余6个复用checkpoint，脚本的断点续跑机制自动跳过）。
3. 只重跑这一个原语类型的直接作答baseline（新增了`--task=`过滤参数，避免重跑其余1200条）。
这两步都需要GPU，**临时暂停了GPU4-7上的35B P2第二分片约15分钟**（该分片当时在跑的`mvbench_action_count`这个chunk有1h11m进度因为不是完整chunk的checkpoint而重新从头开始，其余已完成的chunk不受影响），做完后已经把分片重新启动接上。

**6种能做的原语和对应的物理量**：
- Rotation_Direction / Rotation_Count → **累积旋转角度θ(t)**（弧度，随时间线性增长，比角速度本身更有回归意义——角速度在单个样本内是常数）
- Acceleration_Identification → **速度标量|v|**（3种motion_type对应的速度曲线是硬编码常数，可精确反推；完整(x,y)位置做不了，因为起始点/运动轴没有存下来）
- Complex_Direction_Identification / Directional_Event_Counting → **瞬时2D速度向量(vx,vy)**（按分段线性公式精确重建，含Directional_Event_Counting的"尾段静止"逻辑）
- Bouncing_Counting → **瞬时2D速度向量(vx,vy)**（生成器改动后直接dump真值，不是解析重建）

**1种做不了、如实留白的原语**：
- **Event_Sequence**：物体是离散的出现/消失，不是连续运动，没有自然的物理量可回归（呼应文档对03-10号生成器的排除逻辑，虽然它属于02号脚本，但物理性质相同，同样的道理适用）。

**结果**（线性回归probe，R²，70/30按视频切分后展开成帧级行。**2026-07-14更新：之前这里只列了3层里的best层，被指出后改成完整展示shallow/mid/deep——这里层间差异比§2的分类probe明显更大，之前只报best层确实比展示全部3层"更好看"，如实改正**）：

| 原语 | 物理量 | 9B shallow/mid/deep | 35B shallow/mid/deep |
|---|---|---|---|
| Rotation_Count | 累积角度 | 0.997 / 0.992 / 0.986 | 0.992 / 0.983 / 0.964 |
| Bouncing_Counting | 速度向量 | 0.886 / 0.885 / 0.709 | 0.812 / 0.862 / 0.686 |
| Acceleration_Identification | 速度标量 | 0.780 / 0.915 / 0.875 | **0.361 / 0.692 / 0.599** |
| Complex_Direction_Identification | 速度向量 | 0.817 / 0.792 / 0.678 | 0.736 / 0.700 / **0.381** |
| Rotation_Direction | 累积角度 | 0.715 / 0.587 / 0.645 | 0.585 / 0.552 / **0.262** |
| Directional_Event_Counting | 速度向量 | 0.311 / 0.400 / 0.303 | **-0.263 / 0.058 / -0.262** |

**层间差异的诚实讨论**：35B在好几个原语上深层R²明显低于中/浅层（Rotation_Direction深层仅0.262，是shallow的45%；Complex_Direction_Identification深层0.381，是shallow的52%；Acceleration_Identification反而是shallow最低0.361、mid最高0.692）。9B的层间差异相对小一些，但Bouncing_Counting/Complex_Direction_Identification深层也比浅/中层低20-30%。这说明"物理状态信息"在这几个原语上**不是均匀分布在所有深度**，深层（接近输出、更偏向"决策"而非"感知"的层）R²往往更低——这本身是一个有意义的信号（信息可能在深层被部分丢弃/压缩用于别的用途），不是噪声，但报告只挑best层确实让PGR的信号强度显得比"随便挑一层"更乐观。下面的结论用的是"5种原语里每种至少有一层R²不低"这个更保守的表述，不是暗示每一层都表现同样好。

**结论（PGR判定）**：除了Directional_Event_Counting，其余5种原语**至少有一层**的R²在0.59-1.0之间，意味着**hidden state在某个深度上确实携带了大量关于连续物理状态的信息**——但不是每一层都同样强，尤其35B的深层在多个原语上明显偏弱，这一点在决定PGR具体接入哪一层的读出头时需要纳入考虑，不能默认"任意层都行"。**这里引用novelty.md §2.3.1需要纠正一处（感谢指出）**：R²对应信息量下界的论证依据是结论二（Barber & Agakov 2003变分互信息下界：$I(h_t;s_{t+k}) \ge H(s)-\mathbb E[-\log q_\phi(s\mid h)]$，取高斯变分族时代入MSE/R²），不是结论一（数据处理不等式，DPI）——DPI证明的是反方向的**上限/天花板**（下游任何函数都不能超过$h_t$本身含有的信息量，这是probe-experiment.md整体诊断逻辑、以及上面§3判定TRD时用的论证依据），结论二证明的才是**下界**（回归损失低→$I(h_t;s_{t+k})$不低），方向相反，不能混用。数字本身不受影响，但如果这段论述要写进正式方法部分，引用需要改成结论二。**R²不低就说明这个信息量下界不低，PGR训练有信号来源可以利用，不是无米之炊**。这是本次实验第一次对PGR方向给出正面的、有数据支持的判断——之前完全没有证据。**Bouncing_Counting补做后R²(0.86-0.89)和其他运动类原语量级一致**，进一步支持这个结论具有跨原语的一致性，不是个别类型的偶然。

Directional_Event_Counting在回归probe下依然表现差（9B勉强及格，35B为负——比"永远预测均值"还差），和§2的分类结果交叉验证一致，进一步坐实"过滤+计数"这类复合运算在两个模型的表示层里都编码得不好，这类原语不管是TRD还是PGR都可能收效有限（PSD没有测，见§5/§6的范围说明）。

**旋转类原语的高R²需要单独核实一个混淆变量（感谢指出，这里补做了两组对照实验）**：Rotation_Count/Rotation_Direction的目标量θ(t)=frame_idx×ω，如果ω在样本间几乎不变，probe完全可能只是学会了"这是第几个temporal-group"（transformer天然带位置信息）再乘一个大致固定的ω，而不需要真的读懂视觉上的旋转——这样高R²就是位置编码的副产品，不是模型理解了旋转运动。实际检查：

1. **ω的样本间分布**：Rotation_Direction的|ω|在全部200个样本里**完全恒定**(=π/30，只有符号变化)——这正是最容易被位置confound的情况。Rotation_Count的ω有真实方差（std=0.044，覆盖0.052-0.209，因为`total_rotations∈{2,3,4}`和视频时长都会变）。
2. **position-only baseline**（完全不用hidden state，只用归一化时间位置g/T去线性回归θ）：Rotation_Count的R²=0.782（！），Rotation_Direction的R²≈-0.001（约等于预测均值，说明纯位置对Direction毫无解释力，因为符号大约各占一半，边际化掉符号后θ的位置线性趋势接近于零）。
3. **残差检验**（Rotation_Count，扣掉position-only能解释的部分，看hidden state能不能解释剩下的残差）：残差方差占总方差约21.8%，hidden state在这部分残差上的R²是**0.85-0.98**（9B）/**0.85-0.97**（35B）——说明**hidden state确实携带了超出纯位置的、样本特异的信息**（大概率是每个样本具体的total_rotations/视频时长这类只能从画面读出来的东西），不是纯粹的位置泄漏，但**Rotation_Count的头条数字（R²=0.997）里有相当一部分（0.782/0.997≈78%）是位置结构本身贡献的，不是"模型看懂了旋转"这个叙事单独能解释的**，需要用残差R²(0.85-0.98)而不是原始R²来衡量"真的读懂了多少"。
4. **Rotation_Direction的sign-oracle分析**：假设完美知道方向符号+精确位置，理论R²=1.0；符号错误率5%时R²≈0.82-0.90，10%时R²≈0.65-0.82。观测到的实际R²(9B 0.585-0.715，35B 0.262-0.585)落在"符号错误率5-15%"这个区间内，和§2分类probe测出的方向判断准确率(9B 95%/35B 87-97%，对应5-13%错误率)**量级吻合**——说明Rotation_Direction的R²基本可以理解为"能较准判断符号+精确知道位置"的直接结果，不存在position-only那种confound（因为position-only baseline本身就是≈0，不存在"位置单独解释大部分方差"的问题）。

**修正后的判读**：Rotation_Direction的R²是干净的，可以按原样解读为"读出了旋转方向信息"；**Rotation_Count的R²需要修正解读**——真正体现"读懂了视觉旋转"的部分是残差R²(0.85-0.98)，不是头条的0.997，虽然结论方向不变（hidden state确实携带真实的、样本特异的旋转信息），但数字的解释力度应该调低，从"近乎完美"调整为"位置提供了大部分基础，视觉信息在此之上补充了一块有意义但没那么夸张的额外解释力"。

---

## 5. 方法局限（如实记录）

1. **单帧/单组粒度**：受Qwen3.5视觉tokenizer的`temporal_patch_size=2`限制，probe的时间粒度是"2个采样帧合并后的组"，不是原始单帧。已用实际forward+token计数验证过这个边界（连续run检测，not固定stride），不是假设。
2. **分类probe的eval集较小**：`n_val=60`（70/30切分200条/类），100%和95%之间的差异在这个规模下置信区间较宽，不宜过度解读小数点后第二位的差异，但85%+ vs 25%这种量级的差异是稳的。
3. **只测了一个teacher**：按你之前的拍板，只跑了Qwen3.5-35B-A3B，没有跑Qwen3.6-35B-A3B做交叉验证（spec里是"建议"不是"必须"）。
4. **Event_Sequence的majority baseline很高**（87.3%），100%虽然是完美但相对基线的提升(~13pp)不如其他几类原语的提升幅度大，可能部分反映"形状出现/消失"本身是强视觉突变、容易被单帧特征捕捉，不完全是"时序推理"信号。
5. **回归probe的物理量简化**：Acceleration_Identification只回归了速度标量，没有回归完整(x,y)位置（起始点/运动轴信息未存入metadata，无法重建）；Complex_Direction_Identification/Directional_Event_Counting的速度向量是按renderer公式精确重建，但如果renderer公式本身有我没读到的边界情况（比如帧数极端小的clamp逻辑），个别样本可能有轻微误差，多数样本的验证（如上文Complex_Direction_Identification的手算核对）显示重建是准的。Bouncing_Counting是生成器直接dump真值，不存在这个重建误差的风险。
5b. **Bouncing_Counting的200条样本是重新生成的**，和最初§1.5筛出"哪些benchmark子任务需要时序推理"时用的不是同一批随机抽样（但同分布、同规模），§4.2的分类结果在新旧两批数据上高度一致（95-100%），说明这个差异不影响结论稳定性。
6. **Directional_Event_Counting的反常结果**：分类和回归两种方法交叉验证都表现差，但具体原因（模型是否在生成阶段用了其他机制）未深入验证，如实记录，不过度归因（§3已补充DPI边界条件的解释，说明这不是理论矛盾）。
7. **本次实验只覆盖了plan.md定义的"短时感知"原语（01/02号生成器），完全没有测"长时认知"原语（03-10号生成器：shell game、滑块、纸牌堆这类需要跨遮挡追踪状态的任务）**。这一点很重要：PSD(novelty.md §2.2)最初要解决的核心场景正是"长时状态追踪"，而这次7类原语全部是短时的（单个动作/单段旋转/单次反弹计数，没有一个涉及物体被遮挡后还要记住它的状态），所以**这次的负面结论只能约束TRD，不能同时约束PSD**——PSD的判定需要另外找03-10号生成器补一批探针，不能直接套用这批短时原语的结果。
8. **旋转类原语的位置混淆已经用position-only baseline和残差检验排查过**（见§4），Rotation_Direction干净，Rotation_Count的头条R²需要用残差R²重新解读，细节见§4正文，这里不重复。

---

## 6. 结论与建议

- **TRD方向**：证据**不支持**投入TRD（教表示）。7类原语里4类明确显示"表示已经够好，问题在读出/组合推理"，且没有教师优势可蒸，错误模式抽查也支持"真的在推理只是不精确"而非选项文字混淆；1类（Directional_Event_Counting）反常，也不支持"教表示"的叙事（分类和回归两种方法都显示读不出，且有清楚的DPI边界条件解释，不是理论矛盾）；其余2类（Complex_Direction_Identification、Event_Sequence）端到端本来就没问题。**建议改investigate RLVR（用outcome reward逼模型正确使用已有信息）或读出头/答案格式的轻量干预，而不是representation distillation。**
- **PSD方向**：**这次实验没有测，不能下结论**。7类原语全部来自01/02号生成器（plan.md定义的"短时感知"），PSD真正要解决的"长时状态追踪"（03-10号生成器：shell game/滑块/纸牌堆）一个都没碰。上面TRD的负面结论不能直接套用到PSD，如果要判定PSD值不值得投，需要另外在03-10号生成器上补一批探针实验。
- **PGR方向**：证据**支持**继续投入，而且现在是6个原语类型里5个都验证过、结论一致。R²在0.59-1.0之间（含补做后的Bouncing_Counting），说明连续物理状态在hidden state里有充分的信息量可供回归——novelty.md §2.3.1**结论二**（Barber-Agakov变分下界，不是结论一DPI，见§4的引用纠正）的下界论证在这几类原语上是"下界不低"，值得继续往下做PGR训练设计。Directional_Event_Counting是唯一例外，PGR在这类"过滤+计数"复合原语上可能同样收效有限。**Rotation_Count的R²需要按§4的残差检验重新解读**（头条0.997里约78%是位置结构贡献的，真正的"视觉旋转信息"体现在残差R²0.85-0.98上），量级仍然支持PGR有信号可用，但不宜引用0.997这个未修正的数字。
- **§1.5筛出的7类"真的需要时序"的原语**（MVBench 5类+TOMATO 2类）和§4.2/4.3/4.4测的7类合成原语之间有部分对应（rotation↔TOMATO rotation、count↔MVBench action_count/moving_count等），但不是完全一一映射，如果要更严谨地把两边的结论对上，可以进一步核对§3映射表。

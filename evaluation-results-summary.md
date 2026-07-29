---
title: "全部评估结果汇总"
date: "2026-07-21"
status: "汇总现有各报告里的实测数字,不包含新实验;每个数字标注来源文件,便于溯源核对"
---

# 全部评估结果汇总

本文档汇总项目至今所有评估/诊断实验的实测数字。每节标注来源文件,数字均为原文档中的实测值,未做任何加工或估算。文末单列"已知修正/存疑项",如实说明哪些数字是后来纠正过的、哪些还有未解决的小出入,不隐藏。

涉及模型:**anchor**=Qwen3-VL-8B-Instruct;**student**=Qwen3.5-9B(dense);**teacher候选**=Qwen3.5-35B-A3B、Qwen3.6-35B-A3B(均为MoE,256专家/8激活)。

---

## 1. P1(关闭思考)基线评估

来源:`baseline.md`(Qwen3-VL-8B-Instruct,自写评测脚本,非官方harness,显式标注仅供sanity check)

| Benchmark | 指标 | 数值 | 样本量 |
|---|---|---|---|
| MVBench(19/20任务,fine_grained_pose因视频缺失跳过) | mean-of-tasks acc | 0.6637 | 3800 |
| TemporalBench-short | Binary / MBA | 0.6839 / 0.2827 | 9867 / 2179 |
| TemporalBench-long | Binary / MBA | 0.6514 / 0.2573 | 5485 / 1574 |
| TOMATO | acc | 0.3464 | 1484 |

MVBench 19任务明细(高→低):moving_attribute .93、scene_transition .925、object_existence .86、unexpected_action .845、action_antonym .775、character_order .73、state_change .725、object_interaction .72、action_prediction .70、action_sequence .695、counterfactual_inference .675、moving_count .67、moving_direction .615、episodic_reasoning .565、action_count .48、fine_grained_action .455、action_localization .44、object_shuffle .405、egocentric_navigation .40。

已知协议偏差(§六如实列出):4个MVBench任务未做时间裁剪、fine_grained_pose缺失跳过、分辨率降至256×28×28(官方约672×896)、固定16帧、TemporalBench的caption子任务(1891样本,需GPT judge)跳过。

---

来源:`probe-experiment-report.md`§1 / `ppt-outline.md`(9B、35B-A3B,正式lmms-eval harness)

**P1标准配置下的benchmark总分(9B vs 35B-A3B)**:

| Benchmark | 9B | 35B-A3B | teacher优势 |
|---|---|---|---|
| MVBench(20子任务均值) | 69.42% | 73.35% | +3.93pp |
| TOMATO(总体) | 36.19% | 40.77%(或36.25%,见文末存疑项) | +4.58pp |
| Video-MME | 67.07% | — | — |

---

## 2. P2(打开思考)结果与P1对比

来源:`ppt-outline.md`

**9B: P1 vs P2**

| Benchmark | P1 | P2 | Δ |
|---|---|---|---|
| MVBench | 69.42% | 63.17% | -6.25pp |
| TOMATO | 36.19% | 36.12% | -0.07pp |
| Video-MME | 67.07% | 63.37% | -3.70pp |

**max_new_tokens扫描(TOMATO,40样本子集,n=40)**:2048→50.0%;**4096→55.0%**;8192→50.0%;32768→45.0%。最终P2配置采用max_new_tokens=4096。

注:TOMATO全量1484样本分4个shard跑出的总分36.12%,是4个shard(52.02%/27.22%/28.57%/36.66%)的平均——40样本子集55.0%只是第一个shard、不能代表整体,已在报告里说明。

**结论**:打开思考(不训练,仅推理时切换)在9B上对3个benchmark全部没有正向收益,MVBench反而明显下降。

---

## 3. 时序推理预检(帧乱序实验)

来源:`probe-experiment-report.md`§1

**MVBench 20子任务乱序前后对比(9B / 35B-A3B)**,按9B掉分幅度降序:

| 子任务 | 9B baseline→shuffle(Δ) | 35B baseline→shuffle(Δ) |
|---|---|---|
| action_localization | 57.50%→35.00%(+22.50pp) | 59.00%→41.00%(+18.00pp) |
| moving_direction | 75.00%→59.00%(+16.00pp) | 76.00%→54.00%(+22.00pp) |
| action_count | 59.00%→45.00%(+14.00pp) | 68.00%→41.50%(+26.50pp) |
| moving_count | 66.50%→52.50%(+14.00pp) | 81.00%→63.00%(+18.00pp) |
| moving_attribute | 86.00%→74.00%(+12.00pp) | 91.50%→81.00%(+10.50pp) |
| object_existence | 87.50%→76.50%(+11.00pp) | 93.00%→81.50%(+11.50pp) |
| action_sequence | 79.00%→69.50%(+9.50pp) | 84.00%→74.50%(+9.50pp) |
| action_prediction | 70.00%→62.00%(+8.00pp) | 69.00%→57.50%(+11.50pp) |
| state_change | 61.50%→53.50%(+8.00pp) | 65.50%→56.00%(+9.50pp) |
| character_order | 82.00%→75.00%(+7.00pp) | 87.50%→74.50%(+13.00pp) |
| fine_grained_pose | 64.00%→57.50%(+6.50pp) | 73.00%→66.00%(+7.00pp) |
| object_interaction | 84.50%→78.00%(+6.50pp) | 85.00%→79.50%(+5.50pp) |
| action_antonym | 88.00%→84.50%(+3.50pp) | 79.50%→73.00%(+6.50pp) |
| counterfactual_inference | 72.00%→68.50%(+3.50pp) | 73.50%→72.50%(+1.00pp) |
| object_shuffle | 39.00%→37.00%(+2.00pp) | 50.00%→44.50%(+5.50pp) |
| scene_transition(对照) | 96.50%→94.50%(+2.00pp) | 97.50%→98.00%(-0.50pp) |
| episodic_reasoning | 55.00%→56.00%(-1.00pp) | 65.50%→60.00%(+5.50pp) |
| egocentric_navigation(对照) | 32.00%→33.50%(-1.50pp) | 28.00%→26.50%(+1.50pp) |
| fine_grained_action(对照) | 46.50%→48.50%(-2.00pp) | 50.50%→49.00%(+1.50pp) |
| unexpected_action(对照) | 83.50%→86.00%(-2.50pp) | 90.00%→89.50%(+0.50pp) |
| **20任务均值** | **69.25%→62.30%(+6.95pp)** | **73.35%→64.15%(+9.20pp)** |

**TOMATO按reasoning_type乱序前后对比**:

| reasoning_type | n | 9B baseline→shuffle(Δ) | 35B baseline→shuffle(Δ) |
|---|---|---|---|
| direction | 403 | 48.39%→32.75%(+15.63pp) | 57.32%→29.28%(+28.04pp) |
| shape&trend | 223 | 39.46%→24.22%(+15.25pp) | 39.91%→28.70%(+11.21pp) |
| count | 292 | 34.93%→27.74%(+7.19pp) | 36.30%→34.59%(+1.71pp) |
| visual cues | 70 | 51.43%→48.57%(+2.86pp) | 52.86%→44.29%(+8.57pp) |
| rotation(对照) | 286 | 26.92%→25.87%(+1.05pp) | 22.38%→24.13%(-1.75pp) |
| velocity&frequency | 210 | 18.57%→18.10%(+0.48pp) | 37.14%→30.00%(+7.14pp) |
| **总体** | **1484** | **36.19%→27.83%(+8.36pp)** | **40.77%→30.05%(+10.71pp)** |

**结论**:两个模型在"真正需要时序推理"的任务上乱序后明显掉分(最高22-28pp),对照组(scene_transition/egocentric_navigation/fine_grained_action/unexpected_action/rotation)基本不受影响甚至反向——说明模型确实在利用帧序信息,不是瞎猜,这是后续所有"读出层gap"诊断实验的前提依据。

---

## 4. Probe实验(隐藏状态诊断)

来源:`probe-experiment-report.md`§2/§3/§4,`probe_data/probe_results_9b.json`、`probe_results_35b_35.json`(已核对,数字与报告完全一致)

### 4.1 分类probe准确率(shallow/mid/deep三层,1400样本,70/30切分,n_val=60)

| 原语 | 9B (shallow/mid/deep) | 35B-A3B (shallow/mid/deep) | 多数类基线 |
|---|---|---|---|
| Complex_Direction_Identification | 88.33% / 96.67% / 91.67% | 85.00% / 88.33% / 90.00% | 12.5% |
| Rotation_Direction | 95.00% / 93.33% / 88.33% | 96.67% / 91.67% / 86.67% | 53.0% |
| Rotation_Count | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 37.0% |
| Bouncing_Counting | 98.33% / 100.00% / 95.00% | 100.00% / 100.00% / 96.67% | 24.0% |
| Directional_Event_Counting(异常项) | 25.00% / 18.33% / 21.67% | 23.33% / 18.33% / 23.33% | 32.2% |
| Acceleration_Identification | 98.33% / 100.00% / 96.67% | 98.33% / 100.00% / 93.33% | 39.5% |
| Event_Sequence | 100.00% / 100.00% / 100.00% | 100.00% / 100.00% / 100.00% | 87.3%(按group) |

### 4.2 Probe(最优层) vs 直答准确率——核心发现

| 原语 | 9B probe | 9B直答 | 35B-A3B probe | 35B-A3B直答 |
|---|---|---|---|---|
| Complex_Direction_Identification | 96.67% | 99.50% | 90.00% | 100.00% |
| Event_Sequence | 100.00% | 99.50% | 100.00% | 100.00% |
| **Rotation_Direction** | **95.00%** | **52.00%** | **96.67%** | **52.00%** |
| **Rotation_Count** | **100.00%** | **67.00%** | **100.00%** | **69.00%** |
| **Bouncing_Counting** | **100.00%** | **40.50%** | **100.00%** | **29.00%** |
| **Acceleration_Identification** | **100.00%** | **60.00%** | **100.00%** | **64.00%** |
| Directional_Event_Counting(异常,probe低但直答反而高) | 25.00% | 71.50% | 23.33% | 82.50% |

**这是全项目最核心的诊断发现**:4类原语(粗体)probe能以95-100%读出正确答案,但模型直答时只有29-69%——"读出层gap":信息在,读不出来。误差语义(`probe-experiment-report.md`补充)：Rotation_Direction错误100%是顺/逆时针搞反;Rotation_Count错误67%(9B)/77%(35B)是"差1";Acceleration_Identification错误集中在"加速→误判为匀速";Bouncing_Counting错误66-85%是"差距>1",系统性偏向数多。

### 4.3 回归probe R²(连续物理量,shallow/mid/deep)

| 原语 | 物理量 | 9B (shallow/mid/deep) | 35B-A3B (shallow/mid/deep) |
|---|---|---|---|
| Rotation_Count | 累积转角 | 0.997 / 0.992 / 0.986 | 0.992 / 0.983 / 0.964 |
| Bouncing_Counting | 速度矢量 | 0.886 / 0.885 / 0.709 | 0.812 / 0.862 / 0.686 |
| Acceleration_Identification | 速度标量 | 0.780 / 0.915 / 0.875 | 0.361 / 0.692 / 0.599 |
| Complex_Direction_Identification | 速度矢量 | 0.817 / 0.792 / 0.678 | 0.736 / 0.700 / 0.381 |
| Rotation_Direction | 累积转角 | 0.715 / 0.587 / 0.645 | 0.585 / 0.552 / 0.262 |
| Directional_Event_Counting(异常) | 速度矢量 | 0.311 / 0.400 / 0.303 | -0.263 / 0.058 / -0.262 |

位置混淆排查(Rotation_Count):位置-only基线R²=0.782(约占0.997里的78%),去除位置后残差R²仍有0.85-0.98,说明不是纯粹靠位置作弊。Rotation_Direction位置-only基线R²≈-0.001,确认干净。

---

## 5. Distillation-readiness三项实验

来源:`distillation-readiness-report.md`

### 5.1 实验0:打开思考(thinking-on)直答重测,9B,120样本(4原语×30)

| 原语 | 直答(关闭思考) | thinking-on(1024预算) | thinking-on(4096预算) | hit_cap(1024) | hit_cap(4096) |
|---|---|---|---|---|---|
| Rotation_Direction | 52.00% | 10.00% | 20.00% | 96.7% | 96.7% |
| Rotation_Count | 67.00% | 16.67% | 10.00% | 100.0% | 96.7% |
| Bouncing_Counting | 40.50% | 13.33% | 30.00% | 96.7% | 90.0% |
| Acceleration_Identification | 60.00% | 33.33% | 46.67% | 66.7% | 26.7% |

即使把token预算从1024拉到4096,4类原语依然全部低于直答(关闭思考)基线,降幅13.3-57.0pp,且3/4原语撞满预算比例仍高达90-96.7%(唯Acceleration_Identification降到26.7%)。**结论**:打开思考对这4类原语没有免费收益,反而普遍有害。

### 5.2 实验1:合成↔真实benchmark一致性

| Real子任务(MVBench/TOMATO) | 9B | teacher(35B-A3B) | teacher优势 | 对应合成原语teacher优势 | 方向一致? |
|---|---|---|---|---|---|
| MVBench action_localization | 57.50% | 59.00% | +1.50pp | Complex_Direction_Identification: +0.50pp | ✅ |
| MVBench moving_direction | 75.00% | 76.00% | +1.00pp | 同上 | ✅ |
| MVBench action_count | 59.00% | 68.00% | +9.00pp | Bouncing_Counting: **-11.50pp** | ❌ |
| MVBench moving_count | 66.50% | 81.00% | +14.50pp | 同上 | ❌ |
| MVBench moving_attribute | 86.00% | 91.50% | +5.50pp | (无对应) | — |
| TOMATO direction | 48.39% | 57.32% | +8.93pp | Complex_Direction_Identification | ❌ |
| TOMATO shape&trend | 39.46% | 39.91% | +0.45pp | — | — |
| TOMATO rotation(对照) | 26.92% | 22.38% | -4.54pp(teacher更差) | Rotation_Direction: 0.00pp | ✅ |
| Acceleration_Identification | — | — | — | +4.00pp(无真实数据直接对应,探索性) | N/A |

**核心矛盾**:计数类(MVBench action_count/moving_count)真实数据显示teacher明显更强(+9~14.5pp),但对应的合成原语Bouncing_Counting显示teacher反而更差(-11.5pp)——方向相反,未解决,是novelty.md标记的"待验证矛盾"(§7.1)。

### 5.3 实验2:PSD遮挡试点

**probe(遮挡后核心阶段,200样本)**:9B 5.00%-6.67%,35B-A3B 3.33%-6.67%(均低于多数类基线约10-12%)。遮挡前(sanity check)两模型均100%。

**直答准确率(截断bug修复前后对比)**:

| 模型 | n | 修正前(32 token) | 修正后(1024 token) | unresolved修正前 | unresolved修正后 | hit_cap(1024) |
|---|---|---|---|---|---|---|
| 9B | 200 | 4.00% | **15.50%** | 78.5% | 22.0% | 27.5% |
| 35B-A3B | 200 | 0.00% | **8.50%** | 60.5% | 9.5% | 20.5% |

4选1MCQ随机基线约25%,修正后两模型仍低于随机基线——加强了"PSD(遮挡)条件下模型确实丢失了关键信息,不是token预算不够导致的假阴性"这一结论。

---

## 6. 因果验证(消融)实验

来源:`causal-verification-report.md`(仅测8个full_attention层,24个linear_attention层未测)

### 6.1 唯一干净信号:Acceleration_Identification

| 层(整层消融) | mean消融delta | zero消融delta | 方向一致? |
|---|---|---|---|
| 层7 | 32.0pp | 30.0pp | ✅ |
| 层15 | 32.0pp | 26.0pp | ✅ |
| 层19 | 20.0pp | 12.0pp | ✅ |
| 层11 | 6.0pp | 32.0pp | ✅(方向一致,幅度差较大) |

### 6.2 其余3类原语:无干净信号

- **Rotation_Count**:8层里6层负delta或~0,层19甚至消融后准确率反涨16pp;mean/zero交叉验证经常反向(层19:mean=-16.0 vs zero=+2.0;层7:mean=-8.0 vs zero=+10.0)。
- **Bouncing_Counting**:仅整层级别方向一致(层3/7/11),head级别经常翻转(层3 head13:mean=12.0 vs zero=6.0)。
- **Rotation_Direction**:所有delta都很小(多在0-12pp),mean/zero经常方向不一致(层11整层:mean=2.0 vs zero=4.0)。

### 6.3 head级别可靠性

n=50下,15个共同(layer,head)组合里mean/zero方向一致率约60%(9/15),幅度经常差异很大(层11 head3 Acceleration_Identification:mean=18.0 vs zero=10.0;层3 head13同任务:mean=2.0 vs zero=14.0,几乎反过来)。**head级别结论目前不可靠,只能按整层粒度设计监督**。

样本量:第一轮n=15(60样本,136配置,因整层-vs-head加总不一致太大而未采信)→第二轮n=50(mean消融,采信)→第三轮zero交叉验证(仅覆盖第二轮筛出的约15个候选配置,非全部136个)。

---

## 7. 已知修正/存疑项(如实列出,不隐藏)

1. **Bouncing_Counting直答准确率**:曾误报9B=46.50%/35B=34.50%(旧run留存数字),已于2026-07-14核实修正为9B=40.50%/35B=29.00%(`probe-experiment-verification.md`有完整mtime证据链)。本文档全部采用修正后数值。
2. **PSD直答准确率**:32-token截断bug导致的原始数字9B=4.00%/35B=0.00%,修复为1024-token后的9B=15.50%/35B=8.50%。本文档表格已列出修正前后两组数字,供对比。
3. **因果验证第一轮(n=15)已弃用**:整层消融与16个head加总差异过大(如层19 Rotation_Count:整层delta=0.0pp vs 16head加总=66.7pp),判定为噪声,未用于任何结论,已用n=50重跑。
4. **TOMATO 9B总分小出入**:`probe-experiment-report.md`/`ppt-outline.md`一致使用36.19%,但`probe-experiment-verification.md`第176行出现过一次36.25%——两个源文档均未标注/解决这个小差异,本文档统一采用36.19%(出现频率更高、且是多处交叉引用的版本)。
5. **R²=0.997引用出处小误差**:`opsd-r-implementation-plan.md`把Rotation_Count的R²=0.997归因于"novelty.md §2.3.1",但该数字实际只出现在`probe-experiment-report.md`§4,novelty.md §2.3.1只有数学推导(Barber-Agakov界),没有这个具体数字——引用文件写错了,数字本身没有问题。
6. **novelty.md未包含独立的P1/P2四模型汇总表**:该文件只有零散的prose引用(如"Qwen3.6-35B-A3B:MVBench 74.3、Video-MME 70.9 四模型最强"),完整的四模型×三benchmark表未在novelty.md里单独成表,散落在ppt-outline.md和probe-experiment-report.md里,本文档已按benchmark汇总在§1/§2。

---

## 8. 模型架构背景(供解读上述数字参考)

- **student Qwen3.5-9B**:dense,32层混合attention(8层`full_attention`标准GQA + 24层`linear_attention`/Gated DeltaNet),hidden_size=4096。
- **teacher候选 Qwen3.5-35B-A3B / Qwen3.6-35B-A3B**:均为MoE(`Qwen3_5MoeForConditionalGeneration`,256专家/8激活/token,"A3B"=约3B激活参数),text hidden_size=2048,40层。
- **2026-07-21新增**:因OPRD类方法(隐藏状态对齐蒸馏)需要teacher/student架构可比,已开始下载dense的**Qwen3.5-27B**(`Qwen3_5ForConditionalGeneration`,与9B同一模型类、同一大版本号,非MoE)作为更匹配的teacher候选——与9B同属3.5系列,比`novelty.md:545`原本预留讨论的Qwen3.6-27B(同一模型类但大版本不同)更贴合"同源约束"。下载完成后需要补跑本文档§1/§2同等的P1/P2评测才能纳入比较。附带下载了Qwen3.6-27B(52GB,已完整下载,架构同为dense,留作备用候选,未删除)。

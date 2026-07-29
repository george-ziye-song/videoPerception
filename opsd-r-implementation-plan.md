---
title: "OPSD-R工程实现计划——第一个真正投入训练的idea"
status: "计划,交给负责GPU/训练的会话执行;本文档不监控进程,只定协议"
date: "2026-07-19"
related:
  - "oprd-ral-informed-redesign.md §2.1(OPSD-R的原始设计和数学论证)"
  - "probe-experiment-report.md(probe/直答两条基线数字的来源)"
  - "probe_data/extract_hidden_states.py、probe_data/train_probes.py(直接复用的代码和数据)"
---

# OPSD-R工程实现计划

## 0. 为什么是这个idea先做,不是CAD——把决策依据摆在最前面

`oprd-ral-informed-redesign.md`定义了5个idea,每个都有自己的"门槛"(要满足什么前提才能投入)。逐个过一遍,只有OPSD-R现在**零门槛**:

| idea | 需要teacher有优势? | 需要attention因果落点验证通过? | 现在能不能做 |
|---|---|---|---|
| CAD | 需要,**4类原语目前都没过**(3类打平、1类更差) | 需要,只有2/4原语过(`causal-verification-report.md`) | 不能 |
| PGA | 不需要 | 需要,同上只有2/4原语过 | 只能在2/4原语上做 |
| ARD | 需要,同CAD,没过 | 不需要 | 不能 |
| AP-RLVR | 不需要 | 需要,同CAD | 只能在2/4原语上做,而且是RLVR |
| **OPSD-R** | **不需要**(自蒸馏,不涉及teacher) | **不需要**(监督整个hidden state,不针对具体attention机制) | **能,4类原语全部能做** |

**结论**:OPSD-R是唯一一个不需要等任何前置验证结果、现在就能对全部4类"读出层gap"原语(Rotation_Direction、Rotation_Count、Bouncing_Counting、Acceleration_Identification)投入的方案,而且是这几个里工程最简单、收敛证明最干净的。这是这份计划要落地的对象。

**还有一件事要说清楚**:到目前为止,项目里所有的实验(probe、直答baseline、乱序测试、消融)**都是在冻结/零样本(zero-shot)模型上做的诊断**,从来没有真正训练过。**OPSD-R是这个项目第一次要跑真正的训练**,所以这份计划比之前的诊断类文档(probe-experiment.md、causal-verification-plan.md)多一层——不仅要说"怎么测",还要说"怎么训、训多久、怎么防止训坏"。

---

## 1. 目标与假设

**要验证的假设**:student(Qwen3.5-9B)在这4类原语上,"看视频编码阶段"的hidden state(记为$h^{enc}$)已经验证能被线性probe以85-100%准确率读出正确答案(`probe-experiment-report.md`§2);但"生成答案那一刻"的hidden state(记为$h^{gen}$)导致的直答准确率只有35-69%。**如果直接用一个辅助loss,把$h^{gen}$往$h^{enc}$方向拉,直答准确率会不会明显提升,同时不破坏其它任务的表现。**

**这不是在训一个新能力,是在修一个已经诊断出来的具体断点**——$h^{enc}$不是理论上的东西,是`probe_data/hidden_states/9b/*.pt`里已经存在的、已经验证过的真实数据。

---

## 2. 复用清单(这一步不写在这里,后面才不会漏)

| 需要的东西 | 从哪来 | 要不要重新算 |
|---|---|---|
| $h^{enc}$(编码阶段hidden state) | `probe_data/hidden_states/9b/{task}.pt`,每条record有`shallow`/`mid`/`deep`三层,形状`(T, 4096)` | **不需要**,2026-07-12已经抽好,直接复用 |
| 每个原语该用哪一层 | `probe_data/probe_results_9b.json` | 不需要重算,已有数据,见§3.1的表 |
| 训练数据(视频+问题+真值答案) | `probe_data/Atomic/sft.jsonl` + `Atomic2/sft.jsonl` | 不需要 |
| 70/30切分 | 需要**新固定**一个seed,和`train_probes.py`的`random_state=0`保持一致,避免训练集和之前probe/直答baseline的测试集混在一起 | 需要写,逻辑照抄`train_probes.py`的`train_test_split(...,random_state=0)` |
| prompt构造 | `probe_data/direct_answer_baseline.py`的message构造代码,逐行复用 | 不需要改 |
| 答案解析 | `lmms_eval.tasks._task_utils.mcq_extract.extract_mcq_answer`(共享函数) | 不需要改 |

---

## 3. 具体设计选择——每一条都给"为什么"和"证据",不是拍脑袋

### 3.1 $h^{enc}$用哪一层、哪种池化——按已有数据挑,不是猜

`probe_results_9b.json`里每个原语在shallow(层8)/mid(层16)/deep(层29)三层的分类probe准确率(实测数字,2026-07-19从文件里现查的):

| 原语 | shallow | mid | deep | 选哪层 |
|---|---|---|---|---|
| Rotation_Direction | **95.0%** | 93.3% | 88.3% | **shallow**(明显更好) |
| Rotation_Count | 100% | 100% | 100% | 三层打平,选**mid**(理由见下) |
| Bouncing_Counting | 98.3% | **100%** | 95.0% | **mid** |
| Acceleration_Identification | 98.3% | **100%** | 96.7% | **mid** |

**Rotation_Count三层打平时为什么选mid,不是shallow**:①4个原语里3个(Bouncing_Counting、Acceleration_Identification,Rotation_Count打平)都是mid最优或并列最优,选mid能让4类原语里3类用同一层,减少工程复杂度(不用为每个原语单独配置);②`train_regression_probes.py`的回归probe(物理量,连续值,比分类probe更精细)里,Rotation_Count在mid层的R²也是这几层里较高的一档(`novelty.md`已经记录过Rotation_Count的头条R²=0.997,该数字来自`train_regression_probes.py`默认扫的三层里表现最好的一层,和分类probe选层逻辑一致)。**只有Rotation_Direction例外,单独配置用shallow**,不能为了省事强行让4个原语共用一层——shallow(95.0%)和mid(93.3%)差距虽然不大,但deep(88.3%)明显更差,说明这个原语的信息在浅层更集中,不该忽略这个信号。

**池化方式:均值池化,不是别的**——因为`train_probes.py`训分类probe时用的就是`r[layer_name].mean(dim=0)`(§4.2的"85-100%"这个数字,是在均值池化后的向量上测出来的)。**如果OPSD-R的目标换成别的池化方式(比如只取最后一个temporal group),就不能再引用"85-100%可靠"这个证据了**——目标必须和已验证的东西严格一致,这是这个idea能不需要teacher、直接引用已有结论的前提,不能偷换。

### 3.2 $h^{gen}$(生成阶段的hidden state)——不需要真的调用generate(),重要的工程简化

**关键发现,直接决定了实现复杂度**:`probe_data/direct_answer_9b.json`里的真实生成结果,4类gap原语的response几乎都是"字母+换行"(比如`'C\n'`,之前核实ARD/CAD设计时已经现场验证过)——**答案本质上是1个token**。这意味着:决定这个答案的,就是"喂完整段prompt(视频+问题+`The best answer is:\n  `这段结尾)之后,最后一个位置在forward时的hidden state"——**这个量,在一次forward(prefill)里就能拿到,不需要真的调用`model.generate()`做自回归采样**。

具体来说:`extract_hidden_states.py`本来就已经对完整prompt做了一次`model(**inputs, output_hidden_states=True, use_cache=False)`的forward(这是抽$h^{enc}$用的那次)——**这同一次forward里,最后一个位置(`inputs["input_ids"].shape[1]-1`)的hidden state,就是$h^{gen}$**,不需要另开一次forward,更不需要采样。这比字面照搬OPRD原论文"在student自己采样的rollout上跑teacher forward"要轻量得多——因为我们的答案只有1个token,"生成"这个动作本身几乎不携带额外信息,决定性的计算全部发生在prefill这一步。

**$\ell_{gen}$选哪一层**:建议和$\ell_{enc}$**用同一层**(即§3.1表里选定的那一层),不引入第二个自由的层选择超参数——这样比较的是"同一个模型、同一深度的处理,在两个不同位置(视频token vs 最后一个文本位置)上的差异",不多引入一个可以调的旋钮,减少过拟合到某个巧合层组合的风险。**如果验证后发现效果不好,再单独扫$\ell_{gen}\ne\ell_{enc}$的组合**,不作为默认设计。

### 3.3 投影$P_\theta$——要不要

$\ell_{enc}$和$\ell_{gen}$都在同一个模型(student自己)上,hidden_dim天然相同($d=4096$),**不存在OPRD原文那种student/teacher维度不匹配的问题,严格意义上不需要投影**。但FitNets(Romero et al. 2015)的经验是:就算维度相同,"编码阶段的表示"和"决策阶段该有的表示"可能天然扮演不同角色(决策阶段除了"看到了什么"还要额外编码"我现在要产出哪个token"这类信息),强行要求两者的**原始向量**相等可能过度约束、把决策阶段必要的额外计算也一起压掉。

**建议**:默认加一个**轻量的可训练线性层**$P_\theta:\mathbb R^{4096}\to\mathbb R^{4096}$(初始化成单位矩阵+小扰动,不是随机初始化,这样训练开始时$P_\theta\approx\text{id}$,不会一上来就大幅扰动),给模型一点调整空间,不强行要求两个向量完全重合。**同时把"不加投影(纯恒等)"作为一个消融对照组**,两者都跑,看$P_\theta$有没有必要——如果去掉投影效果差不多甚至更好,后续版本可以把这个部件去掉,简化实现。

### 3.4 完整损失和训练目标

$$L_{\text{OPSD-R}}=\frac1d\big\|P_\theta(h^{gen}_{\ell})-\text{sg}(\bar h^{enc}_{\ell})\big\|_2^2,\qquad \bar h^{enc}_{\ell}=\frac1T\sum_i h^{enc}_{\ell,i}$$

$\ell$按§3.1表选(Rotation_Direction用shallow,其余3类用mid)。**主任务loss**:标准的SFT交叉熵,预测真值答案字母(teacher-forcing,输入是完整prompt,目标是真值字母token)。**总loss**:

$$L_{\text{total}}=L_{\text{CE}}+\beta\cdot L_{\text{OPSD-R}}$$

$\beta$按OPRD原论文的$\mu$-sweep方法学,扫$\beta\in\{0, 0.1, 1, 10\}$——**$\beta=0$这一档就是"纯SFT,不加OPSD-R"这个必须要有的对照组**(见§5.2,这是这次实验最重要的一个对照,不能省)。

---

## 4. 训练protocol

### 4.1 数据切分

每个原语200条,按`train_probes.py`一致的`train_test_split(test_size=0.3, random_state=0)`切成140训练/60测试——**必须用同一个random_state**,这样训练用的140条和之前probe/直答baseline报告里的测试集样本不重合的部分能对齐,验证时可以直接和`probe-experiment-report.md`已经报告过的35-69%这个基线数字比较,不需要重新定义一套新的评测集。

### 4.2 是否需要LoRA

Qwen3.5-9B是稠密9B模型,项目`plan.md`里已经有过资源评估(CLAUDE.md引用):"student off-policy SFT:3×A100-40G(舒适)或4×4090-24G(LoRA+grad-ckpt+ZeRO-3,偏紧)"——**这次沿用同一个结论**:如果用4090(24G)跑,建议LoRA(不做全参数微调)+梯度检查点+ZeRO-3;如果能申请到A100(40G),可以考虑全参数微调,但这不是必须的,LoRA对这种"修一个具体断点"的小规模干预大概率够用,不需要一上来就上全参数微调的成本。

### 4.3 超参数(建议起点,不是最终值)

- LoRA rank=16-32,target modules覆盖attention的q/k/v/o和MLP的gate/up/down(标准LoRA配置)
- 学习率:1e-5到5e-5(LoRA常用范围),cosine decay,warmup 3%
- 每个原语单独训(不要把4类原语混在一起训一个模型),因为§3.1已经确认不同原语用的层不一样,混训会让$\ell$的选择失去意义
- epoch数:**给的数据只有140条/类,容易过拟合**,建议3-5个epoch,每个epoch结束在60条测试集上评一次直答准确率,**准确率开始下降或者训练集loss远低于测试集loss时就停**,不要死板跑满固定epoch数
- batch size:参考`extract_hidden_states.py`/`direct_answer_baseline.py`已验证过的单样本处理速度,结合4-8卡数据并行,batch size 4-8/GPU起步

---

## 5. 评估protocol

### 5.1 主指标

每个原语,60条测试集上的直答准确率(用共享的`extract_mcq_answer`,和`probe-experiment-report.md`§3同一套评测代码),对比:
- $\beta=0$(纯SFT基线)
- $\beta\in\{0.1,1,10\}$(OPSD-R各档)
- 训练前(当前的35-69%,直接引用`probe-experiment-report.md`已有数字,不需要重测)

### 5.2 §5.2这个对照组必须做,解释为什么

**如果只对比"训练前35-69%" vs "训练后OPSD-R的准确率",没法说清楚提升是OPSD-R这个辅助loss带来的,还是"随便在这140条上做SFT,模型见过这类题就会了"带来的**——毕竟到目前为止模型从来没在这类合成数据上训练过,哪怕不加任何特殊设计的纯SFT都可能有提升。**$\beta=0$这个纯SFT对照组,就是用来把这两种可能性分开的**:如果$\beta=0$已经能把准确率拉到很高,说明问题主要是"没见过这类数据",不是"读出层用歪了"这个我们诊断出来的具体机制,OPSD-R这个辅助loss的贡献需要重新评估;如果$\beta=0$提升有限、$\beta>0$才明显更好,才能说明OPSD-R这个辅助对齐机制确实在起作用,不是白设计的。

### 5.3 回归检查(不能只看gap原语,还要看别的原语有没有被破坏)

同样的训练协议(哪怕只是$\beta=0$的SFT),额外跑一遍在Complex_Direction_Identification、Event_Sequence这两类"probe-experiment-report.md里已经是96-100%、没有gap"的原语上——**确认训练没有把本来就好的能力训坏**。Directional_Event_Counting(probe低、直答反而高的反常项)建议**不纳入OPSD-R训练**(因为它的$h^{enc}$本身probe读不出来,拿一个不可靠的目标去对齐没有意义,§2.1的诚实边界已经提过),但可以作为一个"没训练过、纯粹看会不会被连带影响"的旁观测试项。

### 5.4 内部诊断指标(照抄OPRD原论文的"内部视角"检查,§4.3)

训练过程中记录$\|P_\theta(h^{gen}_\ell)-\bar h^{enc}_\ell\|$(或者余弦相似度)随step的变化——**这条曲线应该单调下降/单调上升(相似度)**,如果不是,说明这个loss没有被正常端到端优化,训练配置(学习率、$\beta$取值)需要重新调,不能只看最终准确率而忽略这个中间诊断信号,这正是OPRD论文自己验证"loss确实在被优化"用的同一招(原文Figure 5)。

---

## 6. 风险与应对(如实列出,不回避)

| 风险 | 应对 |
|---|---|
| 140条/类的训练数据太少,过拟合 | 提前设好早停;如果效果不稳定,评估是否用SynRL生成器再生成更多同分布数据(01/02号脚本理论上可以生成任意数量,不受200条这个数字限制,只是之前probe实验固定用了200条) |
| $\beta$太大,把主任务(答对题)的梯度压制掉 | 按OPRD的$\mu$-sweep方法学,从小到大扫,观察主任务loss有没有被拖慢收敛 |
| $\ell_{gen}=\ell_{enc}$这个"共用一层"的简化假设不成立 | 如果4档$\beta$都没有明显提升,下一步应该做的消融是解耦$\ell_{gen}$和$\ell_{enc}$分别扫,而不是直接判定OPSD-R无效 |
| "编码阶段的表示"本身就有第4章提过的边界问题(比如旋转类的position confound) | Rotation_Count是三层打平选的mid,如果实测发现效果不如预期,要重新检查是不是选层时应该更倾向shallow(和Rotation_Direction保持一致),而不是默认"多数服从"这个理由 |
| 只在9B(student)上做,没有同步验证teacher | 这是刻意的范围缩小(OPSD-R本来就是自蒸馏,不需要teacher),如果student结果为正,后续可以复制同一套协议到35B-A3B上做交叉验证,但不是这一轮必须做的 |

---

## 7. 交付物清单

1. 4个原语(或3个,如果决定排除Directional_Event_Counting)× 4档$\beta$的直答准确率表,和训练前基线（35-69%区间）对比
2. §5.3的回归检查结果(Complex_Direction_Identification/Event_Sequence训练前后对比)
3. §5.4的内部诊断曲线(相似度/距离随训练step变化)
4. 如实记录:哪个原语提升明显、哪个没有、$\beta$最优值是多少、$P_\theta$投影有没有必要(§3.3消融结果)

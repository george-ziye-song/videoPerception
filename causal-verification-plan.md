---
title: "因果验证实验计划——TRD/CAD该监督哪几层哪几个head,先用ablation查清楚再训练"
status: "计划,交给负责GPU/训练的会话执行,本文档本身不监控进程"
date: "2026-07-14"
related:
  - "novelty.md §2.1(TRD)、§2.5(CAD/RAD)、§2.1.2(ARD)"
  - "probe-experiment-report.md(§4.1的hidden state抽取协议,本计划复用同一套输入构造代码)"
---

# 因果验证实验计划:TRD/CAD该监督哪几层哪几个head

## 0. 为什么做这个实验,解决什么问题

novelty.md §2.1/§2.5讨论TRD/CAD(蒸teacher的attention分布)时,一直有一条没解决的软肋:**attention权重≠信息被因果使用**——一层attention算出来的权重高,不代表模型最终答案真的依赖这个位置的信息(残差连接、MLP、多层间接传递,都可能让信息绕开我们盯着的这一层attention)。这条软肋不是猜的,§1.2从文档最早的版本就写了("信息可走value/MLP路径"),§2.5又用Qwen3.5的真实代码结构确认了一遍(每层都是attention→残差→MLP→残差,包括最后一层)。

如果真要投入TRD/CAD这类"监督某一层/某几个head的attention"的训练,**在写训练代码之前,应该先低成本地查清楚:这几类任务(Rotation_Direction/Rotation_Count/Bouncing_Counting/Acceleration_Identification)上,模型的最终答案到底因果依赖哪几层、哪几个head**,而不是凭"25%/50%/90%深度"这种和probe实验借来的惯例瞎选——这个惯例是给"抽hidden state"用的,不是给"该监督哪层attention"用的,§2.5已经发现这套深度比例换算出来的层(第8/16/29层)在Qwen3.5-9B上**全部是linear_attention层,不是标准softmax attention**,直接照搬会选错。

**这个实验要回答的问题**:对Qwen3.5-9B(student)在这4类"probe读得出、直答答不对"的原语上,把每一个`full_attention`层(Qwen3.5-9B共32层,只有第4/8/12/16/20/24/28/32层——0-indexed为3/7/11/15/19/23/27/31——是`full_attention`,其余24层是`linear_attention`)、以及该层每个attention head,分别做"临时抹掉"处理,看模型的直答准确率掉多少——掉得多的,说明这一层/这个head是**真正被用上的**,值得作为TRD/CAD监督的目标;掉得少或不掉,说明监督它即使训练收敛了也没有意义,因为模型压根没依赖它。

**这不是训练实验,是诊断实验**,和probe-experiment.md的定位一样——回答"该不该投入、投入在哪",不产出任何模型权重。

## 1. 方法:Activation Patching / 消融(Ablation)

**核心技术**:对选定的某一层某个head(或整层全部head),在forward过程中把该head的attention输出替换成一个"中性"值,再看最终直答准确率相对baseline(不做任何替换)掉了多少。这是可解释性研究里的标准做法(因果中介分析/activation patching),常见先例:

- Vig et al., NeurIPS 2020,*Investigating Gender Bias in Language Models Using Causal Mediation Analysis*——用activation patching定位模型内部哪些component对某个行为有因果贡献。
- Meng, Bau, Andonian & Belinkov, NeurIPS 2022,*Locating and Editing Factual Associations in GPT*(ROME)——用类似的因果追踪方法定位事实性知识存在模型的哪一层。
- Wang, Variengien, Conmy, Shlegeris & Steinhardt, ICLR 2023,*Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small*——展示了逐head的ablation如何定位一个具体任务背后的因果电路,方法论和这里要做的事几乎一样,只是换成了我们自己的4类合成原语任务。

**"中性值"怎么选,两种做法都做,交叉验证**:
1. **零消融(zero ablation)**:把该head的attention输出直接置零(该head对最终结果的贡献完全抹除,只剩其余head+残差路径)。实现最简单,但会引入一个"分布外"的激活值(模型训练时从没见过这个head输出恰好是0的情况),可能高估重要性。
2. **均值消融(mean ablation)**:把该head的attention输出替换成**这一批样本上该head输出的均值**(不是0,是"一个典型但不携带这条样本具体信息的值")。更贴近"这个head存在,但没有针对这条样本给出有意义信息"这个反事实,业界认为比零消融更干净,作为主要结论来源;零消融的结果作为交叉验证,两者方向一致才采信。

## 2. 具体步骤

### 2.1 目标层/head范围

Qwen3.5-9B(`num_hidden_layers=32`,`num_attention_heads=16`,`num_key_value_heads=4`,GQA分组)里`layer_types`确定是`full_attention`的层(0-indexed):**[3, 7, 11, 15, 19, 23, 27, 31]**,共8层。每层16个query head(4个KV head,每个KV head被4个query head共享——消融时按query head粒度做,不需要单独处理KV head)。**只在这8层上做,不碰24层linear_attention层**(那些层是Gated DeltaNet机制,概念上没有"某个位置的attention权重"这个东西,ablation的方式完全不同,不在本次范围内,如果8层的结果不足以解释清楚,再单独设计linear_attention层的消融方案,不在这次计划里)。

### 2.2 数据

复用`probe_data/`里已经生成好的4类"读出层gap"原语数据(`Rotation_Direction`/`Rotation_Count`/`Bouncing_Counting`/`Acceleration_Identification`),每类200条,**不需要新生成数据**。为控制总计算量,先用每类**50条**(4类共200条)做第一轮粗筛,如果某几层/head表现出明显信号(掉分幅度显著大于其它层),再对这几个候选目标补跑到全部200条做确认。

### 2.3 具体流程(伪代码级别,交给负责GPU的会话时展开成实际脚本)

```
baseline_acc = direct_answer_baseline(model, task_samples)  # 复用 direct_answer_baseline.py,不做任何消融,这是对照

for layer_idx in [3, 7, 11, 15, 19, 23, 27, 31]:
    for head_idx in range(16):
        # 注册一个forward hook,在self_attn模块内部,把该head对应的输出通道
        # (hidden_size切成16份,每份对应一个head)替换成:
        #   (a) 全零(零消融)
        #   (b) 这一批样本在该head上的输出均值(均值消融,先跑一遍拿到均值再第二遍替换)
        ablated_acc_zero = direct_answer_baseline(model_with_hook_zero, task_samples)
        ablated_acc_mean = direct_answer_baseline(model_with_hook_mean, task_samples)
        delta_zero = baseline_acc - ablated_acc_zero
        delta_mean = baseline_acc - ablated_acc_mean
        record(layer_idx, head_idx, delta_zero, delta_mean)

    # 同时做一次"整层消融"(该层全部16个head一起消融),作为head级别结果的交叉验证:
    # 如果16个head的delta加起来远小于整层消融的delta,说明head之间有非线性的协同效应,
    # 不能简单把head级别的重要性线性相加去解释整层的重要性,这一点要如实记录,不能假装线性可加。
```

### 2.4 输出

每个(layer_idx, head_idx)一个`delta_zero`和`delta_mean`,按`delta_mean`降序排列。**同时对4类原语分别排,不要合并成一个总分**——TRD/CAD的判定表(novelty.md §2.5.1)已经显示这4类原语teacher-student关系并不一致(Bouncing_Counting teacher更差,其余3类打平),不同原语的因果电路很可能也不完全一样,合并成一个数字会掩盖这种差异。

## 3. 判定逻辑(消融结果怎么用)

| 消融结果模式 | 解读 | 对TRD/CAD的意义 |
|---|---|---|
| 某几个(layer, head)消融后掉分明显(比如>10pp),其余大多数消融后几乎不掉分 | 找到了真正因果载荷重的稀疏子集 | 只监督这几个(layer, head)的attention分布,不要盲目监督所有8层16head(128个组合),省算力也更准 |
| 消融任何单一(layer, head)都掉分很少,但同时消融一整层(16个head)掉分明显 | 信息分散在同一层的多个head里冗余编码,单个head不关键,整层关键 | 监督对象应该是"整层pool后的attention分布"(TRD原始公式里"按head平均"那种做法),不是挑单个head |
| 消融任何单层/单head都几乎不掉分(哪怕整层消融) | 这几类任务的答案信息主要不是靠这8层full_attention层的attention机制拿到的(可能主要靠24层linear_attention,或者主要靠MLP) | **TRD/CAD在full_attention层上没有着力点**,如果连整层消融都不掉分,说明监督这些层的attention分布,就算训练收敛,也大概率对最终答案没有帮助——这时候应该回到ARD(§2.1.2,不碰attention,直接蒸最终答案分布)或者干脆放弃这条attention监督的路线 |
| 不同原语类型(比如Rotation_Direction vs Bouncing_Counting)因果依赖的层/head完全不一样 | 这4类"读出层gap"原语的底层机制本来就不是同一回事 | 不能设计一个"通用"的TRD/CAD配置套用到所有4类原语上,需要分别调 |

## 4. 和已有工作的衔接

- 这个实验的输出(哪几层/head该监督)是novelty.md §2.5(CAD)、§2.1(TRD)**投入训练之前的前置条件**,不是可选项——§2.5.1结论四已经写了"KL收敛但最终答案没提升,说明信息绕道走了",这个实验就是**提前**把"会不会绕道"这件事查清楚,而不是等训练跑完了才发现选错层。
- 和probe-experiment.md的关系:probe-experiment测的是"hidden state里有没有信息"(表示层诊断);这个实验测的是"模型生成答案时,因果依赖哪个attention机制"(读出层诊断)——两者是互补的两层诊断,不是重复。
- 复用的代码:`direct_answer_baseline.py`的prompt构造和评分逻辑原样复用,只需要在模型forward时插入hook,不需要重新设计数据/评测协议。

## 5. 局限(如实记录,不回避)

- 消融是"必要性"检验(拿掉这个东西答案变差),不完全等同于"这个东西携带的信息内容是什么"——某个head消融后掉分明显,只说明模型依赖它,不直接告诉我们它具体在传递"哪几帧""什么信息",这一层解释仍然需要结合attention权重本身去看(比如掉分明显的head,它的attention权重具体落在哪些视觉位置上),消融实验和"看attention权重"这两件事要结合着看,不是互相替代。
- 零消融/均值消融给出的排序如果明显不一致,说明消融方式本身引入了较大的分布外效应,这时候排序结果的可信度要打折扣,需要如实报告,不能挑对自己有利的那个消融方式的结果。
- 这次只测了8层full_attention层,如果结果显示"哪层都不关键",不代表任务真的和attention无关——有可能关键机制在24层linear_attention里,这需要另外设计针对Gated DeltaNet机制的消融方法,不在这次范围内,需要另外立项。

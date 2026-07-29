---
title: 真实benchmark新发现gap任务——representation层还是readout层？诊断实验计划
date: 2026-07-21
status: 计划，尚未执行（本会话不跑GPU实验，交给专门的训练/评估会话）
---

# 新发现的真实gap任务：是"读不出"还是"根本不在"？

> 背景：这次重新核实了 MVBench 全 20 任务 + TOMATO 全 6 类别 + VideoMME 12 个 task_category 的 teacher/student 真实准确率（数据来自 `probe-experiment-report.md` §1 和本次现算的 VideoMME 分解），筛出一批之前完全没在我们 4 个 gap primitive 名单里的、teacher 明显更强的真实任务（object_shuffle +11.0pp、episodic_reasoning +10.5pp、fine_grained_pose +9.0pp 等，完整名单见对话记录）。
>
> 用户要求：这些任务如果找不到对应真实训练视频，也必须先搞清楚 gap 出在哪一层——是像原来 4 个 gap primitive 那样"信息在 hidden state 里、只是没读出来"（readout gap，训练成本低，加个读出头/RLVR 就可能解决），还是像 PSD 的 shell-game 发现那样"信息在 hidden state 里就已经没了"（representation gap，需要更根本的训练干预）。这份文档就是那个诊断计划。

## 0. 核心方法论障碍（必须先解决，否则整个计划做不了）

`probe-experiment-verification.md` §1.1 已经把这件事讲得很清楚，直接决定了这份计划怎么设计：

> "这个probe要训出来,前提是每条样本得有一个**独立于MCQ答案字母**的、精确的真值...这种真值只有SynRL生成器自己造视频的时候才知道...P1测的MVBench/TOMATO/VideoMME是真实世界视频,只有'这道题标准答案是哪个字母'这一种标注,没有'这个动作背后精确的物理参数是什么'这种标注。没有这个真值,probe根本没法训。"

原来 4 个 gap primitive 能训 probe，是因为 SynRL 生成器画视频的时候就知道"这个球到底转了几圈""方向是顺时针还是逆时针"——这个真值不依赖 MCQ 选项字母，能直接拿来当 probe 的分类标签。**真实的 MVBench/TOMATO/VideoMME 视频没有这种东西**，只有"正确答案是C"这一条标注。如果直接拿"正确答案的字母"当 probe 标签去训，训出来的东西没有意义（字母是随机打乱的，probe 学到的可能只是字母和某种视觉偏置的巧合关联，不是真的读出了任务相关信息）。

### 解决办法：追溯到 MVBench/TOMATO 每个子任务的原始源数据集，拿它的原生标注当真值

这正好是同时在跑的另一路调研（"给这些任务找真实训练视频"）要做的事——**如果一个真实任务的训练视频来源，是靠追溯到它在 MVBench/TOMATO 论文里引用的原始源数据集（比如 object_shuffle 可能来自 CLEVRER，CLEVRER 自己的标注就是"每个物体在每一帧的精确位置/颜色/形状"，不是MCQ字母），那么这个源数据集的原生标注，同时也能直接拿来当 probe 的真值标签**——不需要另外发明一套。这是这份计划最重要的设计决策：**probe 真值 = 该任务的 MVBench/TOMATO 溯源到的原始数据集的原生标注**，不是重新去标注。

**如果某个任务追溯不到一个有结构化标注的源数据集**（比如 VideoMME 的 "Information Synopsis" 这种开放式任务，源头就是 YouTube 视频+人工出题，没有什么"精确物理真值"可言）——这类任务用下面 §3 的备用方案（对比 probe，不是分类 probe）。

## 1. 复用已验证的抽取方法，只换输入数据

`probe_data/extract_hidden_states.py` 的方法本身不用改：
- 一次性 forward（`output_hidden_states=True`），不 `generate()`
- 抓 3 层（浅25%/中50%/深90%）
- `find_video_runs()` 做连续视频 token 检测，避免把时间戳文字 token 混进 pooling
- 帧数、`enable_thinking=False`、prompt 构造和 `direct_answer_baseline.py` 保持逐行一致（`probe-experiment-verification.md`已经验证过这套纪律的重要性——帧数、`max_new_tokens`、prompt 措辞不对齐,观察到的差异就会混进配置confound,不是真的readout gap）

**要新写的部分**：输入从 `sft.jsonl`（SynRL 生成）换成从源数据集原始标注 + 对应真实视频构造的 prompt。每个任务需要一个"该任务的样本 → (视频, 结构化真值标签)"的数据准备脚本，类似 `01_atomic_motion.py` 生成 `metadata.jsonl` 的角色，只是这次是从已有真实数据集提取，不是从头渲染。

**direct-answer 基线数字不需要重新跑**——已经有了：`eval/results_q35_9b`/`eval/results_q35_35b` 里每个任务的真实 baseline 准确率（本次汇总的 20+6+12 张表），这就是"模型自己答"这条线，probe 数字只需要和这个已有数字比较，不用重新测一遍直答。

## 2. 优先级：先做哪几个任务的诊断

不建议一次性对所有 15+ 个新发现任务都跑——按 gap 幅度从大到小、且**源数据集有结构化标注（能训 probe）**的任务优先：

| 优先级 | 任务 | teacher优势 | 前提：能否追溯到有结构化标注的源数据集 |
|---|---|---|---|
| 1 | MVBench object_shuffle | +11.0pp | 待另一路调研确认（怀疑CLEVRER，位置标注现成） |
| 2 | MVBench episodic_reasoning | +10.5pp | 大概率不行（需要TV剧情内容，源头是TVQA这类，标注多是人工QA，没有独立于答案的结构化真值） |
| 3 | MVBench fine_grained_pose | +9.0pp | 可能可以（如果源头是骨骼关键点类数据集，关键点坐标就是现成真值） |
| 4 | MVBench action_count | +9.0pp | 已经在做（Bouncing_Counting/Directional_Event_Counting路线，不用重新设计） |
| 5 | TOMATO velocity&frequency | +18.57pp（全表最大） | 待查（如果源头视频有可提取的速度/频率物理量，比如运动轨迹数据集，能提取；否则退到§3方案） |

**其余的**（unexpected_action、object_existence、moving_attribute、character_order等）等另一路"找训练视频"调研的结果回来后，看每个能不能追溯到结构化标注源，再决定要不要单独设计 probe。

## 3. 追溯不到结构化真值时的备用方案：对比式 probe，不是分类 probe

对 episodic_reasoning 这类开放式任务（大概率追溯不到独立于答案的结构化真值），提议一个不同的探针设计，诚实地说，这个比原来的分类 probe 弱、干净程度不如原方案，但至少能给出"信息在不在"的方向性判断：

1. 用模型自己的文本 embedding（或者一个独立的 text encoder）分别编码"正确答案"和"错误选项"的文本。
2. 训一个探针，输入是 hidden state（pooling 后），目标不是"分类到哪个类别"，而是"hidden state 和正确答案 embedding 的相似度，是否显著高于和错误选项 embedding 的相似度"（对比式，类似 CLIP 那种对比目标）。
3. 如果 hidden state 里确实"知道"正确答案相关的信息，即便模型最后生成时选错了，这个相似度差异应该是正的、显著的；如果 hidden state 里根本没有这个信息，相似度差异应该接近随机。

这个方案的局限（如实说）：不像原来的分类 probe 那样有一个干净的"70%训/30%测,准确率"数字，解读起来没那么直接，需要做统计检验（比如 paired t-test 看相似度差异是否显著大于0）而不是简单看accuracy。这是"能做，但证据强度弱于主方案"的备选，不是首选。

## 4. 输出解读框架（复用原来 TRD vs PSD 的判断标准）

对每个跑了诊断的任务：

- **probe 准确率明显高于直答准确率**（类似原来 4 类 gap primitive 的 85-100% vs 35-69%）→ **readout gap**，信息在，没读出来，适用 TRD 那条思路（读出头/轻量干预/RLVR，训练成本相对低）。
- **probe 准确率接近或低于 majority baseline，且接近/低于直答准确率**（类似 PSD 在 shell-game 上的发现：3.3-6.7% vs 直答8.5-15.5%,两条都远低于随机基线）→ **representation gap**，信息在模型看完视频那一刻就已经不在了，需要更根本的训练干预（不是加读出头能解决的）。
- 中间地带（probe 比直答高但没有原来 4 类primitive那么悬殊）→ 如实报告，不要强行归类到两档里的任何一档。

## 5. 待办

1. 等另一路"找真实训练视频"调研结果回来，确认 object_shuffle/episodic_reasoning/fine_grained_pose/velocity&frequency 等任务能不能追溯到有结构化标注的源数据集。
2. 能追溯到的，写数据准备脚本（源数据集标注 → probe 真值标签），复用 `extract_hidden_states.py`/`train_probes.py` 原有代码跑 probe。
3. 追溯不到的，考虑 §3 的对比式 probe，或者先不做诊断，只标注"暂时不知道是哪层的gap"，不勉强凑数字。
4. 这份计划本身不在本会话执行（本会话是"novelty brainstorm"分支，不跑GPU/监控训练评估进程）——需要交给专门跑训练/评估的会话执行。

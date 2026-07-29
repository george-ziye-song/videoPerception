---
title: 训练数据说明
date: 2026-07-19（首版）/ 2026-07-25（重写精简）/ 2026-07-27（按真实gap重新配比）
status: 数据采集 + pass-rate过滤 + 按gap质量重新配比，全部完成；尚未转换成训练代码能直接吃的messages格式
---

# 训练数据——现状 & 怎么用

> 这是给OPRD/OPSD-R准备的视频感知训练数据（不是plan.md里那套已作废的"合成CoT蒸馏"方案）。详细的调研/踩坑过程见 `training-data-source-survey.md`；这份文档只讲现在有什么、在哪、怎么用。

## 一句话现状

最终 **3,323条**"16帧图 + 选择题 + 答案"样本，专项:常规 = 70:30。条数比07-25版本(8,248条)少很多，但质量高得多——07-25版本里一大半"专项"数据其实是teacher和student都会做的陪跑样本，07-27按"kept里真正有效"这个指标重新筛过一遍，砍掉陪跑、补了新专项。

## 文件在哪，用哪个

都在 `/remote-home/ziyesong/videoPerception/`：

| 文件 | 条数 | 说明 |
|---|---|---|
| `FINAL_TRAINING_DATASET.json` | 10,880 | 过滤前的完整整合数据(含新增CLEVRER两个专项+PerceptionTest补充) |
| **`FINAL_TRAINING_DATASET_final.json`** | **3,323** | **训练用这个**——按真实gap重新配比后的最终版 |
| `eval/passrate_teacher_full.json` / `passrate_student_full.json` | 各10,880 | 逐条teacher/student pass-rate原始结果 |

字段：`id` / `source` / `layer`(专项\|常规) / `task` / `frames`(16张已抽好的帧图路径，**原视频已删除**) / `num_frames` / `question` / `options` / `answer` / `teacher_correct` / `student_correct` / `teacher_pred` / `student_pred`。

## 为什么从8,248条降到3,323条：gap-ratio方法论

07-25版本的过滤规则只有一条：`teacher_correct=True`。跑完后发现`unexpected_action`(3,065条kept)保留率84.4%看着很健康，但用户追问"student在这里正确率是不是也很高"后深挖发现：**这3,065条里只有14条是"teacher对、student错"的真正有效样本(0.5%)，剩下99.5%是teacher和student都答对的陪跑数据**——对representation distillation来说几乎没有信号，因为student的隐藏状态已经能得出正确答案了，没有"东西"可学。

所以现在每个task看两个数字：
- **teacher_correct过滤后的kept数**（原来唯一的标准）
- **kept里"teacher对、student错"的真gap样本占比**（新增的、真正决定训练价值的指标）

按这个指标重新处理：
- **gap占比很低的task**(unexpected_action 0.5%、Rotation_Count 6.6%、fine_grained_pose 9.1%)：降重——保留全部真gap样本 + 有限的陪跑样本(封顶=3倍真gap数，至少100条撑住覆盖面)，不再让陪跑样本占主体积。
- **gap占比健康的task**：全部保留，不裁。
- **常规层(broad_coverage)**：本来就不是奔着gap去的(防退化保险)，但既然student在这层也基本都会(gap仅1.3pp)，用户要求缩小占比，按来源比例缩到1,000条。

## 数据内容 & 每个task的真实gap质量

**专项层（2,323条）**：

| task | 条数 | kept里真gap占比 | 处理 | 来源 |
|---|---|---|---|---|
| **Bouncing_Counting** | 123 | 55.6% | 全留 | OVR/Ego4D(58) + Perception Test补充(165，含新挖的60条"last-set"追问，白捡的——同一批视频官方还有一道没用过的题) |
| **character_order** | 38 | 22.6% | 全留 | Perception Test（原来和object_shuffle合并成一个task，标签一压根本看不出这俩差很多，拆开后发现character_order单独gap+14.5pp、object_shuffle反而是负的） |
| **counterfactual_inference**(新) | 251 | 32.7% | 全留 | **CLEVRER**(ICLR 2020, MIT/DeepMind，MVBench官方自己也用它做这个task——虽是合成物理仿真数据，但社区认可度上比咱们自己的SynRL生成器高得多，所以选它不选SynRL) |
| Acceleration_Identification | 591 | 19.1% | 全留 | LLaVA-Video-178K关键词筛选 |
| **moving_direction**(新) | 317 | 24.9% | 全留 | CLEVRER（同上，方向标签用物体真实位移轨迹算出来的，坐标系映射拿MVBench已知200条答案校准过，52/52验证通过） |
| fine_grained_pose | 740(降重前2,037) | 9.1% | 降重 | NTU RGB+D |
| Rotation_Count | 112(降重前181) | 6.6% | 降重 | ActivityNet Captions窄子集 |
| unexpected_action | 114(降重前3,065) | 0.5% | 降重 | FunQA |
| object_shuffle | 37(降重前116) | 负gap | 降重 | Perception Test |

**常规层（1,000条，按来源比例从2,200缩量）**：ActivityNet academic(643) + STAR/Charades(297) + Perception Test(60)，来源构成不变，只是整体缩小。

**counterfactual_inference/moving_direction怎么来的**：MVBench自己这两个task用的CLEVRER视频，我们本地已经下载了915个（MVBench自己的200条eval只用了390个，390个之外还空闲525个)。从CLEVRER官方标注(`data.csail.mit.edu`直连下载，~116MB，不含在代理规则限制范围)里，counterfactual_inference直接复用CLEVRER原生的反事实推理问题，moving_direction从物体运动轨迹标注自己生成方向判断题——都用的是MVBench自己没碰过的525个视频，不存在和评测集重叠/泄漏的问题。

**还没做、以后可以补的**（评估过，这次没做，原因见括号）：
- Kinetics trampoline关键词还能挖到~132个新的Bouncing_Counting候选视频（需要重新走一遍yt-dlp下载+裁剪+质量过滤，工作量和这次做的CLEVRER相当，这次没做）
- LLaVA-Video Acceleration_Identification有12条当初下载失败但数据其实还在本地一个tar包里（需要先重建原始968候选的index映射关系，投入产出比低，这次跳过）

## Pass-rate过滤方法

teacher(Qwen3.5-9B)/student(Qwen3.5-4B)对全部数据直接作答，`enable_thinking=False`，只要求输出选项字母（和P1协议一致，不用CoT）：teacher最终正确率81.2%(10,880条)，student 79.3%。

⚠️**踩过的坑**：`PerceptionTest_object_shuffle_character_order`和`PerceptionTest_Bouncing_Counting_supplement`两个来源(共343条)的`options`/`answer`最初没有"A. "字母前缀——不是打分脚本的问题，是prompt本身没给模型看到字母标号却要求"回答字母"，模型当时是在瞎猜，导致这两个task初次算出0.0%/8.1%这种低于随机蒙(25%)的荒谬低分。教训：**凡是正确率显著低于随机蒙的信号，先怀疑数据格式，不要直接当成模型能力不足**。已修复并重新推理过。

## 怎么接入OPRD/OPSD-R训练

现在的schema是上面那张扁平字段表，**还不是训练代码直接吃的格式**。项目里已有的训练格式范例是 `probe_data/*/sft.jsonl`（chat messages格式：system+user+assistant，每段内容带`loss:true/false`标记，assistant部分是答案字母、`loss:true`）。

还差一步轻量转换：把每条记录的 `frames`（16张图）拼成user消息里的16个image内容块（因为原视频已删，不能像`probe_data`那样传单个video块）+ `question`/`options`拼成文本；`answer`里的字母作为assistant消息、`loss:true`。这一步还没做，是下一步的工作。

teacher/student的hidden state怎么从这份数据里提取、distillation loss怎么算，参考 `oprd-reproduction-plan.md` / `oprd-ral-informed-redesign.md` / `opsd-r-implementation-plan.md`（那几份文档已经用`probe_data`验证过"答案是1个token、单次forward就能拿到决定性hidden state"这个机制，同样适用于这批数据）。

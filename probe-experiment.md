---
type: experiment-spec
field: 时序理解 gap 定位 —— linear probe 决策关卡
created: 2026-07-05
language: zh-CN
status: 待执行。这份 spec 是在"novelty" 分支会话里讨论出来的,写给**负责跑实验的会话**(有 GPU、管着 P1/P2 评测进程的那个根会话)执行,讨论本身不在这份文档里。
depends_on:
  - novelty.md 的 §0.6(两个阻塞性前提)、§5①②③(验证实验列表)
  - 评测提取器 bug(见下方"前置澄清")——**已由负责评测的会话在 lmms-eval 包内直接修复**(2026-07-05,`tomato/utils.py`/`mvbench/utils.py` 两处 local patch),本实验直接用官方函数即可,不用另写
outputs_expected:
  - "TRD 专属(§1.5,新增):MVBench/TOMATO 各子任务的'打乱帧后掉分幅度'表——筛出哪些子任务真的考关系推理,值得再往下做 hidden state probe"
  - 每个(模型, 原语类型, 层)的 probe 准确率 表格(分类型,§4.2)+ 物理量回归 probe 的 R²/误差表格(§4.4,支撑 novelty.md §2.3 PGR)
  - 每个模型的"直接作答"准确率(用 lmms-eval 官方的 `parse_multi_choice_response`/`mcq_acc`,已修复,见 §3)
  - 一个明确的判定:gap 在感知(表示层)还是在读出/组合推理层,还是 teacher 本身也没有 —— 决定下一步投 TRD/PSD、PGR、纯 RLVR、还是换目标
note: "2026-07-06 更新:①novelty.md 的 GCR 已删除换成 PGR(不需要teacher,监督来自模拟器精确物理真值)。本文档新增 §4.4,把 probe 从'分类标签(方向/计数)'扩展到'连续物理量回归(速度/角速度)',直接支撑 PGR 的数学论证(novelty.md §2.3.1 结论一:DPI 给出的信息量上限,对物理量回归同样成立)。②新增 §1.5,TRD 专属的前置检验——先确认benchmark子任务本身考不考关系推理,再决定要不要往下做hidden state probe,避免在不吃这套的子任务上白费功夫。"
---

# Probe 实验:时序理解的 gap 到底在哪一层?

## 0. 为什么要做这个实验(不要跳过,决定了后面怎么解读结果)

`novelty.md` §0.6 已经确认两件事:①没有一个现成模型是"强时序 teacher"(TOMATO 最高分~31%,随机基线~20%);②合成→真实的迁移 gap 没人验证过。

**前置澄清(评测提取器 bug——已修复,2026-07-06 更新)**:lmms-eval 官方的答案提取逻辑(TOMATO 的 `parse_multi_choice_response`、MVBench 的 `mcq_acc`)在长 CoT 输出上原本有个真实 bug——当模型的最终答案不是"(X)"括号格式时(比如写"The correct option is E."而不是"...is (E)"),两个函数的兜底逻辑会**默认选中候选列表里字母序最靠前的那个**,而候选列表是从"哪些字母当裸字符出现过"生成的——英文长文本里"**B**ased on..."、"**A**lso"、"**B**ut"这类大写句首词天然含有 A/B,于是被误判成"候选答案"。实测(9B thinking-off 的 TOMATO 数据):**626 条模型明确说出答案的样本里,406 条(65%)提取错了**,系统性偏向 A/B。

**现状**:负责评测的会话已经在 `lmms_eval/tasks/{tomato,mvbench}/utils.py` 里直接打了 local patch(补回括号匹配 + 覆盖 markdown粗体`**X**`/`\boxed{X}`/`<answer>X</answer>`/dict-echo 等收尾格式,找不到答案时返回 `None` 而不是瞎猜"A")。修复后 A+B 占比从 82-96% 降到 37-49%(GT 基线约40%),量级正常了。**本实验(和 §5①②③ 一样)直接调用 `lmms_eval.tasks.tomato.utils.parse_multi_choice_response` / `lmms_eval.tasks.mvbench.utils.mcq_acc` 这两个官方函数即可,不用再另写提取器**——但要注意这是打在 conda env 里的本地补丁,不是升级 pip 包,GPFS 环境备份要跟着刷新(`setup_lmmseval.txt` 顶部命令),重装环境时容易漏打。

## 1. 目标

对每个候选模型、每类时序原语,回答:**"模型的 hidden state 里到底有没有编码正确的时序信息(方向/次序/计数/关键事件时刻),不管它最后答没答对"**。四种可能结果:

| probe 读得出来吗 | 最终答案对吗 | 结论 |
|---|---|---|
| 能 | 对 | 端到端没问题,这类原语不用管 |
| **能** | **不对** | **gap 在"读出/组合推理"层,不在感知层** → 表示已经有了,TRD/PSD 这类"教表示"的方法用错方向;该修的是读出头/答案格式/轻量微调,或者直接上 RLVR(outcome reward 逼它把已有信息用对) |
| **不能** | 不对 | **gap 真的在感知层** → 表示里就没有这个信息 → 值得投 TRD/PSD(教表示);还要看 teacher 的 probe 是否比 student 高(见 §4 判定表) |
| 不能 | 对 | 反常,可能是猜对/走了捷径,记下来但不特殊处理 |

这个实验只需要**冻结前向 + 训一个线性/浅层 probe**,不需要 RL 训练基建,几个小时能出结果。

## 1.5 ★ TRD 专属前置检验(2026-07-06 追加):这道benchmark题本身考不考"关系推理"?

> 这一节单独存在是因为讨论中发现:§4 的 hidden state probe 只能回答"student的表示里有没有关系推理需要的信息",但**回答不了"这道题本身需不需要关系推理"**——如果benchmark某个子任务本身能靠单帧表观/语言先验蒙对,那不管TRD把student的关系推理练得多好,这个子任务的分数都不会变,因为分数瓶颈根本不在这。这个实验就是先把这一层筛掉,是§1目标表格再往前一步的、专门针对 TRD(novelty.md §2.1) 的检验,不通用于 PSD/PGR。

**这个实验做什么(机制)**:把 MVBench 20个子任务、TOMATO 6个reason_type,各自的帧序做**随机打乱**,用已经测过的四个模型(锚8B/9B/3.5-35B/3.6-35B,复用现成 eval infra,不需要新写代码)重新跑一遍,**按子任务/reason_type分别记录"打乱前 vs 打乱后"的准确率差**——不是像 novelty.md §5⑤ 那样只看一个笼统的全局对比数字。

**为什么做这个实验(动机)**:TRD 整个方案的前提是"student 的 gap 来自不会正确地把多帧关联起来推理"。但这个前提要成立,有一个更基础、必须先满足的条件——**它想修的这个能力,得是这道benchmark题真的在考的东西**。doc02(`02-社区时序理解与推理方法-cn.md`)开篇引用的经典判据就是这个:"MVBench 打乱帧只掉 ≤1.2%"——说明**大多数视频LLM在大多数子任务上,压根不需要真的做时序/关系推理就能蒙对**。如果我们不先筛出"哪些子任务打乱后真的掉分明显",直接假设"TRD练好了关系推理就能涨分",可能会在一堆压根不吃这套的子任务上白费训练成本。

**这是为哪个idea铺垫的**:**专门服务于 TRD(novelty.md §2.1)**,不是给 PSD/PGR 用的通用检验——PSD 关心的是"状态追踪"(需要另一种子任务映射,比如 object_shuffle 这类跨越遮挡的追踪任务),PGR 关心的是"物理量预测"(需要能定义连续/离散物理状态的原语),这两个不是这同一个 shuffle 实验能覆盖的,**不要把三个idea的前置检验混在一起看**。

**具体产出**:一张"子任务 × 打乱后掉分幅度"的表。**只有掉分幅度大的子任务,才是TRD理论上可能有用武之地的战场**——§4 的 hidden state probe 要不要做、值不值得做,应该**优先在这些子任务对应的样本上做**,没必要对着"根本不吃这套"的子任务浪费probe实验的功夫。如果掉分幅度大的子任务,恰好也是 student 现在分数偏低的子任务(比如 P1 数据里 MVBench 的 object_shuffle/episodic_reasoning),"TRD 值得投"这个假设才算在**我们真正会拿去汇报的 benchmark 上**坐实了,不只是在一个我们自己设计的合成诊断环境里成立。

**和已有内容的关系(避免和别的实验混淆)**:
- 这个实验**先于**§4 对 TRD 的 hidden state probe 验证,是个更上游、更便宜的过滤器——§4 只在这里筛出来的子任务上做才有效率。
- 和 novelty.md §5⑤(排列敏感性 probe)**不是同一件事**:§5⑤比的是"OPRD训练 vs TRD训练,哪个更能应对乱序"(**训练之后**的效果对比,需要跑训练);这个实验比的是"现有的、没训练过的模型在乱序前后的benchmark分数"(**不训练**,纯诊断benchmark任务本身的性质)。两者是不同阶段的检验,都要做,不能互相替代。

## 2. 模型(同源约束,2026-07-05 用户拍板)

teacher 和 student 必须**同源**(同 tokenizer/同代际,避免 OPD 蒸馏时分布不一致)。Qwen3.6 官方没发 ~10B 模型(只有 27B dense 和 35B-A3B MoE,已用 WebSearch 核实),所以:

- **student**:`Qwen/Qwen3.5-9B`(`/root/models/Qwen3.5-9B`,dense,已下载校验,19.3G)
- **teacher(至少跑一个,建议两个都跑做交叉验证)**:
  - `Qwen/Qwen3.5-35B-A3B`(`/root/models/Qwen3.5-35B-A3B`,MoE,已下载校验)
  - `Qwen/Qwen3.6-35B-A3B`(`/root/models/Qwen3.6-35B-A3B`,MoE,已下载校验,P1 综合分最高)
  - (可选,dense-dense 更干净的架构对照,需要新下载)`Qwen/Qwen3.6-27B` —— 如果要做 TRD/PSD 这类需要对齐 hidden dim/attention head 的白盒方法,dense-dense(9B+27B)比 dense-MoE(9B+35B-A3B)更少一层"架构不匹配"的混淆,值不值得多下载这一个模型,留给执行的人权衡(下载命令模板见 `setup_lmmseval.txt`,记得 `env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com`)。

## 3. 数据:SynRL 合成原语(复用已 clone 的生成器,不需要新代码)

用 `repos/Synthetic-Video` 的生成器,选**和真实 benchmark 时序原语能对应上**的几类(这批数据后面 §5①②③ 也要复用,一次生成够用):

| 生成器脚本 | 任务类型(GENERATORS 里的 key) | 对应真实 benchmark |
|---|---|---|
| `01_atomic_motion.py` | `Complex_Direction_Identification` | TOMATO reason_type=direction、MVBench moving_direction |
| `01_atomic_motion.py` | `Rotation_Direction` / `Rotation_Count` | TOMATO reason_type=rotation |
| `01_atomic_motion.py` | `Bouncing_Counting` / `Directional_Event_Counting` | TOMATO reason_type=count、MVBench action_count/moving_count |
| `01_atomic_motion.py` | `Acceleration_Identification` | 速度类(无直接TOMATO对应,当探索项) |
| `02_atomic2_extended_motion.py` | `Event_Sequence` | MVBench action_sequence/action_prediction(次序) |

每类生成 **~200条**(够训+验证一个线性 probe,不需要 SynRL 论文的 1500/类那么大)。改 `generate_dataset(samples_per_type=200, ...)` 调用即可,每个脚本单独跑(不需要跑全部 GENERATORS,注释掉不需要的或者写个小 wrapper 只调目标 key)。产出:
- `sft.jsonl`:直接可用的 zero-shot QA(message 格式,含 `assistant` 里的 GT 答案字母)。
- `metadata.jsonl`:**这次实验真正要用的东西**——`ground_truth_details.video_events_timeline_ms`(关键事件的精确毫秒时刻)+ `other_details`(方向/计数等结构化 GT,不是字母,是数值/类别真值,不受任何文本提取器影响)。

## 4. 具体步骤

### 4.1 抽 hidden state(冻结前向,不训练)

对每个模型(9B student + 1-2 个 teacher),每条合成样本:
- 用**直答协议**(不开 thinking,不用 reasoning_prompt——这一步要的是"模型看完视频那一刻的表示",不是"CoT 之后的表示",所以故意不用 P2 的 thinking 协议),按 `sft.jsonl` 的 message 格式组 prompt,跑到**生成第一个 token之前**(即最后一个输入 token 位置的 forward),抓这个位置在若干层(建议:一层浅层、一层中层、一层接近输出的深层,3层够,不用每层都抓)的 hidden state,按帧做 pooling(参考 novelty.md §2.1 的帧块 pooling 写法:每帧对应的 visual token 块取 mean)得到 $h_i \in \mathbb{R}^d$,$i=1..N$($N$=帧数)。
- 存下来:`{model, task_type, sample_id, layer, h_1...h_N}`(pooled,不是原始 attention,不需要存 attention 矩阵,省很多显存/磁盘)。

### 4.2 训 probe(线性或 2 层小 MLP,故意浅,避免 probe 自己学会推理)

对每个 (model, task_type, layer) 组合,用 §3 表格里对应的 **GT 数值/类别**(来自 `metadata.jsonl`,不是字母答案)当监督目标,训一个线性分类器/回归器:
- direction/rotation-direction 类:pooled 表示(比如 mean over frames,或最后一帧)→ 方向类别(分类)。
- count 类:pooled 表示 → 计数(分类或回归,取决于计数范围)。
- event-sequence 类:更细一点,可以对每帧 $h_i$ 训一个"该帧是否是关键事件帧"的二分类,用 `video_events_timeline_ms` 转成帧级标签(帧的时间戳落在关键事件窗口内=正类)。
- 80/20 或 70/30 切 train/val(每类~200条,够用)。CPU 或单卡几分钟内训完,不需要占用评测 GPU 太久。

### 4.3 同一批数据,记录模型自己的直接作答(直接用 lmms-eval 官方函数,已修复)

对每条样本走同样的 zero-shot 直答协议,拿模型的生成文本,用 `lmms_eval.tasks.tomato.utils.parse_multi_choice_response`(或按任务类型换成对应 benchmark 的官方解析函数)提取答案字母,和 `sft.jsonl` 的 `assistant` GT 字母比较算 accuracy。这两个函数已经打过 2026-07-05 的 local patch(见 §0),不用再自己写:找不到显式声明句时会按 markdown粗体/boxed/dict-echo 等惯例兜底,**都找不到时返回 `None`(计为 unparseable,不瞎猜字母)**。执行前确认一下 `conda activate lmmseval` 用的是打了补丁的那个 env(本地 `/root/miniconda3/envs/lmmseval`,不是 GPFS 备份——备份要是还没刷新可能是旧版本)。

### 4.4 ★ 新增:物理量回归 probe(2026-07-06 追加,支撑 novelty.md §2.3 PGR 的数学论证)

§4.2 的 probe 目标是**分类/离散**标签(方向类别、计数值分箱)。PGR(novelty.md §2.3)需要的是**连续物理量**(位置、速度、角速度)的回归 probe——这需要生成器多吐一点东西,不是白嫖 `metadata.jsonl` 现成就有的。

**需要的工程改动(轻量,在 `repos/Synthetic-Video` 生成器里加)**:目前 `metadata.jsonl` 的 `ground_truth_details.video_events_timeline_ms` 只存**事件发生的时刻**,不存**逐帧的连续物理状态**。需要在生成器渲染循环里,每帧额外 dump 一份 $s_t$(比如 `01_atomic_motion.py` 的匀速/碰撞类:每帧的 (x, y, vx, vy);`Rotation_*` 类:每帧的角度、角速度)到 `metadata.jsonl` 新增字段(比如 `physical_state_per_frame`),不需要改渲染逻辑本身,只是把模拟器已经在算的中间量多存一份。

**probe 训练**:和 §4.2 一样,冻结主干,在 $h_t$ 上训一个**线性回归头** $g_\phi(h_t)=Wh_t+b$,监督目标是 $k$ 帧之后的真实物理状态 $s_{t+k}$(从上面新增的 dump 里取)。评估用 $R^2$(可解释方差比例)而不是准确率,这样能感受"probe 解释了多少物理状态的变化"而不只是"分对了没有"。

**为什么这个值得做(不只是"多测一个维度")**:回归 probe 的 $R^2$ 直接对应 novelty.md §2.3.1 结论一里那条数据处理不等式划的信息量上限——如果 $R^2$ 很低(probe 解释不了物理状态的变化),说明 $I(h_t;s_{t+k})$ 本身就低,那么无论后面 PGR 训练/RLVR 训练怎么加码,答案里能用到的物理信息都不可能超过这个上限。这是**在投入 PGR 训练之前,最先应该看的数字**——比 §4.2 的分类 probe 更贴近 PGR 要解决的具体问题(PGR 关心的正是这类连续物理量,不是抽象的"方向对不对")。

**范围**:只在 01/02 脚本里"有干净连续物理量"的原语类型上做(匀速运动、碰撞、旋转);03-10 号生成器(shell game、滑块、纸牌堆这类符号操作/记忆类任务)大概率没有自然的连续物理量可回归,不用勉强凑,留白就是留白,如实报告哪些原语类型做得了这个 probe、哪些做不了。

## 5. 判定表(出结果后怎么读)

对每个 task_type,列一张这样的表(teacher 用综合分最高的先跑,teacher 之间有分歧再互相对照):

| | student(9B) probe acc | teacher probe acc | student 直答 acc | teacher 直答 acc |
|---|---|---|---|---|
| direction | ? | ? | ? | ? |
| rotation | ? | ? | ? | ? |
| count | ? | ? | ? | ? |
| event_sequence | ? | ? | ? | ? |

**读法**:
- 若某个 task_type 上 **teacher probe acc ≫ student probe acc**(比如差 >20pp),且 teacher 直答 acc 也明显更高 → 这类原语上 teacher 确实有更好的表示,**TRD/PSD 在这类原语上有信号可蒸**。
- 若 **teacher probe acc 和 student 差不多,但 teacher 直答 acc 更高** → gap 不在表示层,是"同样的信息,teacher 更会读出来"——这时候蒸表示(TRD/PSD)没意义,该学的是 teacher 的"读出方式"(更接近读出头微调,或者压根不需要蒸,直接 RLVR 让 student 自己练出读出能力更省事)。
- 若 **两个模型的 probe acc 都很低**(比如都 <随机基线+10pp)→ **这类原语上没有强 teacher**,§0.6 的结论在这个具体原语上进一步坐实,TRD/PSD 都没有信号来源可以蒸(**但 PGR 不受影响**——它的监督来自模拟器不来自 teacher,§4.4 的回归 probe 结果单独看),只能靠纯 RLVR(novelty.md §5⑥)或者——按用户的底线("如果验证发现真正的gap不是这个时序感知问题,就换个目标")——**考虑换一个 teacher probe acc 明显更高的原语类型/任务维度**当主战场,不必执着于这几个 task_type。
- 若某类原语两个模型 probe 都高、直答 acc 都低(且不是提取器的问题,因为这次用了新提取器)→ 值得单独记录,可能是"读出机制"本身有普遍性缺陷(比如选项格式敏感),跳出 novelty.md 现有三个 idea 的范围,可能要考虑答案格式/prompt 层面的干预。

## 6. 范围声明 / 不做什么

- 不训练任何大模型,只训小 probe——GPU 占用主要是**抽 hidden state 的冻结前向**,几个模型 × 几百条样本 × 单帧几百 token,量级不大,几小时应该够。
- 不需要等 P1/P2 全部跑完,这是独立的、更小的实验,可以插空跑。
- 不在这份 spec 里展开 TRD/PSD/PGR 的具体训练设计——那是拿到这个实验结果**之后**的事。
- 这份文档是"novelty"讨论分支产出的交接件,执行、调试、跑 GPU 的工作由负责评测/训练基建的会话接手;有疑问在那边的会话上下文里解决,不要带回这条讨论分支。

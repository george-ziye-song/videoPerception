# videoPerception / OPRD-Bridge：交接说明

这份文件是给接手这台 A100 服务器的 Claude Code 看的——你没有前面调试这套流程那个会话的记忆，这里把任务背景、数据从哪来、Stage 0/1/2 各自干了什么/要干什么讲清楚，让你能快速上手，不用重新摸索。

## 任务是什么

复现 OPRD-Bridge(on-policy representation distillation，跨架构表示蒸馏)论文方法，适配到视频 MCQ(选择题)数据上。

- **Student**：`Qwen3.5-4B`，要训练的模型。
- **Teacher**：`Qwen3.5-9B`，全程冻结，只做推理。
- **目标**：不是简单地让 student 学 teacher 的答案(student 自己做这批题已经能到 82%+ 正确率，答案层面没多少可学的)，而是让 student 生成答案时的**内部 hidden state 表示**更像 teacher 的——即"内部怎么想"更接近，而不只是"嘴上说的答案"接近。这是 representation-level distillation，不是普通的知识蒸馏或 SFT。
- 两个模型 hidden_size 不同(student 2560 / teacher 4096)，没法直接比较，所以需要先学一个"桥"(低秩投影器)，这就是 Stage 1 在做的事。

## 数据链路(input 路径)

原始数据整理过程见 `training-data-decision.md`(如果这份文件在仓库里)，这里只说 Stage 0/1/2 真正读取的东西：

- **训练/验证集**：`OPRD/datasets/video_qa_mc_train.parquet`(3138 行)、`OPRD/datasets/video_qa_mc_val.parquet`(185 行)。verl `RLHFDataset` 能直接读的 schema：`data_source`/`prompt`/`images`/`reward_model`/`extra_info`。**注意行数**——这份数据经过"按 teacher/student 真实能力 gap 重新配比"，剔除了大量"teacher/student 都对、蒸馏没有信号"的陪跑样本，如果哪天看到这两个文件变成 8000+ 行，大概率是有人把没重新配比的旧版本传上来了，需要核对是不是数据出错(`OPRD/scripts/data_preprocess/video_qa_mc.py` 的 `SRC` 应该指向 `FINAL_TRAINING_DATASET_final.json`，3323 行，不是 `_passrate_filtered.json`)。
- **帧图片**：`data/{activitynet_rotation,activitynet_videor1,clevrer,funqa,llavavideo_accel,ntu_rgbd,ovr,perceptiontest,star_videor1}/`，每条样本 16 帧。parquet 的 `images` 列存的是这些目录下文件的**绝对路径**，不是相对路径。如果这份数据是从 HuggingFace dataset 下载下来、存放位置和原始机器不一样，要跑 `OPRD/scripts/data_preprocess/rewrite_dataset_image_paths.py` 把路径前缀改成当前机器的实际路径，否则训练时找不到图片文件。
- **Stage 1 产出(Stage 2 训练必需，不是可选项)**：`OPRD/outputs/stage1_rank8_full/rank_8/ps_bank.pt`——低秩(rank=8)投影器权重，Stage 2 靠 `REP_LOW_RANK_INIT_CHECKPOINT` 这个参数拿它做 warm start。没有这个文件 Stage 2 会从随机初始化/PCA 冷启动开始，效果和"桥接"的本意不一样。
- **模型权重**：`/root/models/Qwen3.5-4B`(student/actor)、`/root/models/Qwen3.5-9B`(teacher，在 verl 配置里挂在 `reward_model` 这个 worker 角色下——这只是复用 verl 已有的角色分工，不是真的在打分，实际用途是跑 teacher 前向、抽 hidden state)。

## Stage 0：on-policy 生成 + 抽 hidden state(已完成)

脚本：`OPRD/scripts/analysis/cross_arch_repr_analysis.py`。用 student(4B)对训练集里每条视频 MCQ 样本做生成(不走 vLLM，直接 `processor` + `model.generate`，因为真实回复很短——就是选项字母，4-12 个 token，不需要长 CoT 那套复杂生成栈)，然后把 prompt+response 这整条序列分别喂给 student 和 teacher，抽每一层的 hidden state(response 位置)。按 shard 分开跑的，产出在 `OPRD/outputs/cross_arch_video_shard_*/`，之后合并成 `OPRD/outputs/merged_on_policy_pairs.jsonl` 供 Stage 1 用。

## Stage 1：PCA + 低秩投影器训练(已完成)

脚本：`OPRD/scripts/analysis/cross_arch_preexp2_train_ps.py`。用 Stage 0 抽出来的 hidden state，先做 PCA、再训练一个低秩(rank=8)投影器，把 teacher 的 4096 维空间和 student 的 2560 维空间对齐——这是 OPRD-Bridge 方法论的核心步骤，没有这一步 Stage 2 没法做 representation-level 的蒸馏 loss。产出 `OPRD/outputs/stage1_rank8_full/rank_8/ps_bank.pt`。

## Stage 2：真正的训练(你要做的事，还没在真实视频数据上跑过)

脚本：`OPRD/on_policy_distillation.sh` → verl 的 `main_ppo` 训练循环(Ray + FSDP + vLLM)。student 一边正常 on-policy 生成回复，一边它的 hidden state 被(经过 Stage 1 投影器桥接后)拉向 teacher 的 hidden state，这是真正在改 student 权重的一步，Stage 0/1 都只是准备工作。

**当前状态**：这条完整的训练链路(训练→reward/teacher 前向计算→checkpoint 保存→resume→对验证集跑验证)已经在**纯文本数据**上跑通过、5 个 step 零报错——目的是排除代码层面的 bug。但**从没有在真实视频数据 + A100 硬件上跑过**，本地只有 24GB 显存的卡，跑不动真实视频的显存需求，所以这最后一步必须在你这台机器上完成。

启动命令、每个关键参数为什么这么选(尤其 `REP_DISTILLATION_LAYERS=all`、`MAX_PROMPT_LENGTH=16384`、几个显存开关的取舍)——都写在 `UPLOAD_GUIDE.md`(国内服务器用，含代理/hf-mirror 相关步骤)或 `A100_SETUP_SINGAPORE.md`(新加坡服务器，不用代理)。直接照着第 4 节(或第 6 节)的命令跑就行，不需要重新设计这套配置，那是根据这套代码之前在 24GB 卡上反复 OOM 调试出来的结果。

**已知的不确定性/风险**(诚实告知，不是隐瞒)：
- 3 卡 A100 colocate 跑 student+ref+teacher 三个模型，是否真的够用没有实测过，guide 里给了 OOM 时的调参顺序建议。
- `REWARD_USE_DYNAMIC_BSZ=True` 这个选项下 verl 自己的 rep-distillation 合并代码有一个真实 bug(不同 micro-batch 样本数不一致时报 shape 不匹配)，没有修，保持 `False`(静态分批)即可绕开，除非你打算修这个 verl 自身的 bug。
- 训练完之后如果要导出能直接部署的单体 HF 模型，FSDP 分片 checkpoint 不能直接用，需要跑 `OPRD/verl/scripts/legacy_model_merger.py`，具体用法没验证过。

## 遇到问题时

- 先看 `UPLOAD_GUIDE.md`/`A100_SETUP_SINGAPORE.md` 里"这条命令里几个做了选择、但你应该知道的点"那部分，很多参数选择的 why 已经写清楚了。
- 环境搭建报错优先怀疑版本号(`vllm==0.19.1` 这类锁定的版本都是踩过坑锁死的，不要因为想用新版本随便升级)。
- 这是纯本地 GPU 训练任务，不涉及网络请求(除了 wandb 上报和一开始下载模型/数据)，遇到问题大概率是显存/参数配置问题，不是网络问题。

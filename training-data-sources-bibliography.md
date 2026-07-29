---
title: 训练数据子类来源总表（含引用与网址）
date: 2026-07-19
status: 完成；配合 training-data-decision.md 使用
---

# 训练数据：每个子类从哪里来（引用 + 网址）

> 配合 `training-data-decision.md` 使用，那份文档是决策（用什么、为什么、比例多少），这份文档是**溯源**（具体每一条数据的出处、引用、下载地址）。
>
> **抽样方法**：Stratum B 采用**分层抽样**（不是随机抽样）——先按子类分组，每组定量抽取，不受原始数据集里各子类天然占比大小影响，保证小众子类不会被抽样时忽略掉。这是唯一符合"全面覆盖、不漏子类"这个要求的方法，随机抽样做不到这一点。

---

## Stratum A：定向 gap primitive（4 个子类）

### A1. Bouncing_Counting（有精确计数标签，已下载完成）

- **数据集**：OVR (Open Vocabulary Repetition)
- **论文引用**：Dwibedi, D., Aytar, Y., Tompson, J., Zisserman, A. "OVR: A Dataset for Open Vocabulary Temporal Repetition Counting in Videos." arXiv:2407.17085, 2024. (Google DeepMind)
- **论文网址**：https://arxiv.org/abs/2407.17085
- **代码/数据网址**：
  - GitHub: https://github.com/google-deepmind/ovr
  - 官方数据页: https://sites.google.com/view/openvocabreps/
  - 标注文件直链: `https://storage.googleapis.com/semantic_repetitions/ovr_kinetics_release.json`、`https://storage.googleapis.com/semantic_repetitions/ovr_ego4d_release.json`
- **底层视频来源**（OVR 本身不含视频，只有标注+指向下面两个数据集的 video_id）：
  - Kinetics-700-2020：Carreira, J., Zisserman, A. 等（视频来自 YouTube，通过 CVDF 镜像 `s3.amazonaws.com/kinetics/700_2020/`）——本次决策里**这部分暂缓**（无 video_id→分片索引，逐个探测不现实）
  - Ego4D：Grauman, K. et al. "Ego4D: Around the World in 3,000 Hours of Egocentric Video." CVPR 2022. 官网 https://ego4d-data.org/ ——**本次决策实际采用的部分**
- **实际获取记录**：关键词过滤出 106 条 "bouncing" 标注（Ego4D 来源），对应 28 个不重复视频；已通过 `ego4d` CLI（个人许可协议，已签署批复）下载 28 条视频，用 ffmpeg 按标注时间段裁剪出 75 个片段，每段抽 16 帧存图，原视频/片段已删除。最终产物：`/remote-home/ziyesong/videoPerception/data/ovr/ego4d_frames_16/`（439MB，1,200 张图）。

### A2. Acceleration_Identification（COIN 的 ThrowHammer 类，可读未跑抽帧）

- **数据集**：COIN (Comprehensive Instructional videONs)
- **论文引用**：Tang, Y., Ding, D., Rao, Y., Zheng, Y., Zhang, D., Zhao, L., Lu, J., Zhou, J. "COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis." CVPR 2019.
- **论文网址**：https://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_COIN_A_Large-Scale_Dataset_for_Comprehensive_Instructional_Video_Analysis_CVPR_2019_paper.pdf
- **官网**：https://coin-dataset.github.io/
- **标注 GitHub**：https://github.com/coin-dataset/annotations（`COIN.json`，本次直接从 `raw.githubusercontent.com` 下载）
- **实际获取记录**：本地已有完整视频压缩包 `/remote-home/share/datasets/coin.tar.gz`（286GB，实测可读）。从 `COIN.json` 的 11,827 个视频、按任务类别关键词过滤，"accelerate" 相关标注 137 条全部来自 `ThrowHammer`（链球投掷技术教学）这一个任务类别，约 40 个视频，原文标注 "rotate body and accelerate the hammer"。**尚未执行**裁剪+抽帧（流程已在 OVR 批次验证，方法相同）。

### A3. Rotation_Count（ActivityNet Captions，标注已下载，视频待本地存储修复）

- **数据集**：ActivityNet Captions（对 ActivityNet 视频的密集时序字幕标注）
- **标注论文引用**：Krishna, R., Hata, K., Ren, F., Fei-Fei, L., Niebles, J.C. "Dense-Captioning Events in Videos." ICCV 2017.
- **底层视频数据集引用**：Caba Heilbron, F., Escorcia, V., Ghanem, B., Niebles, J.C. "ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding." CVPR 2015.
- **标注下载网址**（本次实际使用）：https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip （Stanford 官方托管，train.json/val_1.json/val_2.json）
- **ActivityNet 官网**：http://activity-net.org/
- **实际获取记录**：关键词过滤出 rotate/spin 相关 827 条标注、652 个不重复视频（体操转体、花样滑冰旋转、棍棒 twirl 等真实旋转动作，质量优于 OVR 自身的"手转轮子"）。视频文件本地路径 `/remote-home/share/datasets/ActivityNet/[Update]_Anet_videos_15fps_short256.zip`，**实测读取报 Input/output error**，存储层问题待修复，标注本身已经在手、与视频可读性无关。

### A4. Rotation_Direction（无真实数据来源）

- **数据集**：无。查过 OVR、ActivityNet Captions、Epic-Kitchens-100 三个数据源的标注文本，均不含方向信息（顺/逆时针）。
- **实际方案**：继续使用项目自有的 SynRL 合成生成器（`repos/Synthetic-Video`，项目内部工具，非外部引用文献）。

---

## Stratum B：宽口径任务覆盖（5 个子类，MVBench 任务名），分层抽样

以下 5 项是 §覆盖矩阵 里我们现有数据集能自然覆盖、且置信度较高的 MVBench 任务（不是 20 项全覆盖，其余任务见 Stratum C）。分层抽样的"层"，具体到每个数据集的原生分类：COIN 按其 ~180 个任务类别分层、Epic-Kitchens 按其 97 个 verb class 分层、ActivityNet Captions 按其约 200 个活动类别分层——从每一层定量抽取，保证冷门子类不被忽略。

### B1. Object Interaction / Object Existence
- **主要来源**：Epic-Kitchens-100
- **论文引用**：Damen, D., Doughty, H., Farinella, G.M., Furnari, A., Ma, J., Kazakos, E., Moltisanti, D., Munro, J., Perrett, T., Price, W., Wray, M. "Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100." International Journal of Computer Vision (IJCV), 130, 33–55, 2022.
- **论文网址**：https://arxiv.org/abs/2006.13256
- **官网**：https://epic-kitchens.github.io/2023
- **标注 GitHub**：https://github.com/epic-kitchens/epic-kitchens-100-annotations
- **实际获取记录**：本地视频 `/remote-home/share/datasets/epic-kitchens-100/EPIC-KITCHENS/`（1.5TB，实测可读，通过官方 BitTorrent 下载脚本获取）；标注 `EPIC_100_train.csv`/`EPIC_100_validation.csv`（76,885 条叙事片段）+ `EPIC_100_verb_classes.csv`（97个受控词表 verb class）。
- **次要来源（补充）**：COIN 的 Object Interaction 类内容（教程类视频天然包含大量物体操作）。

### B2. State Change
- **主要来源**：Epic-Kitchens-100（切/煮/洗＝状态变化，见 B1 引用）
- **次要来源**：COIN（步骤化教程的设计初衷就是"前一步→状态变化→后一步"，见 A2 引用）

### B3. Action Sequence / Action Prediction
- **主要来源**：ActivityNet Captions（密集时序字幕的本质就是"事件按顺序发生"，见 A3 引用）
- **次要来源**：COIN（多步骤教程天然是有序序列，见 A2 引用）

### B4. Scene Transition
- **主要来源**：ActivityNet Captions（很多视频跨越多个场景，见 A3 引用）

**（Charades-Ego 作为潜在补充来源，标注尚未挖掘）**
- **论文引用**：Sigurdsson, G.A., Gupta, A., Schmid, C., Farhadi, A., Alahari, K. "Actor and Observer: Joint Modeling of First and Third-Person Videos." CVPR 2018. 另有数据集描述文档：arXiv:1804.09626 "Charades-Ego: A Large-Scale Dataset of Paired Third and First Person Videos"
- **论文网址**：https://arxiv.org/abs/1804.09626
- **官网**：https://prior.allenai.org/projects/charades-ego
- **实际获取记录**：本地视频 `/remote-home/share/datasets/charades-ego/`（227GB，实测可读），标注尚未按关键词/分层挖掘，留作后续补充候选。

---

## Stratum C：目前没有数据来源（诚实列出，不勉强凑数）

| 缺口 | 说明 |
|---|---|
| VideoMME 的 Film & Television / Multilingual / Artistic Performance 内容域 | 现有候选数据集（COIN/Epic-Kitchens/ActivityNet/OVR/Charades-Ego）全部是教程/厨房/体育/日常记录性质，没有影视剧/多语种/艺术表演内容 |
| MVBench 的 Episodic Reasoning、Character Order | 需要"剧集里多角色/多事件"叙事内容，本质上也是需要 Film&TV 类数据，同上无来源 |
| VideoMME 的中/长时长档（4分钟以上，最长1小时） | 现有候选数据的可用片段全部是秒级到最多几分钟，不覆盖长视频 |

这三项如果对最终评测重要，需要另开一轮数据源调研（大概率要找类似 TVQA 的剧集/电影类数据集），这次没有做。

---

## 数据集总览表（去重后，含许可协议信息）

| 数据集 | 引用 | 网址 | 许可 | 本地状态 |
|---|---|---|---|---|
| OVR | Dwibedi et al., arXiv:2407.17085, 2024 | https://arxiv.org/abs/2407.17085 | CC BY 4.0（软件Apache 2.0） | 标注已下载；Ego4D来源视频已下载+处理完成 |
| Ego4D | Grauman et al., CVPR 2022 | https://ego4d-data.org/ | 需签署许可协议（已完成，个人身份） | 28条目标视频已下载完成 |
| Kinetics-700-2020 | Carreira & Zisserman | https://www.deepmind.com/open-source/kinetics | 需遵守 YouTube ToS；CVDF镜像见 https://github.com/cvdfoundation/kinetics-dataset | 暂缓（无索引） |
| COIN | Tang et al., CVPR 2019 | https://coin-dataset.github.io/ | 需在官网注册登录下载 | 本地已有 286GB，可读 |
| ActivityNet Captions | Krishna et al., ICCV 2017 | https://cs.stanford.edu/people/ranjaykrishna/densevid/ | 学术研究用途 | 标注已下载；视频本地 I/O error 待修 |
| ActivityNet（底层视频） | Caba Heilbron et al., CVPR 2015 | http://activity-net.org/ | 学术研究用途 | 同上 |
| Epic-Kitchens-100 | Damen et al., IJCV 2022, arXiv:2006.13256 | https://epic-kitchens.github.io/2023 | CC BY-NC 4.0 | 本地已有 1.5TB，可读 |
| Charades-Ego | Sigurdsson et al., CVPR 2018, arXiv:1804.09626 | https://prior.allenai.org/projects/charades-ego | 学术研究用途 | 本地已有 227GB，可读；标注未挖 |

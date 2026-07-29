# 上传 GitHub + 部署到 A100 训练指南

写这份指南时的验证状态：这台服务器上用纯文本数据(dapo-math-17k)跑通了 Stage 2 完整流程——训练、reward 计算、checkpoint 保存(含 HF 格式导出)、对真实测试集验证，5 个 step 全部零报错。OPRD 已经并入 `videoPerception/OPRD/` 子目录一起上传。

## 0. 关键前提

- **OPRD 原来是独立 git 仓库，remote 指向别人的 GitHub 账号，传不了**，已删掉 `OPRD/.git`、整个目录挪进 `videoPerception/OPRD/`，用 OPRD 自己的 `.gitignore`(已验证嵌套情况下正常生效)。以后只传 `videoPerception` 这一个仓库。
- `videoPerception` 的 remote 已确认是你自己的账号：`https://github.com/george-ziye-song/videoPerception.git`，分支 `main`。
- 根目录 `.gitignore` 里新加了几条排除规则(纯数据/缓存，不是代码，故意不传)：`probe_data/hidden_states*`(里面有单文件 >100MB，会直接把 push 挡住)、`eval/results_*`(评测产物)、`FINAL_TRAINING_DATASET*.json`(中间产物)。加完之后完整 `git add -A` 验证过——3398 个文件、约 400MB、没有单文件超过 50MB，可以安全 push。
- **发现并修复了一个数据过期问题**：`OPRD/datasets/video_qa_mc_train.parquet`(verl 训练直接读的那份)是 7-26 用当时的 `FINAL_TRAINING_DATASET_passrate_filtered.json`(8,248 行，过滤前一版)转出来的，但 7-27 你又做了一轮按真实 gap 重新配比("kept 里真 gap 占比"那次分析)，产出了更小但质量更高的 `FINAL_TRAINING_DATASET_final.json`(3,323 行)——**转 parquet 这一步没跟着重新跑**，Stage 1 的 `ps_bank.pt` 用的已经是新数据，但 Stage 2 原本要读的 `video_qa_mc_train.parquet` 还是旧数据，两边对不上。已经用 `scripts/data_preprocess/video_qa_mc.py`(改了 `SRC` 指向 `_final.json`)重新生成，现在 train/val 是 **3138/185 行**，旧文件挪到了 `OPRD/datasets/_stale_20260726/` 留底(没删)。下面所有数字已经按新的来。

## 1. 推送代码到 GitHub

```bash
cd /remote-home/ziyesong/videoPerception
git add -A
git status              # 扫一眼，确认没有意外的东西
git commit -m "OPRD-Bridge Stage 2: 并入 OPRD、视频 MCQ 数据管线、显存优化"
git push origin main
```

## 2. 上传视频数据到 HuggingFace Dataset(手动)

Stage 2 训练要用、但不适合放 git 的东西：
- 2 个 parquet：`OPRD/datasets/video_qa_mc_{train,val}.parquet`(train 3138 行、val 185 行，都是重新生成过的——文件本身很小，大头是它们引用的帧图片。`video_qa_mc_smoke.parquet` 是本地冒烟测试用的旧文件，A100 正式训练用不到，不用传)
- 9 个帧图片文件夹，共 12GB：`data/{activitynet_rotation,activitynet_videor1,clevrer,funqa,llavavideo_accel,ntu_rgbd,ovr,perceptiontest,star_videor1}/`
- Stage 1 已算出的桥接产物 `OPRD/outputs/stage1_rank8_full/rank_8/{ps_bank.pt,results.json}`(7.4MB，Stage 2 要用它做 rep-distillation 投影的 warm start，不是可选项)

**注意**：parquet 里 `images` 列存的是这台机器上的**绝对路径**(`/remote-home/ziyesong/videoPerception/data/...`)，不是相对路径，也没嵌入图片字节。下载到 A100 后如果存放位置不完全一样，需要跑一下 `OPRD/scripts/data_preprocess/rewrite_dataset_image_paths.py` 把路径前缀改过来(下面第 3.4 节)。

### 2.1 创建 dataset 仓库(网页操作)
1. 登录 https://huggingface.co → 右上角头像 → New Dataset
2. Owner 选你自己账号，起个名字(比如 `video-mcq-oprd-bridge`)，Public/Private 随意，建议先 Private
3. 创建后去 Settings → Access Tokens 页面(或 https://huggingface.co/settings/tokens )确认有一个 **write** 权限的 token，下面命令行登录要用

### 2.2 命令行上传(在这台服务器上做)
```bash
env -u http_proxy -u https_proxy -u all_proxy pip install -U huggingface_hub   # 如果还没装
env -u http_proxy -u https_proxy -u all_proxy huggingface-cli login            # 粘贴上一步的 write token

cd /remote-home/ziyesong/videoPerception
REPO=你的用户名/video-mcq-oprd-bridge   # 换成实际起的名字

env -u http_proxy -u https_proxy -u all_proxy huggingface-cli upload $REPO OPRD/datasets/video_qa_mc_train.parquet video_qa_mc_train.parquet --repo-type dataset
env -u http_proxy -u https_proxy -u all_proxy huggingface-cli upload $REPO OPRD/datasets/video_qa_mc_val.parquet video_qa_mc_val.parquet --repo-type dataset
env -u http_proxy -u https_proxy -u all_proxy huggingface-cli upload $REPO OPRD/outputs/stage1_rank8_full/rank_8/ps_bank.pt stage1_bridge/ps_bank.pt --repo-type dataset
env -u http_proxy -u https_proxy -u all_proxy huggingface-cli upload $REPO OPRD/outputs/stage1_rank8_full/rank_8/results.json stage1_bridge/results.json --repo-type dataset

# 12GB 帧图片，9 个文件夹逐个传(每个可能要几分钟到十几分钟)
for d in activitynet_rotation activitynet_videor1 clevrer funqa llavavideo_accel ntu_rgbd ovr perceptiontest star_videor1; do
  env -u http_proxy -u https_proxy -u all_proxy huggingface-cli upload $REPO data/$d data/$d --repo-type dataset
done
```

## 3. A100 服务器：环境搭建

### 3.1 克隆代码
```bash
env -u http_proxy -u https_proxy -u all_proxy git clone https://github.com/george-ziye-song/videoPerception.git
cd videoPerception
```

### 3.2 conda 环境(和这台服务器的 `oprd_repro` 完全一致，版本号都是踩过坑定下来的，照抄，不要图新版本升级)
```bash
env -u http_proxy -u https_proxy -u all_proxy conda create -n oprd_repro python=3.12 -y
conda activate oprd_repro

env -u http_proxy -u https_proxy -u all_proxy pip install "vllm==0.19.1" "transformers>=5.2.0"
env -u http_proxy -u https_proxy -u all_proxy MAX_JOBS=32 pip install "flash-attn==2.8.3.post1" --no-cache-dir --no-build-isolation
env -u http_proxy -u https_proxy -u all_proxy MAX_JOBS=32 pip install "git+https://github.com/fla-org/flash-linear-attention.git" --no-deps --no-build-isolation
env -u http_proxy -u https_proxy -u all_proxy MAX_JOBS=32 pip install "causal-conv1d" --no-deps --no-build-isolation --no-cache-dir

cd OPRD/verl && pip install -e . --no-deps && cd ../..
env -u http_proxy -u https_proxy -u all_proxy pip install math-verify wandb
```
- `transformers` 必须 ≥5.2.0(才认识 `qwen3_5` 模型类型)，`vllm` 必须是 `0.19.1`——更新版本会把 torch 连带拉到 cu13，如果 A100 服务器驱动是 cu12.x 会导致 GPU 检测不到。如果确认驱动支持更新 CUDA、想升级，装完先用下面的验证命令确认 GPU 能用再继续。
- `flash-linear-attention` 必须从 GitHub 装(命令已经是)，PyPI 上同名包是空壳，`import fla.modules` 会报错。
- Qwen3.5 跑通所需要的源码 patch(FSDP 多模态输入穿线、`AutoModelForImageTextToText` 改名等)全部已经在 git 里，clone 下来就有，不用重新改。

验证：
```bash
python -c "
import torch, transformers, vllm
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(transformers.__version__, vllm.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print('qwen3_5' in CONFIG_MAPPING)
"
```
期望：`torch.cuda.is_available()` 是 `True`，`'qwen3_5' in CONFIG_MAPPING` 是 `True`。

### 3.3 下载模型
```bash
env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen3.5-4B --local-dir /root/models/Qwen3.5-4B
env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen3.5-9B --local-dir /root/models/Qwen3.5-9B
```
4B 约 9.3GB、9B 约 19.3GB。如果 A100 服务器不在国内墙内、能直连 huggingface.co，把 `HF_ENDPOINT=...` 去掉即可。`on_policy_distillation.sh` 里模型路径写死指向 `/root/models/...`；如果要放别处，启动训练时额外传 `ACTOR_MODEL_PATH=`/`REWARD_MODEL_PATH=` 覆盖。

### 3.4 下载数据集
```bash
env -u http_proxy -u https_proxy -u all_proxy huggingface-cli download 你的用户名/video-mcq-oprd-bridge \
  --repo-type dataset --local-dir /tmp/video_mcq_download

cp /tmp/video_mcq_download/video_qa_mc_*.parquet OPRD/datasets/
mkdir -p OPRD/outputs/stage1_rank8_full/rank_8
cp /tmp/video_mcq_download/stage1_bridge/* OPRD/outputs/stage1_rank8_full/rank_8/
mkdir -p data
cp -r /tmp/video_mcq_download/data/* data/

# 如果 clone/存放路径不是 /remote-home/ziyesong/videoPerception，改一下 parquet 里的图片路径前缀：
python OPRD/scripts/data_preprocess/rewrite_dataset_image_paths.py \
  OPRD/datasets/video_qa_mc_train.parquet OPRD/datasets/video_qa_mc_train.parquet --new-root "$(pwd)/data"
python OPRD/scripts/data_preprocess/rewrite_dataset_image_paths.py \
  OPRD/datasets/video_qa_mc_val.parquet OPRD/datasets/video_qa_mc_val.parquet --new-root "$(pwd)/data"
```

### 3.5 wandb
```bash
wandb login   # 交互式，粘贴 https://wandb.ai/authorize 的 API key
```

## 4. 启动 Stage 2 训练(3×A100)

```bash
cd videoPerception/OPRD/verl
conda activate oprd_repro

N_GPUS_PER_NODE=3 \
MODEL_DTYPE=bfloat16 \
TRAIN_DATASET=../datasets/video_qa_mc_train.parquet \
TEST_FILE='["../datasets/video_qa_mc_val.parquet"]' \
MAX_PROMPT_LENGTH=16384 MAX_RESP_LENGTH=256 MAX_VAL_RESP_LENGTH=256 \
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
REWARD_MICRO_BATCH_SIZE_PER_GPU=1 REWARD_USE_DYNAMIC_BSZ=False \
ENFORCE_EAGER=False OPTIMIZER_OFFLOAD=False USE_TORCH_COMPILE=False \
USE_REP_DISTILLATION=True REP_LOW_RANK=8 \
REP_LOW_RANK_INIT_CHECKPOINT=../outputs/stage1_rank8_full/rank_8/ps_bank.pt \
REP_DISTILLATION_LAYERS=all \
SAVE_FREQ=50 TEST_FREQ=50 MAX_ACTOR_CKPT_TO_KEEP=3 \
LOGGER="['console','wandb']" LOG_VAL_GENERATIONS=10 IS_PLOT=False \
CKPT_PATH=outputs/stage2_video_bridge_rank8 \
bash ../on_policy_distillation.sh
```

**崩了/断了怎么办**：原样重跑同一条命令(同样的 `CKPT_PATH`)，会自动从最近 checkpoint 续训，`SAVE_FREQ=50` 最多损失 50 步。

### 这条命令里几个我替你做了选择、但你应该知道的点

1. **`REP_DISTILLATION_LAYERS=all`**——OPRD 论文原文监督全部层(28 层，对应 Qwen3.5 是 32 层)，脚本自己的默认值其实是 `last`(只监督最后一层，更省显存但不是论文协议)。这次专门优化的 hidden-state hook 也只在 `all` 模式下才生效(避免 32 层的完整张量同时留在内存里)。论文协议和这次的性能优化都指向 `all`，我按这个定的；如果想换回更省显存的 `last`，改这一个变量就行。
2. **三个"恢复速度"的开关**——都是这次在 24GB 卡上为了不 OOM 才关掉的，A100 40GB 理论上够，这条命令里已经恢复成开：`ENFORCE_EAGER=False`(开 CUDA graph)、`OPTIMIZER_OFFLOAD=False`(优化器状态留在 GPU)。**如果 3 卡 A100 还是 OOM**，按顺序改回去最可能有效：先 `OPTIMIZER_OFFLOAD=True`(3 张卡同时装 student+ref+teacher 三个模型，比这次测试过的配置更紧张)，不够再 `ENFORCE_EAGER=True`(这次验证过 CUDA graph 确实占真实显存，不是无关的坑)。
3. **`USE_TORCH_COMPILE=False`**——脚本自己的默认值其实是 `True`，但这次全程验证成功的组合用的是 `False`；`True` 从没在 Qwen3.5+rep-distillation 这个组合上试过。建议先按 `False` 跑通，之后单独找一次运行试 `True` 看能不能提速，别和第一次正式跑混在一起冒风险。
4. **`MAX_PROMPT_LENGTH=16384`**——不是随便定的，是 Stage 0/1 阶段拿真实 16 帧视频样本实测出来的(`image_patch_size=16` 匹配视觉塔真实 patch size 之后，360×640×16 帧的 token 数普遍超过 8192)。`MAX_RESP_LENGTH=256`——这批数据是 MCQ 选字母，真实回复长度只有 4-12 个 token，256 留了充足冗余，没有沿用数学任务那种上千 token 的默认值。
5. **`ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=24576`**——必须大于单条样本最长可能长度(prompt+response，实测上限约 16512)，这里留了约 1.5 倍冗余；卡太紧会直接触发 `max_token_len must be greater than the sequence length` 报错(这次调试踩过)。
6. **`MAX_ACTOR_CKPT_TO_KEEP=3`**——这次在文本数据上验证时实测一个 checkpoint(FSDP 分片+优化器状态)约 **26GB**。约 392 步、`SAVE_FREQ=50` 意味着约 8 个 checkpoint，不加限制会占约 200GB 磁盘。这是新加的一个纯 shell、不碰训练逻辑的环境变量开关(`trainer.max_actor_ckpt_to_keep` 本来就是 verl 自带的配置项，只是这个脚本之前没接上)，只保留最近 3 个，按磁盘需要调大。训练完之后如果要导出能直接部署的单体 HF 模型，FSDP 分片 checkpoint 不能直接用，需要另外跑 `OPRD/verl/scripts/legacy_model_merger.py` 合并(具体用法没在这次验证范围内，需要的话告诉我)。

没有改动、按脚本/论文原有设置跑的：`N_RESPONSES=2`、`TEMPERATURE=1.0`、`TEACHER_TEMPERATURE=1.0`、`MINI_BATCH_SIZE=8`、`GPU_MEMORY_UTILIZATION`(脚本默认 0.8)——这些是训练方法本身的超参，这次没有理由动它们。

### 训练要跑多少 step

`train_batch_size = MINI_BATCH_SIZE(8) × PARALLEL_SIZE(1) = 8` 条 prompt/step。视频训练集 `video_qa_mc_train.parquet`(gap 重新配比后的最终版)共 **3138 行**。命令里没设 `TOTAL_TRAINING_STEPS`，脚本按 `total_epochs=1` 自然跑完 **3138÷8 ≈ 392 step** 后停止——这是起点，不是终点。用 wandb 盯着 reward/loss 曲线，如果 1 个 epoch 结束时还在明显下降，可以设 `TOTAL_TRAINING_STEPS` 接着跑第 2 个 epoch(注意 `data.shuffle=False` 是脚本写死的，多个 epoch 会重复同样的数据顺序)。

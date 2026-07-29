# A100(新加坡，直连不用代理)环境搭建 + 启动训练

和 `UPLOAD_GUIDE.md` 第 3/4 节内容一样，唯一区别：**这台机器不在墙内，所有命令都不用 `env -u http_proxy ...` 去代理、也不用 `HF_ENDPOINT=hf-mirror.com` 走镜像**，直接连真实的 huggingface.co/github.com 就行(不加多余前缀，也不会有影响——只是没必要)。国内那台 A100 还是照 `UPLOAD_GUIDE.md` 第 3/4 节来。

## 0. 前提

代码已经推到 `https://github.com/george-ziye-song/videoPerception.git`，视频数据(parquet + 12GB 帧图片 + Stage 1 产出 `ps_bank.pt`)已经传到 HuggingFace dataset 仓库——这两步是在国内那台机器上做的，细节见 `UPLOAD_GUIDE.md` 第 1/2 节，这台机器不用重做，直接拉就行。

## 1. 克隆代码

```bash
git clone https://github.com/george-ziye-song/videoPerception.git
cd videoPerception
```

## 2. conda 环境

版本号照抄，都是在国内那台机器上踩坑定下来的组合，不是随便选的（原因见 `oprd-repro-env-recovery.md`）：

```bash
conda create -n oprd_repro python=3.12 -y
conda activate oprd_repro

pip install "vllm==0.19.1" "transformers>=5.2.0"
MAX_JOBS=32 pip install "flash-attn==2.8.3.post1" --no-cache-dir --no-build-isolation
MAX_JOBS=32 pip install "git+https://github.com/fla-org/flash-linear-attention.git" --no-deps --no-build-isolation
MAX_JOBS=32 pip install "causal-conv1d" --no-deps --no-build-isolation --no-cache-dir

cd OPRD/verl && pip install -e . --no-deps && cd ../..
pip install math-verify wandb
```
- `transformers` 必须 ≥5.2.0(才认识 `qwen3_5` 模型类型)，`vllm` 必须是 `0.19.1`——更新版本会把 torch 连带拉到 cu13，如果这台机器驱动是 cu12.x 会导致 GPU 检测不到。如果确认驱动支持更新 CUDA、想升级，装完先用下面的验证命令确认 GPU 能用再继续。
- `flash-linear-attention` 必须从 GitHub 装(命令已经是)，PyPI 上同名包是空壳，`import fla.modules` 会报错。
- `MAX_JOBS=32` 不是随便定的，是在 192 核/~500GB 内存的机器上试出来的安全值(`MAX_JOBS=64` 编译时直接把内存打爆、进程被 OOM Killed)——这台机器如果核数/内存差异很大，酌情调整，太高有编译期被杀掉的风险。
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

## 3. 下载模型

```bash
hf download Qwen/Qwen3.5-4B --local-dir /root/models/Qwen3.5-4B
hf download Qwen/Qwen3.5-9B --local-dir /root/models/Qwen3.5-9B
```
4B 约 9.3GB、9B 约 19.3GB。`on_policy_distillation.sh` 里模型路径写死指向 `/root/models/...`；如果要放别处，启动训练时额外传 `ACTOR_MODEL_PATH=`/`REWARD_MODEL_PATH=` 覆盖。

## 4. 下载数据集

```bash
hf download 你的用户名/video-mcq-oprd-bridge --repo-type dataset --local-dir /tmp/video_mcq_download

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
下载完确认一下行数对不对(train 应该是 3138 行、val 185 行——不是 8000+ 那个过期版本，之前在国内机器上发现并修过这个数据过期问题):
```bash
python -c "
import pandas as pd
print('train', len(pd.read_parquet('OPRD/datasets/video_qa_mc_train.parquet')))
print('val', len(pd.read_parquet('OPRD/datasets/video_qa_mc_val.parquet')))
"
```

## 5. wandb

```bash
wandb login   # 交互式，粘贴 https://wandb.ai/authorize 的 API key
```
(实测过 `api.wandb.ai` 国内那台机器不用代理也能直连，这台就更不用担心了)

## 6. 启动 Stage 2 训练(3×A100)

命令和参数选择的完整理由在 `UPLOAD_GUIDE.md` 第 4 节，一字不差地照抄那条命令即可(和代理/网络环境完全无关，纯本地 GPU 训练)：

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

崩了/断了：原样重跑同一条命令(同样的 `CKPT_PATH`)，自动从最近 checkpoint 续训。参数选择的理由(为什么是 `all`、为什么是 `16384`、OOM 了怎么办)不再重复，见 `UPLOAD_GUIDE.md`。

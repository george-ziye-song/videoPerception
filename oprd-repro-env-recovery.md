---
title: OPRD 复现环境重建指南——容器重启/重建后按此恢复
date: 2026-07-24
status: 已知问题——nvidia-smi报"Failed to initialize NVML"时，用户确认重启实例能解决。/root下的东西会清空，本文档就是为了不用重新摸索一遍环境搭建过程。
---

# 触发条件

`nvidia-smi` 报 `Failed to initialize NVML: Unknown Error`，且确认不是残留进程导致（`ray stop --force` + 检查 `ps aux | grep main_ppo|raylet` 干净后依然报错）。用户说重启实例能解决，原因不明，是复现出的问题。

**重启后会丢的东西**（都在 `/root` 下，Docker overlay）：
- conda 环境 `oprd_repro` 整个没了
- `/root/models/` 下的模型权重（Qwen3.5-4B、Qwen3.5-9B 等）没了

**不会丢的东西**（都在 `/remote-home/ziyesong`，GPFS 持久盘）：
- `/remote-home/ziyesong/OPRD` 整个仓库，包括所有已经改过的源码（见下面"已应用的源码补丁"）
- 训练 checkpoint：`/remote-home/ziyesong/OPRD/verl/outputs/grpo_phase1_500steps/`（目前应该有 `global_step_30`，可能更新）
- 本文档

# 恢复步骤

## 1. 检查 GPU 恢复正常

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```
应该显示8张4090，都是 `12 MiB` / `0%`（空闲）。如果还报NVML错误，说明重启没解决，需要用户再想办法。

## 2. 重建 conda 环境（约10-15分钟，主要是下载耗时）

```bash
env -u http_proxy -u https_proxy -u all_proxy conda create -n oprd_repro python=3.12 -y
source /root/miniconda3/etc/profile.d/conda.sh && conda activate oprd_repro
```

**关键：装的顺序和版本都有讲究，踩过的坑直接抄下面的，不要自己按 README 走一遍。**

### 2.1 torch + vllm + transformers（最容易踩坑的一步）

Qwen3.5 是2026-02才合入 transformers 主干的全新架构（混合线性注意力+原生多模态），必须用新版本栈：

```bash
env -u http_proxy -u https_proxy -u all_proxy pip install "vllm==0.19.1" "transformers>=5.2.0"
```

**为什么是这两个版本，不能随便改**：
- `transformers` 必须 ≥5.2.0 才认识 `qwen3_5` 这个 model_type（4.x系列完全不支持）。
- `vllm` 必须 ≥0.19.1（0.19.0之前硬锁 `transformers<5`，装不到一起）。
- **千万不要装 vllm 0.20+ 或更新版本**——会把 torch 连带拉到 cu13（CUDA 13），而这台机器驱动是 `550.90.07`，`nvidia-smi` 报的上限是 CUDA 12.4，装cu13的torch会导致GPU完全检测不到。之前验证过 `vllm==0.19.1` 不会动 torch 的 CUDA 大版本（还是 cu12.8 系），这是精心选出来的安全版本，不要图新版本bugfix多就升级。
- 这条命令会连带把 torch 装到 2.10.0+cu128（vllm 0.19.1 的依赖决定的，正常现象）。

装完验证：
```bash
python -c "
import torch, transformers, vllm
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(transformers.__version__, vllm.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print('qwen3_5' in CONFIG_MAPPING)  # 必须是 True
"
```
期望输出：`2.10.0+cu128 12.8 True` / `5.14.1 0.19.1`（或更新的5.x）/ `True`。

### 2.2 flash-attn（要重新编译，注意 MAX_JOBS）

```bash
env -u http_proxy -u https_proxy -u all_proxy MAX_JOBS=32 pip install "flash-attn==2.8.3.post1" --no-cache-dir --no-build-isolation
```

**为什么要 `--no-build-isolation`**：不加的话 pip 会在隔离沙箱里装一个全新版本的 torch 来探测wheel该下哪个，容易搞乱。加上之后才会用当前环境里已经装好的 torch 2.10.0 来编译。

**为什么 `MAX_JOBS=32` 不能更高**：这台机器有192核、~500GB内存，但 flash-attn 编译单文件峰值能到8-10GB，`MAX_JOBS=64` 试过一次直接把内存打爆导致编译被 OOM Killed。32是验证过安全的值。

编译要几分钟到十几分钟，正常。

### 2.3 flash-linear-attention + causal-conv1d（给24层线性注意力层提速，不装也能跑但慢很多）

```bash
env -u http_proxy -u https_proxy -u all_proxy MAX_JOBS=32 pip install "git+https://github.com/fla-org/flash-linear-attention.git" --no-deps --no-build-isolation
env -u http_proxy -u https_proxy -u all_proxy MAX_JOBS=32 pip install "causal-conv1d" --no-deps --no-build-isolation --no-cache-dir
```

**注意：flash-linear-attention 必须从 GitHub 装，不能用 PyPI 上的包**——PyPI 上的 `flash-linear-attention` 包（`pip install flash-linear-attention`）是空壳/版本不对，装完 `import fla.modules` 会报 `ModuleNotFoundError`，必须用 `git+https://github.com/fla-org/flash-linear-attention.git`。

验证快速路径生效（不应该看到 "fast path is not available" 警告）：
```bash
python -c "
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('/root/models/Qwen3.5-4B', dtype=torch.bfloat16, device_map='cuda:0')
print('OK', type(m).__name__)
"
```

### 2.4 verl（仓库自带的fork，editable装）

```bash
cd /remote-home/ziyesong/OPRD/verl
pip install -e . --no-deps
```
`--no-deps` 是为了不让 pip 因为装 verl 又去乱动前面精心装好的 torch/vllm 版本。

### 2.5 math-verify（reward打分要用）

```bash
env -u http_proxy -u https_proxy -u all_proxy pip install math-verify
```

## 3. 检查模型权重还在不在

```bash
ls /root/models/Qwen3.5-4B /root/models/Qwen3.5-9B
```

如果没了，需要重新下载。当时是怎么下的不确定（不是本会话下载的），如果需要重新下载，用 hf-mirror：
```bash
env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen3.5-4B --local-dir /root/models/Qwen3.5-4B
env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen3.5-9B --local-dir /root/models/Qwen3.5-9B
```
（具体 HF repo 名字可能不是 `Qwen/Qwen3.5-4B`，需要确认，之前会话里没有记录原始下载来源。4B模型约9.3GB，9B模型约19.3GB。）

## 4. 已应用的源码补丁（这些在 `/remote-home/ziyesong/OPRD` 里，不会因为容器重建丢失，不需要重新做，这里列出来是为了知道改过什么）

在 `/remote-home/ziyesong/OPRD` 目录下 `git diff` 能看到完整 diff。改过的文件：

- `grpo.sh` / `on_policy_distillation.sh`：模型路径改成本地路径、去掉强制 `enable_thinking=False`、去掉强制多模态输入处理、`MODEL_DTYPE`/`PPO_MAX_TOKEN_LEN_PER_GPU`/`GPU_MEMORY_UTILIZATION`/`ENABLE_ACTIVATION_OFFLOAD`/`OPTIMIZER_OFFLOAD`/`SAVE_FREQ`/`CKPT_PATH`/`DATA_SHUFFLE`/`ENTROPY_CHUNKING` 等一批参数改成可以用环境变量覆盖
- `verl/verl/workers/fsdp_workers.py`：①`AutoModelForVision2Seq`（transformers 5.x已移除）改名`AutoModelForImageTextToText`；②FSDP wrap policy 逻辑改成"找不到的类名跳过，不是任何一个找不到就报错"；③Auto类选择顺序改成优先`AutoModelForImageTextToText`（不是`AutoModelForCausalLM`），因为vLLM自己的权重加载mapper是按前者的命名设计的，选错了rollout阶段权重同步会失败
- `verl/verl/utils/model.py`、`verl/verl/model_merger/base_model_merger.py`：同样的 `AutoModelForVision2Seq`→`AutoModelForImageTextToText` 改名
- `verl/verl/utils/fsdp_utils.py`：`get_fsdp_wrap_policy` 里"部分class找不到就报错"的bug，改成只在全部找不到时才报错
- `verl/verl/utils/vllm/utils.py`：`from vllm.lora.models import LoRAModel` 改成 `from vllm.lora.lora_model import LoRAModel`（vllm 0.19内部模块重组）

## 5. 恢复训练（跳过 Phase 1 已经跑过的部分，从 checkpoint 继续）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate oprd_repro
cd /remote-home/ziyesong/OPRD/verl
cat outputs/grpo_phase1_500steps/latest_checkpointed_iteration.txt   # 确认目前存到第几步

MINI_BATCH_SIZE=8 N_RESPONSES=2 MODEL_DTYPE=bfloat16 \
PPO_MAX_TOKEN_LEN_PER_GPU=16384 ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
SAVE_FREQ=10 CKPT_PATH=/remote-home/ziyesong/OPRD/verl/outputs/grpo_phase1_500steps \
PYTORCH_ALLOC_CONF=expandable_segments:True DATA_SHUFFLE=True \
bash ../grpo.sh
```

会自动从 checkpoint 续训（日志里能看到 `Resuming from .../global_step_XX`），不是从头开始。目标是跑满 `trainer.total_training_steps=500`（脚本里已经写死，对应论文原文的训练协议）。

**这个配置是经过反复试错定下来的稳定配置**（batch=8/response=2是这台8×24GB 4090上能稳定跑的上限，更大会稳定OOM；`DATA_SHUFFLE=True` 是因为发现`shuffle=False`时数据顺序固定，会在同一个step反复复现同一个"最坏情况"batch导致OOM），不需要再调参，除非又遇到新问题。

**已知问题**：训练过程中大约每15-40步会因为显存碎片/某些batch响应特别长而偶发OOM崩溃——这是已知的、在容忍范围内的行为，不是bug。每次崩溃后用**同样的命令**（同样的`CKPT_PATH`）重新跑一遍就行，断点续训会自动接上，最多损失10步进度（`SAVE_FREQ=10`）。

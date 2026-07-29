#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate oprd_repro
cd /remote-home/ziyesong/OPRD/verl

# All 8 GPUs now free -- also sidesteps the prime-number (7) DataProto-chunk-divisibility issue
# from earlier attempts entirely, so batch/response sizes can go back to normal (non-multiple-of-7)
# values.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Attempt 10 (token-budget dynamic batching for the reward model) STILL OOM'd at nearly identical
# figures to before -- the weights themselves (9B, always resident on GPU DURING active compute;
# offload only helps BETWEEN calls) genuinely don't fit alongside the 4B actor + vLLM rollout
# engine on the same GPU set, regardless of how the reward model's own batching is bounded.
# Switching mechanism entirely: give the reward model its OWN dedicated GPUs (verl's
# reward_model.enable_resource_pool, a separate Ray resource pool from actor/rollout/ref's
# "global_pool" -- verl/trainer/main_ppo.py's init_resource_pool_mgr) instead of colocating
# everything. 6 GPUs for actor+rollout+ref (needs vLLM's memory + FSDP-sharded 4B weights+
# optimizer state), 2 dedicated for the 9B reward/teacher model (forward-only, no optimizer
# state, and no longer fighting the vLLM engine or actor for the same card).
# Attempt 11 (2 dedicated GPUs) shifted the memory profile (20-21 GiB in use vs 22-23 before,
# "private pools" no longer mentioned) but still OOM'd on the same-size allocation -- real but
# insufficient improvement. More dedicated GPUs = finer FSDP sharding specifically for the 9B
# model. Going to an even 4+4 split.
export N_GPUS_PER_NODE=4
export REWARD_ENABLE_RESOURCE_POOL=True
export REWARD_N_GPUS_PER_NODE=4
export REWARD_NNODES=1

export USE_REP_DISTILLATION=True
export REP_LOW_RANK=8
export REP_LOW_RANK_INIT_CHECKPOINT=/remote-home/ziyesong/OPRD/outputs/stage1_rank8_full/rank_8/ps_bank.pt

export TRAIN_DATASET=../datasets/video_qa_mc_smoke.parquet
export TEST_FILE='["../datasets/video_qa_mc_val.parquet"]'
export TRAIN_DATASET_NAME=video_qa_mc_smoke

export RETURN_MULTI_MODAL_INPUTS=True
export IMAGE_PATCH_SIZE=16
export ENABLE_THINKING=False
export REWARD_FUNCTION_PATH=verl/utils/reward_score/video_mc/__init__.py

# Empirically measured: 16 frames of 360x640 with the correct image_patch_size=16 need well over
# 8192 tokens for most samples (confirmed via a real crash: 32-row smoke set filtered to only 8
# rows at the old 8192 ceiling).
export MAX_PROMPT_LENGTH=16384
export MAX_RESP_LENGTH=128
export MAX_VAL_RESP_LENGTH=128
# Global pool is now 4 GPUs -- batch*n_responses must divide evenly by 4 (both pools are 4 now).
# N_RESPONSES=2 restored to the script's own default (more faithful to actual OPRD on-policy
# distillation now that memory has a structural fix in progress instead of a volume cut).
export MINI_BATCH_SIZE=4
export N_RESPONSES=2
export REWARD_MICRO_BATCH_SIZE_PER_GPU=1
export GPU_MEMORY_UTILIZATION=0.6
export USE_TORCH_COMPILE=False
export MODEL_DTYPE=bfloat16
# Root cause (confirmed by reading verl's actual source, not guessed): RewardModelWorker already
# hardcodes FSDP CPUOffload unconditionally at construction -- param_offload was genuinely a
# no-op for this worker (verified: identical OOM figures with it True vs False). The real OOM
# driver is UNBOUNDED activation memory per forward call: rep_distillation_layers=all requests
# output_hidden_states=True, i.e. the FULL 32-layer hidden-state tuple, and reward_model's fixed
# micro_batch_size_per_gpu packs samples by COUNT not by token budget, so one long (~16.5K-token)
# video sample's forward can spike far past a fixed-size micro-batch's assumed memory. Switching
# to token-budget dynamic batching (same mechanism actor/ref/critic already use successfully)
# directly bounds this instead of guessing at a sample-count. 20000 is just above one sample's
# worst case (16384 prompt + 128 response), so at most one long sample is ever in flight per
# forward call on a given GPU.
# Attempt 12 (4+4 GPU split, up from 2+6) made ZERO difference to the OOM figures (still
# ~19.28-19.97 GiB allocated by PyTorch, byte-for-byte close to attempt 11's numbers). Read
# verl's actual RayResourcePool/RayWorkerGroup plumbing to confirm FSDP sharding IS scoped
# correctly to the dedicated pool's own (smaller) world_size -- it is, verified at the code
# level. The real reason more GPUs didn't help: `n_gpus_per_node` only buys more concurrent
# micro-batches (data parallelism), it does NOT shrink any single micro-batch's own tensors.
# The actual OOM (confirmed via the real traceback: _compute_entropy_safe's log_softmax over a
# [total_nnz, vocab_size] logits tensor) is bounded by SEQUENCE LENGTH per GPU, and the missing
# lever is `reward_model.ulysses_sequence_parallel_size` -- this SPLITS one long sequence's
# tokens across the pool's GPUs before the forward pass (verl/workers/fsdp_workers.py's
# ulysses_pad_and_slice_inputs), which is what actually shrinks the per-GPU activation tensor
# for a single long (~16.5K-token) video sample, unlike plain data-parallel GPU count.
export REWARD_PARAM_OFFLOAD=True
export REWARD_USE_DYNAMIC_BSZ=True
# Attempt 17 (SP=2) got past the shape-mismatch bug entirely -- confirms SP=4 was producing an
# empty shard for short sequences, now fixed -- and reached a NEW, purely arithmetic error:
# "max_token_len must be greater than the sequence length. Got max_token_len=16384 and
# max_seq_len=16512". Ulysses SP splits ONE sequence across SP_size GPUs, so a sequence's total
# length must fit within forward_max_token_len_per_gpu * SP_size. At SP=2, that was 8192*2=16384
# -- 128 tokens short of the worst case (16384 prompt + 128 response = 16512). Raising the budget
# to comfortably clear that ceiling.
export REWARD_FORWARD_MAX_TOKEN_LEN_PER_GPU=10000
# Attempt 16 (after fixing the actor_module->reward_module bug) hit a NEW error one level deeper:
# transformers' own create_causal_mask/_preprocess_mask_arguments does
# position_ids.expand(batch_size,-1) where position_ids has shape [0, 28388] -- an EMPTY first
# dimension. Hypothesis: some sample in the smoke set is short enough that 4-way Ulysses
# sequence-splitting leaves one of the 4 shards with zero tokens. Testing a less aggressive SP
# degree (2, still splits across half the dedicated pool -- DP=2 x SP=2 -- keeping some memory
# relief for the longest samples while being less likely to produce an empty shard for short ones).
export REWARD_ULYSSES_SP=2
# Attempt 13: the reward-model OOM is GONE (Ulysses SP worked) -- real progress, first time past
# that point. NEW, different error now: vLLM's CUDA graph capture fails with "torch.AcceleratorError:
# CUDA error: operation not permitted" inside cuda_graph.py's capture_end(). This is a distinct
# vLLM/CUDA-graph-capture issue, not a memory problem -- disabling CUDA graph capture (fall back
# to eager execution, standard vLLM escape hatch for graph-capture issues in constrained/shared-GPU
# or containerized environments) trades some rollout speed for sidestepping this class of failure.
export ENFORCE_EAGER=True

export SAVE_FREQ=1
export TEST_FREQ=100
export TOTAL_TRAINING_STEPS=3
export CKPT_PATH=/remote-home/ziyesong/OPRD/verl/outputs/stage2_smoke_test
# vLLM's CuMemAllocator hard-asserts this env var is absent (confirmed via a real crash earlier).
# Deliberately NOT set here.

env -u http_proxy -u https_proxy -u all_proxy \
  bash ../low_rank_rep_distillation.sh \
  > /tmp/claude-0/-remote-home-ziyesong/8259ba29-de35-42a3-adc8-c7ded51e6d99/scratchpad/stage2_smoke.log 2>&1
echo "STAGE2_SMOKE_EXIT_CODE=$?" >> /tmp/claude-0/-remote-home-ziyesong/8259ba29-de35-42a3-adc8-c7ded51e6d99/scratchpad/stage2_smoke.log

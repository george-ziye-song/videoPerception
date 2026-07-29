---
title: OPRD 复现计划——分阶段，交给负责实验的会话执行
date: 2026-07-22
status: 计划，本会话不跑GPU/不监控训练，只定协议；student=Qwen3.5-4B，teacher=Qwen3.5-9B
related:
  - "/remote-home/ziyesong/OPRD（已clone的官方repo，README/脚本均以此为准）"
  - "oprd-ral-informed-redesign.md（后续视频novelty设计，本文档只覆盖原生文本复现部分）"
---

# OPRD 复现计划

> 目的：在把 OPRD 的机制搬到视频任务之前，先在它自己原生的设置（纯文本数学推理，verl训练栈）上把它跑通、验证论文声称的几个具体结论站不站得住。这样后续做视频改造时，遇到问题能分清楚"是我们改视频这部分的问题"还是"OPRD本身在我们的硬件/模型规模上就没跑对"。
>
> 本文档只覆盖**原生文本复现**这部分。视频相关的novelty（运动能量attention正则化等）还在讨论中，不在这次分阶段里，留了 Phase 7 占位。

## 0. 模型确认（已核实，不是假设）

| 角色 | 模型 | 路径 | hidden_size | 层数 | full_attention层数 |
|---|---|---|---|---|---|
| Student | Qwen3.5-4B | `/root/models/Qwen3.5-4B` | **2560** | 32 | 8 |
| Teacher（脚本里叫`REWARD_MODEL_PATH`，见§2.2说明） | Qwen3.5-9B | `/root/models/Qwen3.5-9B` | **4096** | 32 | 8 |

**关键发现，直接决定走哪条复现路径**：两个模型**hidden_size不同**（2560 vs 4096），层数和full_attention层的分布位置相同（都是32层、每4层1个full_attention）。这意味着：

**必须走 OPRD-Bridge（跨架构），不能走 OPRD-Vanilla（同架构）**——README原话："When teacher and student have different hidden dimensions, depths, or even different tokenizers, direct hidden-state alignment fails. OPRD-Bridge addresses this via a two-stage pipeline"。Vanilla假设维度相同、直接对hidden state做MSE，我们这里维度对不上，做不了，必须用Bridge的PCA+可学习投影这条路。

**⚠️ 模型在`/root/models/`，这是`/root`下——按CLAUDE.md，这是Docker overlay，容器重建会清空。** 如果这两个模型没有对应的持久化重下载脚本（类似`data/benchmarks/redownload_temporalbench_to_root.sh`那种"数据在/root、脚本+日志在GPFS"的模式），执行前建议先确认一下，避免跑到一半模型没了。

## 1. Finished

## 2. Phase 1：GRPO baseline（最先跑，确认基建通不通）

```bash
bash /remote-home/ziyesong/OPRD/grpo.sh
```

**为什么第一个跑这个**：不涉及teacher、不涉及hidden state蒸馏，只需要student自己+verl的RL训练管线。如果这一步跑不通，问题一定出在"verl装得对不对/GPU配置对不对/Qwen3.5-4B能不能被这套代码正常加载"这些基建层面，和OPRD的蒸馏机制本身没关系，值得先单独排除。

## 3. Phase 2：Token-level OPD baseline

```bash
export ACTOR_MODEL_PATH=${MODEL_DIR}/Qwen3.5-4B
export REWARD_MODEL_PATH=${MODEL_DIR}/Qwen3.5-9B   # 见下方说明
USE_REP_DISTILLATION=False bash /remote-home/ziyesong/OPRD/on_policy_distillation.sh
```

**一个容易搞混的命名,提前说清楚**：脚本里叫`REWARD_MODEL_PATH`/`REWARD_MODEL_NAME`的这个变量,在OPRD/OPD这套代码里,实际扮演的是**teacher**的角色(继承自上游OPD代码库的命名习惯,因为它的作用是给student生成的token打分,架构上和"reward model"的调用方式一样)，不是真的reward model，执行的时候不要被名字误导。

**这一步的目的**：跑通"teacher给student的on-policy rollout打分、按token级别log-prob做蒸馏"这条路径（`LOG_PROB_TOP_K`控制top-k），作为和Phase 4（表示级别蒸馏）对比的baseline——README里论文的核心卖点之一就是"比token-level OPD快1.44倍、省54%显存"，没有这个baseline就没法比。

## 4. Phase 3：OPRD-Bridge——真正要复现的机制，三阶段

### 3.1 Stage 0：On-Policy Pair Collection

```bash
bash /remote-home/ziyesong/OPRD/scripts/analysis/run_cross_arch_analysis.sh
```
Student（4B）用共享prompt生成on-policy回复，teacher（9B）和student各自跑一遍forward，把hidden state存成`on_policy_pairs.jsonl`。

### 3.2 Stage 1：Bridge Construction

```bash
bash /remote-home/ziyesong/OPRD/scripts/analysis/run_cross_arch_preexp2.sh
```
用Stage 0存下来的pairs，算teacher hidden state协方差的top-r主成分（$P_T$），训一个student投影$P_S$去最小化在这个共享r维子空间里的重建误差，两者都冻结，产出bridge（`ps_bank.pt`）。

**参数需要定**：`REP_LOW_RANK`默认cross-architecture是8——这个是bridge的"带宽"，rank越大保留的信息越多但监督越不精确，建议先用默认值8跑一版，不要一上来就调。

### 3.3 Stage 2：Distillation（用冻结的bridge实际训练student）

```bash
export REP_PROJECTOR_MODE=low_rank
export REP_LOW_RANK=8
export REP_LOW_RANK_INIT_CHECKPOINT=<Stage 1产出的ps_bank.pt路径>
export REP_FREEZE_PS=True
bash /remote-home/ziyesong/OPRD/low_rank_rep_distillation.sh
```

## 5. Phase 4：复现保真度检查——论文的具体数字claim站不站得住

```bash
export ACTOR_UPDATE_MEM_PROFILE=1
# 分别跑一遍 rep-only (Phase 3) 和 token-level OPD (Phase 2)，对比：
# mem/actor_update_peak_alloc_GB, mem/actor_update_delta_peak_GB, mem/actor_update_peak_reserved_GB
```
论文claim"1.44倍更快、最多省54%显存"是在他们自己的模型规模/硬件上测的，我们这里模型规模不同（4B/9B vs 论文可能用的其他配置）、硬件也不同（4090+A100 vs 论文的硬件）——**不需要精确复现这两个数字，但方向应该一致**（rep-only确实比token-level OPD省显存、收敛更快），如果方向都对不上，说明复现本身有问题，需要先排查，不要直接怀疑是我们规模小的原因。

## 6. Phase 5：验证——推理准确率有没有真的提升

```bash
cd /remote-home/ziyesong/OPRD/scripts/val/eval
python gen_vllm.py   # 设置 MODEL_NAMES 和 workers
python grade.py
```
对比 GRPO(Phase1) / Token-OPD(Phase2) / OPRD-Bridge(Phase3) 三者在AIME24/AMC23等测试集上的准确率——这是判断"表示级别蒸馏是不是真的比token级别蒸馏效果更好"这个核心论文claim在我们的4B/9B配置下站不站得住的最终标准，前面几个阶段都是为这一步做准备。

## 7. Phase 6（待定）：视频适配

这一阶段还没有具体设计，正在这个会话里继续讨论——核心方向是"用视觉/运动信息设计一个针对读出层的机制，不是简单照搬OPRD-Bridge的PCA投影"。等设计定下来会补充成独立文档，不阻塞前面6个阶段先执行。

## 8. 待办 / 需要负责实验的会话确认的事项（汇总）

1. ~~确认torch装cu126/cu128/cu129哪个变体~~ → **已解决**：`lmmseval`环境实测`torch 2.10.0+cu128`可用，直接装`torch==2.8.0+cu128`。
2. ~~确认`verl`/`verl2`两个现有环境的verl版本，决定是否需要新建`oprd_repro`环境~~ → **已解决**：直接新建了专门的`oprd_repro`环境，没有复用/动现有的`verl`/`verl2`。
3. ~~Phase 0装完后pip报的4条依赖冲突（outlines/numba/mistral-common/decord）是否要处理~~ → **已解决**（见§1.3）：逐条实测排查完，全部不影响Phase 1-5，没有改动numpy版本。
4. 确认`/root/models/`下的模型有没有持久化重下载脚本（`/root`会被容器重建清空）。
5. **新增**：`oprd_repro`环境本身也在`/root`下（见§1.3末尾），目前没有重建脚本——如果容器重建导致环境丢失，照着§1.2的命令手动重新跑一遍即可，暂不需要单独写自动化脚本，除非中途真的遇到容器重建。
6. 确定实际用几张卡、哪几张（4090 vs A100），改掉脚本里硬编码的`n_gpus_per_node=8`。
7. Phase 3 Stage 1 的`REP_LOW_RANK`，先用默认值8起步。
8. `on_policy_distillation.sh`里的`TEST_DATASET`默认只测AMC23一个集，其余（AIME24/AIME25/MATH-500等）被注释掉了，跑Phase 5验证时按需要打开。

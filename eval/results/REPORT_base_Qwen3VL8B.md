# Base 评测报告 — Qwen3-VL-8B-Instruct（实验阶梯 arm (a) = 地板）

- 日期：2026-07-01
- 模型：`/remote-home/ziyesong/models/Qwen3-VL-8B-Instruct`（未训练，intrinsic 能力）
- 目的：plan.md §4 的 arm (a) student-base 地板，作为后续 teacher / student-distill / student-direct 的对照基线。

## 配置（关键决策）
| 项 | 值 | 说明 |
|---|---|---|
| 推理后端 | **transformers `Qwen3VLForConditionalGeneration`** | ⚠️ vLLM 0.8.5 未收录 `qwen3_vl` 架构，无法用；transformers 4.57.6 原生支持 |
| 并行 | 4×RTX4090 数据并行（manifest 按 shard 均分） | 单卡 bf16 ~16.5G，batch 4 时显存 ~19G/24G |
| 帧数 | 16 帧（全 benchmark 统一） | plan §3.4：短视频 8–16 帧饱和 |
| 分辨率 | `max_pixels = 256×28×28`（~448px, ~1500 视觉tok/16帧） | 平衡精度与速度 |
| 解码 | greedy（do_sample=False），max_new_tokens=32，直答无 CoT | 测 intrinsic，MCQ 只需选项字母 |
| ⭐时间戳 | **传 `video_metadata` 给 processor** | Qwen3-VL 会把 `<t.t seconds>` 插进 prompt；不传则 fps 默认 24、11s 片段被压成 0.6s，严重误导时序任务 |
| 环境 | conda `qwen3vl`；`env=qwen-vl-utils 0.0.14 + decord` 抽帧 | |

## 总览结果
| Benchmark | 主指标 | 值 | 样本 | 对照（发表/同级）| 判读 |
|---|---|---|---|---|---|
| **MVBench** | mean-of-tasks acc | **0.6637** | 3800（19任务）| Qwen2.5-VL-7B ~0.69 | ✅ 同量级，可信 |
| **TemporalBench-short** | Binary Acc | **0.6839** | 9867 | GPT-4o binary ~0.70 | ✅ 接近闭源，可信 |
|  | Multiple-Binary Acc（按视频全对） | 0.2827 | 2179视频 | GPT-4o MBA ~0.16 | ✅ 高于 GPT-4o |
| **TemporalBench-long** | Binary Acc | **0.6514** | 5485 | — | ✅ 略低于 short，合理 |
|  | Multiple-Binary Acc | 0.2573 | 1574视频 | — | |
| **TOMATO** | acc | **0.3464** | 1484 | GPT-4o ~0.31 / 人类 ~0.95 | ✅ 极难基准，8B>GPT-4o，可信 |

> 预测分布非退化（不总选同一字母），pred_None=23（媒体/解码错误，<0.1%），MVBench media_errors=18（STAR 14 缺失视频 + 少量解码失败），TOMATO/TB media_errors=0。

## MVBench 逐任务（19，`fine_grained_pose` 因缺 NTU 视频跳过）
| 任务 | acc | | 任务 | acc |
|---|---|---|---|---|
| moving_attribute | 0.930 | | object_interaction | 0.720 |
| scene_transition | 0.925 | | action_prediction | 0.700 |
| object_existence | 0.860 | | action_sequence | 0.695 |
| unexpected_action | 0.845 | | counterfactual_inference | 0.675 |
| action_antonym | 0.775 | | moving_count | 0.670 |
| character_order | 0.730 | | moving_direction | 0.615 |
| state_change | 0.725 | | episodic_reasoning | 0.565 |
| action_count | 0.480 | | action_localization | 0.440 |
| fine_grained_action | 0.455 | | egocentric_navigation | 0.400 |
| object_shuffle | 0.405 | | | |

- 强项：静态属性/场景切换/存在性（0.86–0.93）；弱项：细粒度动作、时序定位、egocentric、object_shuffle（0.40–0.48）——**正是感知/时序类，与本项目要提升的能力吻合**。

## TemporalBench 细分
**short — per-category**：Others 0.802 / Action Type 0.716 / Event Order 0.696 / Motion Magnitude 0.678 / Action Effector 0.635 / Action Order 0.612 / Motion Direction 0.590 / **Action Frequency 0.566（最弱）**
**short — per-dataset**：Movie_Description 0.742 / ActivityNet 0.731 / Oops 0.719 / EgoExo4D 0.712 / COIN 0.682 / Charades 0.678 / **FineGym 0.539（体操细粒度最难）**
**long — per-dataset**：ActivityNet 0.666 / EgoExo4D 0.663 / COIN 0.658 / Charades 0.630 / FineGym 0.557

## TOMATO 逐 reasoning-type
direction 0.454（最好）/ visual_cues 0.357 / shape&trend 0.336 / count 0.329 / velocity&frequency 0.329 / **rotation 0.231（最差）**
- 全类都低（0.23–0.45），旋转方向/计数几乎接近随机——**合成运动感知正是 SynRL 训练要攻的短板，delta 空间大**。

## 与 SynRL 预期增益的关系（go/no-go 参考）
plan §3.3 预期训练后：accuracy +1.7~4.9、grounding +8~12.6。当前 base 作为地板：
- **TOMATO 0.346 / TemporalBench 0.65–0.68 / MVBench 0.664**，均有明确上行空间；
- 弱项集中在细粒度时序 / 合成运动（FineGym、action_localization、rotation、Action Frequency），与合成 grounded-CoT 训练目标高度对齐，**方向信号可观测**。

## 复现
```bash
conda activate qwen3vl
cd /remote-home/ziyesong/videoPerception/eval
bash run_all.sh          # 4卡DP跑 mvbench + temporalbench_short/long
# tomato 单独：for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python run_eval.py --benchmark tomato --num-shards 4 --shard $g --out results/tomato.shard$g.jsonl & done
python score.py          # 合并分片 -> results/scores.json
```

## 遗留 / 备注
- **caption 子任务**（TemporalBench short_caption 1891）未评：需 GPT judge 打分，无 API 暂跳过。
- **数据落盘**：MVBench 视频解压在 `/root/benchmarks/MVBench_video`、TemporalBench 在 `/root/benchmarks/TemporalBench`、TOMATO 在 `/root/benchmarks/TOMATO_lmms`（均 overlay 易失盘，容器重建需重新解压/下载；原始 zip：MVBench/TB 见各自目录，TOMATO 走 `lmms-lab/TOMATO` hf-mirror 重下）。
- **TOMATO 视频来源**：原 HF `yale-nlp/TOMATO` 不含视频（只在被墙的 Google Drive）；改用 **`lmms-lab/TOMATO`**（HF-mirror 可直连，parquet 自带 `video_path`，3 zip ~11.5G）。

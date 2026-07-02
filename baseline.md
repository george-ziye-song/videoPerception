核验完毕。所有关键事实都取证了。下面是**最完整、不粉饰**的汇报，包含和官方口径的偏差——这些偏差意味着我的绝对分数**不能直接当官方 leaderboard 数字引用**，只能当作"内部一致、量级正确"的基线。

---

# 一、一句话诚实定性

我**没有用现成评测框架**（VLMEvalKit/lmms-eval），而是**自己写了一个 transformers 推理+打分脚本**。分数经过"对照已发表数字在合理量级内"的 sanity check，但**未与官方 harness 逐位对齐**。它的价值是：**同一套脚本跨 arm 复用时，base 与 student 的对比是可信的**；绝对值不要当作"Qwen3-VL-8B 的官方 MVBench 分"去引用。

---

# 二、框架 / 模型 / 环境

| 项                  | 事实                                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **模型**            | `/remote-home/ziyesong/models/Qwen3-VL-8B-Instruct`（Instruct 版，**未训练**，测 intrinsic），bf16，flash-attention-2 |
| **框架**            | ❌ 非 VLMEvalKit/lmms-eval（都没装）；✅ **自写** `videoPerception/eval/run_eval.py`（~270 行）                       |
| **推理后端**        | **transformers 4.57.6 `Qwen3VLForConditionalGeneration.generate()`**                                                  |
| **为什么不用 vLLM** | 实测 `vLLM 0.8.5.post1` 的模型注册表里**没有 `Qwen3VL` 架构**（模型太新，vLLM 是 2025-04 的），只能走 transformers    |
| **并行**            | 4×RTX4090，**数据并行**：4 个进程各占 1 卡，manifest 按 `[shard::4]` 均分，各写各的 jsonl，最后合并                   |
| **抽帧**            | `qwen-vl-utils 0.0.14` + `decord`；conda 环境 `qwen3vl`                                                               |
| **解码**            | **greedy**（`do_sample=False`，已核实），`max_new_tokens=32`，**直答、无 CoT**                                        |
| **每帧**            | **16 帧**；`max_pixels=256×28×28`（≈448px，实测每样本输入 ~1200–1600 token）                                          |

---

# 三、数据：怎么下、怎么解压（逐 benchmark）

**通用姿势**（CLAUDE.md 规则）：`env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com hf download ...`（去代理直连 + HF 走国内镜像）。解压一律用 **Python `zipfile`**（这台机器没装 `unzip`），且**解压到 `/root`**（overlay 本地盘，无配额、无 GPFS ×2 翻倍；代价：容器重建即清空）。

| Benchmark         | 下载源                                           | 原始位置(GPFS持久)                                      | 解压目标(/root易失)                               | 解压后                                       |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------- |
| **MVBench**       | `OpenGVLab/MVBench`                              | `data/benchmarks/MVBench/`（12 个 video zip + 20 json） | `/root/benchmarks/MVBench_video/`                 | star/clevrer/perception/tvqa(帧)…            |
| **TemporalBench** | `microsoft/TemporalBench`（gated，前一会话下的） | `/root/benchmarks/TemporalBench/`（6 zip + 4 parquet）  | 就地解压 → `short_video/`、`long_video/<数据集>/` | 与 parquet 的 `video_name` 对齐              |
| **TOMATO**        | ⚠️见下                                           | —                                                       | `/root/benchmarks/TOMATO_lmms/`（3 zip ~11.5G）   | `videos/{human,object,simulated}/`，1417 mp4 |

**TOMATO 的坑（实话）**：原计划的 `yale-nlp/TOMATO` 这个 repo **根本不含视频**（只有 QA parquet + 几张示例 png），视频只在**被墙的 Google Drive**（实测 http=000 连不上）。我改用了 **`lmms-lab/TOMATO`**——它的 parquet 直接带 `video_path` 字段、视频打包成 3 个 zip 放在 HF，走 hf-mirror 能直连。所以**你不用手动下**，但要知道：**TOMATO 用的是 lmms-lab 重新打包的版本**，不是 yale 原版。

解压脚本核心（`eval/extract_videos.py` / `get_tomato.sh`）就是 `zipfile.ZipFile(z).extract(...)` 循环，完事写个 `.extract_done` 标记。

---

# 四、每个 benchmark 的完整流程（附真实逐样本轨迹取证）

**统一流程**：读 parquet/json → 建 manifest（视频路径 + 问题 + 选项字母化 + GT字母）→ 组 Qwen 消息 `[{video}, {text}]` → **`process_vision_info(..., return_video_metadata=True)` 抽16帧并带回真实时间戳** → processor 把 `<t.t seconds>` 时间戳插进 prompt → greedy 生成 → 正则解析首个选项字母 → 与 GT 比对。

**⭐时间戳修复（最关键的一处）**：Qwen3-VL 是时间戳感知模型，会在 prompt 里插 `<秒数>`。如果不把真实 `video_metadata` 喂给 processor，它会 fps 默认 24，把一个 11 秒的视频当成 0.6 秒——时序任务全被误导。我专门修了这个。下面轨迹里"TIMESTAMPS the model saw"就是证据。

### 真实轨迹（我刚重新跑的，非记忆）

**MVBench / action_sequence[0]** — `ZS9XR.mp4`
```
问题: What happened after the person took the food?
(A) Ate the medicine. (B) Tidied up the blanket. (C) Put down the cup... (D) Took the box.
时间戳: 1.5, 7.4, 13.3 ... 42.8   ← 覆盖整段 ~43s
原始输出: '(D) Took the box.'   解析: D  | GT: A   （答错）
```

**TemporalBench-short[0]** — 二选一"哪个 caption 更准"
```
A. ...run over ... five times.   B. ...run over ... three times.
时间戳: 0.4, 1.9 ... 10.9   ← 11.26s 片段，正确
原始输出: 'B'   解析: B | GT: A   （答错）
```

**TOMATO[0]** — 5 选 1 手部移动方向
```
时间戳: 0.2 ... 5.4   原始输出: '(E) First to the left then to the right.'  解析: E | GT: C
```

### 打分口径（`eval/score.py`）
- **MVBench**：per-task acc（每任务恰好 200 题，已核实 19×200=3800、无重复 uid）→ mean-of-tasks（=micro，因等样本量）。
- **TemporalBench**：**Binary Acc**（逐题）+ **Multiple-Binary Acc / MBA**（按 `video_name` 分组，一个视频下所有题**全对**才算对；实测 short 2179 视频、每视频 1–14 题、均值 4.53）——这符合 TemporalBench 官方 MBA 定义。
- **TOMATO**：overall acc + per reason_type。

---

# 五、结果（完整）

| Benchmark           | 主指标            | 值                  | 样本            |
| ------------------- | ----------------- | ------------------- | --------------- |
| MVBench             | mean-of-tasks acc | **0.6637**          | 3800（19 任务） |
| TemporalBench-short | Binary / MBA      | **0.6839 / 0.2827** | 9867 / 2179视频 |
| TemporalBench-long  | Binary / MBA      | **0.6514 / 0.2573** | 5485 / 1574视频 |
| TOMATO              | acc               | **0.3464**          | 1484            |

MVBench 逐任务（高→低）：moving_attribute .93 / scene_transition .925 / object_existence .86 / unexpected_action .845 / action_antonym .775 / character_order .73 / state_change .725 / object_interaction .72 / action_prediction .70 / action_sequence .695 / counterfactual .675 / moving_count .67 / moving_direction .615 / episodic_reasoning .565 / action_count .48 / fine_grained_action .455 / action_localization .44 / object_shuffle .405 / egocentric_navigation .40。

预测分布非退化（不总选同一字母），unparseable 仅 23（<0.1%），MVBench 18 个媒体错误（14 个 STAR 视频两 zip 都没有 + 少量解码失败，**计为答错**）。

---

# 六、和官方口径的偏差（你要严格核查，重点看这里）

这些是**实话**，会让我的分数偏离官方 leaderboard：

1. **不是官方 harness**：自写脚本，prompt 措辞、答案解析都是我自己的（官方 VLMEvalKit 有它自己的 prompt 和抽取器）。
2. **MVBench 没做时间裁剪**：官方对 4 个带 `start/end` 的任务（action_sequence/prediction/object_interaction/action_localization）会把视频裁到标注区间；**我没裁，喂的是整段视频**（轨迹里 1.5→42.8s 是整段，而该题标注区间只有 1.5–17.1s）。这大概率**压低**了这几个任务的分。
3. **MVBench 少 1 个任务**：`fine_grained_pose`（NTU RGB+D）视频有版权、官方只给下载清单没给视频，我们没有 → **跳过，只跑 19/20**。
4. **分辨率压过**：`max_pixels=256×28×28`（每样本 ~1.2–1.6k token），比模型默认（~4.7k token/672×896）低。省时间省显存，但**细粒度任务（如 TOMATO 旋转/计数）可能被压低**。
5. **固定 16 帧**：长视频（TemporalBench-long 最长到分钟级）也只采 16 帧，会丢时序细节。
6. **TemporalBench 指令行重复**：parquet 的 `question` 末尾已有"Answer with the option's letter..."，我又追加了一句 MCQ 指令 → prompt 里出现两遍。纯冗余、不改变答案分布，但确实是个瑕疵。
7. **TemporalBench caption 子任务（1891）没评**：需要 GPT judge 打分，无 API，跳过。
8. **TOMATO 用 lmms-lab 重打包版**，非 yale 原版；16 帧、我的 prompt。
9. **只做 sanity check，未跑官方 harness 对拍**：我用"MVBench≈Qwen2.5-VL-7B 的 0.69 量级、TemporalBench binary≈GPT-4o 的 0.70、TOMATO≈GPT-4o 的 0.31"判断量级正确，**但没有逐位复现官方数字**。

**结论**：这套数字作为**arm (a) 地板**、并用**同一脚本**去比 teacher/student 的 delta 是站得住的；但**别把它当"Qwen3-VL-8B 官方分"对外引用**。如果你要可对外的官方分，我需要装 VLMEvalKit、补时间裁剪/全分辨率/NTU 视频重跑。

---

# 七、怎么自己复核

```bash
conda activate qwen3vl && cd /remote-home/ziyesong/videoPerception/eval
python score.py                      # 重新合并分片算分 -> results/scores.json
# 逐样本原始记录（可抽查任意一条的 gt/pred/raw）：
head results/mvbench.shard0.jsonl results/temporalbench_short.shard0.jsonl results/tomato.shard0.jsonl
```
- 代码：`run_eval.py`（推理+解析）、`score.py`（指标）、`run_all.sh`（4卡DP）
- 报告：`results/REPORT_base_Qwen3VL8B.md`、`results/scores.json`
- 逐样本：`results/*.shard*.jsonl`（每行含 uid/gt/pred/correct/raw）

要我把上面偏差里最影响分数的两条（**MVBench 时间裁剪** + **全分辨率**）补上重跑，还是先推进 plan §6 造数据？你定。
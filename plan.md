---
type: implementation-spec
created: 2026-06-22
language: zh-CN
note: 初步计划(合成视频→grounded CoT→off-policy SFT student→测感知)的实验/实现细节。配 [[实验总计划-cn]] §2。算力:4×4090(24G,可临时升8) + 3×A100(40G);base = Qwen3-VL-8B。走一步看一步,只写这一步。
hardware: 4×RTX4090-24G(可升8) + 3×A100-40G
---

# 初步计划 — 实现细节(实验维度)

> 目标:验证 **off-policy SFT on 合成 grounded CoT 能否提升 student 的 intrinsic perception 并迁移到真实视频**。

## 1. 模型清单
| 角色 | 模型(HF repo) | 用途 | 备注 |
|---|---|---|---|
| **student(主)** | `Qwen/Qwen3-VL-8B-Instruct` | 被提升的对象;off-policy SFT | 8B 是社区主流 base;Instruct 起 |
| **teacher(训出强感知)** | `Qwen/Qwen3-VL-8B-Instruct` 起(资源够可上 32B) | **在合成数据上训**(SFT 直答 → 可选 +GRPO)→ 真获得感知;训完当蒸馏源 | **验证**:训后 VIUBench 由弱变强(SynRL 19.5→51.9)才算"强感知 teacher";teacher≥student 才有得蒸 |
| student(调试) | `Qwen/Qwen3-VL-4B-Instruct` | 管线冒烟 / 显存红线备胎 | |
| 评测锚 | base 8B + 引用 VideoSSR/SynRL 发表数字 | VIUBench/真实集对照 | 不复测闭源 |

> 32B teacher 在 40G 卡上:bf16 ~64GB → **2×A100 TP** 或 **AWQ/FP8 单卡**;嫌重可先用 `Qwen3-VL-8B-Thinking` 当 CoT 生成器(单卡,质量稍弱,够探针)。

## 2. 软件 / 环境
| 组件 | 选择 | 说明 |
|---|---|---|
| 合成视频生成 | **SynRL repo** `repos/Synthetic-Video`(已 clone) | 10 个生成器;deps:matplotlib(Agg)/numpy/PIL/cv2/imageio/tqdm;纯 CPU 渲染 |
| teacher CoT 生成 | **vLLM** 服 32B + 自写脚本(喂 帧+metadata+answer → CoT) | 一次性;`prompt_logprobs` 不需要,纯生成 |
| student SFT | **LLaMA-Factory**(VLM SFT 最省心)或 ms-swift / veRL-SFT;**LoRA** | qwen-vl-utils + decord 抽帧 |
| 评测 | **VLMEvalKit / lmms-eval** + VIUBench(VideoSSR repo `lcgysl/VideoSSR`) | VIUBench 若未放码 → 按其 3 pretext(anomaly/counting/jigsaw)自重建 |
| 记账 | wandb | |

## 3. 数据集怎么造(已读 SynRL 代码,具体)
`repos/Synthetic-Video` = **10 个生成器,纯程序化、自包含**。

**3.1 跑生成器 → 视频 + 两个 jsonl(repo 自产)**
- 命令:`python 01_atomic_motion.py`(默认 `samples_per_type=1500`,改小即探针);或 `python run_all_examples.py` 一键跑全 10 类各 1 条做 sanity。
- 渲染:matplotlib(Agg)+cv2,512×512 / 30fps / 黑底,CPU 并行(`max_workers`)。
- 覆盖:**短时感知**(01/02:bouncing-count / direction / trajectory-shape / directional-event-count / accel-decel / rotation-dir / rotation-count / speed)+ **长时认知**(03-10:shell game / sliding puzzle / card pile / chip cup / terminal tracking / trajectory-math / grid tracking / long-term)。
- 每条产:
  - **`sft.jsonl` = 直答 QA**(已 message 格式,可直接 SFT):`{id, task, messages:[system, user:[{type:video,...},{type:text, MC问题}], assistant:"C"]}` —— **只有答案字母,无 CoT**。
  - **`metadata.jsonl` = 富 GT**(造 CoT 的原料):`{answer, ground_truth_details:{question_data, video_events_timeline_ms(逐事件毫秒时间线), other_details(形状/颜色…)}}` —— **关键 = frame-level 事件时间线,可验证、可条件化**。
- 规模(探针):每类 ~500–1000 → 总 **~5–8k**(SynRL 全量 7.7K,探针够用)。

**3.2 grounded CoT(repo 不含,要自建)**
- 自写脚本:`(视频帧 + metadata 的 timeline + question + answer)` → VLM 生成"逐时间戳追踪"CoT → 用 timeline/answer **核对**(不过则重生 ≤5 次,仍不过丢弃)→ filter 后 ~5–7k 高质 CoT。(= SynRL Stage 2 generate→verify→reflect→polish。)
- **用途**:① 训 teacher(比直答更强);② teacher 产的 CoT 当蒸 student 的监督料。
- (可选)混 ~10–15% 真实视频**直答**(LLaVA-Video)防分布漂移 / 防 video-SFT 退化 spatial(VideoSFT temporal-trap)。

**3.3 评测集**:**主(感知)= VIUBench**;**真实迁移 = MVBench / TemporalBench / RexTime / Video-MME**。

## 4. 实验阶梯(teachability:感知能不能被学到)
| arm | 是什么 | 目的 |
|---|---|---|
| (a) student-base | 不训 | 地板 |
| **teacher** | 在合成上训**强配方**(SFT on **grounded CoT**,可选 +GRPO) | **天花板** + 坐实"teacher 有强感知"(VIUBench 验证) |
| **(b) student-distill**(主探针)| student off-policy SFT on **teacher 生成的 grounded CoT** | **核心:感知能不能被 student 学到** |
| (c) student-direct | student SFT on **合成直答 QA**(`sft.jsonl` 原始标签,不经 teacher CoT) | 对照:"经 teacher 蒸" vs "直接从原始数据学" |
> ⚠ 三者监督形式必须互不相同(否则 teacher≡student-direct,退化):teacher 学 CoT(+RL)、direct 学原始标签、distill 学 teacher 的 CoT。
| (可选 d) +GRPO | (b) 上加 GRPO | 下一步,资源够再做 |

**待你定的一个设计点**:teacher 给 student 的监督**生成在什么数据上**?① 合成视频(干净、隔离 teachability)；② 真实视频(teacher 用感知去标真实,更接近"迁移",但混入"合成→真实泛化"的混淆)；③ 两者混。**建议先①(最干净,纯测可教性),再②看迁移。**

**go/no-go**:(b) student-distill 在 **VIUBench 显著 > (a) 且接近 teacher**、真实集不退 → **感知可教** → 和师姐沟通定下一步;若 student-distill ≈ base → 感知不易教 / off-policy 不够(需 on-policy 或 RL)。

## 5. 计算映射(贴你的卡)
| 步骤 | 跑在哪 | 说明 |
|---|---|---|
| 合成视频生成 | **CPU**(任意机) | matplotlib 渲染,并行,无 GPU |
| teacher CoT 生成 | **vLLM @ A100**(32B 用 2×40G TP 或 AWQ/FP8 单卡;或 8B-Thinking 单卡) | **一次性**,生成完即释放 |
| **student off-policy SFT** | **3×A100-40G**(舒适)或 **4×4090-24G**(LoRA+grad-ckpt+ZeRO-3,偏紧) | **最轻训练**:无 rollout、无 teacher 共驻、纯 SFT |
| 评测 | 单卡(4090/A100)vLLM | |

**显存预算(8B LoRA-SFT)**:大头是**视频 token**(16 帧 × ~200–260 tok ≈ 3–4k)≫ 文本。
- 设定:**训练 16 帧 / max_pixels 限单帧 / response ≤1024 / LoRA(r=16–32) + grad-ckpt**。
- 4090-24G:ZeRO-3 + 上述 → 可行但紧;OOM 则降到 8 帧 / batch=1 + grad-accum / 用 4B 调试。
- A100-40G ×3:全程宽松,可放 32 帧 / 更大 batch。

## 6. 执行步骤(checklist)
- [ ] 环境:clone SynRL(已) + VideoSSR(取 VIUBench);装 LLaMA-Factory + vLLM + qwen-vl-utils/decord;WebSearch torch-cu 兼容矩阵再装
- [ ] 跑 10 生成器 → ~5–8k 合成视频 + metadata + QA;抽查渲染正确
- [ ] vLLM 起 teacher → 脚本产 grounded CoT → verify/filter → ~5–7k SFT corpus(+可选 10–15% 真实直答)
- [ ] (b) student LoRA-SFT;(c) direct-SFT;(a) base 直接评
- [ ] 评测三 arm:VIUBench(主)+ MVBench/TemporalBench/RexTime/Video-MME
- [ ] go/no-go 判定 → 写一页结果 → 和师姐沟通

## 7. 风险 / 预案
| 风险 | 预案 |
|---|---|
| VIUBench 未放码 | 按其 3 pretext(anomaly/counting/jigsaw)用同类合成自重建评测集 |
| 32B teacher 服不动 | AWQ/FP8 或退 8B-Thinking 当 CoT 生成器 |
| 合成→真实不迁移 | 混真实直答数据;SynRL 已证可迁移,概率低 |
| off-policy SFT 过拟合合成风格 / 退化真实(temporal-trap) | 混真实数据 + **真实集必评**;若退化 → 这本身是"off-policy 不够、需 RL"的证据 |
| 4090 显存 | 8 帧 + LoRA + grad-accum;或借升 8 卡 / 上 A100 |

> 只写这一步。完成后和师姐沟通再定下一步(是否上 GRPO、novelty 在哪)。

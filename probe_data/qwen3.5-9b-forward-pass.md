---
title: "Qwen3.5-9B 完整forward流程 —— 从输入到hidden state抽取"
date: "2026-07-14"
status: "全部shape均为实际运行验证,不是从文档/代码推断的理论值(验证脚本见文末)"
---

# Qwen3.5-9B 完整forward流程

本文档以`probe_data/Atomic/videos/Complex_Direction_Identification_0007.mp4`（16帧、512×512）为具体样例，逐步展示输入到输出每一步的真实shape。所有数字来自2026-07-14在GPU4上实际执行`model(**inputs, output_hidden_states=True)`得到的结果，不是理论推算。

## 0. 模型整体结构（来自`AutoConfig`实际读取，非猜测）

```
Qwen3_5ForConditionalGeneration
├── model: Qwen3_5Model
│   ├── visual: Qwen3_5VisionModel          (视觉塔,处理pixel_values_videos)
│   │   ├── patch_embed: Qwen3_5VisionPatchEmbed   (Conv3D-like patchify)
│   │   ├── blocks: 27 × Qwen3_5VisionBlock         (depth=27,每块=Attention+MLP)
│   │   └── merger: Qwen3_5VisionPatchMerger        (1152维 → 4096维,并做2×2空间下采样)
│   └── language_model: Qwen3_5TextModel     (文本塔,32层混合注意力decoder)
│       └── layers: 32 × Qwen3_5DecoderLayer
└── lm_head: Linear(4096 → 248320)           (tie_word_embeddings=False,独立权重)
```

**关键配置数字**（`text_config`/`vision_config`实际字段）：

| 模块 | 字段 | 值 |
|---|---|---|
| 文本塔 | num_hidden_layers | 32 |
| 文本塔 | hidden_size | 4096 |
| 文本塔 | intermediate_size(MLP) | 12288 |
| 文本塔 | num_attention_heads | 16 |
| 文本塔 | num_key_value_heads(GQA) | 4 |
| 文本塔 | head_dim | 256 |
| 文本塔 | vocab_size | 248320 |
| 文本塔 | hidden_act | silu(SwiGLU MLP) |
| 文本塔 | rms_norm_eps | 1e-6 |
| 视觉塔 | depth | 27 |
| 视觉塔 | hidden_size | 1152 |
| 视觉塔 | out_hidden_size(投影后) | 4096 |
| 视觉塔 | patch_size | 16 |
| 视觉塔 | spatial_merge_size | 2 |
| 视觉塔 | temporal_patch_size | 2 |
| 特殊token | image_token_id / video_token_id | 248056 / 248057 |
| 特殊token | vision_start_token_id / vision_end_token_id | 248053 / 248054 |

**⚠️ 重要架构细节（容易被忽略）：文本塔32层不是清一色的标准self-attention，是"3层linear attention + 1层full attention"循环8次的混合架构**，来自`config.text_config.layer_types`实际读取：

```
[linear, linear, linear, FULL] × 8   (共32层)
```

即第4,8,12,...,32层(1-indexed,对应下面hidden_states的index 4,8,12,16,20,24,28,32)是标准self-attention(`Qwen3_5Attention`)，其余24层是**门控delta网络**(`Qwen3_5GatedDeltaNet`，线性注意力/类SSM机制，`linear_num_value_heads=32, linear_key_head_dim=128`)。这解释了运行日志里那条警告：`[transformers] The fast path is not available... install flash-linear-attention / causal-conv1d`——这两个包是给`GatedDeltaNet`用的高效kernel，没装就退化成慢速`torch`实现（本次实验用的就是慢速回退路径，正确性不受影响，只是速度慢一些）。

---

## 1. 输入构造（chat template + 视觉预处理）

```python
message = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": [
      {"type": "video", "video": ".../Complex_Direction_Identification_0007.mp4", "nframes": 16},
      {"type": "text", "text": "question here"},
  ]},
]
text = processor.apply_chat_template(..., enable_thinking=False)
image_inputs, video_inputs, processed_video_kwargs = process_vision_info(..., return_video_kwargs=True, image_patch_size=16)
inputs = processor(text=text, videos=video_inputs, ...)
```

**实测shape**：

| 张量 | shape | 含义 |
|---|---|---|
| `input_ids` | `(1, 2139)` | 1条样本，2139个token（system+video占位符+question文本） |
| `pixel_values_videos` | `(8192, 1536)` | 见下方"视觉塔"一节的拆解 |
| `video_grid_thw` | `[[8, 32, 32]]` | T=8(时间组,已经是temporal_patch_size=2合并后), H=32,W=32(合并前的patch网格) |

`pixel_values_videos`的`8192`和`1536`不是随便两个数，是patchify公式算出来的：
- `8192 = T(8) × H(32) × W(32)`：视觉塔真正吃到的patch总数（合并前）。
- `1536 = patch_size(16) × patch_size(16) × in_channels(3) × temporal_patch_size(2)`：每个patch展平后的原始像素维度（`16×16=256`像素 ×3通道=768 ×2帧合并=1536）。

---

## 2. 视觉塔前向：`pixel_values_videos (8192, 1536)` → 视觉embedding

```
patch_embed (Conv3D-like线性层): (8192, 1536) → (8192, 1152)          # 1152=vision hidden_size
27 × Qwen3_5VisionBlock (Attention+MLP,维度不变): (8192, 1152) → (8192, 1152)
merger (Qwen3_5VisionPatchMerger,2×2空间下采样+投影,实际读代码验证的3步):
    (8192, 1152) → 每4个空间相邻patch拼成一组(2×2 spatial_merge) → reshape (2048, 4608)   # 4608=1152×2²
    → LayerNorm → Linear(4608→4608) → GELU → Linear(4608→4096) → (2048, 4096)
```

`merger`之后的`2048 = T(8) × Hm(16) × Wm(16)`，`Hm=H/spatial_merge_size=32/2=16`，这就是最终塞进文本塔的视觉token数量——和`input_ids`里`video_token_id`(=248057)出现的次数**实测精确相等**（2048次，不多不少）。

**⚠️ 关键细节（本次实验踩过的坑，已用代码验证）**：这2048个视觉token在`input_ids`序列里**不是连续一整块**。Qwen3.5会在每256个视觉token（=一个时间组T对应的256个token）之间插入一小段**时间戳文本token**，实测解码出来是这样的字面文本：

```
<|vision_end|>0.8 seconds<|vision_start|>
```

所以2048个视觉token实际被切成**8段、每段256个、段间隔着8个时间戳文本token**。这意味着抽取hidden state时不能按固定stride切片（会把时间戳文字的embedding也平均进去），必须用"连续run检测"（`find_video_runs()`，按`input_ids == video_token_id`找连续区间）。

---

## 3. 文本塔前向：视觉token替换 + 32层混合decoder

1. `input_ids (1, 2139)` 先过词嵌入表 → `inputs_embeds (1, 2139, 4096)`
2. 视觉塔算出的`(2048, 4096)`视觉特征，按`video_token_id`所在位置**原地替换**`inputs_embeds`里对应位置的embedding（`get_placeholder_mask`+`scatter`）
3. `get_rope_index`计算M-RoPE的3D位置编码（时间/高/宽三个维度各自的position_ids，多模态位置编码，Qwen2-VL系列同款机制）
4. 32层`Qwen3_5DecoderLayer`依次处理，每层：
   ```
   residual = x
   x = RMSNorm(x)
   x = 该层类型对应的token mixer(x)     # linear_attention用GatedDeltaNet, full_attention用标准GQA Attention
   x = residual + x
   residual = x
   x = RMSNorm(x)
   x = SwiGLU_MLP(x)                    # (4096→12288→4096)
   x = residual + x
   ```
   **每一层输入输出shape全程不变**：`(1, 2139, 4096)`。

**实测`output_hidden_states=True`拿到的33个张量**（index 0=词嵌入表输出/视觉token替换后，未经任何decoder层；index 1..32=经过第1..32层decoder后的输出）：

| hidden_states index | shape | 对应哪一层之后 |
|---|---|---|
| 0 | `(1, 2139, 4096)` | 词嵌入+视觉特征替换后，**未过任何transformer层** |
| 1 | `(1, 2139, 4096)` | 第1层(linear_attention)之后 |
| 8 | `(1, 2139, 4096)` | 第8层(**full_attention**)之后 |
| 16 | `(1, 2139, 4096)` | 第16层(**full_attention**)之后 |
| 24 | `(1, 2139, 4096)` | 第24层(**full_attention**)之后 |
| 29 | `(1, 2139, 4096)` | 第29层(linear_attention)之后 |
| 32 | `(1, 2139, 4096)` | 第32层(**full_attention**,最后一层)之后 |

最后：
```
x = final_norm(hidden_states[32])                     # (1, 2139, 4096)
logits = lm_head(x)                                    # (1, 2139, 4096) → (1, 2139, 248320)
```

---

## 4. probe实验怎么从这33个张量里抽hidden state

### 4.1 三层抽样版（`extract_hidden_states.py`，probe-experiment.md正式实验用的版本）

```python
shallow, mid, deep = round(0.25*32), round(0.5*32), round(0.9*32)   # = 8, 16, 29
```

对每个选中的层index，取该层`hidden_states[li][0]`（去掉batch维，得到`(2139, 4096)`），用`find_video_runs()`找到视觉token的8个连续区间，对每个区间做`mean(dim=0)`（每个时间组的256个视觉token → 1个`(4096,)`向量），8个时间组的向量再取一次`mean`，得到该层最终的单个`(4096,)`池化向量。3层各存一份，一条样本最终存3个`(4096,)`向量。

**⚠️ 如实说明一个巧合**：按25%/50%/90%取整算出来的8/16/29，恰好前两个(8,16)都落在**full_attention**层输出上，第三个(29)落在**linear_attention**层输出上——这不是刻意设计的，是取整结果和"每4层1个full_attention"的周期恰好碰到了这个组合，见下方§5的讨论。

### 4.2 全33层扫描版（`extract_all_layers.py`，这次专门为了回答"是不是U型"补做的）

不再只抽3层，对`hidden_states[0]`到`hidden_states[32]`每一层都做同样的池化，一条样本存一个`(33, 4096)`的张量。这样才能画出完整的逐层曲线，不是只看3个采样点。

---

## 5. 33层完整曲线长什么样，为什么mid经常比deep好（9B实测，Acceleration_Identification原语）

**分类probe（判断加速/匀速/减速，3类）**：第5层（16%深度）就已经100%准确率，一路到第32层都是100%——**这条曲线看不出U型**，是天花板效应：这个任务在很浅的层就线性可分了，后面根本没有下降空间可以观察。

**回归probe（预测连续速度标量，更敏感的度量）**：

```
层0(embedding):  R²=0.925
层1-4(线性/线性/线性/FULL):  0.944 → 0.956 → 0.958 → 0.976  (持续上升)
层5-8(线性/线性/线性/FULL):  0.980 → 0.978 → 0.982 → 0.982
层9-14:                       0.977 → ... → 0.985 (第14层,44%深度,全局峰值)
层15-21:                       0.982 → ... → 0.980  (高位平台期)
层22-32:                       0.977 → ... → 0.960 (第32层,持续下滑)
```

**这是一条清晰的"先升后降"驼峰曲线**：从embedding的0.925爬升，中间层（44%-66%深度区间）达到平台/峰值(~0.98-0.985)，然后**持续下滑**到最后一层的0.960。中间层比最后一层高出约2-3个百分点的R²——这就是"mid比deep好"这个现象的真实来源，不是3层采样的偶然。

**full_attention层是否能解释这个曲线的具体起伏**：检查过，**不能完全解释**。把8个full_attention层的输出index(4,8,12,16,20,24,28,32)单独拎出来看，有的确实比前一层(linear_attention输出)高（比如第4层0.976 vs 第3层0.958，明显跳升），但也有几乎持平甚至更低的（比如第32层0.960反而低于第31层0.964）。所以full/linear的交替**不是**曲线起伏的主要解释，真正主导曲线形状的是一个更大尺度的、跨越整个深度的趋势：**中间层达到峰值、深层持续衰减**。

**架构层面的合理解释（有实测曲线支持，但不是被证明过的定论）**：Qwen3.5是自回归decoder，每个位置在深层的表示，训练目标是预测"这个位置的下一个token"，不是服务某个下游任务。对视频token这种夹在长prompt中间的位置来说，"下一个token"往往只是下一个patch或者时间戳文字，跟"这个物体多快"这种语义没关系。越接近输出，表示就越可能被这个局部的next-token目标"特化"，把探针需要的通用语义信息挤掉一部分；中间层还没被这么强地拉向局部预测任务，反而保留了更多能被下游探针复用的信息。这个解释和其他transformer模型可解释性研究里"中间层表示更通用、末层表示更任务特化"的发现方向一致，但没有专门针对Qwen3.5这个具体模型验证过因果关系，是一个和实测曲线吻合、有理论依据的假设，不是定论。

---

## 6. 验证方法说明（可复现）

本文档所有shape和曲线数字来自以下脚本的实际执行，不是从代码/文档推断：
- `probe_data/test_extract_one.py`：验证video token布局(连续run检测)
- 2026-07-14现场跑的一次性forward脚本：验证`input_ids`/`pixel_values_videos`/`video_grid_thw`/`hidden_states`/`logits`的实际shape
- `probe_data/extract_all_layers.py` + `probe_data/analyze_all_layers.py`：33层完整probe曲线(9B, Acceleration_Identification/Rotation_Direction, 各200条样本)

#!/usr/bin/env python3
"""§4.1验证性测试:在1条合成样本上跑通"冻结前向+定位video token块+按temporal group pooling",
确认video_grid_thw × temporal_patch_size/spatial_merge_size的token计数假设成立,
再决定要不要在1400条上批量跑。不训练、不生成,只做一次forward。
"""
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

PRETRAINED = "/root/models/Qwen3.5-9B"
DEVICE = "cuda:4"

cfg = AutoConfig.from_pretrained(PRETRAINED, trust_remote_code=True)
video_token_id = cfg.video_token_id
print("video_token_id:", video_token_id, "image_token_id:", cfg.image_token_id)
print("temporal_patch_size:", cfg.vision_config.temporal_patch_size, "spatial_merge_size:", cfg.vision_config.spatial_merge_size)

processor = AutoProcessor.from_pretrained(PRETRAINED)
model = AutoModelForImageTextToText.from_pretrained(PRETRAINED, dtype=torch.bfloat16, trust_remote_code=True).eval().to(DEVICE)

message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "video", "video": "/remote-home/ziyesong/videoPerception/probe_data/Atomic/videos/Complex_Direction_Identification_0007.mp4", "nframes": 16},
        {"type": "text", "text": "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D...) of the correct option. What is the movement path of the object? Possible answer choices: A. First to the right, then to the left\nB. First to the up, then to the left\nC. First to the down, then to the right\nD. First to the down, then to the up\nThe best answer is:\n  "},
    ]},
]

text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, enable_thinking=False)
image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
    [message], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True,
)
video_metadata_list = None
if video_inputs is not None:
    video_inputs, video_metadata_list = map(list, zip(*video_inputs))

inputs = processor(
    text=text, images=image_inputs, videos=video_inputs, video_metadata=video_metadata_list,
    **processed_video_kwargs, do_resize=False, return_tensors="pt",
)
inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}

print("\ninput_ids shape:", inputs["input_ids"].shape)
print("video_grid_thw:", inputs.get("video_grid_thw"))
grid_thw = inputs["video_grid_thw"][0].tolist()
T, H, W = grid_thw
merge = cfg.vision_config.spatial_merge_size
Hm, Wm = H // merge, W // merge
print(f"T={T} H={H} W={W} (pre-merge patch grid); post-merge per-t tokens = {Hm}*{Wm}={Hm*Wm}; total={T*Hm*Wm}")

vid_mask = (inputs["input_ids"][0] == video_token_id)
n_vid_tokens = vid_mask.sum().item()
print("video tokens found in input_ids:", n_vid_tokens)
assert n_vid_tokens == T * Hm * Wm, "TOKEN COUNT MISMATCH — layout assumption wrong!"

vid_positions = vid_mask.nonzero(as_tuple=True)[0]
print("video token position range:", vid_positions.min().item(), "-", vid_positions.max().item())
is_contiguous = (vid_positions.max() - vid_positions.min() + 1) == n_vid_tokens
print("contiguous block:", is_contiguous.item())

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, use_cache=False)

n_layers = len(out.hidden_states)
print("\nnum hidden_states (incl. embedding layer):", n_layers)
for li in [1, n_layers // 2, n_layers - 1]:
    hs = out.hidden_states[li][0]  # (seq_len, hidden)
    print(f"layer {li}: hidden_states shape {hs.shape}")

# per-temporal-group pooling: video tokens laid out row-major (t,h,w) -> block of Hm*Wm per t
per_t = Hm * Wm
start = vid_positions.min().item()
hs_deep = out.hidden_states[-1][0]
pooled = []
for t in range(T):
    blk = hs_deep[start + t * per_t: start + (t + 1) * per_t]
    pooled.append(blk.mean(dim=0))
pooled = torch.stack(pooled)
print("\npooled per-temporal-group hidden states shape (T, hidden):", pooled.shape)
print("sample sanity: pooled[0][:5] =", pooled[0][:5].float().cpu().tolist())
print("\nOK — layout assumption verified end-to-end.")

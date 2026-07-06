# 评测框架说明书(videoPerception 项目)

> 更新:2026-07-03。配套文件:`setup_lmmseval.txt`(环境安装/重建手册)、`run_lmmseval.sh`(一键评测)。
> 本文回答三个问题:**我们用什么评测、它内部怎么工作、哪些细节决定数字可不可信**。

---

## 0. 一句话总览

**lmms-eval 0.7.2(社区标准 harness)+ Qwen 官方预处理(qwen-vl-utils)+ transformers 5.12.1 后端**,
四个 Qwen 模型 × 三个 benchmark(MVBench / TOMATO / Video-MME w/o sub),全部社区默认配置,
唯一本地偏离是一个防崩溃钳帧补丁(见 §6)。

---

## 1. 三层职责分工(谁在干什么)

```
┌─────────────────────────────────────────────────────────┐
│ lmms-eval(裁判层)                                       │
│  · 任务定义:题目 prompt 怎么拼、选项怎么排                  │
│  · 判分:官方答案解析器 + 指标(acc / MBA / 分类细分)          │
│  · 生成控制:max_new_tokens、greedy、批大小                  │
│  · 数据分发:4 卡 accelerate 数据并行(每卡一个完整模型副本)     │
├─────────────────────────────────────────────────────────┤
│ qwen-vl-utils(备菜层,Qwen 官方包 = 官方教程本体)             │
│  · 抽帧:均匀采样 N 帧(decord 后端)                          │
│  · 分辨率:每帧上限 786432 px(官方视频上限,≈250 token/帧)      │
│  · 时间戳:把每帧真实秒数算出来交给 processor                   │
├─────────────────────────────────────────────────────────┤
│ transformers 5.12.1(模型层)                              │
│  · Qwen3VL / Qwen3_5(Moe) ForConditionalGeneration        │
│  · bf16 + sdpa attention,greedy 生成                      │
│  (vLLM 0.19.1 备用:批量 CoT 生成/加速评测时切 vllm 后端)      │
└─────────────────────────────────────────────────────────┘
```

## 2. 一条样本的完整流水线

1. lmms-eval 从 HF hub 加载任务标注(带 HF token,走 hf-mirror);
2. `doc_to_visual` 把样本映射到本地视频文件(路径布局见 §7);
3. `doc_to_text` 拼 prompt(题干 + 选项 + 任务自带指令,例如 "Only give the best option.");
4. qwen-vl-utils 均匀抽 N 帧、缩放、算时间戳;processor 把 `<t 秒>` 标记插进输入序列
   (**Qwen3-VL 是时间戳感知模型,漏传时间戳会把 11s 视频当 0.6s——lmms-eval 的 qwen3_vl 类
   已内置正确处理**,源码 `models/simple/qwen3_vl.py:370-397`,`return_video_metadata=True`);
5. 模型 greedy 生成;
6. 任务自带解析器从输出里抽答案字母,与 GT 比对;
7. 汇总指标写入 `results_lmmseval/`(含逐样本 log,`--log_samples`)。

## 3. 三个 benchmark 的协议表(第一轮 base 评测)

| | MVBench | TOMATO | Video-MME |
|---|---|---|---|
| 任务名 | `mvbench`(20 子任务×200 题) | `tomato`(1484 题) | `videomme`(2700 题,= w/o subtitles) |
| 帧数 | 32(lmms-eval qwen3_vl 默认) | **16(TOMATO 论文官方协议)** | 32 |
| 输入长度 | ≈8.3k tokens(32×~250+文本) | ≈4.3k | ≈8.3k |
| max_new_tokens | 16 | **1024**(任务作者允许先推理再作答) | 16 |
| 解码 | greedy(temperature=0) | greedy | greedy |
| 指标 | 各子任务 acc + 总 acc | 总 acc + 6 推理类型细分 | 总 acc + 时长/类别细分 |
| 视频来源 | OpenGVLab/MVBench video 分支(含官方预裁剪 segment) | yale-nlp 原版视频(标注走 lmms-lab) | lmms-lab/Video-MME |

补充事实:
- **MVBench 时间裁剪**:官方实现 = 直接提供裁好的段视频(`star/Charades_segment` 等),我们用的就是它 → 20/20 任务全量,含 NTU(fine_grained_pose);
- 抽帧方式(均匀采样)是全社区共识;**帧数没有全行业统一值**(8~768 都有人用),这是"绝对分无法逐位对齐"的主因——对策 = 固定自家协议 + 同协议比 delta(SynRL 论文同样没公布帧数);
- Qwen3-VL 上下文 256K,8.3k 输入毫无压力;4090-24G 实测占用 ~22G(含 17G 权重)。

## 4. 参赛模型清单(统一入口 `/root/models/`)

| 目录(`/root/models/`) | 角色 | 大小 | 架构 | 存储属性 |
|---|---|---|---|---|
| `Qwen3-VL-8B-Instruct` → 软链至 GPFS `/remote-home/ziyesong/models/` | **锚 arm**(旧 baseline 可对拍;SynRL 表有此行:MVBench 67.2/TOMATO 33.2/VideoMME 63.4→+SynRL 69.1/38.1/65.2) | 16.5G | qwen3_vl | GPFS 持久 |
| `Qwen3.5-9B` | student 候选 + teacher 横评小杯 | 19.3G | qwen3_5 | /root 易失,可重下 |
| `Qwen3.5-35B-A3B` | teacher 候选(3.5 代) | 71.9G | qwen3_5_moe | /root 易失,可重下 |
| `Qwen3.6-35B-A3B` | teacher 候选(3.6 代,与 3.5-35B 同构可干净对比) | 71.9G | qwen3_5_moe | /root 易失,可重下 |

- 全部通过完整性校验(分片齐全 + safetensors 头可解析 + config 架构确认)。
- 35B 推理布局:4×4090 TP=4(vLLM)或 transformers device_map=auto;**A100 上 TP=3 不可用**(16 头除不尽)。
- 重下命令模板:`env -u http_proxy -u https_proxy -u all_proxy HF_ENDPOINT=https://hf-mirror.com hf download Qwen/<名> --local-dir /root/models/<名>`

## 5. 运行方法

```bash
cd /remote-home/ziyesong/videoPerception/eval
# 冒烟(每任务 8 样本):
LIMIT='--limit 8' bash run_lmmseval.sh
# 正式(默认模型=8B 锚):
bash run_lmmseval.sh
# 换模型(例:Qwen3.6-35B):
MODEL=/root/models/Qwen3.6-35B-A3B OUT=results_lmmseval_q36_35b bash run_lmmseval.sh
```
- 脚本自带**启动守卫**:任一 GPU 显存 >1G 拒跑(防叠跑 OOM);
- 杀残留进程:容器里 nvidia-smi 显示的是**宿主机 PID 不能直接 kill**,用
  `pgrep -f 'lmms[_]eval'` 找容器内 PID(注意模式加方括号防自匹配)。

## 6. 与社区默认的全部偏离(诚实清单)

1. **短视频钳帧补丁**(唯一行为偏离):lmms-eval 0.7.2(含 main)bug——视频总帧数 < N 时
   qwen-vl-utils 抛异常、整 rank 崩 + 其余 rank 卡死。补丁:仅此时把采样数钳到实际帧数(向下取偶)。
   协议表述:*uniform N-frame sampling; clips shorter than N use all frames (floor-to-even)*。
2. **补齐 pip 包缺失的 247 个任务文件**(打包缺陷,补的是官方原文件,非行为偏离)。
3. 其余全部默认:prompt、解析器、greedy、分辨率上限、system prompt("You are a helpful assistant.")。

## 7. 数据布局(评测数据全在易失盘,重建见 setup_lmmseval.txt §11)

```
HF_HOME=/root/hf_home
├── mvbench_video/   ← 真目录(不能软链,否则触发整库重下!)20 任务全量,零缺失(已逐标注审计)
├── videomme -> /root/benchmarks/VideoMME    (data/ 900 mp4 + subtitle/)
├── TOMATO   -> /root/benchmarks/TOMATO      (yale 原版 1417 mp4,lmms 标注全命中)
├── token    ← HF token(任务 yaml token:True 必需)
└── datasets/ ← 标注缓存
```

## 8. 已知 dial / 待办

- [ ] Video-MME 长视频(最长 1h)32 帧偏保守,社区 Qwen 系常用 64+ → 第一轮出数后抽子集做 64 帧敏感性;
- [ ] 评 CoT student 时:mvbench/videomme 的 max_new_tokens=16 会截断推理 → 用 `--gen_kwargs` 提高并写入协议;
- [ ] TemporalBench 暂缓(用户拍板;数据已删,GPFS 有重下脚本,gated 需 R3ca1c1trant 的 token);
- [ ] 给 lmms-eval 上游提 nframes bug issue;
- [ ] 环境有改动后刷新 GPFS 备份(`setup_lmmseval.txt` 顶部命令)。

## 9. 结果与对表口径

- 结果目录:`results_lmmseval*/`(每次跑含 aggregate json + 逐样本 jsonl);
- 对外引用:写明本文件 §3 协议;绝对分与他家论文比较时注明协议差异;
- **核心论证永远是同协议 delta**:base → distill/direct/teacher 各 arm 全部用本框架同配置;
- SynRL 对表(其协议未公开,仅作量级参照):Qwen3-VL-8B 行 base 33.2/67.2/63.4(TOMATO/MVBench/VideoMME)→ +SynRL 38.1/69.1/65.2。

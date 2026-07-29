# P2 思考模式"卡住不收敛"样例（组会用）

模型：Qwen3.5-9B，`enable_thinking=True`，`max_new_tokens=4096`，TOMATO benchmark（16帧协议）。
数据来源：`eval/results_transformers_40_4096/models__Qwen3.5-9B/*samples_tomato.jsonl`（transformers后端，40条样本里25%属于这一类）。
完整原始输出（未做任何裁剪/处理）另存于（已拷贝到GPFS持久盘，不会因容器重建丢失）：
- doc_id=3　→ `eval/stuck_examples_raw/doc3_full.txt`
- doc_id=20 → `eval/stuck_examples_raw/doc20_full.txt`
- doc_id=26 → `eval/stuck_examples_raw/doc26_full.txt`
- doc_id=33 → `eval/stuck_examples_raw/doc33_full.txt`

---

## 样例1：doc_id=3 —— "反复得出正确答案，但停不下来"（最典型）

- **视频**：`/root/benchmarks/TOMATO/videos/human/0234-02.mp4`
- **问题**：In which direction(s) did the person's hand move?
- **选项**：A. Upwards then downwards. / B. Not moving at all / C. Downwards then upwards. / **D. Up.** / E. Down.
- **正确答案**：D（Up.）
- **模型输出长度**：14514 字符，撞满4096 token预算，没有真正"结束"（硬截断）

**关键现象**：模型全文反复达成同一个结论"答案是D/Up"**至少13次**，每次之后都自己推翻重来。摘录几段（原文顺序）：

> Let's re-evaluate the sequence. ... So the movement is primarily **upwards**. ... So the movement is "Up".
>
> Let's check option C "Downwards then upwards". ... So "Up" is the most accurate description of the *change* in position.
>
> The movement is clearly from down to up. Option D is "Up". ... **Therefore, the answer is likely D.**
>
> Wait, let me look at the options again.
> A: Upwards then downwards.
> B: Not moving at all
> C: Downwards then upwards.
> D: Up.
> E: Down.
> Is it possible that in the first few frames, the hand is actually moving *down*?

> Okay, I'm confident the movement is Up.
>
> Wait, let me look at the options one more time.
> Is it possible that the answer is C "Downwards then upwards"?

（最后被截断在）：

> ...This doesn't fit "Downwards then upwards" as a sequence of *movements*. It fits "Down then Up" as a sequence of *positions*. But the question asks "In which direction(s) did the person's hand *move*?". Movement implies velocity. **The velocity is Up.**

**要点**：模型不是"推不出来"，是从很早就知道答案，但没有停下来的机制，每次自我怀疑就把同样的画面证据重新过一遍，如此循环到token预算耗尽。

---

## 样例2：doc_id=20 —— 被自己的"帧数疑虑"带偏，最终没能收敛

- **视频**：`/root/benchmarks/TOMATO/videos/human/0251-03.mp4`
- **问题**：What was the orientation of the person's hand movement?
- **选项**：A. Down then left. / B. Not moving at all / C. Left. / **D. Left then Down.** / E. Down.
- **正确答案**：D（Left then Down.）
- **模型输出长度**：14366 字符，硬截断，没有给出最终答案

**关键现象**：模型中途开始怀疑自己没拿到足够的帧：

> "You will be provided with 16 separate frames...". I only see 7 images in the prompt. Wait, let me count the images in the prompt. There are 7 images.

**已核实这是模型自己的错误认知，不是我们数据管线的bug**：用`decord`直接读了这个视频文件，真实总帧数370帧（60fps），远超16帧协议要求，不存在因视频过短被钳帧的情况——16帧是确实被采样并送进去的。模型在这里是凭空怀疑、把注意力引到一个不存在的问题上，进一步挤占了原本该用来推理画面内容的预算。

---

## 样例3：doc_id=26 —— 同类"帧数疑虑"，另一个视频

- **视频**：`/root/benchmarks/TOMATO/videos/human/0531-00.mp4`
- **问题**：What is the direction of the person's hand movement?
- **选项**：A. Rightwards then leftwards. / B. Downwards then leftwards. / C. Downwards then rightwards. / **D. Leftwards then leftwards.** / E. Upwards then rightwards. / F. Upwards then leftwards.
- **正确答案**：D（Leftwards then leftwards.）
- **模型输出长度**：13179 字符，硬截断

**关键现象**：和样例2同一类问题，模型又一次怀疑帧数不对：

> "You will be provided with 16 separate frames...". This is a standard prompt. But the actual input only has 6 frames. I must work with what I have.

同样核实：该视频真实总帧数320帧，16帧协议正常执行，模型的怀疑同样是凭空的。**两个独立样例出现同一种"自我怀疑帧数不足"的模式，值得作为一个单独的观察点在组会上提——不是我们的bug，但可能反映这个模型在长CoT下容易被自己生成的"元问题"（关于任务设置本身的疑问）带偏，而不是专注在视觉证据本身。**

---

## 样例4：doc_id=33 —— 经典"let's look at the options again"循环，最后卡在"Maybe"

- **视频**：`/root/benchmarks/TOMATO/videos/human/0537-01.mp4`
- **问题**：In which way did the person's hand move?
- **选项**：A. Rightwards then leftwards. / B. Downwards then leftwards. / **C. Upwards then Upwards.** / D. Upwards then rightwards. / E. Leftwards then downwards. / F. Downwards then rightwards.
- **正确答案**：C（Upwards then Upwards.）
- **模型输出长度**：14426 字符，硬截断，最后一个词是"Maybe"（字面意义上被切在半句话中间）

**关键现象**：

> ...But "Leftwards then..." is not an option with "Upwards". So it has to be C or D. In frame 6, the hand is pointing right. In frame 5, it's pointing up. So there is a rightward component at the end. But the primary action is raising the hand. Let's look at the options again. Maybe

模型已经把范围收窄到C/D两个选项（且C是正确答案），但在做最后判断前，预算耗尽，句子直接断在"Maybe"这个词上。

---

## 汇总数据（40条样本量级，供组会引用）

| | 占比 |
|---|---|
| 几乎不写推理、直接给答案（如"The correct answer is E."仅24字符） | 70% |
| 正常推理后收敛 | 5% |
| 长篇推理但撞满预算，未给出可提取的最终答案（本文档4个样例这一类） | 25% |

对照测试：给模型额外加"请逐帧仔细推理"式的system/user提示（reasoning_prompt），40条样本的输出**逐字节与不加提示时完全一致**——说明这不是"没被提醒要认真想"，是模型自己的策略决定要不要认真想、以及想完之后是否停下来，运行时的提示语不起作用。

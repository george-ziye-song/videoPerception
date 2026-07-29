---
title: "probe-experiment-report.md 工程实现核实"
status: "核实完成,方法论成立,发现一处数字过期(不影响结论方向)"
date: "2026-07-13"
---

# probe-experiment-report.md 工程实现核实

目的:不看报告文字,直接读代码和数据文件,核实"probe读得出、直答只对35-69%"这类关键结论是怎么算出来的,以及数字本身对不对得上。方法:通读`probe_data/`下全部脚本(不是抽查),核对脚本输出和报告正文逐字对比,交叉核对文件时间戳。

---

## 0. 结论先行

- **方法论站得住**:核心对比("probe读得出" vs "模型自己答不对")是两条独立的代码路径产生的,不是同一次计算换个名字说两遍,细节见§1。分类probe、回归probe、position-confound检查、错误语义分类,这几处我读了全部代码,实现和报告描述的一致,没有偷懒或者掉包。
- **发现一处真实问题**:报告正文表格里`Bouncing_Counting`的直答准确率(9B=46.50%, 35B=34.50%)是**过期数字**,来自一次后来被覆盖掉的中间运行。当前实际保存的数据文件显示的是9B=40.50%、35B=29.00%。证据在§2,是硬时间戳链,不是我猜的。
- **这处错误不影响结论方向,反而让结论更成立**:两个数字修正后都是更低,gap比报告写的还要大。但这是报告应该修正的一处硬伤,细节见§2末尾。

---

## 1. "probe读得出、直答答不对"是怎么算出来的

这个结论的核心,是拿**同一批合成视频、同一个模型**,跑**两条完全独立的代码路径**,比较各自的准确率。不是同一次forward换个说法讲两遍。

**为什么不能直接用P1的结果,非要新跑一个实验**:两个原因,都是硬性的,不是图省事重复造轮子。

1. **P1根本没有这批数据需要的"真值"**。§1.1这个probe要训出来,前提是每条样本得有一个**独立于MCQ答案字母**的、精确的物理真值(比如"这个物体到底是顺时针还是逆时针转""球到底弹了几次")。这种真值只有SynRL生成器自己造视频的时候才知道(它是按这个真值把视频画出来的),P1测的MVBench/TOMATO/VideoMME是真实世界视频,只有"这道题标准答案是哪个字母"这一种标注,没有"这个动作背后精确的物理参数是什么"这种标注。没有这个真值,§1.1那个probe根本没法训——不是不想用P1,是P1这批数据里根本不存在训probe要用的监督信号。

2. **P1从来没有抽过hidden state**。P1全程只调用过`generate()`拿最终答案去打分,从没执行过`output_hidden_states=True`这一步(这一步很占显存和时间,P1真实benchmark体量下没有做这个的必要,也没做过)。就算P1测的是同一批视频,没有hidden state数据,§1.1还是拿不出来东西,得重新抽一遍。

**如果硬要凑合用P1的直答数字对比,会不会正好踩中"配置不一样"这个坑**:会,而且是实打实的坑,不是假设。P1里不同benchmark用的帧数、`max_new_tokens`、post_prompt措辞都不一样(比如MVBench用32帧+`max_new_tokens:16`,TOMATO用16帧+`max_new_tokens:1024`,各benchmark的post_prompt也不同),这些配置从来不是为了"和某一次hidden state抽取严格对齐"而设计的。§1.1和§1.2这两个脚本能放在一起比,靠的恰恰是`extract_hidden_states.py`和`direct_answer_baseline.py`两处**几乎逐行相同**的prompt构造代码(同样的`nframes=16`、同样的消息结构、同样的`enable_thinking=False`)——这保证了两条路径吃进去的是完全同一份输入,差异只出在"看一眼内部表示"还是"真的生成"这一步。如果换成拿P1现成的直答数字(某个benchmark用32帧測的)去跟这次新抽的16帧hidden state比,那观察到的准确率差异就可能一部分来自"用了不同帧数看视频",而不是"信息在不在读出层"——这就把想验证的东西和一个新引入的confound混在一起了,结论没法看。所以宁可专门新写`direct_answer_baseline.py`跟`extract_hidden_states.py`配套用,把输入死死锁定成同一份,也不能图省事直接拿P1的数字来凑。

上一轮已经验证过的是**反过来**的一层核对:不是拿P1数字替代这次实验,而是拿P1的system prompt/解码协议**当外部参照**,确认这次新脚本的协议设置没有偏离已验证过的P1配置太远(见1.2节"和P1能不能对上"),这是"用P1做交叉校验"和"用P1的结果替代新实验"两件不同的事。

### 1.1 第一条路径:抽取hidden state → 训一个线性probe

代码:`probe_data/extract_hidden_states.py`

关键调用(第146-147行):
```python
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, use_cache=False)
```
这是**一次性forward,不generate**。模型看完整段视频+问题prompt,只往前走一遍,拿三层(浅/中/深,第91行`shallow=round(0.25*L) mid=round(0.5*L) deep=round(0.9*L)`)的hidden state。

有个容易踩的坑,这个脚本处理对了:Qwen的视频token不是一整块连续区域,每隔一段会插入时间戳文本token(比如`<|vision_end|>3.2 seconds<|vision_start|>`),把视频token切成好几段。如果直接按固定stride切片做pooling,会把时间戳文字的embedding也平均进去,数据是脏的。脚本用`find_video_runs()`(第59-73行)专门做"连续run检测",按视频token真实的连续区间分组,再对每组做`hs[s:e].mean(dim=0)`(第155行)。我读了这段逻辑,处理是对的。

然后`probe_data/train_probes.py`拿这些pooling后的向量,配上`metadata.jsonl`里的**真值**(不是模型自己生成的答案,是生成器造视频时就知道的ground truth,比如`total_rotations`、`direction`这些字段,第38-70行的`label_*`函数),训一个`sklearn.linear_model.LogisticRegression`(第100行)。切分是`train_test_split(..., test_size=0.3, stratify=y)`(第97行),200条里70%训、30%测(n_val=60,和报告写的一致)。

**这一步得到的准确率,衡量的是:hidden state里这个信息能不能被一个简单线性分类器读出来。它完全不依赖模型自己怎么回答问题。**

**label具体是什么——举两个真实例子**:label**不是**MCQ答案字母(那个是随机打乱的,`train_probes.py`里从来不用它),而是`ground_truth_details.other_details`里生成器造视频时就知道的真值字段。抽两条实际记录:

例1,Rotation_Direction样本`rotation_direction_203`:
```python
ground_truth_details = {
  'question_data': {'question': 'Is the l shape spinning clockwise or counter-clockwise?',
                     'options': {'A': 'Clockwise', 'B': 'It is not spinning', 'C': 'Counter-clockwise'},
                     'correct_answer_value': 'Clockwise'},
  'other_details': {'shape': 'l_shape', 'color': 'green', 'direction': 'Clockwise'}
}
```
这条样本的MCQ答案字母是`A`(选项顺序打乱后A对应Clockwise),但`train_probes.py`第42-43行的`label_rotation_direction()`直接取`other_details["direction"]`="Clockwise"当分类标签,完全不看字母A。probe学的是"两分类:Clockwise vs Counter-clockwise"。

例2,Bouncing_Counting样本`bouncing_counting_600`:
```python
ground_truth_details.other_details = {'shape': 'circle', 'color': 'yellow', 'target_bounce_count': 4}
```
这条的MCQ答案字母是`C`,但`label_bouncing_count()`(第50-51行)取的是`target_bounce_count`=4,标签是字符串`"4"`。实测200条Bouncing_Counting样本里`target_bounce_count`实际出现的取值是`{3,4,5,6,7}`(5个),probe学的是这5分类。

结论:label来源和MCQ字母是两条完全不相交的信息通路,不存在"probe偷看了答案字母"这种数据泄漏。

**2类和5类是怎么用同一套代码处理的**:不是分task写不同分类器,类别数是**运行时从数据里自动数出来的**,不是写死的。`train_one_probe()`第90-91行:
```python
le = LabelEncoder()
y = le.fit_transform(y_raw)          # 有多少个不同的字符串值,就有多少类,自动决定
```
`LabelEncoder`扫一遍这一个task全部200条样本的`y_raw`(label字符串列表),有几个不同的值就编几个整数类别。Rotation_Direction的`direction`字段实测只出现`Clockwise`/`Counter-clockwise`两种值→K=2;Bouncing_Counting的`target_bounce_count`实测出现5种值→K=5;我顺手也查了Complex_Direction_Identification的`path`字段,实测12种值(`Down_Left`、`Up_Right`……方向两两组合)→K=12。K完全由这个task当次实际出现的label取值数量决定,同一个`train_one_probe()`函数不用改一行代码就能适配K=2/5/12的情况,因为下一步:
```python
clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(Xtr, ytr)
```
`sklearn`的`LogisticRegression`本身就是通用多分类实现——K=2时是标准二分类,K>2时内部自动切换成multinomial(softmax)多分类,`clf.fit()`这一行代码本身不用因为K是2还是12而改。

**输入到底是不是hidden state,是不是纯视觉token(先说人话,再看证据)**

人话版本:

- 模型读视频+读题,不是"看一眼直接蹦出答案"这么简单,中间有个逐层计算的过程。做法是:先把视频和文字**都切成一个个"位置"**——每个字是一个位置,每一小块视频画面(准确说是每2帧合并、再切成小方块patch)也是一个位置,所有位置排成一条长队,一起喂给模型。模型每往前计算一层,长队里**每个位置**都会更新出一个几千维的数字向量——这个向量就叫"hidden state",你可以理解成"模型这一刻对这个位置内容的内部印象/理解",还没变成它最后嘴上说出来的文字。probe要读的就是这个"内部印象",不是视频原始像素,也不是模型最终说出口的答案文字。

- 但这条长队里,不全是视频画面的位置——还夹着文字的位置:开头的system prompt("你是一个helpful assistant")、问题原文、选项A/B/C的文字,甚至视频放到一半还会插进几个字的"时间戳"提示(比如告诉模型"现在演到第0.8秒了")。如果抽取代码一不小心把这些**文字位置**的印象也一起平均进去,那probe学到的就不是"这个模型是不是真的看懂了视频画面",而是掺了文字的大杂烩,实验就不干净了。所以必须证明:代码精确挑出了"纯视频画面"对应的那些位置,一个文字位置都没混进来。

再看证据(不是我看代码猜的,是实际跑数据验证过的):

1. **确实是hidden state,不是像素/不是最终答案**:第32行`out = model(**inputs, output_hidden_states=True, ...)`,这里`out.hidden_states[layer_idx]`就是上面说的"每个位置的内部印象向量",HuggingFace transformers库的标准输出,不是原始像素、也不是模型最后要预测下一个字用的那个"打分表"(logits)。

2. **确实只挑了纯视频位置,文字被排除在外**:`find_video_runs()`用的判断条件是`input_ids == video_token_id`(第60行)——`video_token_id`是Qwen模型词表里专门留给"这是一块视频画面"的一个特殊编号,只要某个位置不是这个编号(不管是system prompt、问题文本、选项文字,还是插在视频中间的时间戳文字),就不会被选中、不会被拿去平均。

   我实际重跑了一遍这条流水线来验证(不需要加载几十G的模型权重,只用负责"切词"的processor,几秒钟跑完),拿一条真实的16帧`Rotation_Direction`样本实测结果:
   ```
   总video token数: 2048
   分成几段连续的视频位置(=T): 8段
   每段长度: [256, 256, 256, 256, 256, 256, 256, 256]   ← 都一样长
   每段之间隔了几个"文字"token: [8, 8, 8, 8, 8, 8, 8]
   把第一段间隔的8个token解码出来看是什么文字: '<|vision_end|><0.8 seconds><|vision_start|>'
   ```
   翻译一下这个结果:16帧视频,每2帧被模型自动合并成一组(这就是`temporal_patch_size=2`的意思),所以整段视频被拆成了8个时间段;每个时间段对应256个"纯视频画面"位置;每两个时间段之间,模型会插入一小段文字提示当前演到第几秒(比如`0.8 seconds`),这8个字的提示**不算在256个视频位置里面**,代码也确实没有把它们的印象抽进来。这和代码注释里写的机制完全对得上,不是凭空写的说明。

3. **2048这个数字是每条样本都一样,还是设了上限、不够就padding**:是**每条样本都一样、真实算出来的,不是设上限+padding**。我把`hidden_states/`下全部1400条记录(7个task×200条,两个模型都查了)扫了一遍,`num_temporal_groups`(=T)和每层张量的shape**没有一条例外**,9B全部是`(8, 4096)`,35B-A3B全部是`(8, 2048)`(35B隐藏维度和9B不同,这是模型架构差异,不是这个问题要关心的点)。这不是因为设了"最多8段、不够补齐"这种截断/填充机制,而是两个前提天然锁死了这个数字:①代码调用时`nframes`这个参数**每次都固定传16**(不管视频原始时长多长——我抽查了几条视频的真实帧数,120~240帧不等,但处理流水线统一只从里面均匀抽16帧喂给模型,不是全喂进去);②这批合成视频**画布分辨率统一是512×512**(我用`decord`抽查了8条不同task的视频文件,像素尺寸`(512, 512, 3)`没有例外)。`16帧÷2(合并)=8段`、`512×512这个固定画布切出的patch数=每段256个`,这两个数只要输入的nframes和分辨率不变,结果就必然不变,所以是"确定性算出来的一样",不是"有上限、短的padding凑够"。

4. **怎么把8段变成1个向量喂给probe(两级平均)**:`extract_hidden_states.py`第155行先对每一段的256个视频位置做一次平均,得到"这一小段时间在说什么"的印象,8段一共存成`(8, 4096)`的一个表格;`train_probes.py`第154行再把这8段的印象再平均一次,压缩成"整段视频在说什么"的单个`(4096,)`向量,这才是真正喂进`LogisticRegression`去分类的X。因为刚才实测确认这8段长度完全一样(都是256),"先分段平均、再把8段平均"和"把2048个视频位置直接一次性全部平均"结果是完全一样的——不会因为某一段特别长或特别短而被算得偏重或偏轻。

（例外:Event_Sequence这个task不做"整段视频mean成一个向量"这一步,是把每个temporal group当成独立一行、配一个"是否落在关键事件时间窗口内"的二分类标签,细节见`run_event_sequence_probe()`,和上面6个task的处理方式不同,这是report和代码都明确说明了的。）

### 1.2 第二条路径:真实generate → 官方答案提取器判分

代码:`probe_data/direct_answer_baseline.py`

Prompt构造和1.1**完全相同**(同样的`enable_thinking=False`、同样的video/text message结构,第83-89行 vs extract_hidden_states.py第126-132行,两处代码几乎逐行一致),这保证了两条路径的输入是同一个东西。

区别在下一步(第104-105行):
```python
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
```
这是**真正的自回归生成**——模型要自己吐出token,不是只做一次forward看内部表示。生成完之后(第107-109行)解码成文本,用下面这个函数提取答案:
```python
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
pred = extract_mcq_answer(resp, choices=choices)
```
这个函数和P1/P2评测用的**是同一个共享函数**(不是另写的、可能有bug的解析器),然后和`d["messages"][2]["content"]`(真实答案字母)比较。

**这一步得到的准确率,衡量的是:模型自己生成完整答案的能力。**

**先从"forward"这个词本身讲起,彻底的大白话版**:不管是§1.1还是§1.2,模型能做的事只有一种,叫"forward"——可以理解成"过一遍电路":先把视频画面和文字都转换成一串数字,从模型第一层塞进去,一层一层往后传,每层做固定的乘法加法运算,传到最后一层,吐出一组数字(可以翻译成"接下来最可能该说哪个字")。这个"数字从第一层流到最后一层"的完整过程,就是一次"forward"。模型没有第二种运作方式,它不会凭空"想"——它能做的就是这一件事:塞数字进去,吐数字出来。**§1.1和§1.2的全部差别,只是"用了几次forward、怎么处理这些数字"的差别,模型本身的运作原理没有变过。**

- **§1.1只做1次forward就结束**:数字进去,流到最后一层,但我们**不看**"最后应该说哪个字"这个结果,而是在流到半路(比如流到90%那一层)的时候,偷看一眼每个位置上正在流淌的那一串数字——这串数字就是"hidden state"。看完就结束,不会继续往下流,模型没有说出任何一个字,也不知道自己被"偷看"了。

- **§1.2要循环做很多次forward**,一次接一次:
  1. 第1次forward,和§1.1一样把"视频+问题"整段塞进去流一遍,但这次**看的是最后一层吐出来的结果**,选出第1个要说的字。
  2. 第2次forward,把这第1个字**接到刚才的输入后面**,再流一遍电路(靠`use_cache=True`做了个工程优化,不用把前面全部重新算一遍,只算新加这一个字带来的计算量,但从效果上等同于把"输入+已经说的第1个字"完整地重新走了一遍),吐出第2个字。
  3. 第3次forward,把第1、2个字都接上再流一遍,吐出第3个字……这样循环下去,最多循环32次(`max_new_tokens=32`),或者模型自己决定"说完了"提前停。

关键差别在这里:§1.2从第2次forward开始,**输入里多了"模型自己刚刚吐出来的字"**——它是在根据自己已经说出口的话,接着往下说,这也是这个过程叫"自回归(auto-regressive)"的原因:自己的输出变成了下一步的输入。如果它前面选错了方向,后面没法反悔,只能顺着这个已经"说出去的话"接下去。§1.1完全没有这个"自己说过的话喂回给自己"的循环,只流了1次电路就结束了,模型压根没有"说话"这个动作。这正是"probe读得出、直答答不对"能发生的机制层面原因:§1.1看到的是模型半路上的一次性"内部印象",§1.2测的是模型"一个字一个字往外蹦、自己对自己已经说的话负责"的循环过程——这是两件不同的事。

**除了数据集不一样,和P1还有哪些配置不同(我逐项查过,不是只查了dataset这一项)**:

| 配置项 | P1实际用的(`run_lmmseval.sh`+对应task yaml) | 这次`direct_answer_baseline.py` | 一致吗 |
|---|---|---|---|
| system prompt | `"You are a helpful assistant."` | 同上 | ✅ 一致 |
| 解码方式 | `do_sample=False`(贪婪) | `do_sample=False` | ✅ 一致 |
| thinking开关 | `enable_thinking=False`(P1协议) | `enable_thinking=False` | ✅ 一致 |
| dtype | `bfloat16` | `torch.bfloat16` | ✅ 一致 |
| batch_size | `--batch_size 1`(逐条跑) | 逐条for循环,相当于batch_size=1 | ✅ 一致 |
| do_resize | `False`(不额外缩放视频) | `False` | ✅ 一致 |
| **采样帧数nframes** | MVBench/VideoMME**32帧**,TOMATO**16帧**(`run_lmmseval.sh`第29-31行`max_num_frames=32/16/32`) | **统一16帧**(为了和§1.1的hidden state抽取协议对齐) | ❌ 不一致,而且P1内部三个benchmark彼此都不一样 |
| **max_new_tokens** | MVBench**16**、TOMATO**1024**(各自task yaml里的`generation_kwargs`) | **32**(自己选的折中值) | ❌ 不一致 |
| **问题文本的具体措辞/模板** | lmms-eval自己的`doc_to_text`函数拼的,比如MVBench是`"Question:"+问题+"\nOption:\n"+选项+"Only give the best option.\n"` | SynRL数据生成阶段就写死在`sft.jsonl`里的模板:`"Select the best answer...Respond with only the letter...The best answer is:\n  "` | ❌ 遣词造句不同(但都是"只给字母"这个意图) |
| **调用路径** | 走完整lmms-eval框架(`Instance`/`doc_to_visual`/`doc_to_text`/task自己的`process_results`打分) | 单独的standalone脚本,直接调`transformers`的`AutoModelForImageTextToText`+`generate()`,只借用了共享的`extract_mcq_answer`做打分 | ❌ 不是同一套代码,是照着协议重新写的 |

所以准确的结论是:**贪婪解码、thinking开关、system prompt、dtype、batch_size、do_resize这几项底层协议是严格对齐的;但帧数、`max_new_tokens`、问题模板措辞、以及整个调用框架都不是照抄P1的**——这也印证了上一轮的结论:不是"偷懒复用了P1",而是专门为了让§1.2和§1.1的输入严格一致(都是16帧、都用`sft.jsonl`里那份模板),新写的一个独立脚本,牺牲了"和某个P1 benchmark的具体配置逐项相同",换来的是"和自己这次hidden state抽取协议逐项相同"——这个取舍是对的,因为这次实验要回答的问题是"probe vs 直答",不是"和P1哪个benchmark比"。

**和P1能不能对上——三个具体核查点**:

**(a) system prompt是否一致**:第84行写的是`{"role": "system", "content": "You are a helpful assistant."}`。查了P1实际用的`lmms_eval/models/simple/qwen3_vl.py`第104行,这**正是P1跑真实benchmark时用的默认system_prompt**(`system_prompt: Optional[str] = "You are a helpful assistant."`),而且确认`run_lmmseval.sh`里没有传任何`system_prompt=`覆盖它。两边system prompt逐字一致,不是巧合。

**(b) 有没有专门指示模型直接给字母**:有,而且不是靠system prompt,是**直接写死在user turn的问题文本里**(来自SFT生成时的模板,不是这个脚本自己加的):
```
"Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter (A, B, C, D...) of the correct option. Is the t shape
spinning clockwise or counter-clockwise? Possible answer choices: A. It is not spinning
\nB. Clockwise\nC. Counter-clockwise\nThe best answer is:\n  "
```
这个指示比P1的MVBench任务用的`post_prompt: "Only give the best option.\n"`(见`lmms_eval/tasks/mvbench/mvbench_moving_direction.yaml`)更明确。解码参数上`do_sample=False`(贪婪解码)也和P1的`generation_kwargs: do_sample: false, temperature: 0`一致,`enable_thinking=False`同样对齐P1的thinking-off协议。再加上两个模型的`unresolved`字段全是0(提取器每一条都找到了合法字母,没有"模型输出乱七八糟解析不出来"的情况),这些一起说明35-69%这个低分不是"没让模型好好答"造成的假象。

**(c) 数字层面能不能互相印证**:P1记分板(thinking-off协议,和这次直答baseline同协议)里,35B-A3B在真实benchmark上稳定强于9B(MVBench 73.35 vs 69.42、TOMATO 43.53 vs 36.25)。这次7类合成原语算了个总平均,9B=70.00%、35B-A3B=70.93%,而且**7类里有6类35B≥9B**(Acceleration_Identification +4.00、Directional_Event_Counting +11.00、Rotation_Count +2.00、两类打平),模式和P1一致——说明"35B在合成数据上普遍不比9B差"这个大方向和P1对得上,排除了"合成数据pipeline对35B模型整体不兼容/跑挂了"这种解释。**唯独Bouncing_Counting一项35B反而低了11.5pp**(29.00% vs 40.50%),是7类里唯一的反常项,不是整体模式的一部分——这让报告"Bouncing_Counting这一项teacher更差"的说法更像一个具体、局部的真实效应,而不是数据管线系统性出错的信号。

### 1.3 为什么"probe读得出、直答答不对"不矛盾

这是用户之前审novelty.md数学论证时确认过的点,这里用代码层面重新说一遍:1.1和1.2虽然起点相同(同一个模型、同一帧输入),但走的是两个不同的**下游函数**——1.1是"hidden state → 线性分类器"这个函数,1.2是"hidden state → 继续自回归生成32个新token → 文本解析"这个函数。数据处理不等式(DPI)只保证:下游任何函数的表现都不能超过输入本身含有的信息量上限。它不保证两个不同下游函数的表现互相之间有大小关系。一个简单的线性读出函数完全可能比"自回归生成+看着自己已经写出的token续写"这个更复杂的函数更擅长利用某一种信息——尤其当模型在生成时把注意力/推理力气花在了别的地方。所以观测到的"probe 85-100%、直答35-69%"这个组合,是两个不同函数在同一份信息上的不同利用效率,不违反任何理论保证。

### 1.4 "拿hidden state直接做逻辑回归"这套方法正规吗——有没有有影响力的先例

这个质疑很合理,值得正面回应。**先说结论:这不是我们自创的土办法,是深度学习可解释性/表示学习领域一个有十年历史、非常主流的标准方法,专门的名字叫"linear probing"(线性探针)或"probing classifier"。**分三条线索说:

**(a) 这个方法本身的奠基论文,就是"冻结网络+训一个线性分类器,看某一层里有没有编码某种信息"**——Alain & Bengio,*"Understanding intermediate layers using linear classifier probes"*(2016,ICLR workshop,后来发表在JMLR)。这篇论文提出的做法,和我们`train_probes.py`里做的事情在方法论骨架上是同一件事:冻结主网络、只训一个线性层、拿准确率当"这一层里这个信息线性可读程度"的度量。NLP可解释性领域后续沿着这条路做了大量工作,比如Hewitt & Manning *"A Structural Probe for Finding Syntax in Word Representations"*(2019 NAACL,在冻结的BERT/ELMo hidden state上探测句法树)、Tenney et al. *"BERT Rediscovers the Classical NLP Pipeline"*(2019 ACL,用浅层分类器逐层探测BERT编码了什么语言学信息)、Conneau et al. *"What you can cram into a single vector"*(2018 ACL,SentEval探测任务合集,引用量很高)。这条方法路线本身也有专门的综述文章讨论它的适用边界和局限(Belinkov,*"Probing Classifiers: Promises, Shortcomings, and Advances"*,Computational Linguistics期刊2022)——这点我觉得值得如实提一句,免得显得只挑对自己有利的引用。

**(b) 在视觉自监督表示学习领域,"linear probing accuracy"不是小众技巧,而是这个子领域近十年的**标准评测指标本身****——SimCLR(Chen et al. 2020)、MoCo(He et al. 2020)、BYOL(Grill et al. 2020)、MAE(He et al. 2022)这些自监督视觉表示学习的代表作,论文里报的头条数字**就是**"冻结预训练encoder+只训一个线性分类头"在ImageNet上的准确率——这几篇论文比较不同预训练方法优劣,靠的就是这一个数字。CLIP(Radford et al. 2021)论文里,linear probe准确率也是除zero-shot外的另一条主线评测方式,跨约30个数据集报告。也就是说,我们探针实验里"probe看某一层能不能读出XX信息"这套操作,和这些视觉领域最有影响力的论文评价"这个表示学不学得好"用的是**同一件事**。

**(c) 和我们这次探针实验的具体故事(表示层里有信息、但模型自己嘴上说不对)最直接对应的先例,是LLM"内部知道但不说/说错"这条研究线**——Burns et al. *"Discovering Latent Knowledge in Language Models Without Supervision"*(2022,俗称CCS方法)训练一个探针去读LLM hidden state里"这句话真假"的信号,发现即使模型最终生成的回答是错的,hidden state里仍然线性编码着正确答案;Azaria & Mitchell *"The Internal State of an LLM Knows When It's Lying"*(2023 EMNLP)标题本身就是这套叙事,用简单分类器探测hidden state判断模型是否在"说谎"(即嘴上答案和内部表示不一致)。这两篇的实验设计骨架——冻结hidden state训探针、对比"探针读出的" vs "模型自己嘴上说的"——和`probe-experiment-report.md`这次做的事几乎是同一个模板,只是我们的领域是视频时序感知,他们的领域是文本事实性。

**老实说一句局限**:probing classifier这条方法路线本身在文献里不是没有争议——常见的质疑是"探针的分类准确率,到底反映的是表示里'真的有'这个信息,还是反映了探针自己的拟合能力(给足够复杂的探针,几乎什么都能从任意表示里线性/非线性地拟合出来)"。这也是为什么这条文献线里的经验法则是**优先用线性探针**(而不是深的MLP)——线性探针的假设空间受限,读出来的准确率更能归因于"表示本身线性可分",不容易靠探针自己"硬记住"训练集把分数刷上去。`train_probes.py`用的正是`sklearn.linear_model.LogisticRegression`(线性),没有偷偷换成更强的分类器,这一点是符合文献惯例的谨慎选择,报告里也如实报了majority baseline做对照(§3.2提到的70/30切分、n_val=60),这些都是这条方法论要求的标准操作,不是自己发明的花活。

---

## 2. 发现的问题:Bouncing_Counting 的直答数字过期

### 2.1 报告正文写的数字

`probe-experiment-report.md` 第114行:
```
| Bouncing_Counting | 100.00% | 46.50% | 100.00% | **34.50%** | **能→不对,gap在读出层,且teacher更差** |
```
(列顺序:9B-probe / 9B-直答 / 35B-probe / 35B-直答)

### 2.2 当前数据文件里的实际数字

```bash
$ python3 -c "import json; print(json.load(open('direct_answer_9b.json'))['per_task_acc']['Bouncing_Counting'])"
40.5
$ python3 -c "import json; print(json.load(open('direct_answer_35b_35.json'))['per_task_acc']['Bouncing_Counting'])"
29.0
```
9B: 报告写46.50%,文件实际40.50%。35B: 报告写34.50%,文件实际29.00%。**其余6个原语类型(Rotation_Direction/Count、Acceleration_Identification等)逐一核对,报告数字和文件完全一致,只有Bouncing_Counting这一项对不上。**

### 2.3 证据链:为什么会不一致(不是随便猜的,是时间戳锁死的)

按mtime从早到晚排列`probe_data/`下的相关文件:

| 时间(UTC) | 文件 | 内容 |
|---|---|---|
| 07-13 02:32:15 | `direct_answer_35b35.log` | 35B模型**全量**跑了7×200=1400条,日志里`Bouncing_Counting acc=34.50%`,和报告数字吻合 |
| 07-13 03:51:27 | `hidden_states/9b/Bouncing_Counting.pt` | 重新生成/重新抽取(比同目录其他6个task的.pt文件晚了将近22小时) |
| 07-13 03:55:24 | `hidden_states/35b_35/Bouncing_Counting.pt` | 同上,重新生成/重新抽取 |
| 07-13 03:56:09 | `direct_answer_baseline.py` | 脚本本身被改过(报告第148行提到"新增了`--task=`过滤参数,避免重跑其余1200条") |
| 07-13 03:58:53 | `direct_answer_9b.json` | 在脚本改动**之后**生成,9B的Bouncing_Counting=**40.50%** |
| 07-13 04:02:54 | `direct_answer_35b_35.json` | 在脚本改动**之后**生成,35B的Bouncing_Counting=**29.00%** |

再看`direct_answer_35b_35.json`里除Bouncing_Counting外的另外6个task,和02:32那份log**逐字节吻合**(Acceleration_Identification 64.00%、Complex_Direction_Identification 100.00%、Directional_Event_Counting 82.50%、Event_Sequence 100.00%、Rotation_Count 69.00%、Rotation_Direction 52.00%,一个不差)。这精确对应`direct_answer_baseline.py`第137-143行的合并逻辑:

```python
if task_filter and os.path.exists(out_path):
    with open(out_path) as f:
        existing = json.load(f)
    for key in ["per_task_acc", "per_task_n", "per_task_unresolved"]:
        existing.setdefault(key, {}).update(new_data[key])
    ...
```
也就是说:**02:32那次是全量跑,产出的JSON被之后一次`--task=Bouncing_Counting`的补跑用这段合并逻辑"打了个补丁"——只更新了Bouncing_Counting这一个key,其余6个原样保留**。这完全解释了为什么6个task数字不变、只有1个变了。

### 2.4 这次补跑是为什么

报告正文自己也交代了这件事(第5行、第145-149行):Bouncing_Counting最初因为生成器没把初始速度`random.uniform(-4,4)`存进metadata,后来按要求给生成器加了3行代码把逐帧`(x,y,vx,vy)`真值存下来,**重新生成了这200条样本**,重新抽了hidden state,也重新跑了一次直答baseline。

问题在于:报告描述"补做"这件事本身是准确的,但**补做之后,报告正文表格里的数字没有跟着换成补做后的结果**,还留着补做前(即02:32那次全量跑)的旧数字。分类probe那边(第4.2节)报告明确写了"新旧两批数据上高度一致(95-100%)"(第193行),说明probe这条线的数字后来是有人去核对更新过的;但直答baseline这条线,似乎补跑完之后忘了把表格里的46.50%/34.50%换成新产出的40.50%/29.00%。

### 2.5 这处错误影响结论吗

不影响方向,反而让结论更站得住:
- 9B: 100%(probe) vs 40.50%(直答,不是46.50%)——gap比报告写的更大。
- 35B: 100%(probe) vs 29.00%(直答,不是34.50%)——gap比报告写的更大,"teacher比student更差"这个反常现象(29.00% < 40.50%)依然成立,而且差距比报告写的(34.50% vs 46.50%)更悬殊。

`check_wrong_answer_semantics.py`(错误语义分类,第76行直接读`direct_answer_{model_tag}.json`)的执行时间(脚本mtime 04:57,晚于两个json文件的03:58/04:02)晚于补跑完成时间,所以它读到的是补跑后的最终数据,报告里"66-85%是差>1的离谱错误、误差系统性偏正方向"这段定性分析**不受这处数字过期问题影响**,是基于最终数据算出来的。

**建议**:告诉负责报告的会话,把第114行表格的46.50%/34.50%改成40.50%/29.00%,第121行"34.5% vs 46.5%"也要同步改成"29.0% vs 40.5%"。这是我核实中发现的唯一一处硬数字问题。

---

## 3. 其他关键环节的代码核查(逐项确认无造假/无偷懒)

### 3.1 回归probe(支撑PGR方向,`train_regression_probes.py`)

物理量目标(比如Rotation_Count用累积角度、Acceleration_Identification用速度标量)都是从`ground_truth_details`里存的确定性生成参数**解析重建**的,不是模型输出、也不是猜的——渲染器的运动公式是确定性的(比如`target_rotation_count`第59-67行直接用`total_rotations`和帧数反推角速度,再乘帧号得到累积角度)。切分同样是video-level的70/30(第173行),避免同一个视频的不同temporal group同时出现在训练和验证集里造成信息泄漏。这个防泄漏的处理是对的,我特意确认了这点。

Bouncing_Counting在这个脚本里第100-109行用的是"生成器改造后直接dump的真值"(`physical_state_per_frame`),不是解析重建——脚本顶部docstring(第9-13行)还留着"跳过Bouncing_Counting"的旧说明没删,和第129-136行`TARGET_FNS`字典里实际已经包含它的代码不一致,这是一处**文档滞后**(纯注释问题,不影响输出正确性,只是能看出这个脚本也是在补做Bouncing_Counting时被现改的)。

### 3.2 Position-confound检查(`check_rotation_confound.py`)

这是用户之前质疑"旋转类probe的高R²会不会只是位置编码的副产品"之后,补的检查。代码做了三件事,都读过了:
1. `position_only_baseline()`:只用`(g+0.5)/T`这个归一化位置标量(完全不用hidden state)去回归角度,结果Rotation_Count的R²=0.782——证实confound确实存在,报告如实写了这一点,没有藏着掖着。
2. `residual_test()`:先用position-only回归扣掉能被位置解释的部分,再看hidden state能不能解释**残差**——如果这里R²还很高,说明hidden state里确实有位置之外的、和视觉输入相关的真实信息。
3. `sign_oracle_direction()`:专门针对Rotation_Direction(它的角速度大小是常数,只有正负号在变),模拟"完美知道方向 vs 按不同噪声率掺假"几种情况下纯位置能重建出多高的R²,和实际观测到的R²对比,判断观测值是不是"纯靠位置+蒙对方向"就能解释。

这三步是层层递进的,不是随便加个baseline应付了事,数学上是在问"扣掉confound之后还剩多少",这个思路是对的。

### 3.3 直答错误语义分类(`check_wrong_answer_semantics.py`)

核心操作:因为每条样本的选项字母是**随机打乱**的(同一个字母A在不同样本里可能对应不同的语义选项),不能直接比较模型选了哪个字母,必须先把预测字母和真值字母都还原成语义内容再统计。代码第20-32行从原始`sft.jsonl`的prompt文本里把每条样本的"字母→语义"映射解析出来(`id_to_options`字典),后续统计(比如Rotation_Direction的"误选成另一个真实方向 vs 误选成距离无关的distractor")全部基于还原后的语义值,不是字母。这排除了"文字混淆/选项顺序"这个解释,报告用它来支撑"模型确实是不会做,不是选项格式坑了它",这个论证方式是对的。

---

## 4. 总体核实结论

方法论上,报告是可信的:两条独立代码路径(冻结前向 vs 真实生成)分别衡量"信息在不在表示层"和"模型能不能自己读出/生成",工具链和P1/P2共用官方答案提取器,统计口径(70/30分层切分、majority baseline对照、position-confound排查、语义还原后的错误分类)都是站得住的处理,没有发现为了让结论好看而做的手脚。

数字上,我核实过程中发现`Bouncing_Counting`的直答准确率一项过期(46.50%/34.50%应为40.50%/29.00%),根因是"先全量跑、后来针对这一个原语单独补跑"的合并逻辑更新了数据文件,但报告正文表格没跟着换新数字。这处需要修正,但方向不影响结论——差距比报告写的还大。除此之外没有发现其他数字不一致。

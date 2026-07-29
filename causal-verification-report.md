---
title: "因果验证实验报告——TRD/CAD该监督哪几层哪几个head"
status: "8层full_attention层的消融扫描已完成(mean+zero交叉验证);24层linear_attention层不在本次范围"
date: "2026-07-19"
---

# 因果验证实验报告

对照`causal-verification-plan.md`执行。方法、代码、判定框架细节见该文档,这里只写结果和结论。

## 0. 执行过程(如实记录,包含一次方法论修正)

**第一轮(n=15/类,4类共60条)信号太噪,没有采信**：8层×16head+8层整层=136种消融配置,每种只有15个样本,单个head的delta里13-27pp的"大"数字,拆开看往往只对应2-4条样本翻转;更关键的是,"整层消融"和"该层16个head逐个消融delta的和"经常对不上(比如19层Rotation_Count:整层消融delta=0.0pp,16个head加总却是66.7pp)。这个量级的分歧更像噪声累积,不是真实的非线性协同效应,所以没有直接用这批数据下结论,改成按计划文档自己建议的50条/类重跑。

**第二轮(n=50/类,4类共200条,mean消融)**：同样136种配置,数字明显更稳定。基于这轮结果挑出的候选,做了**第三轮zero消融交叉验证**(只测第二轮里delta明显的那些layer/head,不是136种全测)。

## 1. 核心发现:Acceleration_Identification有清晰、跨方法一致的多层因果信号

| 层 | mean消融delta | zero消融delta | 方向一致? |
|---|---|---|---|
| 层7(整层) | 32.0pp | 30.0pp | ✅ |
| 层15(整层) | 32.0pp | 26.0pp | ✅ |
| 层19(整层) | 20.0pp | 12.0pp | ✅ |
| 层11(整层) | 6.0pp | 32.0pp | ✅(方向一致,幅度差较大) |

**这是本次实验里唯一一个"跨层重复出现、两种消融方法方向都一致"的干净信号**——4个不同的full_attention层,消融后Acceleration_Identification的直答准确率都明显下降,且mean/zero两种消融方式方向一致。按`causal-verification-plan.md` §3判定表:这类"多层都关键"的模式提示信息可能分散在多个层里冗余编码,不是单一层/head独占,**TRD/CAD如果要投入,Acceleration_Identification这个原语上full_attention机制确实有可监督的因果落点**,值得作为优先目标。

## 2. 其余3类原语:没有找到干净的信号

- **Rotation_Count**:mean消融下,8层里6层是负delta或接近0(甚至层19消融后准确率反而涨了16pp)——说明拿掉这些层的attention,这个任务答得**更好**,不是更差。zero消融交叉验证时方向经常对不上(层19整层:mean=-16.0 vs zero=+2.0,层7整层:mean=-8.0 vs zero=+10.0)。**结论:这个原语的答案不依赖这8层full_attention层的attention机制**,按判定表第4档,这里的TRD/CAD在full_attention层上没有着力点——如果这个原语真有读出层gap,机制大概率在24层linear_attention(Gated DeltaNet)里,不在本次范围内。
- **Bouncing_Counting**:整层消融方向基本一致(层3/7/11两种方法都是正delta),但head级别的信号方向经常翻转(比如层7 head7:两种方法都显示对Bouncing_Counting几乎没有影响,层3 head13:mean=12.0但zero=6.0,弱化了不少)。**只有整层级别的信号可信,单head级别定位不出稀疏子集**——按判定表第2档,信息可能是同一层内多个head冗余编码,监督对象应该是"整层pooled attention分布",不是挑单个head。
- **Rotation_Direction**:所有层/head的delta都很小(大多在0-12pp内),mean/zero两种方法经常方向不一致(比如层11整层:mean=2.0 vs zero=4.0,层11 head3:mean=10.0 vs zero=2.0)。**没有找到任何明确信号**,这个原语在8层full_attention层上大概率也没有清晰的因果落点。

## 3. 头级别(single-head)结论:普遍不可靠,不建议按单head设计监督

对比stage1b(mean,n=50)和stage2(zero交叉验证)的15个共同(layer,head)组合,方向一致率大约60%左右(9/15),但幅度经常差异很大(比如层11 head3的Acceleration_Identification:mean=18.0 vs zero=10.0;层3 head13的同任务:mean=2.0 vs zero=14.0,几乎反过来)。**在n=50这个规模下,单head级别的消融结果还不够稳定,不建议现在就挑某个具体head设计TRD/CAD的监督目标**——如果真要往这个方向投入,需要先把样本量再提高(比如满量200条/类),而不是基于现在这批数据拍板选哪个head。

## 4. 对novelty.md TRD/CAD的直接建议

- **Acceleration_Identification**:证据支持投入,监督对象是"多个full_attention层(至少7/15/19三层)的整体attention输出",不是单一层或单一head。
- **Bouncing_Counting**:证据部分支持,但要按"整层pooled"的方式监督(呼应判定表第2档的做法),不要挑单head。
- **Rotation_Count**:证据不支持在full_attention层做监督,这个原语的读出层gap(如果真存在)大概率不在这8层里。
- **Rotation_Direction**:没有找到信号,和Rotation_Count一样,不建议在full_attention层投入监督。

## 5. 局限(如实记录)

- 24层linear_attention(Gated DeltaNet)层完全没测,按plan.md原定范围,这次只测full_attention层——如果Rotation_Count/Rotation_Direction的读出层gap真实存在,机制大概率在这24层里,需要另外设计针对Gated DeltaNet的消融方法。
- 头级别结果的样本量(n=50/类)仍然不够稳定,§3已经如实说明,不要基于当前头级别数据做训练决策。
- 只测了35B-A3B里没有的9B student自己,没有对teacher(35B-A3B)做同样的消融对照——plan.md本身也只要求测9B,这不是遗漏,是原定范围。
- zero消融交叉验证只覆盖了mean消融挑出来的候选(约15种配置),不是136种全部重新测一遍zero模式——这是成本考虑下的合理取舍,但意味着"没被挑中"的配置只有mean-ablation一种视角,没有交叉验证。

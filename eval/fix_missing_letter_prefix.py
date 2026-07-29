#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复FINAL_TRAINING_DATASET.json里options/answer缺字母前缀的bug:
PerceptionTest_object_shuffle_character_order(178条)和PerceptionTest_Bouncing_Counting_supplement(165条)
的options是纯文本选项、answer是纯文本(和某个option精确相等),没有"A. "这样的字母前缀。
这导致prompt里选项本身没有字母标号,却要求模型"回答字母",teacher/student当时是在瞎猜。
用answer在options里的下标重建字母前缀,写回原地。
"""
import json, re

PATH = "/remote-home/ziyesong/videoPerception/FINAL_TRAINING_DATASET.json"
data = json.load(open(PATH))

fixed_ids = []
for e in data:
    if re.match(r'^[A-H][\.\)]', e['answer']):
        continue
    idx = e['options'].index(e['answer'])
    new_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(e['options'])]
    new_answer = new_options[idx]
    e['options'] = new_options
    e['answer'] = new_answer
    fixed_ids.append(e['id'])

json.dump(data, open(PATH, "w"), indent=2, ensure_ascii=False)
print(f"修复了 {len(fixed_ids)} 条")

# 校验: 全数据集现在应该没有任何一条还缺字母前缀
bad = [e for e in data if not re.match(r'^[A-H][\.\)]', e['answer'])]
print(f"修复后仍缺前缀的条数(应为0): {len(bad)}")

json.dump(fixed_ids, open("/remote-home/ziyesong/videoPerception/eval/letterfix_ids.json", "w"))
print("受影响的id列表已写入 eval/letterfix_ids.json")

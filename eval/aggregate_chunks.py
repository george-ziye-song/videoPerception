#!/usr/bin/env python3
"""汇总分片跑法产出的结果——每个分片自己打印的 aggregate 只是那一片的局部分数,
这里读所有分片的 samples_*.jsonl、用已经修复过的官方 parser(共享 extract_mcq_answer)
重新算整体分数,并把每一条答错的题目(doc_id/task+question+正确答案+模型输出)记录下来
(2026-07-09 补:用户要求正式P2评估的错题必须全部记录,不只是消融测试)。

用法: python3 aggregate_chunks.py <chunks_root_dir> <mvbench|tomato|videomme>
例:   python3 aggregate_chunks.py results_p2_transformers_9b_chunks mvbench

输出:
  - 终端打印总体/分任务准确率
  - <chunks_root_dir>_wrong_answers.json  结构化错题记录(doc_id/task/question/正确答案/模型输出/token数)
  - <chunks_root_dir>_wrong_answers.md    可读表格
"""
import sys
import json
import glob
import os
import collections

sys.path.insert(0, "/root/miniconda3/envs/lmmseval/lib/python3.12/site-packages")


def _save_wrong(chunks_root, bench, wrong_records):
    json_path = f"{chunks_root}_wrong_answers.json"
    md_path = f"{chunks_root}_wrong_answers.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(wrong_records, f, ensure_ascii=False, indent=2)

    lines = [f"# {bench} 正式评估错题记录\n", f"共 {len(wrong_records)} 条\n"]
    lines.append("| doc_id/task | 题目 | 正确答案 | 模型输出(尾部200字符) | 生成token数 |")
    lines.append("|---|---|---|---|---|")
    for r in wrong_records:
        tail = (r["model_output"] or "")[-200:].replace("\n", " ").replace("|", "/")
        q = (r["question"] or "")[:150].replace("\n", " ").replace("|", "/")
        lines.append(f"| {r['id']} | {q} | {r['gt']} | {tail} | {r.get('token_count','?')} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n错题已保存: {json_path} / {md_path}")


def aggregate_mvbench(chunks_root):
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    per_task_scores = collections.defaultdict(list)
    wrong_records = []
    for samples_file in glob.glob(f"{chunks_root}/mvbench_*/models__*/*samples_mvbench_*.jsonl"):
        task_name = os.path.basename(samples_file).split("samples_")[-1].replace(".jsonl", "")
        with open(samples_file) as f:
            for line in f:
                d = json.loads(line)
                pred_text = d.get("filtered_resps", "") or ""
                gt = d.get("mvbench_accuracy", {}).get("gt_answer")
                if gt is None:
                    continue
                pred_letter = extract_mcq_answer(pred_text, choices=["A", "B", "C", "D", "E"])
                score = 1 if pred_letter and pred_letter == gt.upper() else 0
                per_task_scores[task_name].append(score)
                if score == 0:
                    tc = d.get("token_counts")
                    wrong_records.append({
                        "id": f"{task_name}#{d['doc_id']}",
                        "question": d.get("input", ""),
                        "gt": gt,
                        "pred_letter": pred_letter or "(未解析)",
                        "model_output": pred_text,
                        "token_count": tc[0]["output_tokens"] if tc and tc[0] else None,
                    })

    print(f"{'task':<28} {'n':>6} {'acc':>8}")
    task_means = []
    for task_name in sorted(per_task_scores):
        scores = per_task_scores[task_name]
        mean = 100 * sum(scores) / len(scores)
        task_means.append(mean)
        print(f"{task_name:<28} {len(scores):>6} {mean:>7.2f}%")
    if task_means:
        print(f"\n{'MEAN OF TASKS':<28} {'':>6} {sum(task_means)/len(task_means):>7.2f}%")
    total_n = sum(len(v) for v in per_task_scores.values())
    print(f"total subtasks found: {len(per_task_scores)}/20, total samples: {total_n}/4000")
    _save_wrong(chunks_root, "mvbench", wrong_records)


def aggregate_tomato(chunks_root):
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/TOMATO", split="test")
    docs = {i: ds[i] for i in range(len(ds))}

    scores = []
    unresolved = 0
    seen_doc_ids = set()
    wrong_records = []
    for samples_file in glob.glob(f"{chunks_root}/tomato_p*/models__*/*samples_tomato.jsonl"):
        with open(samples_file) as f:
            for line in f:
                d = json.loads(line)
                doc_id = d["doc_id"]
                if doc_id in seen_doc_ids:
                    continue  # 分片边界重叠保险,理论不该发生
                seen_doc_ids.add(doc_id)
                doc = docs[doc_id]
                resp = d.get("filtered_resps", "") or ""
                gt_letter = chr(65 + doc["answer"])
                n_opts = len(doc["options"])
                choices = [chr(65 + i) for i in range(n_opts)]
                pred = extract_mcq_answer(resp, choices=choices)
                correct = bool(pred) and pred == gt_letter
                if not pred:
                    unresolved += 1
                scores.append(1.0 if correct else 0.0)
                if not correct:
                    tc = d.get("token_counts")
                    wrong_records.append({
                        "id": doc_id,
                        "question": doc["question"],
                        "gt": f"{gt_letter}. {doc['options'][doc['answer']]}",
                        "pred_letter": pred or "(未解析)",
                        "model_output": resp,
                        "token_count": tc[0]["output_tokens"] if tc and tc[0] else None,
                        "video_path": doc["video_path"],
                    })

    n = len(scores)
    print(f"n={n}/1484  accuracy={100*sum(scores)/n:.2f}%  unresolved={unresolved}({100*unresolved/n:.1f}%)" if n else "no samples found")
    _save_wrong(chunks_root, "tomato", wrong_records)


def aggregate_videomme(chunks_root):
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/Video-MME", split="test")
    docs = {i: ds[i] for i in range(len(ds))}

    correct = 0
    unresolved = 0
    seen_doc_ids = set()
    wrong_records = []
    total = 0
    for samples_file in glob.glob(f"{chunks_root}/videomme_p*/models__*/*samples_videomme.jsonl"):
        with open(samples_file) as f:
            for line in f:
                d = json.loads(line)
                doc_id = d["doc_id"]
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                doc = docs[doc_id]
                resp = d.get("filtered_resps", "") or ""
                gt_letter = doc["answer"].upper()
                n_opts = len(doc.get("options", [])) or 4
                choices = [chr(65 + i) for i in range(max(n_opts, 4))]
                pred = extract_mcq_answer(resp, choices=choices)
                total += 1
                if not pred:
                    unresolved += 1
                is_correct = bool(pred) and pred.upper() == gt_letter
                if is_correct:
                    correct += 1
                else:
                    tc = d.get("token_counts")
                    wrong_records.append({
                        "id": doc_id,
                        "question": doc.get("question", ""),
                        "gt": gt_letter,
                        "pred_letter": pred or "(未解析)",
                        "model_output": resp,
                        "token_count": tc[0]["output_tokens"] if tc and tc[0] else None,
                        "videoID": doc.get("videoID", ""),
                    })

    print(f"n={total}/2700  accuracy={100*correct/total:.2f}%  unresolved={unresolved}({100*unresolved/total:.1f}%)" if total else "no samples found")
    _save_wrong(chunks_root, "videomme", wrong_records)


if __name__ == "__main__":
    chunks_root, bench = sys.argv[1], sys.argv[2]
    if bench == "mvbench":
        aggregate_mvbench(chunks_root)
    elif bench == "tomato":
        aggregate_tomato(chunks_root)
    elif bench == "videomme":
        aggregate_videomme(chunks_root)
    else:
        print("bench must be mvbench|tomato|videomme")
        sys.exit(1)

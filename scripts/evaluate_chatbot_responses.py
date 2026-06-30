#!/usr/bin/env python3
"""Evaluate ChatBotINF responses with a reproducible question dataset.

The metric is intentionally simple and transparent for the paper:
- A case passes when every expected group has at least one phrase in the answer
  and no forbidden phrase appears.
- Keyword coverage reports how many expected phrases appeared overall.
- Forbidden hit rate estimates hallucination / unsafe / out-of-domain leakage.

This does not replace human review, but it creates a repeatable baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_chatbot  # noqa: E402


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains(answer_norm: str, phrase: str) -> bool:
    return normalize(phrase) in answer_norm


def score_case(case: dict[str, Any], answer: str) -> dict[str, Any]:
    answer_norm = normalize(answer)
    expected_groups = case.get("expected_any", []) or []
    forbidden = case.get("forbidden", []) or []

    group_results = []
    expected_phrase_hits = 0
    expected_phrase_total = 0

    for group in expected_groups:
        hits = [phrase for phrase in group if contains(answer_norm, phrase)]
        expected_phrase_hits += len(hits)
        expected_phrase_total += len(group)
        group_results.append({
            "group": group,
            "matched": hits,
            "passed": bool(hits),
        })

    forbidden_hits = [phrase for phrase in forbidden if contains(answer_norm, phrase)]
    expected_groups_passed = all(item["passed"] for item in group_results) if group_results else True
    passed = expected_groups_passed and not forbidden_hits
    coverage = expected_phrase_hits / expected_phrase_total if expected_phrase_total else 1.0

    return {
        "passed": passed,
        "expected_groups_passed": expected_groups_passed,
        "forbidden_hits": forbidden_hits,
        "group_results": group_results,
        "keyword_coverage": round(coverage, 4),
    }


def write_outputs(results: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total = len(results)
    passed = sum(1 for item in results if item["score"]["passed"])
    forbidden_cases = sum(1 for item in results if item["score"]["forbidden_hits"])
    avg_coverage = statistics.mean(item["score"]["keyword_coverage"] for item in results) if results else 0.0

    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        category = item["category"]
        bucket = by_category.setdefault(category, {"total": 0, "passed": 0, "coverage_values": []})
        bucket["total"] += 1
        bucket["passed"] += int(item["score"]["passed"])
        bucket["coverage_values"].append(item["score"]["keyword_coverage"])

    category_summary = {}
    for category, bucket in sorted(by_category.items()):
        category_summary[category] = {
            "total": bucket["total"],
            "passed": bucket["passed"],
            "accuracy": round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0,
            "avg_keyword_coverage": round(statistics.mean(bucket["coverage_values"]), 4),
        }

    summary = {
        "generated_at": stamp,
        "total_cases": total,
        "passed_cases": passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "avg_keyword_coverage": round(avg_coverage, 4),
        "forbidden_hit_rate": round(forbidden_cases / total, 4) if total else 0.0,
        "category_summary": category_summary,
    }

    json_path = out_dir / f"chatbot_eval_results_{stamp}.json"
    csv_path = out_dir / f"chatbot_eval_results_{stamp}.csv"
    md_path = out_dir / f"chatbot_eval_summary_{stamp}.md"

    json_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "id", "category", "passed", "keyword_coverage", "forbidden_hits", "question", "answer", "notes"
        ])
        writer.writeheader()
        for item in results:
            writer.writerow({
                "id": item["id"],
                "category": item["category"],
                "passed": item["score"]["passed"],
                "keyword_coverage": item["score"]["keyword_coverage"],
                "forbidden_hits": "; ".join(item["score"]["forbidden_hits"]),
                "question": item["question"],
                "answer": item["answer"].replace("\n", " "),
                "notes": item.get("notes", ""),
            })

    lines = [
        "# Evaluación automática de ChatBotINF",
        "",
        f"Fecha de ejecución: `{stamp}`",
        f"Casos evaluados: **{total}**",
        f"Accuracy por rúbrica: **{summary['accuracy'] * 100:.2f}%**",
        f"Cobertura promedio de palabras esperadas: **{summary['avg_keyword_coverage'] * 100:.2f}%**",
        f"Tasa de términos prohibidos: **{summary['forbidden_hit_rate'] * 100:.2f}%**",
        "",
        "## Resultados por categoría",
        "",
        "| Categoría | Casos | Correctos | Accuracy | Cobertura promedio |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, bucket in category_summary.items():
        lines.append(
            f"| {category} | {bucket['total']} | {bucket['passed']} | {bucket['accuracy'] * 100:.2f}% | {bucket['avg_keyword_coverage'] * 100:.2f}% |"
        )

    failed = [item for item in results if not item["score"]["passed"]]
    lines.extend(["", "## Casos que requieren revisión", ""])
    if not failed:
        lines.append("No hubo casos fallidos según la rúbrica automática.")
    else:
        for item in failed:
            lines.extend([
                f"### {item['id']} - {item['category']}",
                f"Pregunta: {item['question']}",
                f"Términos prohibidos detectados: {', '.join(item['score']['forbidden_hits']) or 'ninguno'}",
                f"Cobertura: {item['score']['keyword_coverage'] * 100:.2f}%",
                "Respuesta:",
                "```text",
                item["answer"].strip(),
                "```",
                "",
            ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": summary, "json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChatBotINF evaluation questions.")
    parser.add_argument("--dataset", default="tests/chatbot_eval_questions.json")
    parser.add_argument("--out-dir", default="evaluation_results")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--category", action="append", help="Run only selected category. Can be repeated.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between questions.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    dataset_path = PROJECT_ROOT / args.dataset
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.category:
        selected = set(args.category)
        cases = [case for case in cases if case.get("category") in selected]
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"Cargando chatbot con DB_DIR={os.getenv('DB_DIR')} y modelo={os.getenv('LLM_MODEL_NAME')}", flush=True)
    llm = setup_chatbot()

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['question']}", flush=True)
        start = time.time()
        try:
            answer = llm.generate_response(case["question"])
            error = None
        except Exception as exc:  # keep the report useful even if one case breaks
            answer = ""
            error = repr(exc)
        elapsed = round(time.time() - start, 2)
        score = score_case(case, answer)
        result = {
            "id": case["id"],
            "category": case["category"],
            "discord_message": case.get("discord_message", ""),
            "question": case["question"],
            "answer": answer,
            "elapsed_seconds": elapsed,
            "score": score,
            "notes": case.get("notes", ""),
            "error": error,
        }
        print(f"    pass={score['passed']} coverage={score['keyword_coverage']:.2f} forbidden={score['forbidden_hits']} time={elapsed}s", flush=True)
        results.append(result)
        if args.sleep:
            time.sleep(args.sleep)

    output = write_outputs(results, PROJECT_ROOT / args.out_dir)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

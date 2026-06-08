"""
RAG Evaluation Pipeline — DeepEval

Chạy:
    python group_project/evaluation/eval_pipeline.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(question: str, use_reranking: bool = True) -> dict:
    from src.task10_generation import generate_with_citation

    return generate_with_citation(question, use_reranking=use_reranking)


def evaluate_with_deepeval(golden_dataset: list[dict], use_reranking: bool = True) -> dict:
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_cases = []
    for item in golden_dataset:
        print(f"  Running: {item['question'][:60]}...")
        result = run_pipeline(item["question"], use_reranking=use_reranking)
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=[c["content"] for c in result.get("sources", [])],
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]

    eval_result = evaluate(test_cases=test_cases, metrics=metrics)

    scores = {m.__name__: [] for m in metrics}
    per_case = []
    for tc in test_cases:
        case_scores = {}
        for metric in metrics:
            try:
                metric.measure(tc)
                case_scores[metric.__name__] = metric.score
                scores[metric.__name__].append(metric.score or 0)
            except Exception:
                case_scores[metric.__name__] = 0.0
                scores[metric.__name__].append(0.0)
        per_case.append({"question": tc.input, **case_scores})

    avg = {k: sum(v) / len(v) if v else 0 for k, v in scores.items()}
    avg["overall"] = sum(avg.values()) / len(avg) if avg else 0
    return {"averages": avg, "per_case": per_case, "raw": eval_result}


def compare_configs(golden_dataset: list[dict]) -> dict:
    print("\n=== Config A: Hybrid + Reranking ===")
    config_a = evaluate_with_deepeval(golden_dataset, use_reranking=True)
    print("\n=== Config B: Hybrid, no Reranking ===")
    config_b = evaluate_with_deepeval(golden_dataset, use_reranking=False)
    return {"hybrid_rerank": config_a, "hybrid_no_rerank": config_b}


def _worst(per_case: list[dict], n: int = 3) -> list[dict]:
    return sorted(per_case, key=lambda x: x.get("FaithfulnessMetric", 0))[:n]


def export_results(comparison: dict):
    a = comparison["hybrid_rerank"]["averages"]
    b = comparison["hybrid_no_rerank"]["averages"]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Tổng Quan",
        "",
        "| | |",
        "|-|-|",
        "| **Framework** | DeepEval |",
        f"| **Số test cases** | {len(comparison['hybrid_rerank']['per_case'])} cặp Q&A |",
        "| **Metrics** | Faithfulness, Answer Relevancy, Context Recall, Context Precision |",
        "| **Config A** | Hybrid Search + Reranking |",
        "| **Config B** | Hybrid Search, không reranking |",
        "",
        "---",
        "",
        "## Bảng Điểm Tổng Hợp",
        "",
        "| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |",
        "|--------|-------------|-----------------|---------------|------------------|---------|",
        f"| Config A — Hybrid + Reranking | {a.get('FaithfulnessMetric', 0):.3f} | {a.get('AnswerRelevancyMetric', 0):.3f} | {a.get('ContextualRecallMetric', 0):.3f} | {a.get('ContextualPrecisionMetric', 0):.3f} | {a.get('overall', 0):.3f} |",
        f"| Config B — Hybrid, no Reranking | {b.get('FaithfulnessMetric', 0):.3f} | {b.get('AnswerRelevancyMetric', 0):.3f} | {b.get('ContextualRecallMetric', 0):.3f} | {b.get('ContextualPrecisionMetric', 0):.3f} | {b.get('overall', 0):.3f} |",
        "",
        "---",
        "",
        "## So Sánh A/B",
        "",
        "| Metric | Config A | Config B | Delta (B−A) |",
        "|--------|---------|---------|-------------|",
    ]

    for key, label in [
        ("FaithfulnessMetric", "Faithfulness"),
        ("AnswerRelevancyMetric", "Answer Relevancy"),
        ("ContextualRecallMetric", "Context Recall"),
        ("ContextualPrecisionMetric", "Context Precision"),
        ("overall", "**Overall**"),
    ]:
        va, vb = a.get(key, 0), b.get(key, 0)
        lines.append(f"| {label} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} |")

    winner = "A (Hybrid + Reranking)" if a.get("overall", 0) >= b.get("overall", 0) else "B"
    lines += [
        "",
        f"**Kết luận:** Config **{winner}** cho kết quả tốt hơn.",
        "",
        "---",
        "",
        "## Worst Performers — Config A",
        "",
        "| Câu hỏi | Faithfulness | Answer Relevancy |",
        "|---------|-------------|-----------------|",
    ]
    for row in _worst(comparison["hybrid_rerank"]["per_case"]):
        lines.append(
            f"| {row['question'][:60]}… | {row.get('FaithfulnessMetric', 0):.3f} | {row.get('AnswerRelevancyMetric', 0):.3f} |"
        )

    lines += [
        "",
        "## Worst Performers — Config B",
        "",
        "| Câu hỏi | Faithfulness | Answer Relevancy |",
        "|---------|-------------|-----------------|",
    ]
    for row in _worst(comparison["hybrid_no_rerank"]["per_case"]):
        lines.append(
            f"| {row['question'][:60]}… | {row.get('FaithfulnessMetric', 0):.3f} | {row.get('AnswerRelevancyMetric', 0):.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Đề Xuất Cải Tiến",
        "",
        "1. **Bật reranking** nếu faithfulness thấp — cross-encoder chọn chunk phù hợp hơn.",
        "2. **Mở rộng corpus bài báo** cho câu hỏi về nghệ sĩ và tin tức.",
        "3. **Tăng chunk overlap** để không cắt mất điều khoản pháp luật.",
        "4. **Hạ score threshold** để kích hoạt PageIndex fallback sớm hơn.",
        "",
    ]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    if len(golden_dataset) < 15:
        print("⚠ Cần tối thiểu 15 cặp Q&A trong golden_dataset.json")

    comparison = compare_configs(golden_dataset)
    export_results(comparison)

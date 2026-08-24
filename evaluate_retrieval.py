"""Offline retrieval evaluation and safe weight tuning (no LLM/API calls).

Run:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --tune
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from app import ROOT, Retriever, Settings


CASES_PATH = ROOT / "evaluation" / "retrieval_cases.json"
REPORT_PATH = ROOT / "evaluation" / "retrieval_report.json"
WEIGHT_PRESETS = [
    {"semantic": 0.30, "bm25": 0.30, "rrf": 0.15, "exact": 0.15, "route": 0.10, "focus": 0.10},
    {"semantic": 0.25, "bm25": 0.35, "rrf": 0.15, "exact": 0.15, "route": 0.10, "focus": 0.10},
    {"semantic": 0.25, "bm25": 0.30, "rrf": 0.15, "exact": 0.20, "route": 0.10, "focus": 0.10},
    {"semantic": 0.30, "bm25": 0.26, "rrf": 0.15, "exact": 0.15, "route": 0.10, "focus": 0.14},
    {"semantic": 0.20, "bm25": 0.40, "rrf": 0.15, "exact": 0.15, "route": 0.10, "focus": 0.10},
]


@dataclass
class CaseResult:
    case_id: str
    hit_at_3: float
    reciprocal_rank: float
    precision_at_3: float
    forbidden_rate: float
    threshold_passed: bool
    top_score: float
    top_sources: list[str]


def is_relevant(item: Any, case: dict[str, Any]) -> bool:
    metadata = item.chunk.metadata
    if metadata.get("document_type") not in case["expected_types"]:
        return False
    expected_codes = set(case.get("expected_codes", []))
    if expected_codes and expected_codes.isdisjoint(metadata.get("codes", [])):
        return False
    terms = [term.lower() for term in case.get("must_contain_any", [])]
    if terms and not any(term in item.chunk.search_text.lower() for term in terms):
        return False
    return True


def evaluate(retriever: Retriever, cases: list[dict[str, Any]], weights: dict[str, float]) -> tuple[float, list[CaseResult]]:
    results: list[CaseResult] = []
    for case in cases:
        ranked = retriever.rank_local(case["query"], weights=weights)[:3]
        relevant = [is_relevant(item, case) for item in ranked]
        first_rank = next((index for index, value in enumerate(relevant, start=1) if value), None)
        forbidden = set(case.get("forbidden_types", []))
        forbidden_count = sum(item.chunk.metadata.get("document_type") in forbidden for item in ranked)
        results.append(
            CaseResult(
                case_id=case["id"],
                hit_at_3=float(any(relevant)),
                reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
                precision_at_3=sum(relevant) / max(1, len(ranked)),
                forbidden_rate=forbidden_count / max(1, len(ranked)),
                threshold_passed=bool(ranked and ranked[0].score >= retriever.settings.rag_threshold),
                top_score=round(ranked[0].score, 4) if ranked else 0.0,
                top_sources=[
                    f"{item.chunk.metadata.get('document_type')}:{item.chunk.metadata.get('page', '-')}"
                    for item in ranked
                ],
            )
        )
    count = max(1, len(results))
    hit = sum(item.hit_at_3 for item in results) / count
    mrr = sum(item.reciprocal_rank for item in results) / count
    precision = sum(item.precision_at_3 for item in results) / count
    forbidden_rate = sum(item.forbidden_rate for item in results) / count
    threshold_rate = sum(item.threshold_passed for item in results) / count
    objective = 0.30 * hit + 0.30 * mrr + 0.20 * precision + 0.20 * threshold_rate - 0.40 * forbidden_rate
    return objective, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local hybrid RCM retrieval")
    parser.add_argument("--tune", action="store_true", help="compare safe weight presets")
    args = parser.parse_args()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    settings = Settings()
    retriever = Retriever(settings, ROOT / "knowledge_base")
    configured_weights = {
        "semantic": settings.semantic_weight,
        "bm25": settings.bm25_weight,
        "rrf": settings.rrf_weight,
        "exact": settings.exact_code_weight,
        "route": settings.source_route_weight,
        "focus": settings.focus_term_weight,
    }
    presets = WEIGHT_PRESETS if args.tune else [configured_weights]
    evaluated = [(weights, *evaluate(retriever, cases, weights)) for weights in presets]
    weights, objective, case_results = max(evaluated, key=lambda row: row[1])
    report = {
        "objective": round(objective, 4),
        "recommended_weights": weights,
        "cases": [asdict(item) for item in case_results],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    failures = [
        item.case_id
        for item in case_results
        if not item.hit_at_3 or item.forbidden_rate or not item.threshold_passed
    ]
    if failures:
        raise SystemExit(f"Retrieval evaluation failed: {', '.join(failures)}")
    print("\nPASS: every case hit relevant evidence, passed threshold, and avoided forbidden sources.")


if __name__ == "__main__":
    main()


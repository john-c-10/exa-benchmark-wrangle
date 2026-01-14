from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalMetrics:
    match: float  # R@1: % of queries where rank-1 is correct
    recall_at_10: float  # R@10: % of queries with correct in top 10
    precision: float  # % of results that are relevant
    num_queries: int


def compute_retrieval_metrics(grades: list[dict[str, Any]]) -> RetrievalMetrics:
    by_query: dict[str, list[dict]] = {}
    for g in grades:
        qid = g["query_id"]
        if qid not in by_query:
            by_query[qid] = []
        by_query[qid].append(g)

    for qid in by_query:
        by_query[qid].sort(key=lambda x: x.get("rank", 0))

    per_query = []
    failed_r10_queries: list[tuple[str, str]] = []
    for qid, query_grades in by_query.items():
        first_match_rank = None
        for g in query_grades:
            if g.get("is_match", 0) >= 1.0:
                first_match_rank = g.get("rank", query_grades.index(g) + 1)
                break

        n_results = len(query_grades)
        relevances = [g.get("is_match", 0) for g in query_grades]

        match = 1.0 if first_match_rank == 1 else 0.0
        recall_10 = 1.0 if first_match_rank and first_match_rank <= 10 else 0.0
        n_matches = sum(1 for r in relevances if r >= 1.0)
        precision = n_matches / n_results if n_results > 0 else 0.0

        if recall_10 == 0.0:
            query_text = query_grades[0].get("query_text", "") if query_grades else ""
            failed_r10_queries.append((qid, query_text))

        per_query.append({"match": match, "recall_10": recall_10, "precision": precision})

    if failed_r10_queries:
        print(f"\n=== R@10 FAILED QUERIES ({len(failed_r10_queries)}) ===")
        for qid, qtext in failed_r10_queries:
            print(f"  {qid}: {qtext}")
        print("=" * 40)

    n = len(per_query)
    if n == 0:
        return RetrievalMetrics(0, 0, 0, 0)

    return RetrievalMetrics(
        match=sum(m["match"] for m in per_query) / n,
        recall_at_10=sum(m["recall_10"] for m in per_query) / n,
        precision=sum(m["precision"] for m in per_query) / n,
        num_queries=n,
    )

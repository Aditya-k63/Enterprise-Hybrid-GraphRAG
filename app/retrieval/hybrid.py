import logging
import numpy as np

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    fused_scores = {}
    content_map = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            key = item.get("content", "")[:100]
            if key not in fused_scores:
                fused_scores[key] = 0.0
                content_map[key] = item
            fused_scores[key] += 1.0 / (k + rank + 1)

    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)

    results = []
    for key in sorted_keys:
        item = content_map[key].copy()
        item["rrf_score"] = fused_scores[key]
        results.append(item)

    return results

import logging
from app.graph import graph_search, get_entity_context
from app.ingestion.entity_extractor import extract_query_entities

logger = logging.getLogger(__name__)


def graph_retrieve(query: str, max_hops: int = 2) -> list[dict]:
    entities = extract_query_entities(query)
    if not entities:
        logger.info("No entities found in query for graph search")
        return []

    logger.info(f"Query entities for graph: {entities}")

    graph_results = graph_search(entities, max_hops=max_hops)

    context_texts = get_entity_context(entities)

    seen = set()
    results = []
    for item in graph_results:
        name = item["name"]
        if name not in seen:
            seen.add(name)
            results.append({
                "content": f"{name} ({item['type']}): {item['description']}",
                "score": 1.0 / (1 + item["distance"]),
                "source": "graph",
                "entity": name,
            })

    for ctx in context_texts:
        first_line = ctx.split("|")[0].strip()
        if first_line not in seen:
            results.append({
                "content": ctx,
                "score": 0.8,
                "source": "graph_context",
            })

    return results[:20]

import logging
import json
from groq import Groq
from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def get_groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


CLASSIFY_PROMPT = """Classify the following question into one of these retrieval types:
- "vector": General questions that need semantic search (definitions, explanations, summaries)
- "graph": Questions about specific entities, relationships, or connections between things
- "hybrid": Complex questions that need both factual/semantic info AND entity relationships
- "direct": Simple factual questions that can be answered directly

Return ONLY a JSON object: {{"type": "...", "reason": "..."}}

Question: {query}"""


def classify_query(query: str) -> dict:
    client = get_groq()

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": CLASSIFY_PROMPT.format(query=query),
            }],
            temperature=0.1,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
        result = json.loads(content)
        retrieval_type = result.get("type", "hybrid")
        if retrieval_type not in ("vector", "graph", "hybrid", "direct"):
            retrieval_type = "hybrid"
        return {"type": retrieval_type, "reason": result.get("reason", "")}
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        return {"type": "hybrid", "reason": "classification failed, defaulting to hybrid"}

import logging
import json
from groq import Groq

logger = logging.getLogger(__name__)


def get_groq():
    from app.config import settings
    return Groq(api_key=settings.GROQ_API_KEY)


ENTITY_PROMPT = """Extract all named entities from the following text. For each entity, provide:
- name: the entity name
- type: one of PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT, EVENT, TECHNOLOGY
- description: a brief 1-sentence description of the entity based on the context

Also extract relationships between entities. For each relationship:
- source: entity name
- target: entity name
- relation: the relationship type (e.g., "founded", "located_in", "occurred_in", "works_at", "invented")

Return ONLY valid JSON in this format:
{
  "entities": [{"name": "...", "type": "...", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "relation": "..."}]
}

Text:
{text}"""


def extract_entities(text: str) -> dict:
    from app.config import settings
    client = get_groq()

    truncated = text[:3000] if len(text) > 3000 else text

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": ENTITY_PROMPT.format(text=truncated),
            }],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        valid_entity_names = {e["name"] for e in entities}
        relationships = [
            r for r in relationships
            if r["source"] in valid_entity_names and r["target"] in valid_entity_names
        ]

        logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse entity extraction response: {e}")
        return {"entities": [], "relationships": []}
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def extract_query_entities(query: str) -> list[str]:
    from app.config import settings
    client = get_groq()

    prompt = f"""Extract the key named entities from this question. Return ONLY a JSON array of entity names.
If no clear entities, return an empty array [].

Question: {query}"""

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
        entities = json.loads(content)
        if isinstance(entities, list):
            return [str(e) for e in entities]
        return []
    except Exception as e:
        logger.error(f"Query entity extraction failed: {e}")
        return []

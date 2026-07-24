import logging

logger = logging.getLogger(__name__)

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        from app.config import settings
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.info("Neo4j driver created")
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def init_graph():
    driver = get_driver()
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (d:Document) REQUIRE d.filename IS UNIQUE
        """)
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
    logger.info("Neo4j constraints and indexes created")


def store_entities(filename: str, entities: list[dict], relationships: list[dict]):
    driver = get_driver()
    with driver.session() as session:
        session.run(
            "MERGE (d:Document {filename: $filename})",
            filename=filename,
        )

        for entity in entities:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type, e.description = $description
                MERGE (d:Document {filename: $filename})
                MERGE (e)-[:MENTIONED_IN]->(d)
            """,
                name=entity["name"],
                type=entity["type"],
                description=entity.get("description", ""),
                filename=filename,
            )

        for rel in relationships:
            session.run("""
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[r:RELATED_TO {relation: $relation}]->(b)
            """,
                source=rel["source"],
                target=rel["target"],
                relation=rel.get("relation", "related"),
            )

    logger.info(f"Stored {len(entities)} entities, {len(relationships)} relationships for '{filename}'")


def graph_search(query_entities: list[str], max_hops: int = 2) -> list[dict]:
    driver = get_driver()
    results = []
    with driver.session() as session:
        for entity_name in query_entities:
            cypher = f"""
                MATCH path = (e:Entity {{name: $name}})-[*1..{max_hops}]-(connected:Entity)
                RETURN DISTINCT
                    connected.name AS name,
                    connected.type AS type,
                    connected.description AS description,
                    length(path) AS distance
                ORDER BY distance
                LIMIT 20
            """
            record_result = session.run(cypher, name=entity_name)
            for record in record_result:
                results.append({
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "distance": record["distance"],
                })
    return results


def get_entity_context(entity_names: list[str]) -> list[str]:
    driver = get_driver()
    contexts = []
    with driver.session() as session:
        for name in entity_names:
            result = session.run("""
                MATCH (e:Entity {name: $name})
                OPTIONAL MATCH (e)-[:MENTIONED_IN]->(d:Document)
                OPTIONAL MATCH (e)-[r:RELATED_TO]-(connected:Entity)
                RETURN e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       collect(DISTINCT connected.name) AS related,
                       collect(DISTINCT d.filename) AS documents
            """, name=name)
            for record in result:
                related = [r for r in record["related"] if r]
                ctx = f"{record['name']} ({record['type']}): {record['description']}"
                if related:
                    ctx += f" | Related to: {', '.join(related[:5])}"
                contexts.append(ctx)
    return contexts


def get_document_entities(filename: str) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document {filename: $filename})
            RETURN e.name AS name, e.type AS type, e.description AS description
        """, filename=filename)
        return [dict(record) for record in result]


def delete_document_graph(filename: str):
    driver = get_driver()
    with driver.session() as session:
        session.run("""
            MATCH (d:Document {filename: $filename})
            DETACH DELETE d
        """, filename=filename)
    logger.info(f"Deleted graph for '{filename}'")

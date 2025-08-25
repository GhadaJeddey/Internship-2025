from neo4j import GraphDatabase
from typing import Any, Dict, List, Optional

class Neo4jGraphManager:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_graph(self, limit: int = 25) -> Dict[str, Any]:
        with self.driver.session() as session:
            cypher = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT $limit"
            results = session.run(cypher, limit=limit)
            nodes = set()
            edges = []
            for record in results:
                n = record["n"]
                m = record["m"]
                r = record["r"]
                nodes.add((n.id, dict(n)))
                nodes.add((m.id, dict(m)))
                edges.append((n.id, m.id, r.type))
            return {"nodes": [d for _, d in nodes], "edges": edges}

    def add_node(self, label: str, properties: Dict[str, Any]) -> Optional[int]:
        with self.driver.session() as session:
            cypher = f"CREATE (n:{label} $props) RETURN id(n) as node_id"
            result = session.run(cypher, props=properties)
            record = result.single()
            return record["node_id"] if record else None

    def add_relationship(self, node1_id: int, node2_id: int, rel_type: str, properties: Dict[str, Any] = None) -> bool:
        with self.driver.session() as session:
            cypher = (
                "MATCH (a), (b) "
                "WHERE id(a) = $id1 AND id(b) = $id2 "
                f"CREATE (a)-[r:{rel_type} $props]->(b) "
                "RETURN id(r)"
            )
            result = session.run(cypher, id1=node1_id, id2=node2_id, props=properties or {})
            return result.single() is not None

    def find_node(self, label: str, property_key: str, property_value: Any) -> Optional[int]:
        with self.driver.session() as session:
            cypher = f"MATCH (n:{label}) WHERE n.{property_key} = $value RETURN id(n) as node_id"
            result = session.run(cypher, value=property_value)
            record = result.single()
            return record["node_id"] if record else None

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

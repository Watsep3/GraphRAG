from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jLoader:
    """Charge les données dans Neo4j"""
    
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Nettoie la base (ATTENTION: supprime tout!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Base de données nettoyée")
    
    def create_constraints(self):
        """Crée les contraintes et index"""
        with self.driver.session() as session:
            # Contraintes d'unicité
            session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT relation_id IF NOT EXISTS
                FOR (r:Relation) REQUIRE r.id IS UNIQUE
            """)
            
            # Index pour recherche rapide
            session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)
            
            print("✓ Contraintes et index créés")
    
    def load_fb15k_batch(self, df: pd.DataFrame, batch_size: int = 1000):
        """Charge FB15k-237 par batch"""
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(df), batch_size), desc="Loading FB15k"):
                batch = df.iloc[i:i+batch_size]
                
                # Créer les triplets
                session.run("""
                    UNWIND $triples AS triple
                    MERGE (h:Entity {id: triple.head, name: triple.head})
                    MERGE (t:Entity {id: triple.tail, name: triple.tail})
                    MERGE (h)-[r:RELATION {type: triple.relation}]->(t)
                """, triples=batch.to_dict('records'))
        
        print(f"✓ {len(df)} triplets chargés")
    
    def get_statistics(self):
        """Affiche les statistiques de la base"""
        with self.driver.session() as session:
            # Nombre d'entités
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entities = result.single()['count']
            
            # Nombre de relations
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            relations = result.single()['count']
            
            print(f"\n📊 Statistiques Neo4j:")
            print(f"Entités: {entities:,}")
            print(f"Relations: {relations:,}")

if __name__ == "__main__":
    # Charger les données
    loader = Neo4jLoader()
    
    try:
        # Nettoyer (optionnel - décommenter si tu veux repartir de zéro)
        # loader.clear_database()
        
        # Créer contraintes
        loader.create_constraints()
        
        # Charger FB15k-237 (commencer avec train seulement)
        print("\n=== Chargement FB15k-237 Train ===")
        df_train = pd.read_csv('C:/Projects/GraphRAG/data/processed/fb15k_train.csv')
        loader.load_fb15k_batch(df_train)
        
        # Statistiques
        loader.get_statistics()
        
    finally:
        loader.close()
    
    print("\n✓✓✓ CHARGEMENT NEO4J TERMINÉ ✓✓✓")
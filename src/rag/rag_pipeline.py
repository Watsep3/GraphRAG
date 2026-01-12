import torch
import numpy as np
from typing import List, Dict, Tuple
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class GraphRAGRetriever:
    """Retriever qui combine recherche textuelle et graphe"""
    
    def __init__(self, 
                 text_model_name: str = 'all-MiniLM-L6-v2',
                 neo4j_uri: str = None,
                 neo4j_user: str = None,
                 neo4j_password: str = None):
        
        # Charger le modèle de texte
        print("Chargement du modèle de texte...")
        self.text_encoder = SentenceTransformer(text_model_name)
        
        # Connexion Neo4j
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        if self.neo4j_uri:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            print("✓ Connexion Neo4j établie")
        else:
            self.driver = None
            print("⚠ Neo4j non configuré")
        
        # Index FAISS (sera initialisé lors du chargement)
        self.index = None
        self.entity_names = []
        self.entity_embeddings = None
    
    def load_embeddings(self, embeddings_path: str):
        """Charge les embeddings et construit l'index FAISS"""
        print(f"Chargement des embeddings depuis {embeddings_path}...")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        # Vérifier le format
        print(f"Type de données: {type(data)}")
        
        if isinstance(data, dict):
            # Nouveau format (dict avec 'entities' et 'embeddings')
            if 'entities' in data and 'embeddings' in data:
                self.entity_names = data['entities']
                embeddings = data['embeddings']
                print("✓ Format dict détecté")
            
            # Format DataFrame
            elif 'name' in data and 'embedding' in data:
                self.entity_names = data['name'].tolist()
                embeddings = np.vstack(data['embedding'].values)
                print("✓ Format DataFrame détecté")
            
            else:
                print(f"Clés disponibles: {data.keys()}")
                raise ValueError("Format d'embeddings non reconnu - clés incorrectes")
        
        elif isinstance(data, pd.DataFrame):
            # Format DataFrame direct
            if 'name' in data.columns and 'embedding' in data.columns:
                self.entity_names = data['name'].tolist()
                embeddings = np.vstack(data['embedding'].values)
                print("✓ Format DataFrame direct détecté")
            else:
                print(f"Colonnes disponibles: {data.columns}")
                raise ValueError("Format DataFrame incorrect")
        
        else:
            print(f"Type non reconnu: {type(data)}")
            raise ValueError("Format d'embeddings non reconnu - type incorrect")
        
        self.entity_embeddings = embeddings
        
        # Créer index FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Index FAISS créé: {len(self.entity_names)} entités, dim={dimension}")
    
    def text_search(self, query: str, k: int = 5) -> List[Dict]:
        """Recherche textuelle pure via FAISS"""
        if self.index is None:
            raise ValueError("Embeddings non chargés. Appelez load_embeddings() d'abord.")
        
        # Encoder la requête
        query_embedding = self.text_encoder.encode([query])
        
        # Recherche FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'entity': self.entity_names[idx],
                'score': float(1 / (1 + dist)),  # Convertir distance en score
                'distance': float(dist)
            })
        
        return results
    
    def graph_expand(self, entity_id: str, max_hops: int = 2, limit: int = 10) -> List[Dict]:
        """Expand via le graphe Neo4j"""
        if not self.driver:
            return []
        
        query = """
        MATCH path = (start:Entity {id: $entity_id})-[*1..%d]-(connected:Entity)
        WHERE connected.id <> $entity_id
        RETURN DISTINCT connected.id as id, 
               connected.name as name,
               length(path) as hops
        ORDER BY hops ASC
        LIMIT $limit
        """ % max_hops
        
        with self.driver.session() as session:
            result = session.run(query, entity_id=entity_id, limit=limit)
            
            neighbors = []
            for record in result:
                neighbors.append({
                    'entity': record['name'],
                    'id': record['id'],
                    'hops': record['hops']
                })
            
            return neighbors
    
    def hybrid_search(self, query: str, k_text: int = 5, k_graph: int = 10, max_hops: int = 2) -> Dict:
        """Recherche hybride: texte + graphe"""
        
        # 1. Recherche textuelle
        text_results = self.text_search(query, k=k_text)
        
        # 2. Expansion via graphe
        graph_context = []
        for result in text_results[:3]:  # Expand les 3 meilleurs
            entity_id = result['entity']
            neighbors = self.graph_expand(entity_id, max_hops=max_hops, limit=k_graph)
            graph_context.extend(neighbors)
        
        # Dédupliquer
        seen = set()
        unique_graph = []
        for item in graph_context:
            if item['entity'] not in seen:
                seen.add(item['entity'])
                unique_graph.append(item)
        
        return {
            'query': query,
            'text_results': text_results,
            'graph_context': unique_graph[:k_graph],
            'combined_entities': list(set([r['entity'] for r in text_results] + 
                                         [g['entity'] for g in unique_graph[:k_graph]]))
        }
    
    def get_entity_context(self, entity_id: str) -> Dict:
        """Récupère le contexte complet d'une entité depuis Neo4j"""
        if not self.driver:
            return {'entity': entity_id, 'relations': []}
        
        query = """
        MATCH (e:Entity {id: $entity_id})-[r]->(target:Entity)
        RETURN type(r) as relation, target.name as target, target.id as target_id
        LIMIT 20
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_id=entity_id)
            
            relations = []
            for record in result:
                relations.append({
                    'relation': record['relation'],
                    'target': record['target'],
                    'target_id': record['target_id']
                })
            
            return {
                'entity': entity_id,
                'relations': relations
            }
    
    def close(self):
        """Ferme la connexion Neo4j"""
        if self.driver:
            self.driver.close()

class RAGGenerator:
    """Générateur qui utilise les résultats du retriever"""
    
    def __init__(self, model_name: str = 'gpt2'):
        """
        Pour utiliser GPT-4 via API:
        - Installer: pip install openai
        - Configurer OPENAI_API_KEY dans .env
        
        Pour l'instant, on va juste formater le contexte
        """
        self.model_name = model_name
        print(f"✓ Générateur initialisé (mode: {model_name})")
    
    def format_context(self, retrieval_results: Dict) -> str:
        """Formate les résultats de retrieval en contexte"""
        
        context_parts = []
        
        # Entités textuelles
        if retrieval_results['text_results']:
            context_parts.append("## Entités Pertinentes (Recherche Textuelle):")
            for i, result in enumerate(retrieval_results['text_results'][:5], 1):
                context_parts.append(f"{i}. {result['entity']} (score: {result['score']:.3f})")
        
        # Contexte graphe
        if retrieval_results['graph_context']:
            context_parts.append("\n## Contexte du Graphe (Entités Connectées):")
            for i, ctx in enumerate(retrieval_results['graph_context'][:10], 1):
                context_parts.append(f"{i}. {ctx['entity']} (distance: {ctx['hops']} saut(s))")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, retrieval_results: Dict) -> Dict:
        """Génère une réponse basée sur le contexte récupéré"""
        
        # Formater le contexte
        context = self.format_context(retrieval_results)
        
        # Pour l'instant, on retourne juste le contexte formaté
        # Dans une version avancée, on appellerait un LLM ici
        
        answer = f"""
Query: {query}

Contexte Récupéré:
{context}

Entités Combinées: {', '.join(retrieval_results['combined_entities'][:10])}

Note: Pour générer une réponse naturelle, intégrez un LLM (GPT-4, Claude, etc.)
"""
        
        return {
            'query': query,
            'answer': answer,
            'context': context,
            'entities': retrieval_results['combined_entities']
        }

class GraphRAGPipeline:
    """Pipeline complet RAG avec graphe"""
    
    def __init__(self, embeddings_path: str = None):
        self.retriever = GraphRAGRetriever()
        self.generator = RAGGenerator()
        
        if embeddings_path:
            self.retriever.load_embeddings(embeddings_path)
    
    def query(self, question: str, k_text: int = 5, k_graph: int = 10) -> Dict:
        """Execute une requête complète"""
        
        # 1. Retrieval
        print(f"\n🔍 Recherche pour: '{question}'")
        retrieval_results = self.retriever.hybrid_search(
            question, 
            k_text=k_text, 
            k_graph=k_graph
        )
        
        # 2. Generation
        print("📝 Génération de la réponse...")
        answer = self.generator.generate_answer(question, retrieval_results)
        
        return answer
    
    def close(self):
        self.retriever.close()

# Script de test
if __name__ == "__main__":
    print("="*60)
    print("TEST GRAPHRAG PIPELINE")
    print("="*60)
    
    # Initialiser le pipeline
    pipeline = GraphRAGPipeline(
        embeddings_path='C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl'
    )
    
    # Questions de test
    test_queries = [
        "Who is Barack Obama?",
        "What is machine learning?",
        "Tell me about Python programming"
    ]
    
    try:
        for query in test_queries:
            result = pipeline.query(query, k_text=5, k_graph=10)
            
            print("\n" + "="*60)
            print(result['answer'])
            print("="*60)
            
            # Pause entre requêtes
            input("\nAppuyez sur Entrée pour la requête suivante...")
    
    finally:
        pipeline.close()
    
    print("\n✓✓✓ TEST TERMINÉ ✓✓✓")
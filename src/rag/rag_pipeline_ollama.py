import torch
import numpy as np
from typing import List, Dict, Tuple
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import os
from dotenv import load_dotenv
import requests
import json

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
        self.driver = None
        self.neo4j_available = False
        
        # Tester la connexion Neo4j
        if self.neo4j_uri:
            try:
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri, 
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                # Test de connexion
                with self.driver.session() as session:
                    session.run("RETURN 1")
                self.neo4j_available = True
                print("✓ Connexion Neo4j établie")
            except ServiceUnavailable:
                print("⚠️  Neo4j non accessible - Fonctionnement en mode texte uniquement")
                self.driver = None
                self.neo4j_available = False
            except AuthError:
                print("⚠️  Erreur d'authentification Neo4j - Vérifiez vos identifiants")
                self.driver = None
                self.neo4j_available = False
            except Exception as e:
                print(f"⚠️  Erreur Neo4j: {e} - Fonctionnement en mode texte uniquement")
                self.driver = None
                self.neo4j_available = False
        else:
            print("⚠️  Neo4j non configuré - Fonctionnement en mode texte uniquement")
        
        # Index FAISS
        self.index = None
        self.entity_names = []
        self.entity_embeddings = None
    
    def load_embeddings(self, embeddings_path: str):
        """Charge les embeddings et construit l'index FAISS"""
        print(f"Chargement des embeddings depuis {embeddings_path}...")
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'entities' in data and 'embeddings' in data:
            self.entity_names = data['entities']
            embeddings = data['embeddings']
            print("✓ Format dict détecté")
        else:
            raise ValueError("Format d'embeddings non reconnu")
        
        self.entity_embeddings = embeddings
        
        # Créer index FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Index FAISS créé: {len(self.entity_names)} entités, dim={dimension}")
    
    def text_search(self, query: str, k: int = 5) -> List[Dict]:
        """Recherche textuelle pure via FAISS"""
        if self.index is None:
            raise ValueError("Embeddings non chargés")
        
        query_embedding = self.text_encoder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'entity': self.entity_names[idx],
                'score': float(1 / (1 + dist)),
                'distance': float(dist)
            })
        
        return results
    
    def graph_expand(self, entity_id: str, max_hops: int = 2, limit: int = 10) -> List[Dict]:
        """Expand via le graphe Neo4j (avec gestion d'erreur)"""
        if not self.neo4j_available or not self.driver:
            return []
        
        try:
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
        
        except Exception as e:
            print(f"⚠️  Erreur lors de l'expansion graphe: {e}")
            return []
    
    def hybrid_search(self, query: str, k_text: int = 5, k_graph: int = 10, max_hops: int = 2) -> Dict:
        """Recherche hybride: texte + graphe"""
        
        # Recherche textuelle (toujours disponible)
        text_results = self.text_search(query, k=k_text)
        
        # Expansion via graphe (si disponible)
        graph_context = []
        if self.neo4j_available:
            for result in text_results[:3]:
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
                                         [g['entity'] for g in unique_graph[:k_graph]])),
            'neo4j_used': self.neo4j_available
        }
    
    def close(self):
        if self.driver:
            self.driver.close()

class OllamaGenerator:
    """Générateur utilisant Ollama local"""
    
    def __init__(self, model_name: str = 'llama3.2:3b', base_url: str = 'http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url
        self.ollama_available = self.test_connection()
        
        if self.ollama_available:
            print(f"✓ Générateur Ollama initialisé (modèle: {model_name})")
        else:
            print(f"⚠️  Ollama non accessible - Réponses basiques uniquement")
    
    def test_connection(self):
        """Test si Ollama est accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def format_context(self, retrieval_results: Dict) -> str:
        """Formate les résultats de retrieval en contexte"""
        
        context_parts = []
        
        # Entités textuelles
        if retrieval_results['text_results']:
            context_parts.append("Entités pertinentes trouvées:")
            for i, result in enumerate(retrieval_results['text_results'][:5], 1):
                context_parts.append(f"- {result['entity']} (score: {result['score']:.3f})")
        
        # Contexte graphe
        if retrieval_results['graph_context']:
            context_parts.append("\nEntités connectées dans le graphe:")
            for i, ctx in enumerate(retrieval_results['graph_context'][:10], 1):
                context_parts.append(f"- {ctx['entity']} (à {ctx['hops']} saut(s))")
        elif not retrieval_results['neo4j_used']:
            context_parts.append("\n(Note: Contexte graphe non disponible)")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, retrieval_results: Dict) -> Dict:
        """Génère une réponse avec Ollama"""
        
        context = self.format_context(retrieval_results)
        
        # Si Ollama n'est pas accessible, retour basique
        if not self.ollama_available:
            answer = f"""Basé sur la recherche textuelle, voici les entités les plus pertinentes:

{context}

Note: Ollama n'est pas accessible. Pour des réponses en langage naturel, démarrez Ollama avec: ollama serve"""
            
            return {
                'query': query,
                'answer': answer,
                'context': context,
                'entities': retrieval_results['combined_entities'],
                'ollama_used': False
            }
        
        # Créer le prompt
        prompt = f"""Basé sur le contexte suivant, réponds à la question de manière claire et concise.

Note: Les identifiants comme /m/xxxxx sont des codes Freebase qui représentent des entités réelles.

Contexte:
{context}

Question: {query}

Réponse (en 2-3 phrases maximum):"""
        
        try:
            # Appel à Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'Pas de réponse générée').strip()
            else:
                answer = f"Erreur Ollama: {response.status_code}"
                self.ollama_available = False
        
        except Exception as e:
            answer = f"Erreur lors de la génération: {e}"
            self.ollama_available = False
        
        return {
            'query': query,
            'answer': answer,
            'context': context,
            'entities': retrieval_results['combined_entities'],
            'ollama_used': True
        }

class GraphRAGPipeline:
    """Pipeline complet RAG avec Ollama"""
    
    def __init__(self, embeddings_path: str = None, ollama_model: str = 'llama3.2:3b'):
        self.retriever = GraphRAGRetriever()
        self.generator = OllamaGenerator(model_name=ollama_model)
        
        if embeddings_path:
            self.retriever.load_embeddings(embeddings_path)
    
    def query(self, question: str, k_text: int = 5, k_graph: int = 10) -> Dict:
        """Execute une requête complète"""
        
        print(f"\n🔍 Recherche pour: '{question}'")
        retrieval_results = self.retriever.hybrid_search(
            question, 
            k_text=k_text, 
            k_graph=k_graph
        )
        
        if self.generator.ollama_available:
            print("📝 Génération de la réponse avec Ollama...")
        else:
            print("📝 Génération de la réponse basique...")
        
        answer = self.generator.generate_answer(question, retrieval_results)
        
        return answer
    
    def close(self):
        self.retriever.close()

# Script de test
if __name__ == "__main__":
    print("="*60)
    print("TEST GRAPHRAG PIPELINE AVEC OLLAMA")
    print("="*60)
    
    # Initialiser le pipeline
    pipeline = GraphRAGPipeline(
        embeddings_path='C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl',
        ollama_model='llama3.2:3b'
    )
    
    print("\n" + "="*60)
    print("STATUT DES SERVICES")
    print("="*60)
    print(f"Neo4j: {'✓ Disponible' if pipeline.retriever.neo4j_available else '✗ Non disponible'}")
    print(f"Ollama: {'✓ Disponible' if pipeline.generator.ollama_available else '✗ Non disponible'}")
    print("="*60)
    
    # Questions de test
    test_queries = [
        "Who invented the computer?",
        "What is machine learning?",
        "Tell me about Python programming"
    ]
    
    try:
        for query in test_queries:
            result = pipeline.query(query, k_text=5, k_graph=10)
            
            print("\n" + "="*60)
            print(f"❓ Question: {result['query']}")
            print("="*60)
            print(f"\n💡 Réponse:\n{result['answer']}")
            print("\n" + "-"*60)
            print(f"🔗 Entités: {', '.join(result['entities'][:10])}")
            print("="*60)
            
            input("\n⏎ Appuyez sur Entrée pour la requête suivante...")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur")
    
    finally:
        pipeline.close()
    
    print("\n✓✓✓ TEST TERMINÉ ✓✓✓")
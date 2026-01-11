import chromadb
from chromadb.config import Settings
from typing import List, Dict
import pandas as pd
from sentence_transformers import SentenceTransformer

class ChromaGraphStore:
    """Store vectoriel avec ChromaDB pour GraphRAG"""
    
    def __init__(self, persist_directory: str = "C:/Projects/GraphRAG/data/chroma_db"):
        print(f"Initialisation ChromaDB dans {persist_directory}...")
        
        # Créer client ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Créer/récupérer collections
        self.entities_collection = self.client.get_or_create_collection(
            name="entities",
            metadata={"description": "Entity embeddings from knowledge graph"}
        )
        
        self.questions_collection = self.client.get_or_create_collection(
            name="questions",
            metadata={"description": "Question-answer pairs"}
        )
        
        print("✓ ChromaDB initialisé")
    
    def add_entities(self, entities: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict] = None):
        """Ajoute des entités à la collection"""
        
        ids = [f"entity_{i}" for i in range(len(entities))]
        
        if metadata is None:
            metadata = [{"entity": entity} for entity in entities]
        
        self.entities_collection.add(
            embeddings=embeddings,
            documents=entities,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"✓ {len(entities)} entités ajoutées à ChromaDB")
    
    def search_entities(self, query: str, n_results: int = 5) -> Dict:
        """Recherche des entités similaires"""
        
        results = self.entities_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'entities': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }
    
    def add_qa_pairs(self, questions: List[str], answers: List[str], 
                     embeddings: List[List[float]], metadata: List[Dict] = None):
        """Ajoute des paires question-réponse"""
        
        ids = [f"qa_{i}" for i in range(len(questions))]
        
        if metadata is None:
            metadata = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        
        # Combiner question et réponse dans le document
        documents = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
        
        self.questions_collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"✓ {len(questions)} paires QA ajoutées")
    
    def search_qa(self, query: str, n_results: int = 3) -> Dict:
        """Recherche des QA similaires"""
        
        results = self.questions_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0]
        }
    
    def get_stats(self) -> Dict:
        """Statistiques des collections"""
        return {
            'entities_count': self.entities_collection.count(),
            'qa_count': self.questions_collection.count()
        }

# Script de population
if __name__ == "__main__":
    import pickle
    import numpy as np
    
    print("="*60)
    print("POPULATION CHROMADB")
    print("="*60)
    
    # Initialiser store
    store = ChromaGraphStore()
    
    # Charger les embeddings d'entités
    print("\n1. Chargement des entités...")
    with open('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl', 'rb') as f:
        entity_data = pickle.load(f)
    
    entities = entity_data['name'].tolist()[:1000]  # Limiter pour test
    embeddings = np.vstack(entity_data['embedding'].values[:1000])
    
    # Ajouter à ChromaDB
    print("2. Ajout des entités à ChromaDB...")
    store.add_entities(
        entities=entities,
        embeddings=embeddings.tolist(),
        metadata=[{"entity": e, "source": "fb15k"} for e in entities]
    )
    
    # Tester la recherche
    print("\n3. Test de recherche...")
    test_query = "Barack Obama"
    results = store.search_entities(test_query, n_results=5)
    
    print(f"\nRésultats pour '{test_query}':")
    for i, (entity, dist) in enumerate(zip(results['entities'], results['distances']), 1):
        print(f"{i}. {entity} (distance: {dist:.3f})")
    
    # Statistiques
    print("\n4. Statistiques:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓✓✓ CHROMADB POPULATION TERMINÉE ✓✓✓")
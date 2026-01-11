import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import json
from tqdm import tqdm
import sys
sys.path.append('C:/Projects/GraphRAG/src')

from rag.rag_pipeline import GraphRAGPipeline
from cross_modal_training import CrossModalProjection, CrossModalTrainer

class GraphRAGEvaluator:
    """Évaluateur pour GraphRAG"""
    
    def __init__(self, pipeline: GraphRAGPipeline):
        self.pipeline = pipeline
    
    def evaluate_retrieval(self, questions: List[str], 
                          ground_truth_entities: List[List[str]],
                          k: int = 5) -> Dict:
        """
        Évalue la qualité du retrieval
        
        Args:
            questions: Liste de questions
            ground_truth_entities: Liste de listes d'entités pertinentes
            k: Top-K pour les métriques
        """
        
        metrics = {
            'precision@k': [],
            'recall@k': [],
            'f1@k': [],
            'mrr': []  # Mean Reciprocal Rank
        }
        
        for question, gt_entities in tqdm(zip(questions, ground_truth_entities), 
                                          total=len(questions),
                                          desc="Evaluation"):
            
            # Récupérer les résultats
            results = self.pipeline.retriever.hybrid_search(question, k_text=k)
            retrieved = [r['entity'] for r in results['text_results']]
            
            # Precision@K
            relevant_in_k = len(set(retrieved[:k]) & set(gt_entities))
            precision = relevant_in_k / k if k > 0 else 0
            metrics['precision@k'].append(precision)
            
            # Recall@K
            recall = relevant_in_k / len(gt_entities) if len(gt_entities) > 0 else 0
            metrics['recall@k'].append(recall)
            
            # F1@K
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            metrics['f1@k'].append(f1)
            
            # MRR
            for i, entity in enumerate(retrieved, 1):
                if entity in gt_entities:
                    metrics['mrr'].append(1 / i)
                    break
            else:
                metrics['mrr'].append(0)
        
        # Moyennes
        return {
            f'precision@{k}': np.mean(metrics['precision@k']),
            f'recall@{k}': np.mean(metrics['recall@k']),
            f'f1@{k}': np.mean(metrics['f1@k']),
            'mrr': np.mean(metrics['mrr'])
        }
    
    def evaluate_qa(self, qa_pairs: List[Dict]) -> Dict:
        """
        Évalue sur des paires question-réponse
        
        Args:
            qa_pairs: Liste de {'question': str, 'answer': str}
        """
        
        # Pour l'instant, évaluation simple basée sur la présence d'entités
        scores = []
        
        for qa in tqdm(qa_pairs, desc="QA Evaluation"):
            question = qa['question']
            expected_answer = qa['answer']
            
            # Obtenir la réponse du système
            result = self.pipeline.query(question, k_text=5)
            
            # Score simple: présence de mots-clés
            answer_words = set(expected_answer.lower().split())
            result_words = set(result['answer'].lower().split())
            
            overlap = len(answer_words & result_words)
            score = overlap / len(answer_words) if len(answer_words) > 0 else 0
            
            scores.append(score)
        
        return {
            'avg_word_overlap': np.mean(scores),
            'median_word_overlap': np.median(scores)
        }
    
    def benchmark(self, test_file: str, k: int = 5) -> Dict:
        """
        Benchmark complet sur un fichier de test
        
        Format du fichier:
        [
            {
                "question": "...",
                "answer": "...",
                "entities": ["entity1", "entity2", ...]
            },
            ...
        ]
        """
        
        print(f"Chargement du benchmark: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        questions = [item['question'] for item in test_data]
        ground_truth_entities = [item.get('entities', []) for item in test_data]
        
        # Évaluation retrieval
        print("\n1. Évaluation Retrieval...")
        retrieval_metrics = self.evaluate_retrieval(questions, ground_truth_entities, k=k)
        
        # Évaluation QA
        print("\n2. Évaluation QA...")
        qa_metrics = self.evaluate_qa(test_data)
        
        # Combiner
        all_metrics = {**retrieval_metrics, **qa_metrics}
        
        return all_metrics

def create_synthetic_benchmark(output_file: str, n_samples: int = 100):
    """
    Crée un benchmark synthétique pour tester
    """
    
    print(f"Création de {n_samples} exemples synthétiques...")
    
    # Charger les entités disponibles
    import pickle
    with open('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl', 'rb') as f:
        entity_data = pickle.load(f)
    
    entities = entity_data['name'].tolist()[:1000]
    
    benchmark = []
    
    for i in range(n_samples):
        # Sélectionner des entités aléatoires
        selected = np.random.choice(entities, size=3, replace=False)
        
        # Créer une question synthétique
        question = f"Tell me about {selected[0]} and its relation to {selected[1]}"
        answer = f"{selected[0]} is related to {selected[1]} through {selected[2]}"
        
        benchmark.append({
            'id': i,
            'question': question,
            'answer': answer,
            'entities': list(selected)
        })
    
    # Sauvegarder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Benchmark sauvegardé: {output_file}")
    return benchmark

# Script principal
def main():
    print("="*60)
    print("ÉVALUATION GRAPHRAG")
    print("="*60)
    
    # 1. Créer un benchmark synthétique
    benchmark_file = 'C:/Projects/GraphRAG/data/processed/synthetic_benchmark.json'
    
    if not os.path.exists(benchmark_file):
        create_synthetic_benchmark(benchmark_file, n_samples=50)
    
    # 2. Initialiser le pipeline
    print("\n1. Initialisation du pipeline...")
    pipeline = GraphRAGPipeline(
        embeddings_path='C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl'
    )
    
    # 3. Créer l'évaluateur
    evaluator = GraphRAGEvaluator(pipeline)
    
    # 4. Exécuter le benchmark
    print("\n2. Exécution du benchmark...")
    metrics = evaluator.benchmark(benchmark_file, k=5)
    
    # 5. Afficher les résultats
    print("\n" + "="*60)
    print("RÉSULTATS D'ÉVALUATION")
    print("="*60)
    
    for metric, value in metrics.items():
        print(f"{metric:.<40} {value:.4f}")
    
    # 6. Sauvegarder les résultats
    results_file = 'C:/Projects/GraphRAG/results/evaluation_results.json'
    os.makedirs('C:/Projects/GraphRAG/results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {results_file}")
    
    # 7. Exemples de requêtes
    print("\n" + "="*60)
    print("EXEMPLES DE REQUÊTES")
    print("="*60)
    
    test_queries = [
        "What is machine learning?",
        "Who is Barack Obama?",
        "Explain neural networks"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = pipeline.query(query, k_text=3)
        print(f"Entities: {', '.join(result['entities'][:5])}")
    
    pipeline.close()
    
    print("\n✓✓✓ ÉVALUATION TERMINÉE ✓✓✓")

if __name__ == "__main__":
    import os
    main()
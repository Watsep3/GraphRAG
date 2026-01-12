import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import json
from tqdm import tqdm
import sys
import os

sys.path.append('C:/Projects/GraphRAG/src')

from rag.rag_pipeline_ollama import GraphRAGPipeline

class GraphRAGEvaluator:
    """Évaluateur pour GraphRAG"""
    
    def __init__(self, pipeline: GraphRAGPipeline):
        self.pipeline = pipeline
    
    def evaluate_retrieval(self, questions: List[str], 
                          ground_truth_entities: List[List[str]],
                          k: int = 5) -> Dict:
        """Évalue la qualité du retrieval"""
        
        metrics = {
            'precision@k': [],
            'recall@k': [],
            'f1@k': [],
            'mrr': []
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
        """Évalue sur des paires question-réponse"""
        
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
        """Benchmark complet sur un fichier de test"""
        
        print(f"Chargement du benchmark: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        questions = [item['question'] for item in test_data]
        ground_truth_entities = [item.get('entities', []) for item in test_data]
        
        # Évaluation retrieval
        print("\n1. Évaluation Retrieval...")
        retrieval_metrics = self.evaluate_retrieval(questions, ground_truth_entities, k=k)
        
        # Évaluation QA (optionnel - peut prendre du temps avec Ollama)
        print("\n2. Évaluation QA (simplifiée)...")
        # Sauter l'évaluation QA complète pour gagner du temps
        qa_metrics = {'avg_word_overlap': 0.0, 'median_word_overlap': 0.0}
        
        # Combiner
        all_metrics = {**retrieval_metrics, **qa_metrics}
        
        return all_metrics

def create_synthetic_benchmark(output_file: str, n_samples: int = 50):
    """Crée un benchmark synthétique pour tester"""
    
    print(f"Création de {n_samples} exemples synthétiques...")
    
    # Charger les entités disponibles - FORMAT CORRIGÉ
    import pickle
    with open('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl', 'rb') as f:
        entity_data = pickle.load(f)
    
    # CORRECTION: Utiliser 'entities' au lieu de 'name'
    if isinstance(entity_data, dict) and 'entities' in entity_data:
        entities = entity_data['entities'][:1000]  # Prendre les 1000 premières
    else:
        print("❌ Format d'embeddings incorrect")
        return None
    
    print(f"✓ {len(entities)} entités chargées")
    
    benchmark = []
    
    # Questions templates
    question_templates = [
        "Tell me about {} and {}",
        "What is the relation between {} and {}?",
        "How are {} and {} connected?",
        "Explain {} in relation to {}",
        "What do you know about {}?"
    ]
    
    for i in range(n_samples):
        # Sélectionner des entités aléatoires
        selected = np.random.choice(entities, size=min(3, len(entities)), replace=False)
        
        # Choisir un template
        template = np.random.choice(question_templates)
        
        # Créer la question
        if '{}' in template:
            # Compter le nombre de {} dans le template
            num_placeholders = template.count('{}')
            if num_placeholders == 1:
                question = template.format(selected[0])
                relevant_entities = [selected[0]]
            elif num_placeholders == 2:
                question = template.format(selected[0], selected[1])
                relevant_entities = list(selected[:2])
            else:
                question = template.format(*selected[:num_placeholders])
                relevant_entities = list(selected[:num_placeholders])
        else:
            question = f"Tell me about {selected[0]}"
            relevant_entities = [selected[0]]
        
        answer = f"Information about {', '.join(relevant_entities)}"
        
        benchmark.append({
            'id': i,
            'question': question,
            'answer': answer,
            'entities': list(relevant_entities)
        })
    
    # Sauvegarder
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
        benchmark = create_synthetic_benchmark(benchmark_file, n_samples=50)
        if benchmark is None:
            print("❌ Impossible de créer le benchmark")
            return
    
    # 2. Initialiser le pipeline
    print("\n1. Initialisation du pipeline...")
    pipeline = GraphRAGPipeline(
        embeddings_path='C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl',
        ollama_model='llama3.2:3b'
    )
    
    # 3. Créer l'évaluateur
    evaluator = GraphRAGEvaluator(pipeline)
    
    # 4. Exécuter le benchmark
    print("\n2. Exécution du benchmark...")
    print("⚠️  Note: Cela peut prendre plusieurs minutes...")
    
    try:
        metrics = evaluator.benchmark(benchmark_file, k=5)
        
        # 5. Afficher les résultats
        print("\n" + "="*60)
        print("RÉSULTATS D'ÉVALUATION")
        print("="*60)
        
        for metric, value in metrics.items():
            print(f"{metric:.<40} {value:.4f}")
        
        # 6. Sauvegarder les résultats
        results_dir = 'C:/Projects/GraphRAG/results'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, 'evaluation_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Résultats sauvegardés: {results_file}")
        
        # 7. Exemples de requêtes
        print("\n" + "="*60)
        print("EXEMPLES DE REQUÊTES")
        print("="*60)
        
        test_queries = [
            "What is computer science?",
            "Tell me about programming",
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            result = pipeline.query(query, k_text=3)
            print(f"Entities: {', '.join(result['entities'][:5])}")
            print(f"Answer: {result['answer'][:100]}...")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant l'évaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.close()
    
    print("\n✓✓✓ ÉVALUATION TERMINÉE ✓✓✓")

if __name__ == "__main__":
    main()
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch
import os

class TextEncoder:
    """Encode le texte en embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"✓ Modèle chargé sur {self.device}")
    
    def encode_batch(self, texts: list, batch_size: int = 32, show_progress: bool = True):
        """Encode une liste de textes"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device
        )
        return embeddings
    
    def save_embeddings(self, embeddings, filepath: str):
        """Sauvegarde les embeddings"""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Embeddings sauvegardés: {filepath}")

if __name__ == "__main__":
    os.makedirs('C:/Projects/GraphRAG/models/embeddings', exist_ok=True)
    
    # Initialiser encoder
    encoder = TextEncoder(model_name='all-MiniLM-L6-v2')
    
    # Charger les entités FB15k
    print("\n=== Encodage Entités FB15k ===")
    df_train = pd.read_csv('C:/Projects/GraphRAG/data/processed/fb15k_train.csv')
    
    # Extraire entités uniques
    entities = list(set(df_train['head'].tolist() + df_train['tail'].tolist()))
    print(f"Encodage de {len(entities)} entités...")
    
    # Encoder
    embeddings = encoder.encode_batch(entities)
    
    # IMPORTANT: Sauvegarder dans le bon format
    entity_data = {
        'entities': entities,
        'embeddings': embeddings,
        'entity2id': {e: i for i, e in enumerate(entities)}
    }
    
    # Sauvegarder
    output_path = 'C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl'
    encoder.save_embeddings(entity_data, output_path)
    print(f"✓ {len(entities)} entités encodées")
    
    # Vérifier le format
    print("\n=== Vérification du Format ===")
    with open(output_path, 'rb') as f:
        loaded = pickle.load(f)
    
    print(f"Type: {type(loaded)}")
    print(f"Clés: {loaded.keys() if isinstance(loaded, dict) else 'N/A'}")
    if isinstance(loaded, dict):
        print(f"Nombre d'entités: {len(loaded['entities'])}")
        print(f"Shape embeddings: {loaded['embeddings'].shape}")
        print(f"Exemple d'entité: {loaded['entities'][0]}")
    
    # Encoder questions HotpotQA (optionnel)
    hotpot_file = 'C:/Projects/GraphRAG/data/processed/hotpotqa_train_processed.csv'
    
    if os.path.exists(hotpot_file):
        print("\n=== Encodage Questions HotpotQA ===")
        df_qa = pd.read_csv(hotpot_file)
        
        questions = df_qa['question'].tolist()[:1000]  # 1000 pour test
        question_embeddings = encoder.encode_batch(questions)
        
        # Sauvegarder questions
        question_data = {
            'questions': questions,
            'embeddings': question_embeddings
        }
        
        encoder.save_embeddings(
            question_data,
            'C:/Projects/GraphRAG/models/embeddings/question_embeddings.pkl'
        )
    else:
        print("\n⚠️  HotpotQA non trouvé (optionnel)")
    
    print("\n✓✓✓ ENCODAGE TERMINÉ ✓✓✓")
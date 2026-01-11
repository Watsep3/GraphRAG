from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch

class TextEncoder:
    """Encode le texte en embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Modèles recommandés:
        - all-MiniLM-L6-v2: Rapide, 384 dim
        - all-mpnet-base-v2: Meilleur qualité, 768 dim
        """
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
    
    def encode_entities(self, df: pd.DataFrame, text_column: str = 'name'):
        """Encode les noms d'entités"""
        texts = df[text_column].tolist()
        print(f"Encodage de {len(texts)} entités...")
        
        embeddings = self.encode_batch(texts)
        
        # Ajouter au DataFrame
        df['embedding'] = list(embeddings)
        
        return df
    
    def encode_questions(self, questions: list):
        """Encode des questions"""
        print(f"Encodage de {len(questions)} questions...")
        embeddings = self.encode_batch(questions)
        return embeddings
    
    def save_embeddings(self, embeddings, filepath: str):
        """Sauvegarde les embeddings"""
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Embeddings sauvegardés: {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Charge les embeddings"""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ Embeddings chargés: {filepath}")
        return embeddings

if __name__ == "__main__":
    import os
    os.makedirs('C:/Projects/GraphRAG/models/embeddings', exist_ok=True)
    
    # Initialiser encoder
    encoder = TextEncoder(model_name='all-MiniLM-L6-v2')
    
    # Charger les entités FB15k
    print("\n=== Encodage Entités FB15k ===")
    df_train = pd.read_csv('C:/Projects/GraphRAG/data/processed/fb15k_train.csv')
    
    # Extraire entités uniques
    entities = list(set(df_train['head'].tolist() + df_train['tail'].tolist()))
    entity_df = pd.DataFrame({'name': entities})
    
    # Encoder
    entity_df = encoder.encode_entities(entity_df)
    
    # Sauvegarder
    entity_df.to_pickle('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl')
    print(f"✓ {len(entity_df)} entités encodées")
    
    # Encoder questions HotpotQA
    print("\n=== Encodage Questions HotpotQA ===")
    df_qa = pd.read_csv('C:/Projects/GraphRAG/data/processed/hotpotqa_train_processed.csv')
    
    questions = df_qa['question'].tolist()[:1000]  # 1000 pour test
    question_embeddings = encoder.encode_questions(questions)
    
    # Sauvegarder
    encoder.save_embeddings(
        question_embeddings,
        'C:/Projects/GraphRAG/models/embeddings/question_embeddings.pkl'
    )
    
    print("\n✓✓✓ ENCODAGE TERMINÉ ✓✓✓")
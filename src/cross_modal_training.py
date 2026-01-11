import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from typing import Dict, Tuple
import os

class TextGraphDataset(Dataset):
    """Dataset pour l'entraînement cross-modal"""
    
    def __init__(self, 
                 text_embeddings: np.ndarray,
                 graph_embeddings: np.ndarray,
                 entity_names: list):
        """
        Args:
            text_embeddings: Embeddings textuels (N, text_dim)
            graph_embeddings: Embeddings graphe (N, graph_dim)
            entity_names: Liste des noms d'entités
        """
        assert len(text_embeddings) == len(graph_embeddings), "Tailles incompatibles"
        
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.graph_embeddings = torch.FloatTensor(graph_embeddings)
        self.entity_names = entity_names
    
    def __len__(self):
        return len(self.text_embeddings)
    
    def __getitem__(self, idx):
        return {
            'text': self.text_embeddings[idx],
            'graph': self.graph_embeddings[idx],
            'entity': self.entity_names[idx]
        }

class CrossModalProjection(nn.Module):
    """Projette texte et graphe dans un espace commun"""
    
    def __init__(self, text_dim: int, graph_dim: int, common_dim: int = 512):
        super(CrossModalProjection, self).__init__()
        
        # Projecteurs
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim)
        )
        
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_dim, common_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim)
        )
        
        # Température pour contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def forward(self, text_emb, graph_emb):
        """
        Args:
            text_emb: (batch, text_dim)
            graph_emb: (batch, graph_dim)
        
        Returns:
            text_proj: (batch, common_dim)
            graph_proj: (batch, common_dim)
        """
        text_proj = self.text_projector(text_emb)
        graph_proj = self.graph_projector(graph_emb)
        
        # Normaliser
        text_proj = F.normalize(text_proj, dim=-1)
        graph_proj = F.normalize(graph_proj, dim=-1)
        
        return text_proj, graph_proj

class ContrastiveLoss(nn.Module):
    """Loss contrastive pour alignement cross-modal"""
    
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, text_proj, graph_proj, temperature):
        """
        Contrastive loss bidirectionnel (text->graph et graph->text)
        """
        batch_size = text_proj.shape[0]
        
        # Similarités
        logits = torch.matmul(text_proj, graph_proj.t()) / temperature
        
        # Labels (diagonal = positifs)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Loss text->graph
        loss_t2g = F.cross_entropy(logits, labels)
        
        # Loss graph->text
        loss_g2t = F.cross_entropy(logits.t(), labels)
        
        # Loss total
        loss = (loss_t2g + loss_g2t) / 2
        
        return loss

class CrossModalTrainer:
    """Trainer pour l'alignement cross-modal"""
    
    def __init__(self, 
                 model: CrossModalProjection,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        self.criterion = ContrastiveLoss()
        
        print(f"✓ Trainer initialisé sur {device}")
    
    def train_epoch(self, dataloader: DataLoader, optimizer, epoch: int):
        """Entraîne une époque"""
        
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            text_emb = batch['text'].to(self.device)
            graph_emb = batch['graph'].to(self.device)
            
            # Forward
            text_proj, graph_proj = self.model(text_emb, graph_emb)
            
            # Loss
            loss = self.criterion(text_proj, graph_proj, self.model.temperature)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'temp': self.model.temperature.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader):
        """Évalue le modèle"""
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                text_emb = batch['text'].to(self.device)
                graph_emb = batch['graph'].to(self.device)
                
                text_proj, graph_proj = self.model(text_emb, graph_emb)
                loss = self.criterion(text_proj, graph_proj, self.model.temperature)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def compute_retrieval_metrics(self, dataloader: DataLoader, k: int = 5):
        """Calcule les métriques de retrieval (Recall@K)"""
        
        self.model.eval()
        
        all_text_proj = []
        all_graph_proj = []
        
        with torch.no_grad():
            for batch in dataloader:
                text_emb = batch['text'].to(self.device)
                graph_emb = batch['graph'].to(self.device)
                
                text_proj, graph_proj = self.model(text_emb, graph_emb)
                
                all_text_proj.append(text_proj.cpu())
                all_graph_proj.append(graph_proj.cpu())
        
        # Concatener
        all_text_proj = torch.cat(all_text_proj, dim=0)
        all_graph_proj = torch.cat(all_graph_proj, dim=0)
        
        # Similarités
        similarities = torch.matmul(all_text_proj, all_graph_proj.t())
        
        # Recall@K text->graph
        _, top_k_indices = similarities.topk(k, dim=1)
        correct = torch.any(top_k_indices == torch.arange(len(similarities)).unsqueeze(1), dim=1)
        recall_t2g = correct.float().mean().item()
        
        # Recall@K graph->text
        _, top_k_indices = similarities.t().topk(k, dim=1)
        correct = torch.any(top_k_indices == torch.arange(len(similarities)).unsqueeze(1), dim=1)
        recall_g2t = correct.float().mean().item()
        
        return {
            f'recall@{k}_t2g': recall_t2g,
            f'recall@{k}_g2t': recall_g2t,
            f'recall@{k}_avg': (recall_t2g + recall_g2t) / 2
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer, loss: float):
        """Sauvegarde un checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        print(f"✓ Checkpoint sauvegardé: {filepath}")
    
    def load_checkpoint(self, filepath: str, optimizer=None):
        """Charge un checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Checkpoint chargé: {filepath}")
        return checkpoint['epoch'], checkpoint['loss']

# Script d'entraînement principal
def main():
    print("="*60)
    print("CROSS-MODAL TRAINING")
    print("="*60)
    
    # Hyperparamètres
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    COMMON_DIM = 512
    
    # Créer dossiers
    os.makedirs('C:/Projects/GraphRAG/checkpoints', exist_ok=True)
    
    # 1. Charger les embeddings
    print("\n1. Chargement des embeddings...")
    
    # Text embeddings
    with open('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl', 'rb') as f:
        text_data = pickle.load(f)
    
    text_embeddings = np.vstack(text_data['embedding'].values)
    entity_names = text_data['name'].tolist()
    
    # Graph embeddings
    with open('C:/Projects/GraphRAG/models/graph/graph_embeddings.pkl', 'rb') as f:
        graph_data = pickle.load(f)
    
    graph_embeddings = graph_data['embeddings']
    
    # S'assurer qu'ils ont le même nombre d'entités
    min_len = min(len(text_embeddings), len(graph_embeddings))
    text_embeddings = text_embeddings[:min_len]
    graph_embeddings = graph_embeddings[:min_len]
    entity_names = entity_names[:min_len]
    
    print(f"✓ {min_len} entités chargées")
    print(f"  Text dim: {text_embeddings.shape[1]}")
    print(f"  Graph dim: {graph_embeddings.shape[1]}")
    
    # 2. Créer dataset et dataloader
    print("\n2. Création du dataset...")
    dataset = TextGraphDataset(text_embeddings, graph_embeddings, entity_names)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"✓ Train: {train_size}, Val: {val_size}")
    
    # 3. Créer le modèle
    print("\n3. Initialisation du modèle...")
    model = CrossModalProjection(
        text_dim=text_embeddings.shape[1],
        graph_dim=graph_embeddings.shape[1],
        common_dim=COMMON_DIM
    )
    
    print(f"✓ Modèle créé")
    print(f"  Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Créer trainer et optimizer
    trainer = CrossModalTrainer(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. Entraînement
    print(f"\n4. Entraînement ({EPOCHS} epochs)...")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, epoch)
        
        # Validation
        val_loss = trainer.evaluate(val_loader)
        
        # Scheduler
        scheduler.step()
        
        # Metrics
        if epoch % 5 == 0:
            metrics = trainer.compute_retrieval_metrics(val_loader, k=5)
            print(f"\nEpoch {epoch}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Recall@5: {metrics['recall@5_avg']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                'C:/Projects/GraphRAG/checkpoints/best_cross_modal.pt',
                epoch,
                optimizer,
                val_loss
            )
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            trainer.save_checkpoint(
                f'C:/Projects/GraphRAG/checkpoints/cross_modal_epoch_{epoch}.pt',
                epoch,
                optimizer,
                val_loss
            )
    
    # 6. Évaluation finale
    print("\n" + "="*60)
    print("5. Évaluation finale...")
    
    trainer.load_checkpoint('C:/Projects/GraphRAG/checkpoints/best_cross_modal.pt')
    
    final_metrics = trainer.compute_retrieval_metrics(val_loader, k=5)
    print("\nMétriques finales:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓✓✓ ENTRAÎNEMENT TERMINÉ ✓✓✓")

if __name__ == "__main__":
    main()
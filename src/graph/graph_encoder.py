import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import pickle

class GraphEncoder(nn.Module):
    """Graph Neural Network pour encoder les graphes"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 384, num_layers: int = 2):
        super(GraphEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # Première couche
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        
        # Couches intermédiaires
        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Couche de sortie
        self.conv_out = SAGEConv(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, edge_index):
        # Première couche
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Couches intermédiaires
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Sortie
        x = self.conv_out(x, edge_index)
        
        return x

class GraphDataLoader:
    """Charge les données graphe pour PyG"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_fb15k_graph(self):
        """Charge FB15k comme graphe PyG"""
        # Charger triplets
        df = pd.read_csv(f'{self.data_dir}/fb15k_train.csv')
        
        # Créer mapping entité -> id
        entities = list(set(df['head'].tolist() + df['tail'].tolist()))
        entity2id = {e: i for i, e in enumerate(entities)}
        
        # Créer edge_index
        edge_index = []
        for _, row in df.iterrows():
            src = entity2id[row['head']]
            dst = entity2id[row['tail']]
            edge_index.append([src, dst])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Features initiales (one-hot ou aléatoires)
        num_nodes = len(entities)
        x = torch.randn(num_nodes, 128)  # Features aléatoires pour commencer
        
        # Créer objet Data PyG
        data = Data(x=x, edge_index=edge_index)
        
        print(f"✓ Graphe chargé: {num_nodes} nœuds, {edge_index.shape[1]} arêtes")
        
        return data, entity2id, entities

def train_graph_encoder(data, model, epochs: int = 100, lr: float = 0.01):
    """Entraîne le GNN (self-supervised)"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward
        embeddings = model(data.x, data.edge_index)
        
        # Loss contrastive simple (à améliorer)
        # Pour l'instant, juste normalisation
        loss = F.mse_loss(embeddings, torch.zeros_like(embeddings))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model

if __name__ == "__main__":
    import os
    
    # Charger données
    loader = GraphDataLoader('C:/Projects/GraphRAG/data/processed')
    data, entity2id, entities = loader.load_fb15k_graph()
    
    # Créer modèle
    model = GraphEncoder(
        input_dim=data.x.shape[1],
        hidden_dim=256,
        output_dim=384,  # Même dimension que text embeddings
        num_layers=2
    )
    
    print(f"\n{model}")
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entraîner
    print("\n=== Entraînement GNN ===")
    model = train_graph_encoder(data, model, epochs=50)
    
    # Générer embeddings
    model.eval()
    with torch.no_grad():
        graph_embeddings = model(data.x, data.edge_index)
    
    # Sauvegarder
    os.makedirs('C:/Projects/GraphRAG/models/graph', exist_ok=True)
    
    torch.save(model.state_dict(), 'C:/Projects/GraphRAG/models/graph/graph_encoder.pt')
    
    with open('C:/Projects/GraphRAG/models/graph/graph_embeddings.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': graph_embeddings.numpy(),
            'entity2id': entity2id,
            'entities': entities
        }, f)
    
    print("\n✓✓✓ ENTRAÎNEMENT GNN TERMINÉ ✓✓✓")
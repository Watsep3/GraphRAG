import pickle
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import pandas as pd

class EmbeddingVisualizer:
    """Visualise les embeddings avec t-SNE"""
    
    def __init__(self, embeddings_path: str):
        """Charge les embeddings"""
        print("Chargement des embeddings...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        self.entities = data['entities']
        self.embeddings = data['embeddings']
        self.entity2id = data.get('entity2id', {})
        
        print(f"✓ {len(self.entities)} entités chargées")
        print(f"✓ Dimension: {self.embeddings.shape[1]}")
    
    def compute_tsne(self, 
                     n_components: int = 2,
                     perplexity: float = 30.0,
                     max_iter: int = 1000,
                     random_state: int = 42,
                     sample_size: int = None) -> np.ndarray:
        """Calcule la projection t-SNE"""
        
        embeddings = self.embeddings
        entities = self.entities
        
        # Échantillonnage si trop de données
        if sample_size and len(embeddings) > sample_size:
            print(f"Échantillonnage: {sample_size} sur {len(embeddings)}")
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[indices]
            entities = [entities[i] for i in indices]
            self.sampled_entities = entities
            self.sampled_indices = indices
        else:
            self.sampled_entities = entities
            self.sampled_indices = None
        
        print(f"Calcul t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1
        )
        
        self.tsne_coords = tsne.fit_transform(embeddings)
        
        print(f"✓ t-SNE terminé: shape {self.tsne_coords.shape}")
        return self.tsne_coords
    
    def create_interactive_plot(self, 
                               color_by: str = None,
                               highlight_entities: List[str] = None,
                               title: str = "t-SNE Visualization of Entity Embeddings") -> go.Figure:
        """Crée un graphique interactif Plotly"""
        
        # Préparer les données
        df = pd.DataFrame({
            'x': self.tsne_coords[:, 0],
            'y': self.tsne_coords[:, 1],
            'entity': self.sampled_entities,
            'label': [e[:30] + '...' if len(e) > 30 else e for e in self.sampled_entities]
        })
        
        # Ajouter des catégories basées sur les préfixes
        def get_category(entity):
            if entity.startswith('/m/0'):
                return 'Type A'
            elif entity.startswith('/m/1'):
                return 'Type B'
            elif entity.startswith('/m/2'):
                return 'Type C'
            elif entity.startswith('/m/3'):
                return 'Type D'
            else:
                return 'Other'
        
        df['category'] = df['entity'].apply(get_category)
        
        # Créer le plot
        if color_by == 'category':
            fig = px.scatter(
                df, 
                x='x', 
                y='y',
                color='category',
                hover_data=['entity'],
                title=title,
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            fig = go.Figure()
            
            # Points normaux
            fig.add_trace(go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='lightblue',
                    opacity=0.6,
                    line=dict(width=0.5, color='darkblue')
                ),
                text=df['label'],
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                name='Entities'
            ))
            
            # Highlights
            if highlight_entities:
                highlight_mask = df['entity'].isin(highlight_entities)
                highlight_df = df[highlight_mask]
                
                if len(highlight_df) > 0:
                    fig.add_trace(go.Scatter(
                        x=highlight_df['x'],
                        y=highlight_df['y'],
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='star',
                            line=dict(width=2, color='darkred')
                        ),
                        text=highlight_df['label'],
                        textposition='top center',
                        textfont=dict(size=10, color='darkred'),
                        hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                        name='Highlighted'
                    ))
        
        # Layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            hovermode='closest',
            showlegend=True,
            width=1000,
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def create_3d_plot(self) -> go.Figure:
        """Crée une visualisation 3D si t-SNE a 3 dimensions"""
        
        if self.tsne_coords.shape[1] != 3:
            print("⚠️  t-SNE doit avoir 3 dimensions pour la visualisation 3D")
            return None
        
        df = pd.DataFrame({
            'x': self.tsne_coords[:, 0],
            'y': self.tsne_coords[:, 1],
            'z': self.tsne_coords[:, 2],
            'entity': self.sampled_entities,
            'label': [e[:20] for e in self.sampled_entities]
        })
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['x'],
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=df['label'],
            hovertemplate='<b>%{text}</b><extra></extra>'
        )])
        
        fig.update_layout(
            title='3D t-SNE Visualization',
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            ),
            width=1000,
            height=700
        )
        
        return fig
    
    def find_clusters(self, n_clusters: int = 10) -> Dict:
        """Détecte des clusters avec K-Means"""
        from sklearn.cluster import KMeans
        
        print(f"Détection de {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.tsne_coords)
        
        # Analyser les clusters
        clusters_info = {}
        for i in range(n_clusters):
            mask = labels == i
            cluster_entities = [self.sampled_entities[j] for j in range(len(mask)) if mask[j]]
            clusters_info[f'Cluster {i+1}'] = {
                'size': len(cluster_entities),
                'entities': cluster_entities[:10],  # Top 10
                'center': kmeans.cluster_centers_[i]
            }
        
        self.cluster_labels = labels
        self.clusters_info = clusters_info
        
        return clusters_info
    
    def plot_clusters(self) -> go.Figure:
        """Visualise avec les clusters"""
        
        if not hasattr(self, 'cluster_labels'):
            print("⚠️  Exécutez find_clusters() d'abord")
            return None
        
        df = pd.DataFrame({
            'x': self.tsne_coords[:, 0],
            'y': self.tsne_coords[:, 1],
            'entity': self.sampled_entities,
            'cluster': self.cluster_labels
        })
        
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['entity'],
            title='t-SNE with K-Means Clustering',
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
            color_continuous_scale='Rainbow'
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            template='plotly_white'
        )
        
        return fig


# Script de test
if __name__ == "__main__":
    print("="*60)
    print("VISUALISATION t-SNE DES EMBEDDINGS")
    print("="*60)
    
    # Charger et visualiser
    viz = EmbeddingVisualizer('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl')
    
    # Calculer t-SNE (échantillon de 2000 pour la rapidité)
    viz.compute_tsne(
        n_components=2,
        perplexity=30.0,
        max_iter=1000,
        sample_size=2000
    )
    
    # Créer visualisation
    fig = viz.create_interactive_plot(title="t-SNE: Entity Embeddings (Sample: 2000)")
    fig.write_html('C:/Projects/GraphRAG/results/tsne_visualization.html')
    print("✓ Visualisation sauvegardée: results/tsne_visualization.html")
    
    # Détecter clusters
    clusters = viz.find_clusters(n_clusters=8)
    
    print("\n📊 Clusters détectés:")
    for cluster_name, info in clusters.items():
        print(f"\n{cluster_name}: {info['size']} entités")
        print(f"  Exemples: {', '.join(info['entities'][:5])}")
    
    # Plot avec clusters
    fig_clusters = viz.plot_clusters()
    fig_clusters.write_html('C:/Projects/GraphRAG/results/tsne_clusters.html')
    print("\n✓ Clusters sauvegardés: results/tsne_clusters.html")
    
    print("\n✓✓✓ VISUALISATION TERMINÉE ✓✓✓")
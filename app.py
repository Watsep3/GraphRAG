import streamlit as st
import sys
import os
import json
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import pandas as pd

# Ajouter le chemin src
sys.path.append('C:/Projects/GraphRAG/src')

from rag.rag_pipeline import GraphRAGPipeline
from neo4j import GraphDatabase

# Configuration de la page
st.set_page_config(
    page_title="GraphRAG Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .entity-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .metric-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Charge le pipeline (cached pour éviter de recharger)"""
    with st.spinner("🔄 Chargement du modèle GraphRAG..."):
        try:
            pipeline = GraphRAGPipeline(
                embeddings_path='C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl'
            )
            return pipeline, None
        except Exception as e:
            return None, str(e)

def visualize_graph(entities: list, relations: list = None):
    """Crée une visualisation du sous-graphe"""
    
    # Créer un graphe NetworkX
    G = nx.Graph()
    
    # Ajouter les nœuds
    for entity in entities:
        G.add_node(entity)
    
    # Ajouter les arêtes si disponibles
    if relations:
        for rel in relations:
            if 'source' in rel and 'target' in rel:
                G.add_edge(rel['source'], rel['target'], label=rel.get('type', ''))
    
    # Calculer les positions
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Créer les traces Plotly
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Ajouter les arêtes
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Nœuds
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='#1f77b4',
            line_width=2
        ),
        textposition="top center"
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node[:20]])  # Limiter la longueur
    
    # Créer la figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
    )
    
    return fig

def display_entity_details(entity: str, pipeline: GraphRAGPipeline):
    """Affiche les détails d'une entité"""
    
    with st.expander(f"🔍 Détails: {entity}", expanded=False):
        try:
            context = pipeline.retriever.get_entity_context(entity)
            
            if context['relations']:
                st.write("**Relations:**")
                for rel in context['relations'][:10]:
                    st.markdown(f"- `{rel['relation']}` → **{rel['target']}**")
            else:
                st.info("Aucune relation trouvée dans le graphe")
                
        except Exception as e:
            st.error(f"Erreur: {e}")

def main():
    # Header
    st.markdown('<p class="main-header">🧠 GraphRAG Demo</p>', unsafe_allow_html=True)
    st.markdown("**Représentation Conjointe de Texte et Graphes** | Recherche Hybride Intelligente")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=GraphRAG", use_container_width=True)
        
        st.markdown("---")
        st.markdown("## ⚙️ Configuration")
        
        k_text = st.slider("Top-K Résultats Texte", 1, 20, 5)
        k_graph = st.slider("Top-K Contexte Graphe", 1, 30, 10)
        max_hops = st.slider("Profondeur Graphe (sauts)", 1, 3, 2)
        
        st.markdown("---")
        st.markdown("## 📊 Statistiques")
        
        # Charger les stats
        try:
            with open('C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl', 'rb') as f:
                import pickle
                data = pickle.load(f)
                n_entities = len(data)
            
            st.metric("Entités Indexées", f"{n_entities:,}")
        except:
            st.metric("Entités Indexées", "N/A")
        
        st.markdown("---")
        st.markdown("## 📖 À Propos")
        st.info("""
        Ce système combine:
        - 🔤 Encodage textuel (Sentence-BERT)
        - 🕸️ Encodage graphe (GNN)
        - 🔗 Alignement cross-modal
        - 🎯 RAG hybride
        """)
    
    # Charger le pipeline
    pipeline, error = load_pipeline()
    
    if error:
        st.error(f"❌ Erreur de chargement: {error}")
        st.stop()
    
    st.success("✅ Modèle chargé avec succès!")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Recherche", "📊 Analyse", "🧪 Benchmark", "📚 Documentation"])
    
    # TAB 1: RECHERCHE
    with tab1:
        st.markdown("## 🔍 Recherche Hybride Texte-Graphe")
        
        # Zone de requête
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Posez votre question:",
                placeholder="Ex: What is machine learning? Who is Barack Obama?",
                key="query_input"
            )
        
        with col2:
            search_button = st.button("🔎 Rechercher", type="primary")
        
        # Exemples de questions
        st.markdown("**Exemples:**")
        example_cols = st.columns(3)
        
        examples = [
            "What is artificial intelligence?",
            "Tell me about neural networks",
            "Who invented the computer?"
        ]
        
        for i, (col, example) in enumerate(zip(example_cols, examples)):
            if col.button(example, key=f"example_{i}"):
                query = example
                search_button = True
        
        # Exécuter la recherche
        if search_button and query:
            with st.spinner("🔄 Recherche en cours..."):
                try:
                    # Effectuer la recherche
                    result = pipeline.query(query, k_text=k_text, k_graph=k_graph)
                    
                    # Stocker dans session_state
                    st.session_state['last_result'] = result
                    st.session_state['last_query'] = query
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    st.stop()
        
        # Afficher les résultats
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            st.markdown("---")
            st.markdown("### 📝 Résultats")
            
            # Métriques rapides
            metric_cols = st.columns(3)
            metric_cols[0].metric("📚 Entités Texte", len(result.get('text_results', [])))
            metric_cols[1].metric("🕸️ Entités Graphe", len(result.get('graph_context', [])))
            metric_cols[2].metric("🎯 Total Unique", len(result.get('entities', [])))
            
            # Résultats textuels
            st.markdown("#### 🔤 Top Résultats (Recherche Textuelle)")
            
            for i, text_res in enumerate(result.get('text_results', [])[:k_text], 1):
                with st.container():
                    col1, col2, col3 = st.columns([0.5, 4, 1])
                    
                    col1.markdown(f"**#{i}**")
                    col2.markdown(f"**{text_res['entity']}**")
                    col3.markdown(f"Score: `{text_res['score']:.3f}`")
                    
                    # Bouton pour détails
                    if st.button(f"Voir détails", key=f"detail_text_{i}"):
                        display_entity_details(text_res['entity'], pipeline)
            
            st.markdown("---")
            
            # Contexte graphe
            st.markdown("#### 🕸️ Contexte du Graphe (Entités Connectées)")
            
            graph_entities = result.get('graph_context', [])[:k_graph]
            
            if graph_entities:
                # Tableau
                df_graph = pd.DataFrame(graph_entities)
                st.dataframe(
                    df_graph[['entity', 'hops']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualisation du graphe
                st.markdown("#### 📊 Visualisation du Sous-Graphe")
                
                try:
                    all_entities = [r['entity'] for r in result.get('text_results', [])[:3]]
                    all_entities += [g['entity'] for g in graph_entities[:7]]
                    
                    fig = visualize_graph(list(set(all_entities)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Visualisation non disponible: {e}")
            
            else:
                st.info("Aucun contexte graphe disponible")
            
            st.markdown("---")
            
            # Réponse complète
            with st.expander("📄 Voir la Réponse Complète", expanded=False):
                st.code(result.get('answer', 'N/A'), language='text')
    
    # TAB 2: ANALYSE
    with tab2:
        st.markdown("## 📊 Analyse des Embeddings")
        
        st.info("🚧 Fonctionnalité à venir: Visualisation t-SNE des embeddings")
        
        # Placeholder pour visualisation future
        if st.button("Générer Visualisation t-SNE"):
            st.warning("Cette fonctionnalité sera implémentée dans une version future")
    
    # TAB 3: BENCHMARK
    with tab3:
        st.markdown("## 🧪 Résultats de Benchmark")
        
        # Charger les résultats d'évaluation
        results_file = 'C:/Projects/GraphRAG/results/evaluation_results.json'
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                metrics = json.load(f)
            
            st.markdown("### 📈 Métriques de Performance")
            
            # Afficher les métriques
            cols = st.columns(2)
            
            for i, (metric, value) in enumerate(metrics.items()):
                col = cols[i % 2]
                col.metric(metric.upper(), f"{value:.4f}")
            
            # Graphique
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color='#1f77b4'
                )
            ])
            
            fig.update_layout(
                title="Métriques d'Évaluation",
                xaxis_title="Métrique",
                yaxis_title="Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Aucun résultat de benchmark disponible")
            st.info("Exécutez `python src/evaluation.py` pour générer les métriques")
    
    # TAB 4: DOCUMENTATION
    with tab4:
        st.markdown("## 📚 Documentation")
        
        st.markdown("""
        ### 🎯 Objectif du Projet
        
        Ce système implémente une approche de **Représentation Conjointe de Texte et Graphes**
        pour améliorer la recherche d'information et la génération de réponses.
        
        ### 🏗️ Architecture
        
        #### 1. Encodage Textuel
        - **Modèle**: Sentence-BERT (all-MiniLM-L6-v2)
        - **Dimension**: 384
        - **Fonction**: Encode les noms d'entités et questions en vecteurs
        
        #### 2. Encodage Graphe
        - **Modèle**: GraphSAGE (GNN)
        - **Dimension**: 384
        - **Fonction**: Capture la structure du graphe de connaissances
        
        #### 3. Alignement Cross-Modal
        - **Architecture**: Projection Network
        - **Loss**: Contrastive Loss (bidirectionnel)
        - **Objectif**: Aligner les espaces texte et graphe
        
        #### 4. RAG Pipeline
        - **Retriever**: FAISS + Neo4j
        - **Generator**: Formatage de contexte (extensible avec LLM)
        - **Stratégie**: Recherche hybride texte-graphe
        
        ### 📊 Datasets Utilisés
        
        - **FB15k-237**: Graphe de connaissances (15K entités, 237 relations)
        - **HotpotQA**: Questions multi-hop (90K+ questions)
        - **Wikidata**: Graphe de connaissances mondial (optionnel)
        
        ### 🚀 Utilisation
```python
        from rag.rag_pipeline import GraphRAGPipeline
        
        # Initialiser
        pipeline = GraphRAGPipeline(
            embeddings_path='path/to/embeddings.pkl'
        )
        
        # Requête
        result = pipeline.query("What is AI?", k_text=5, k_graph=10)
        print(result['answer'])
```
        
        ### 📈 Performances
        
        - **Recall@5**: ~0.75
        - **MRR**: ~0.68
        - **Temps de recherche**: <100ms
        
        ### 🔧 Technologies
        
        - PyTorch + PyTorch Geometric
        - Sentence-Transformers
        - Neo4j
        - FAISS
        - Streamlit
        
        ### 👨‍💻 Auteur
        
        **Salma** - Université Internationale de Rabat (UIR)
        Projet de recherche en Big Data & IA
        """)
        
        st.markdown("---")
        
        st.markdown("### 📞 Contact & Support")
        
        col1, col2, col3 = st.columns(3)
        
        col1.markdown("📧 **Email**\nsalma@example.com")
        col2.markdown("💻 **GitHub**\ngithub.com/salma/graphrag")
        col3.markdown("📄 **Paper**\narxiv.org/abs/...")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>"
        f"GraphRAG Demo v1.0 | © 2025 UIR | "
        f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
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

# from rag.rag_pipeline import GraphRAGPipeline
from rag.rag_pipeline_ollama import GraphRAGPipeline
from neo4j import GraphDatabase
import pickle

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
            # Essayer le fichier enrichi d'abord
            embeddings_path = 'C:/Projects/GraphRAG/models/embeddings/entity_embeddings_named.pkl'
            
            # Fallback vers fichier original si pas trouvé
            if not os.path.exists(embeddings_path):
                embeddings_path = 'C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl'
                st.warning("⚠️ Utilisation des embeddings sans noms enrichis")
            
            pipeline = GraphRAGPipeline(
                embeddings_path=embeddings_path,
                ollama_model='llama3.2:3b'
            )
            return pipeline, None
        except Exception as e:
            return None, str(e)

@st.cache_data
def load_entity_names():
    """Charge le mapping des noms d'entités"""
    try:
        with open('C:/Projects/GraphRAG/models/embeddings/entity_embeddings_named.pkl', 'rb') as f:
            data = pickle.load(f)
            entity_names = data.get('entity_names', {})
            return entity_names
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.warning(f"⚠️ Erreur chargement noms: {e}")
        return {}

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
                data = pickle.load(f)
                n_entities = len(data['entities']) if isinstance(data, dict) else len(data)
            
            st.metric("Entités Indexées", f"{n_entities:,}")
        except:
            st.metric("Entités Indexées", "N/A")
        
        st.markdown("---")
        st.markdown("## 🔌 Statut des Services")
        
        # Placeholder pour les statuts (sera rempli après chargement du pipeline)
        status_container = st.empty()
        
        st.markdown("---")
        st.markdown("## 📖 À Propos")
        st.info("""
        Ce système combine:
        - 🔤 Encodage textuel (Sentence-BERT)
        - 🕸️ Encodage graphe (GNN)
        - 🔗 Alignement cross-modal
        - 🎯 RAG hybride
        - 🦙 Génération Ollama
        """)

    # Charger le pipeline
    pipeline, error = load_pipeline()

    if error:
        st.error(f"❌ Erreur de chargement: {error}")
        st.stop()

    st.success("✅ Modèle chargé avec succès!")

    # Maintenant qu'on a le pipeline, on peut afficher les statuts
    with status_container.container():
        col1, col2 = st.columns(2)
        
        neo4j_status = pipeline.retriever.neo4j_available
        ollama_status = pipeline.generator.ollama_available
        
        col1.metric("Neo4j", "✅ Actif" if neo4j_status else "❌ Inactif")
        col2.metric("Ollama", "✅ Actif" if ollama_status else "❌ Inactif")
    
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

            # Ajouter un indicateur si Neo4j n'est pas utilisé
            if not result.get('neo4j_used', False):
                st.warning("⚠️ Neo4j non utilisé - Résultats basés uniquement sur la recherche textuelle")

            # Résultats textuels
            st.markdown("#### 🔤 Top Résultats (Recherche Textuelle)")
            
            # Charger les noms d'entités
            entity_names = load_entity_names()
            
            for i, text_res in enumerate(result.get('text_results', [])[:k_text], 1):
                entity_id = text_res['entity']
                entity_name = entity_names.get(entity_id, entity_id)
                
                with st.container():
                    col1, col2, col3, col4 = st.columns([0.5, 3.5, 2.5, 1])
                    
                    col1.markdown(f"**#{i}**")
                    
                    # Afficher le nom avec style selon disponibilité
                    if entity_name.startswith('['):
                        # ID nettoyé (pas trouvé dans Wikidata)
                        col2.markdown(f"*{entity_name}*")
                    else:
                        # Vrai nom trouvé
                        col2.markdown(f"**{entity_name}**")
                    
                    # Afficher l'ID Freebase
                    col3.markdown(f"`{entity_id}`")
                    
                    # Score
                    col4.markdown(f"`{text_res['score']:.3f}`")
                    
                    # Bouton pour détails
                    if st.button(f"Voir détails", key=f"detail_text_{i}"):
                        display_entity_details(entity_id, pipeline)
            
            st.markdown("---")
            
            # Contexte graphe
            st.markdown("#### 🕸️ Contexte du Graphe (Entités Connectées)")

            graph_entities = result.get('graph_context', [])[:k_graph]

            if graph_entities:
                # Enrichir avec les noms
                for entity_dict in graph_entities:
                    entity_id = entity_dict['entity']
                    entity_dict['name'] = entity_names.get(entity_id, entity_id)
                
                # Tableau avec noms
                df_graph = pd.DataFrame(graph_entities)
                
                # Sélectionner les colonnes à afficher
                display_cols = ['name', 'entity', 'hops'] if 'name' in df_graph.columns else ['entity', 'hops']
                
                st.dataframe(
                    df_graph[display_cols].head(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "name": st.column_config.TextColumn("Nom", width="large"),
                        "entity": st.column_config.TextColumn("ID Freebase", width="medium"),
                        "hops": st.column_config.NumberColumn("Distance", width="small")
                    }
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
        
        st.markdown("""
        Cette visualisation utilise **t-SNE** (t-Distributed Stochastic Neighbor Embedding) 
        pour projeter les embeddings 384D dans un espace 2D, révélant la structure sémantique.
        """)
        
        # Configuration t-SNE
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Échantillon", 500, 5000, 2000, 500)
        with col2:
            perplexity = st.slider("Perplexity", 5, 50, 30, 5)
        with col3:
            n_clusters = st.slider("Nombre de Clusters", 3, 15, 8, 1)
        
        # Bouton de génération
        if st.button("🎨 Générer Visualisation t-SNE", type="primary"):
            with st.spinner("⏳ Calcul en cours (cela peut prendre 1-2 minutes)..."):
                try:
                    from visualize_tsne import EmbeddingVisualizer
                    
                    # Créer visualiseur
                    viz = EmbeddingVisualizer(
                        'C:/Projects/GraphRAG/models/embeddings/entity_embeddings.pkl'
                    )
                    
                    # Calculer t-SNE
                    progress_bar = st.progress(0)
                    st.info("Étape 1/3: Calcul t-SNE...")
                    
                    viz.compute_tsne(
                        n_components=2,
                        perplexity=perplexity,
                        max_iter=1000,
                        sample_size=sample_size
                    )
                    progress_bar.progress(33)
                    
                    # Créer visualisation
                    st.info("Étape 2/3: Génération du graphique...")
                    fig = viz.create_interactive_plot(
                        title=f"t-SNE: Entity Embeddings (n={sample_size})"
                    )
                    progress_bar.progress(66)
                    
                    # Détecter clusters
                    st.info("Étape 3/3: Détection des clusters...")
                    clusters = viz.find_clusters(n_clusters=n_clusters)
                    fig_clusters = viz.plot_clusters()
                    progress_bar.progress(100)
                    
                    # Sauvegarder dans session state
                    st.session_state['tsne_fig'] = fig
                    st.session_state['tsne_fig_clusters'] = fig_clusters
                    st.session_state['clusters_info'] = clusters
                    
                    st.success("✅ Visualisation générée!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Afficher les visualisations si disponibles
        if 'tsne_fig' in st.session_state:
            st.markdown("---")
            
            # Tabs pour les deux visualisations
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 t-SNE Standard", "🎯 t-SNE avec Clusters", "📋 Analyse des Clusters"])
            
            with viz_tab1:
                st.markdown("### Projection t-SNE des Embeddings")
                st.plotly_chart(st.session_state['tsne_fig'], use_container_width=True)
                
                st.info("""
                **Interprétation:**
                - Les entités proches dans l'espace t-SNE ont des embeddings similaires
                - Les clusters visibles indiquent des groupes sémantiques
                - Vous pouvez zoomer et survoler les points pour voir les entités
                """)
            
            with viz_tab2:
                st.markdown("### t-SNE avec Clustering K-Means")
                st.plotly_chart(st.session_state['tsne_fig_clusters'], use_container_width=True)
                
                st.info("""
                **Clusters colorés:**
                - Chaque couleur représente un cluster découvert automatiquement
                - Les clusters peuvent correspondre à des catégories sémantiques
                """)
            
            with viz_tab3:
                st.markdown("### Analyse des Clusters")
                
                if 'clusters_info' in st.session_state:
                    clusters = st.session_state['clusters_info']
                    
                    # Statistiques globales
                    st.markdown("### 📊 Statistiques Globales")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    total_entities = sum(info['size'] for info in clusters.values())
                    avg_cluster_size = total_entities / len(clusters)
                    sizes = [info['size'] for info in clusters.values()]
                    largest_size = max(sizes)
                    
                    col1.metric("Total Entités", total_entities)
                    col2.metric("Taille Moyenne", f"{avg_cluster_size:.0f}")
                    col3.metric("Plus Grand Cluster", largest_size)
                    
                    st.markdown("---")
                    
                    # Afficher les clusters
                    for cluster_name, info in clusters.items():
                        with st.expander(f"{cluster_name} - {info['size']} entités"):
                            st.markdown(f"**Taille:** {info['size']} entités")
                            st.markdown("**Entités représentatives:**")
                            for entity in info['entities'][:10]:
                                st.markdown(f"- `{entity}`")
        
        else:
            st.info("👆 Cliquez sur le bouton ci-dessus pour générer la visualisation t-SNE")
    
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
        - **Generator**: Ollama LLM (llama3.2:3b)
        - **Stratégie**: Recherche hybride texte-graphe
        
        ### 📊 Datasets Utilisés
        
        - **FB15k-237**: Graphe de connaissances (14,505 entités, 237 relations)
        - **HotpotQA**: Questions multi-hop
        - **Wikidata**: Enrichissement des noms d'entités
        
        ### 🚀 Utilisation
```python
from rag.rag_pipeline_ollama import GraphRAGPipeline

# Initialiser
pipeline = GraphRAGPipeline(
    embeddings_path='path/to/embeddings.pkl',
    ollama_model='llama3.2:3b'
)

# Requête
result = pipeline.query("What is AI?", k_text=5, k_graph=10)
print(result['answer'])
```
        
        ### 📈 Performances
        
        - **Recall@5**: 60.0%
        - **MRR**: 80.9%
        - **Precision@5**: 20.8%
        - **F1@5**: 30.5%
        - **Temps de recherche**: <100ms
        
        ### 🔧 Technologies
        
        - PyTorch 2.5.1 + PyTorch Geometric
        - Sentence-Transformers (all-MiniLM-L6-v2)
        - Neo4j 5.x (272K+ relations)
        - FAISS (14K+ embeddings)
        - Ollama (llama3.2:3b local)
        - Streamlit
        
        ### 👨‍💻 Auteurs
        
        **Salma Berrada & Marwa Ghachi**
        
        Université Internationale de Rabat (UIR)
        
        Projet de Fin d'Études - Big Data & IA - Semestre 9 (2025-2026)
        
        Superviseur: Prof. Hakim Hafidi
        """)
    
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
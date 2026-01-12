import bz2
import json
import pandas as pd
from tqdm import tqdm
import re
from typing import Dict, List, Tuple
import pickle
import os
from pathlib import Path

class WikidataProcessor:
    """Traite les dumps Wikidata"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        
    def parse_ntriples_line(self, line: str) -> Dict:
        """Parse une ligne N-Triples"""
        parts = line.strip().split(' ', 2)
        if len(parts) < 3:
            return None
            
        subject = parts[0].strip('<>')
        predicate = parts[1].strip('<>')
        obj = parts[2].rsplit(' .', 1)[0].strip('<>"')
        
        return {
            'subject': subject,
            'predicate': predicate,
            'object': obj
        }
    
    def process_sample(self, max_triples: int = 100000):
        """Traite un échantillon de Wikidata"""
        
        if not os.path.exists(self.input_file):
            print(f"⚠️  Fichier non trouvé: {self.input_file}")
            return None
        
        triples = []
        entities = set()
        relations = set()
        
        print(f"Traitement de {self.input_file}...")
        
        with bz2.open(self.input_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=max_triples, desc="Parsing")):
                if i >= max_triples:
                    break
                    
                triple = self.parse_ntriples_line(line)
                if triple:
                    triples.append(triple)
                    entities.add(triple['subject'])
                    entities.add(triple['object'])
                    relations.add(triple['predicate'])
        
        # Sauvegarder
        os.makedirs(self.output_dir, exist_ok=True)
        
        df = pd.DataFrame(triples)
        df.to_csv(f'{self.output_dir}/wikidata_triples.csv', index=False)
        
        with open(f'{self.output_dir}/entities.txt', 'w', encoding='utf-8') as f:
            for entity in sorted(entities):
                f.write(f"{entity}\n")
        
        with open(f'{self.output_dir}/relations.txt', 'w', encoding='utf-8') as f:
            for rel in sorted(relations):
                f.write(f"{rel}\n")
        
        print(f"✓ {len(triples)} triplets sauvegardés")
        print(f"✓ {len(entities)} entités uniques")
        print(f"✓ {len(relations)} relations uniques")
        
        return df

class FB15kProcessor:
    """Traite FB15k-237"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
    
    def find_fb15k_files(self):
        """Trouve les fichiers FB15k peu importe la structure"""
        
        # Chercher dans différents emplacements possibles
        possible_locations = [
            os.path.join(self.data_dir, 'Release'),           # FB15k-237/Release/
            os.path.join(self.data_dir, 'release'),           # fb15k237/release/
            self.data_dir,                                     # fb15k237/
            os.path.join(self.data_dir, '..', 'fb15k237', 'Release'),
        ]
        
        for location in possible_locations:
            if not os.path.exists(location):
                continue
                
            train_path = os.path.join(location, 'train.txt')
            if os.path.exists(train_path):
                print(f"✓ Fichiers FB15k-237 trouvés dans: {location}")
                return location
        
        # Si toujours pas trouvé, chercher récursivement
        print(f"Recherche récursive dans {self.data_dir}...")
        for root, dirs, files in os.walk(self.data_dir):
            if 'train.txt' in files:
                print(f"✓ Fichiers FB15k-237 trouvés dans: {root}")
                return root
        
        return None
    
    def load_split(self, split: str, base_path: str) -> pd.DataFrame:
        """Charge train/valid/test"""
        filepath = os.path.join(base_path, f"{split}.txt")
        
        if not os.path.exists(filepath):
            print(f"⚠️  Fichier non trouvé: {filepath}")
            return None
        
        print(f"  Chargement de {filepath}...")
        data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) == 3:
                        data.append({
                            'head': parts[0],
                            'relation': parts[1],
                            'tail': parts[2]
                        })
                    else:
                        print(f"    ⚠️  Ligne {line_num} ignorée (format incorrect)")
        
        except Exception as e:
            print(f"  ✗ Erreur lors de la lecture: {e}")
            return None
        
        if not data:
            print(f"  ⚠️  Aucune donnée valide trouvée dans {filepath}")
            return None
        
        return pd.DataFrame(data)
    
    def process_all(self):
        """Traite tous les splits"""
        
        # Trouver les fichiers
        base_path = self.find_fb15k_files()
        
        if base_path is None:
            print("\n❌ ERREUR: Fichiers FB15k-237 non trouvés!")
            print("\n📂 Structure attendue:")
            print("  C:/Projects/GraphRAG/data/raw/fb15k237/Release/")
            print("    ├── train.txt")
            print("    ├── valid.txt")
            print("    └── test.txt")
            print("\n📥 Téléchargez FB15k-237:")
            print("   cd C:/Projects/GraphRAG/data/raw")
            print("   python download_fb15k.py")
            return None
        
        splits = {}
        os.makedirs(self.output_dir, exist_ok=True)
        
        for split in ['train', 'valid', 'test']:
            print(f"\nTraitement {split}...")
            df = self.load_split(split, base_path)
            
            if df is None or len(df) == 0:
                print(f"  ⚠️  Skipping {split} (vide ou erreur)")
                continue
            
            splits[split] = df
            
            # Sauvegarder
            output_file = f'{self.output_dir}/fb15k_{split}.csv'
            df.to_csv(output_file, index=False)
            print(f"  ✓ {len(df)} triplets sauvegardés → {output_file}")
        
        if not splits:
            return None
        
        # Statistiques globales
        all_data = pd.concat(splits.values())
        
        print(f"\n{'='*60}")
        print("STATISTIQUES FB15K-237")
        print(f"{'='*60}")
        print(f"Total triplets: {len(all_data):,}")
        print(f"Entités uniques: {len(set(all_data['head']) | set(all_data['tail'])):,}")
        print(f"Relations uniques: {len(all_data['relation'].unique())}")
        
        # Top relations
        print(f"\nTop 10 relations:")
        top_relations = all_data['relation'].value_counts().head(10)
        for i, (rel, count) in enumerate(top_relations.items(), 1):
            print(f"  {i}. {rel}: {count:,}")
        
        return splits

class HotpotQAProcessor:
    """Traite HotpotQA"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
    
    def process(self, max_samples: int = 10000):
        """Traite et filtre HotpotQA"""
        
        if not os.path.exists(self.input_file):
            print(f"⚠️  Fichier non trouvé: {self.input_file}")
            return None
        
        print(f"Traitement {self.input_file}...")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"✗ Erreur de lecture JSON: {e}")
            return None
        
        # Limiter si nécessaire
        if max_samples and len(data) > max_samples:
            print(f"  Limitation à {max_samples} échantillons (sur {len(data)} disponibles)")
            data = data[:max_samples]
        
        processed = []
        for item in tqdm(data, desc="Processing"):
            processed.append({
                'id': item.get('_id', ''),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'type': item.get('type', ''),
                'level': item.get('level', ''),
                'supporting_facts': str(item.get('supporting_facts', [])),
                'context': str(item.get('context', []))
            })
        
        # Sauvegarder
        os.makedirs(self.output_dir, exist_ok=True)
        
        df = pd.DataFrame(processed)
        output_filename = os.path.basename(self.input_file).replace('.json', '_processed.csv')
        output_path = os.path.join(self.output_dir, output_filename)
        df.to_csv(output_path, index=False)
        
        print(f"✓ {len(df)} questions traitées → {output_path}")
        
        return df

# Script principal
def main():
    print("="*60)
    print("PREPROCESSING GRAPHRAG - VERSION CORRIGÉE")
    print("="*60)
    
    # Créer dossiers de sortie
    output_dir = 'C:/Projects/GraphRAG/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    success = {'wikidata': False, 'fb15k': False, 'hotpotqa': False}
    
    # Traiter Wikidata (optionnel)
    print("\n" + "="*60)
    print("ÉTAPE 1: WIKIDATA (Optionnel)")
    print("="*60)
    
    wikidata_file = 'C:/Projects/GraphRAG/data/raw/wikidata-truthy.nt.bz2'
    
    if os.path.exists(wikidata_file):
        wikidata_proc = WikidataProcessor(wikidata_file, output_dir)
        wikidata_df = wikidata_proc.process_sample(max_triples=100000)
        success['wikidata'] = wikidata_df is not None
    else:
        print(f"⚠️  Wikidata non trouvé (optionnel, peut être ignoré)")
    
    # Traiter FB15k-237 (ESSENTIEL!)
    print("\n" + "="*60)
    print("ÉTAPE 2: FB15K-237 (ESSENTIEL)")
    print("="*60)
    
    # Utiliser le chemin exact que tu as montré dans l'image
    fb15k_base = 'C:/Projects/GraphRAG/data/raw/fb15k237'
    
    fb15k_proc = FB15kProcessor(fb15k_base, output_dir)
    fb15k_splits = fb15k_proc.process_all()
    
    if fb15k_splits is None:
        print("\n❌ ERREUR CRITIQUE: FB15k-237 requis mais non trouvé!")
        print("\n📥 Téléchargez-le maintenant:")
        print("   cd C:/Projects/GraphRAG/data/raw")
        print("   python download_fb15k.py")
        success['fb15k'] = False
    else:
        success['fb15k'] = True
    
    # Traiter HotpotQA (optionnel)
    print("\n" + "="*60)
    print("ÉTAPE 3: HOTPOTQA (Optionnel)")
    print("="*60)
    
    hotpotqa_files = [
        ('C:/Projects/GraphRAG/data/raw/hotpotqa_train.json', 5000),
        ('C:/Projects/GraphRAG/data/raw/hotpotqa_dev_distractor.json', 2000),
    ]
    
    hotpotqa_processed = False
    for hotpotqa_file, max_samples in hotpotqa_files:
        if os.path.exists(hotpotqa_file):
            print(f"\nTraitement: {os.path.basename(hotpotqa_file)}")
            hotpot_proc = HotpotQAProcessor(hotpotqa_file, output_dir)
            hotpot_df = hotpot_proc.process(max_samples=max_samples)
            if hotpot_df is not None:
                hotpotqa_processed = True
    
    if not hotpotqa_processed:
        print(f"⚠️  HotpotQA non trouvé (optionnel, peut être ignoré)")
    
    success['hotpotqa'] = hotpotqa_processed
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ DU PREPROCESSING")
    print("="*60)
    
    status_table = {
        'Wikidata': success['wikidata'],
        'FB15k-237': success['fb15k'],
        'HotpotQA': success['hotpotqa']
    }
    
    for dataset, status in status_table.items():
        status_icon = "✓" if status else "✗"
        status_text = "Traité" if status else "Non traité"
        importance = "(ESSENTIEL)" if dataset == "FB15k-237" else "(Optionnel)"
        print(f"{status_icon} {dataset:.<30} {status_text} {importance}")
    
    # Résultat final
    print("\n" + "="*60)
    if success['fb15k']:
        print("✓✓✓ PREPROCESSING TERMINÉ AVEC SUCCÈS ✓✓✓")
        print("="*60)
        print(f"\n📂 Fichiers créés dans: {output_dir}")
        
        # Lister les fichiers créés
        print("\n📄 Fichiers disponibles:")
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                filepath = os.path.join(output_dir, file)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  ✓ {file} ({size_mb:.2f} MB)")
        
        print("\n📋 PROCHAINES ÉTAPES:")
        print("="*60)
        print("1. Charger les données dans Neo4j:")
        print("   python src/load_neo4j.py")
        print()
        print("2. Créer les embeddings textuels:")
        print("   python src/text/text_encoder.py")
        print()
        print("3. Créer les embeddings graphe:")
        print("   python src/graph/graph_encoder.py")
        print()
        print("4. Entraîner le modèle cross-modal:")
        print("   python src/cross_modal_training.py")
        
    else:
        print("❌ PREPROCESSING INCOMPLET - FB15k-237 MANQUANT")
        print("="*60)
        print("\n📥 Action requise:")
        print("   1. cd C:/Projects/GraphRAG/data/raw")
        print("   2. python download_fb15k.py")
        print("   3. Relancer: python src/preprocessing_fixed.py")

if __name__ == "__main__":
    main()
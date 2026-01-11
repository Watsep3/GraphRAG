import bz2
import json
import pandas as pd
from tqdm import tqdm
import re
from typing import Dict, List, Tuple
import pickle

class WikidataProcessor:
    """Traite les dumps Wikidata"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        
    def parse_ntriples_line(self, line: str) -> Dict:
        """Parse une ligne N-Triples"""
        # Format: <subject> <predicate> <object> .
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
        """
        Traite un échantillon de Wikidata
        max_triples: nombre de triplets à extraire
        """
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
    
    def load_split(self, split: str) -> pd.DataFrame:
        """Charge train/valid/test"""
        filepath = f"{self.data_dir}/{split}.txt"
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    data.append({
                        'head': parts[0],
                        'relation': parts[1],
                        'tail': parts[2]
                    })
        
        return pd.DataFrame(data)
    
    def process_all(self):
        """Traite tous les splits"""
        splits = {}
        
        for split in ['train', 'valid', 'test']:
            print(f"Traitement {split}...")
            df = self.load_split(split)
            splits[split] = df
            
            # Sauvegarder
            df.to_csv(f'{self.output_dir}/fb15k_{split}.csv', index=False)
            print(f"✓ {len(df)} triplets dans {split}")
        
        # Statistiques globales
        all_data = pd.concat(splits.values())
        print(f"\nStatistiques FB15k-237:")
        print(f"Total triplets: {len(all_data)}")
        print(f"Entités uniques: {len(set(all_data['head']) | set(all_data['tail']))}")
        print(f"Relations uniques: {len(all_data['relation'].unique())}")
        
        return splits

class HotpotQAProcessor:
    """Traite HotpotQA"""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
    
    def process(self, max_samples: int = 10000):
        """Traite et filtre HotpotQA"""
        print(f"Traitement {self.input_file}...")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Limiter si nécessaire
        if max_samples:
            data = data[:max_samples]
        
        processed = []
        for item in tqdm(data, desc="Processing"):
            processed.append({
                'id': item['_id'],
                'question': item['question'],
                'answer': item['answer'],
                'type': item['type'],
                'level': item['level'],
                'supporting_facts': item['supporting_facts'],
                'context': item['context']
            })
        
        # Sauvegarder
        df = pd.DataFrame(processed)
        output_file = self.input_file.replace('.json', '_processed.csv')
        df.to_csv(f"{self.output_dir}/{output_file.split('/')[-1]}", index=False)
        
        print(f"✓ {len(df)} questions traitées")
        
        return df

# Script principal
if __name__ == "__main__":
    import os
    
    # Créer dossiers de sortie
    os.makedirs('C:/Projects/GraphRAG/data/processed', exist_ok=True)
    
    # Traiter Wikidata (version réduite pour test)
    print("\n=== WIKIDATA ===")
    wikidata_proc = WikidataProcessor(
        'C:/Projects/GraphRAG/data/raw/wikidata-truthy.nt.bz2',
        'C:/Projects/GraphRAG/data/processed'
    )
    wikidata_df = wikidata_proc.process_sample(max_triples=100000)  # 100K pour commencer
    
    # Traiter FB15k-237
    print("\n=== FB15K-237 ===")
    fb15k_proc = FB15kProcessor(
        'C:/Projects/GraphRAG/data/raw/fb15k237',
        'C:/Projects/GraphRAG/data/processed'
    )
    fb15k_splits = fb15k_proc.process_all()
    
    # Traiter HotpotQA
    print("\n=== HOTPOTQA ===")
    for split in ['train', 'dev_distractor']:
        hotpot_proc = HotpotQAProcessor(
            f'C:/Projects/GraphRAG/data/raw/hotpotqa_{split}.json',
            'C:/Projects/GraphRAG/data/processed'
        )
        hotpot_df = hotpot_proc.process(max_samples=5000)  # 5K par split
    
    print("\n✓✓✓ PREPROCESSING TERMINÉ ✓✓✓") 
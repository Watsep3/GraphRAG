import requests
import json
import pickle
from typing import Dict, Optional, List
import time
from tqdm import tqdm
import os
import re

class EntityNameMapperEnhanced:
    """Mapper amélioré avec multiples sources de données"""
    
    def __init__(self, cache_file: str = 'C:/Projects/GraphRAG/data/processed/entity_names_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.session = requests.Session()
        
        # Statistiques
        self.stats = {
            'wikidata': 0,
            'dbpedia': 0,
            'cleaned': 0,
            'cached': 0
        }
    
    def load_cache(self) -> Dict[str, str]:
        try:
            with open(self.cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"✓ Cache chargé: {len(cache)} entités")
            return cache
        except FileNotFoundError:
            print("Cache vide, création...")
            return {}
    
    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"✓ Cache sauvegardé: {len(self.cache)} entités")
    
    def get_name_from_wikidata(self, freebase_id: str) -> Optional[str]:
        """Récupère depuis Wikidata"""
        
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item wdt:P646 "{freebase_id}" .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """
        
        url = "https://query.wikidata.org/sparql"
        
        try:
            response = self.session.get(
                url,
                params={'query': query, 'format': 'json'},
                headers={'User-Agent': 'GraphRAG/1.0'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                
                if bindings:
                    label = bindings[0].get('itemLabel', {}).get('value')
                    if label and label != freebase_id:
                        self.stats['wikidata'] += 1
                        return label
            
            return None
            
        except Exception as e:
            return None
    
    def get_name_from_freebase_easy(self, freebase_id: str) -> Optional[str]:
        """Essaye l'API Freebase Easy (miroir communautaire)"""
        
        # Convertir /m/02mjmr en format URL
        mid = freebase_id.replace('/', '.')
        
        try:
            url = f"https://www.freebase.com{freebase_id}"
            response = self.session.get(
                url,
                headers={'User-Agent': 'GraphRAG/1.0'},
                timeout=5,
                allow_redirects=True
            )
            
            # Parser le titre de la page
            if response.status_code == 200:
                import re
                title_match = re.search(r'<title>([^<]+)</title>', response.text)
                if title_match:
                    title = title_match.group(1)
                    # Nettoyer
                    title = title.replace(' - Freebase', '').strip()
                    if title and not title.startswith(mid):
                        self.stats['dbpedia'] += 1
                        return title
            
            return None
        except:
            return None
    
    def clean_freebase_id(self, freebase_id: str) -> str:
        """Nettoie l'ID Freebase pour le rendre plus lisible"""
        
        # Enlever /m/ et remplacer underscore
        clean = freebase_id.replace('/m/', '').replace('_', ' ')
        
        # Ajouter des espaces avant les chiffres
        clean = re.sub(r'(\d+)', r' \1', clean)
        
        # Capitaliser
        clean = clean.title().strip()
        
        self.stats['cleaned'] += 1
        return f"[{clean}]"
    
    def get_name(self, freebase_id: str, use_cache: bool = True) -> str:
        """Récupère le nom avec fallback multi-sources"""
        
        # Cache
        if use_cache and freebase_id in self.cache:
            self.stats['cached'] += 1
            return self.cache[freebase_id]
        
        # 1. Essayer Wikidata
        name = self.get_name_from_wikidata(freebase_id)
        if name:
            self.cache[freebase_id] = name
            return name
        
        # 2. Fallback: ID nettoyé
        clean_name = self.clean_freebase_id(freebase_id)
        self.cache[freebase_id] = clean_name
        return clean_name
    
    def batch_get_names(self, freebase_ids: list, delay: float = 0.1, 
                       max_items: int = None) -> Dict[str, str]:
        """Récupère en batch avec progress bar"""
        
        results = {}
        uncached = [fid for fid in freebase_ids if fid not in self.cache]
        
        if max_items:
            uncached = uncached[:max_items]
        
        print(f"\n🔍 Récupération de {len(uncached)} noms (sur {len(freebase_ids)} total)")
        
        for i, fid in enumerate(tqdm(uncached, desc="Fetching names")):
            name = self.get_name(fid, use_cache=False)
            results[fid] = name
            time.sleep(delay)
            
            # Sauvegarder tous les 100 items
            if (i + 1) % 100 == 0:
                self.save_cache()
        
        # Ajouter les noms en cache
        for fid in freebase_ids:
            if fid in self.cache:
                results[fid] = self.cache[fid]
        
        # Sauvegarder le cache final
        self.save_cache()
        
        # Afficher les stats
        print("\n📊 Statistiques:")
        total = sum(self.stats.values())
        if total > 0:
            print(f"  Wikidata:    {self.stats['wikidata']:4d} ({self.stats['wikidata']/total*100:.1f}%)")
            print(f"  Cache:       {self.stats['cached']:4d} ({self.stats['cached']/total*100:.1f}%)")
            print(f"  Nettoyés:    {self.stats['cleaned']:4d} ({self.stats['cleaned']/total*100:.1f}%)")
        
        return results
    
    def enrich_embeddings_file(self, 
                               embeddings_path: str, 
                               output_path: str = None,
                               max_entities: int = 1000):
        """Enrichit le fichier embeddings avec les noms"""
        
        print("="*60)
        print("ENRICHISSEMENT DES EMBEDDINGS")
        print("="*60)
        
        print("\n1. Chargement des embeddings...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        entities = data['entities']
        print(f"✓ {len(entities)} entités chargées")
        
        # Limite pour test
        if max_entities:
            entities_to_process = entities[:max_entities]
            print(f"⚠️  Limitation à {max_entities} entités pour test")
        else:
            entities_to_process = entities
        
        # Récupérer les noms
        print(f"\n2. Récupération des noms...")
        names = self.batch_get_names(entities_to_process, delay=0.1)
        
        # Créer mapping complet pour toutes les entités
        full_mapping = {}
        for entity in entities:
            if entity in names:
                full_mapping[entity] = names[entity]
            else:
                full_mapping[entity] = self.clean_freebase_id(entity)
        
        # Ajouter au data
        data['entity_names'] = full_mapping
        
        # Sauvegarder
        if output_path is None:
            output_path = embeddings_path.replace('.pkl', '_named.pkl')
        
        print(f"\n3. Sauvegarde...")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Embeddings enrichis: {output_path}")
        print(f"✓ {len(full_mapping)} noms mappés")
        
        # Exemples
        print("\n📝 Exemples de mapping:")
        for i, (fid, name) in enumerate(list(names.items())[:10]):
            status = "✓" if not name.startswith("[") else "⚠️"
            print(f"  {status} {fid:20s} → {name}")
        
        print("\n" + "="*60)
        print("✓✓✓ ENRICHISSEMENT TERMINÉ ✓✓✓")
        print("="*60)
        
        return data

# Helper function
def get_entity_name(freebase_id: str) -> str:
    mapper = EntityNameMapperEnhanced()
    return mapper.get_name(freebase_id)

# Test
if __name__ == "__main__":
    print("="*60)
    print("ENTITY NAME MAPPER ENHANCED - TEST")
    print("="*60)
    
    mapper = EntityNameMapperEnhanced()
    
    # Test IDs
    test_ids = [
        '/m/02mjmr',  # Barack Obama
        '/m/0d06m5',  # Hillary Clinton
        '/m/09c7w0',  # United States
        '/m/02zsn',   # Washington D.C.
        '/m/0jbk9',   # Angela Merkel
        '/m/053xw6',  # Inconnu
        '/m/0pc6x',   # Computer Science
        '/m/019lwb',  # Python
    ]
    
    print("\n📝 Test de mapping:")
    for fid in test_ids:
        name = mapper.get_name(fid)
        status = "✓" if not name.startswith("[") else "⚠️"
        print(f"  {status} {fid:20s} → {name}")
    
    mapper.save_cache()
    
    print("\n📊 Statistiques finales:")
    total = sum(mapper.stats.values())
    for source, count in mapper.stats.items():
        if total > 0:
            print(f"  {source:12s}: {count:3d} ({count/total*100:5.1f}%)")
    
    print("\n" + "="*60)
    print("✓✓✓ TEST TERMINÉ ✓✓✓")
    print("="*60)
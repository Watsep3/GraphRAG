import requests
import gzip
import json

# Télécharger le mapping Freebase -> Noms depuis des dumps publics
# Source: https://developers.google.com/freebase

def download_freebase_mapping():
    """Télécharge un mapping Freebase ID -> Name"""
    
    # Option 1: Fichier simplifié (recommandé pour démarrer)
    url = "https://raw.githubusercontent.com/nchah/freebase-triples/master/freebase-names.txt"
    
    print(f"Téléchargement depuis {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            mapping = {}
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        fid, name = parts[0], parts[1]
                        mapping[fid] = name
            
            print(f"✓ {len(mapping)} entités mappées")
            
            # Sauvegarder
            import pickle
            with open('freebase_names.pkl', 'wb') as f:
                pickle.dump(mapping, f)
            
            print("✓ Mapping sauvegardé: freebase_names.pkl")
            return mapping
        
        else:
            print(f"✗ Erreur {response.status_code}")
            return None
    
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None

if __name__ == "__main__":
    download_freebase_mapping()
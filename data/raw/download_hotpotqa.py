import sys
import os
import json
from pathlib import Path

sys.path.append('C:/Projects/GraphRAG/src')
from downloader import ResumableDownloader

def verify_json(filepath: str) -> bool:
    """Vérifie qu'un fichier JSON est valide"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False
    except Exception:
        return False

def main():
    downloader = ResumableDownloader(max_retries=10)
    
    print("="*60)
    print("TÉLÉCHARGEMENT HOTPOTQA")
    print("="*60)
    
    # Définir les URLs
    datasets = {
        'train': {
            'url': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json',
            'file': 'hotpotqa_train.json',
            'desc': 'Training set'
        },
        'dev_distractor': {
            'url': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
            'file': 'hotpotqa_dev_distractor.json',
            'desc': 'Dev set (distractor)'
        },
        'dev_fullwiki': {
            'url': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json',
            'file': 'hotpotqa_dev_fullwiki.json',
            'desc': 'Dev set (fullwiki)'
        },
        'test_fullwiki': {
            'url': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json',
            'file': 'hotpotqa_test_fullwiki.json',
            'desc': 'Test set (fullwiki)'
        }
    }
    
    print("\nDatasets disponibles:")
    for i, (key, info) in enumerate(datasets.items(), 1):
        print(f"{i}. {info['desc']} - {info['file']}")
    
    print(f"{len(datasets)+1}. Tous les datasets")
    
    choice = input(f"\nVotre choix (1-{len(datasets)+1}): ").strip()
    
    # Sélectionner les datasets à télécharger
    if choice == str(len(datasets)+1):
        selected = list(datasets.keys())
    else:
        try:
            idx = int(choice) - 1
            selected = [list(datasets.keys())[idx]]
        except:
            print("✗ Choix invalide")
            return
    
    print(f"\n✓ {len(selected)} dataset(s) sélectionné(s)")
    print("\n💡 Astuce: Interruption possible avec Ctrl+C, reprise automatique!\n")
    
    # Télécharger
    success_count = 0
    failed = []
    
    try:
        for i, key in enumerate(selected, 1):
            info = datasets[key]
            print(f"\n[{i}/{len(selected)}] Téléchargement: {info['desc']}")
            print(f"URL: {info['url']}")
            
            try:
                # Télécharger
                downloader.download_file(info['url'], info['file'])
                
                # Vérifier que le JSON est valide
                print(f"Vérification du fichier JSON...")
                if verify_json(info['file']):
                    print(f"✓ Fichier JSON valide")
                    
                    # Compter les entrées
                    with open(info['file'], 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"✓ {len(data)} entrées dans le dataset")
                    
                    success_count += 1
                else:
                    print(f"✗ Fichier JSON invalide, suppression...")
                    os.remove(info['file'])
                    failed.append(key)
                    
            except Exception as e:
                print(f"✗ Échec: {e}")
                failed.append(key)
        
        # Résumé
        print("\n" + "="*60)
        print(f"RÉSUMÉ: {success_count}/{len(selected)} téléchargements réussis")
        print("="*60)
        
        if success_count == len(selected):
            print("✓✓✓ TOUS LES TÉLÉCHARGEMENTS TERMINÉS ✓✓✓")
        else:
            print(f"\n⚠ Échecs: {', '.join(failed)}")
            print("↻ Relancez le script pour réessayer les échecs")
        
        # Liste des fichiers téléchargés
        print("\nFichiers téléchargés:")
        for key in selected:
            if key not in failed:
                info = datasets[key]
                if os.path.exists(info['file']):
                    size_mb = os.path.getsize(info['file']) / (1024 * 1024)
                    print(f"  ✓ {info['file']} ({size_mb:.2f} MB)")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Téléchargement interrompu")
        print(f"✓ {success_count} fichier(s) téléchargé(s) avec succès")
        print("↻ Relancez ce script pour reprendre")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")

if __name__ == "__main__":
    main()
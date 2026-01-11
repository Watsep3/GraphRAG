import sys
import os
import zipfile
from pathlib import Path

sys.path.append('C:/Projects/GraphRAG/src')
from downloader import ResumableDownloader

def extract_zip(zip_path: str, extract_to: str):
    """Extrait un fichier zip"""
    print(f"\nDécompression de {Path(zip_path).name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Obtenir la liste des fichiers
            files = zip_ref.namelist()
            
            # Extraire avec progression
            for file in files:
                zip_ref.extract(file, extract_to)
                print(f"  ✓ Extrait: {file}")
        
        print(f"✓ Décompression terminée")
        return True
        
    except zipfile.BadZipFile:
        print(f"✗ Fichier zip corrompu: {zip_path}")
        print("↻ Supprimez le fichier et relancez le téléchargement")
        return False
    except Exception as e:
        print(f"✗ Erreur lors de l'extraction: {e}")
        return False

def main():
    downloader = ResumableDownloader(max_retries=10)
    
    print("="*60)
    print("TÉLÉCHARGEMENT FB15K-237")
    print("="*60)
    
    # URL et destination
    url = "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"
    zip_file = "FB15K-237.2.zip"
    extract_dir = "fb15k237"
    
    print(f"\nURL: {url}")
    print(f"Destination: {zip_file}")
    print(f"Extraction: {extract_dir}/")
    print("\n💡 Astuce: Interruption possible avec Ctrl+C, reprise automatique!\n")
    
    try:
        # Télécharger
        print("[1/2] Téléchargement du fichier ZIP...")
        downloader.download_file(url, zip_file)
        
        # Extraire
        print(f"\n[2/2] Extraction des fichiers...")
        os.makedirs(extract_dir, exist_ok=True)
        
        if extract_zip(zip_file, extract_dir):
            # Nettoyer le zip
            print(f"\nNettoyage du fichier ZIP...")
            os.remove(zip_file)
            print(f"✓ Fichier ZIP supprimé")
            
            # Vérifier les fichiers
            print(f"\nFichiers extraits:")
            for file in Path(extract_dir).rglob('*'):
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    print(f"  ✓ {file.name} ({size_kb:.2f} KB)")
            
            print("\n" + "="*60)
            print("✓✓✓ TÉLÉCHARGEMENT FB15K-237 TERMINÉ ✓✓✓")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Téléchargement interrompu")
        print("↻ Relancez ce script pour reprendre")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        print("↻ Relancez le script pour réessayer")

if __name__ == "__main__":
    main()
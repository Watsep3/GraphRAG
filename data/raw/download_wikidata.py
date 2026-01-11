import sys
import os

# Ajouter le chemin src au PATH
sys.path.append('C:/Projects/GraphRAG/src')
from downloader import ResumableDownloader

def main():
    downloader = ResumableDownloader(max_retries=10)
    
    print("="*60)
    print("TÉLÉCHARGEMENT WIKIDATA")
    print("="*60)
    
    # Choisir la version
    print("\nVersions disponibles:")
    print("1. Truthy (version simplifiée) - ~5 GB - RECOMMANDÉ")
    print("2. Full (version complète) - ~100 GB")
    
    choice = input("\nVotre choix (1 ou 2): ").strip()
    
    if choice == "1":
        url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2"
        output = "wikidata-truthy.nt.bz2"
        print("\n✓ Version Truthy sélectionnée")
    elif choice == "2":
        url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
        output = "wikidata-full.json.bz2"
        print("\n✓ Version Full sélectionnée")
        print("⚠ ATTENTION: Cela prendra plusieurs heures!")
    else:
        print("✗ Choix invalide")
        return
    
    # Info sur l'espace disque
    if choice == "2":
        confirm = input("\nConfirmez-vous avoir >150 GB d'espace libre? (oui/non): ")
        if confirm.lower() not in ['oui', 'yes', 'o', 'y']:
            print("Téléchargement annulé")
            return
    
    print(f"\nDébut du téléchargement...")
    print(f"URL: {url}")
    print(f"Destination: {output}")
    print("\n💡 Astuce: Vous pouvez interrompre (Ctrl+C) et relancer - la progression sera sauvegardée!\n")
    
    try:
        downloader.download_file(url, output)
        print("\n" + "="*60)
        print("✓✓✓ TÉLÉCHARGEMENT WIKIDATA TERMINÉ ✓✓✓")
        print("="*60)
        
        # Vérifier la taille
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"\nTaille du fichier: {size_mb:.2f} MB")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Téléchargement mis en pause")
        print("↻ Relancez ce script pour reprendre là où vous vous êtes arrêté")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        print("↻ Relancez le script pour réessayer")

if __name__ == "__main__":
    main()
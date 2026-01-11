import requests
import os
from tqdm import tqdm
import time
import hashlib
from pathlib import Path

class ResumableDownloader:
    """Téléchargeur avec reprise automatique"""
    
    def __init__(self, max_retries: int = 5, chunk_size: int = 8192):
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_file_size(self, url: str) -> int:
        """Obtient la taille du fichier distant"""
        try:
            response = self.session.head(url, allow_redirects=True, timeout=10)
            return int(response.headers.get('content-length', 0))
        except:
            return 0
    
    def download_file(self, url: str, output_path: str, verify_size: bool = True):
        """
        Télécharge un fichier avec reprise automatique
        
        Args:
            url: URL du fichier
            output_path: Chemin de sauvegarde
            verify_size: Vérifier si le fichier existe déjà et est complet
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Obtenir la taille totale
        total_size = self.get_file_size(url)
        
        # Vérifier si le fichier existe déjà
        if output_path.exists():
            existing_size = output_path.stat().st_size
            
            if verify_size and total_size > 0:
                if existing_size == total_size:
                    print(f"✓ Fichier déjà téléchargé: {output_path.name}")
                    return str(output_path)
                elif existing_size > total_size:
                    print(f"⚠ Fichier corrompu, suppression...")
                    output_path.unlink()
                    existing_size = 0
            else:
                existing_size = existing_size
        else:
            existing_size = 0
        
        # Headers pour reprise
        headers = {}
        mode = 'wb'
        if existing_size > 0:
            headers['Range'] = f'bytes={existing_size}-'
            mode = 'ab'
            print(f"↻ Reprise du téléchargement à {existing_size:,} bytes")
        
        # Téléchargement avec retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, 
                    headers=headers, 
                    stream=True, 
                    timeout=30
                )
                response.raise_for_status()
                
                # Obtenir la taille à télécharger
                if 'content-length' in response.headers:
                    remaining_size = int(response.headers['content-length'])
                else:
                    remaining_size = total_size - existing_size if total_size > 0 else 0
                
                # Barre de progression
                progress_bar = tqdm(
                    total=total_size if total_size > 0 else remaining_size,
                    initial=existing_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=output_path.name
                )
                
                # Télécharger par chunks
                with open(output_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                
                progress_bar.close()
                
                # Vérifier taille finale
                final_size = output_path.stat().st_size
                if total_size > 0 and final_size != total_size:
                    print(f"⚠ Taille incorrecte: {final_size} vs {total_size}")
                    if attempt < self.max_retries - 1:
                        print(f"↻ Nouvelle tentative {attempt + 2}/{self.max_retries}...")
                        time.sleep(2)
                        continue
                
                print(f"✓ Téléchargement terminé: {output_path.name}")
                return str(output_path)
                
            except requests.exceptions.RequestException as e:
                print(f"✗ Erreur: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Backoff exponentiel
                    print(f"↻ Nouvelle tentative dans {wait_time}s... ({attempt + 2}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Échec après {self.max_retries} tentatives")
                    raise
            
            except KeyboardInterrupt:
                print("\n⚠ Téléchargement interrompu par l'utilisateur")
                print(f"↻ Progression sauvegardée: {output_path.stat().st_size:,} bytes")
                print("↻ Relancez le script pour reprendre")
                raise
    
    def download_multiple(self, downloads: list):
        """
        Télécharge plusieurs fichiers
        
        Args:
            downloads: Liste de tuples (url, output_path)
        """
        results = []
        for i, (url, output_path) in enumerate(downloads, 1):
            print(f"\n[{i}/{len(downloads)}] Téléchargement: {Path(output_path).name}")
            try:
                result = self.download_file(url, output_path)
                results.append((url, result, True))
            except Exception as e:
                print(f"✗ Échec: {e}")
                results.append((url, output_path, False))
        
        return results

# Fonction helper
def download(url: str, output_path: str):
    """Fonction simple pour télécharger un fichier"""
    downloader = ResumableDownloader()
    return downloader.download_file(url, output_path)
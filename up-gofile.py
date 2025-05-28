#!/usr/bin/env python3
# coding: utf-8

import os
import sys
from datetime import datetime
from pathlib import Path
from gofilepy import gofile_upload
from dotenv import load_dotenv

# load from .env file in current directory
load_dotenv()

def upload_daily_video(gofile_token, folder_id=None, output_dir="./output"):
    """
    Upload la vidéo du jour actuel vers Gofile.
    
    Args:
        gofile_token (str): Token d'authentification Gofile
        folder_id (str, optional): ID du dossier Gofile de destination. 
                                   Si None, Gofile créera automatiquement un nouveau dossier
        output_dir (str): Chemin vers le dossier contenant les vidéos
    
    Returns:
        list: URLs de téléchargement ou None en cas d'erreur
    """
    # Génération du nom de fichier pour aujourd'hui
    today = datetime.now().strftime("%d-%m-%Y")
    video_filename = f"cpva-{today}-no-ads.mkv"
    
    # Chemin complet vers le fichier
    output_path = Path(output_dir)
    video_path = output_path / video_filename
    
    # Vérification de l'existence du dossier
    if not output_path.exists():
        print(f"Erreur: Le dossier '{output_dir}' n'existe pas!")
        return None
    
    if not video_path.exists():
        print(f"Erreur: La vidéo '{video_filename}' n'existe pas dans '{output_dir}'!")
        print(f"Chemin recherché: {video_path.absolute()}")
        return None
    
    if not gofile_token:
        print("Erreur: Token Gofile manquant!")
        return None
    
    print(f"Upload de la vidéo: {video_filename}")
    if folder_id:
        print(f"Vers le dossier Gofile ID: {folder_id}")
    else:
        print("Gofile créera automatiquement un nouveau dossier")
    
    try:
        urls = gofile_upload(
            path=[str(video_path)],
            to_single_folder=True,
            verbose=False,
            export=False,
            open_urls=False,
            existing_folder_id=folder_id
        )
        
        if urls:
            print(f"✅ Upload réussi!")
            print(f"URL de téléchargement: {urls[0]}")
            return urls
        else:
            print("❌ Échec de l'upload")
            return None
            
    except Exception as e:
        print(f"❌ Erreur lors de l'upload: {e}")
        return None

def main():
    gofile_token = os.getenv("GOFILE_TOKEN")
    folder_id = os.getenv("FOLDER_ID")  # Peut être None
    output_dir = os.getenv("OUTPUT_DIR", "./output")  # Valeur par défaut si non définie

    if not gofile_token:
        print("Erreur: La variable GOFILE_TOKEN n'est pas définie dans le fichier .env!")
        sys.exit(1)
    
    urls = upload_daily_video(gofile_token=gofile_token, folder_id=folder_id, output_dir=output_dir)
    if urls:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()


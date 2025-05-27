#!/usr/bin/env python3
# detect-and-delete.py

import argparse
import os
import re
import unicodedata

try:
    from detect_ads import run_analysis
    from delete_ads import main as delete_ads_main
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que detect_ads.py et delete_ads.py sont dans le même dossier que ce script,")
    exit(1)

def slugify(value):
    """
    Normalise la chaîne, convertit en minuscules, supprime les caractères non alphanumériques
    et convertit les espaces et les tirets répétés en un seul tiret.
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    value = re.sub(r'^-+|-+$', '', value) # Supprime les tirets en début/fin
    if not value: # Si la chaîne devient vide après slugification
        return "video" # Retourne un nom par défaut
    return value

def main():
    parser = argparse.ArgumentParser(
        description="Détecte les publicités dans une vidéo puis les supprime, en créant une nouvelle vidéo."
    )
    parser.add_argument(
        "video_path",
        help="Chemin vers le fichier vidéo d'entrée."
    )
    parser.add_argument(
        "logo_dataset", # Retiré default="logo_dataset" car c'est un argument positionnel
        help="Chemin vers le dossier logo_dataset pour detect-ads.py."
    )
    parser.add_argument(
        "--output_dir",
        help="Dossier optionnel pour sauvegarder la vidéo traitée. Par défaut, même dossier que la vidéo d'entrée.",
        default=None
    )
    parser.add_argument(
        "--delete_original",
        action="store_true",
        help="Si spécifié, supprime la vidéo d'entrée originale après le traitement."
    )
    parser.add_argument(
        "--logo_coords",
        nargs=4,
        type=int,
        metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
        default=[120, 905, 163, 103],
        help="Argument 'logo_coords' pour detect-ads.py (défaut: [120, 905, 163, 103] pour CPVA)."
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=60,
        help="Argument 'min_duration' pour detect-ads.py (défaut: 60 secondes)."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Erreur: Le fichier vidéo d'entrée '{args.video_path}' n'a pas été trouvé.")
        return 1 # Retourner un code d'erreur

    abs_video_path = os.path.abspath(args.video_path)
    abs_logo_dataset = os.path.abspath(args.logo_dataset)

    print(f"🎬 Lancement de la détection des publicités pour : {abs_video_path}")
    print(f"🗂️ Utilisation du jeu de données de logos : {abs_logo_dataset}")

    try:
        # Étape 1: Détection des publicités
        # run_analysis retourne hms_results, csv_lines
        hms_results, _ = run_analysis(
            video_path=abs_video_path,
            logo_dataset=abs_logo_dataset,
            logo_coords=args.logo_coords,
            min_duration=args.min_duration
        )
        
        # run_analysis retourne (None, None) en cas d'erreur interne
        if hms_results is None:
            print("❌ Échec de la détection des publicités (run_analysis a retourné None).")
            return 1 # Retourner un code d'erreur
        
        if not hms_results: # Si la liste est vide
            print("✅ Aucune publicité détectée - aucune action nécessaire.")
            return 0 # Succès, mais rien à faire

        print(f"📊 Plages publicitaires détectées (hms_results) : {hms_results}")

    except Exception as e:
        print(f"❌ Erreur critique lors de l'exécution de detect-ads.run_analysis : {e}")
        # Afficher la trace de l'erreur pour plus de détails si nécessaire
        import traceback
        traceback.print_exc()
        return 1 # Retourner un code d'erreur

    video_input_basename = os.path.basename(abs_video_path)
    video_name_part, _ = os.path.splitext(video_input_basename)
    
    slugified_name = slugify(video_name_part)
    output_filename = f"{slugified_name}-no-ads.mkv"

    if args.output_dir:
        output_directory = os.path.abspath(args.output_dir)
        os.makedirs(output_directory, exist_ok=True)
    else:
        output_directory = os.path.dirname(abs_video_path)
    
    output_video_path = os.path.join(output_directory, output_filename)

    print(f"🎞️ Lancement de la suppression des publicités.")
    print(f"    Vidéo d'entrée : {abs_video_path}")
    print(f"    Vidéo de sortie : {output_video_path}")
    print(f"    Plages à supprimer : {hms_results}") # Corrigé ici

    try:
        delete_ads_main(
            input_file=abs_video_path,
            output_file=output_video_path,
            original_remove_ranges=hms_results # Corrigé ici
        )
        print(f"✅ Vidéo traitée et sauvegardée sous : {output_video_path}")

        if args.delete_original:
            print(f"🗑️ Suppression de la vidéo originale : {abs_video_path}")
            try:
                os.remove(abs_video_path)
                print(f"🗑️ Vidéo originale '{abs_video_path}' supprimée avec succès.")
            except OSError as e:
                print(f"❌ Erreur lors de la suppression de la vidéo originale '{abs_video_path}': {e}")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution de delete-ads.main : {e}")
        import traceback
        traceback.print_exc()
        if args.delete_original:
            print("⚠️ La suppression de la vidéo originale a été annulée en raison de l'erreur précédente.")
        return 1 # Retourner un code d'erreur

    if os.path.exists(output_video_path):
        input_size = os.path.getsize(abs_video_path) if os.path.exists(abs_video_path) else 0 # Vérifier si l'original existe encore
        output_size = os.path.getsize(output_video_path)
        
        print(f"\n🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS!")
        if input_size > 0: # Afficher infos de taille seulement si l'original est accessible
            size_reduction = ((input_size - output_size) / input_size) * 100 if input_size > 0 else 0
            print(f"📥 Fichier original: {abs_video_path} ({input_size / (1024*1024):.1f} MB)")
            print(f"💾 Réduction de taille: {size_reduction:.1f}%")
        print(f"📤 Fichier sans pubs: {output_video_path} ({output_size / (1024*1024):.1f} MB)")
        print(f"🎯 Segments supprimés: {len(hms_results)}") # Corrigé ici
    else:
        print(f"❓ Le fichier de sortie attendu '{output_video_path}' n'a pas été trouvé après le traitement.")
        return 1 # Retourner un code d'erreur
    
    return 0 # Succès global

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

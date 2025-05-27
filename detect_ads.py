#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from src.logo_detector import LogoDetector # Assuming src is in PYTHONPATH or main.py is one level up
from src.utils import seconds_to_hms       # Assuming this function is in utils.py

def run_analysis(video_path=None, logo_dataset=None, logo_coords=None, min_duration=60):
    """
    Version modifiée qui peut être appelée directement avec des paramètres
    ou depuis la ligne de commande.
    """
    ##parser = argparse.ArgumentParser(description='Détecteur de publicités basé sur l\'absence de logo')
    ##parser.add_argument('video_path', help='Chemin vers la vidéo à analyser')
    ##parser.add_argument('logo_dataset', help='Chemin vers le dossier contenant les logos PNG')
    ##parser.add_argument('--logo-coords', nargs=4, type=int, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
    ##                    default=[120, 905, 163, 103], help='Coordonnées du logo (x y largeur hauteur)')
    ##parser.add_argument('--min-duration', type=int, default=60,
    ##                    help='Durée minimum en secondes pour un segment publicitaire (défaut: 60)')

    # Si les paramètres sont fournis directement, les utiliser
    if video_path is not None:
        args_dict = {
            'video_path': video_path,
            'logo_dataset': logo_dataset,
            'logo_coords': logo_coords or [120, 905, 163, 103],
            'min_duration': min_duration
        }
        # Créer un objet args simulé
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        args = Args(**args_dict)
    else:
        # Mode ligne de commande original
        parser = argparse.ArgumentParser(description='Détecteur de publicités basé sur l\'absence de logo')
        parser.add_argument('video_path', help='Chemin vers la vidéo à analyser')
        parser.add_argument('logo_dataset', help='Chemin vers le dossier contenant les logos PNG')
        parser.add_argument('--logo-coords', nargs=4, type=int, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                            default=[120, 905, 163, 103], help='Coordonnées du logo (x y largeur hauteur)')
        parser.add_argument('--min-duration', type=int, default=60,
                            help='Durée minimum en secondes pour un segment publicitaire (défaut: 60)')
        args = parser.parse_args()

    # Setup logging (can be more sophisticated)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        detector = LogoDetector(args.logo_dataset, tuple(args.logo_coords))
        detector.min_ad_duration = args.min_duration
        logger.info(f"🎬 Analyse de la vidéo: {args.video_path}")
        logger.info(f"📁 Dataset de logos: {args.logo_dataset}")
        logger.info(f"📐 Coordonnées du logo: {args.logo_coords}")
        logger.info(f"⏱️ Durée minimum des publicités: {args.min_duration}s")
        logger.info("=" * 60)
        
        # analyze_video now returns List[Tuple[float, float]]
        ad_intervals_seconds = detector.analyze_video(args.video_path)
        csv_lines = []
        hms_results = []
        
        if ad_intervals_seconds:
            for start_sec, end_sec in ad_intervals_seconds:
                start_hms = seconds_to_hms(start_sec)
                end_hms = seconds_to_hms(end_sec)
                hms_results.append((start_hms, end_hms))
                csv_lines.append(f"{start_hms},{end_hms}")
        
        return hms_results, csv_lines
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True) # Log traceback
        return None, None # Indicate error

if __name__ == "__main__":
    hms_results, csv_lines = run_analysis()
    print("hms_results :")
    print(hms_results)
    print("-------")
    print(csv_lines)
    if hms_results is not None:
        print("\n🎯 RÉSULTATS FINAUX (HH:MM:SS):")
        print("=" * 60)
        if hms_results:
            print(f"✅ {len(hms_results)} segments publicitaires détectés:")
            for i, (start, end) in enumerate(hms_results, 1):
                print(f"  {i:2d}. {start} -> {end}")
        else:
            print(f"❌ Aucun segment publicitaire détecté (respectant la durée minimum).")
        if csv_lines:
            print("\n💾 Format CSV:")
            print("=" * 60)
            for line in csv_lines:
                print(line)
        
        exit_code = 0
    else:
        print("L'analyse a échoué. Vérifiez les logs pour plus de détails.")
        exit_code = 1
    
    exit(exit_code)


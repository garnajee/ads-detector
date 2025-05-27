#!/usr/bin/env python3

"""
Configuration centralisée pour le détecteur de publicités.
"""

from typing import Tuple


class DetectionConfig:
    """Configuration pour la détection de logo."""
    
    # Seuils de détection
    LOGO_MATCH_THRESHOLD = 0.65
    CONFIDENCE_BOOST_THRESHOLD = 0.55
    
    # Seuils pour écrans extrêmes
    BLACK_SCREEN_THRESHOLD = 15
    WHITE_SCREEN_THRESHOLD = 240
    EXTREME_SCREEN_VARIANCE_THRESHOLD = 100
    
    # Templates de logos
    MAX_LOGO_TEMPLATES = 300
    FIRST_PASS_MAX_TEMPLATES = 100
    SECOND_PASS_MAX_TEMPLATES = 200
    
    # Seuils de matching
    POSITIVE_MATCH_THRESHOLD = 0.5
    FIRST_PASS_MATCH_RATIO_THRESHOLD = 0.15
    SECOND_PASS_MATCH_RATIO_THRESHOLD = 0.12
    EARLY_STOP_THRESHOLD = 0.9
    FIRST_PASS_EARLY_STOP = 0.85


class VideoAnalysisConfig:
    """Configuration pour l'analyse vidéo."""
    
    # Intervalles d'échantillonnage pour la première passe (en secondes)
    SAMPLE_INTERVALS = {
        'very_long': 45,    # Plus de 2h
        'long': 30,         # Plus de 1h
        'medium': 20,       # Plus de 30min
        'short': 15         # Moins de 30min
    }
    
    # Seuils de durée pour catégoriser les vidéos (en secondes)
    DURATION_THRESHOLDS = {
        'very_long': 7200,  # 2h
        'long': 3600,       # 1h
        'medium': 1800      # 30min
    }
    
    # Configuration de la deuxième passe
    SECOND_PASS_BASE_DENSITY = 4
    SECOND_PASS_DENSITY_FACTORS = {
        'ultra_high': 3.0,    # priorité > 8
        'high': 2.5,          # priorité > 5
        'elevated': 2.0,      # priorité > 3
        'standard': 1.5       # priorité normale
    }
    
    # Intervalles d'échantillonnage pour la deuxième passe
    SECOND_PASS_INTERVALS = {
        'very_long': 15,    # Plus de 40min
        'long': 12,         # Plus de 30min
        'medium': 10,       # Plus de 15min
        'short': 8,         # Plus de 5min
        'very_short': 6     # Moins de 5min
    }
    
    # Seuils pour la deuxième passe
    SECOND_PASS_DURATION_THRESHOLDS = {
        'very_long': 2400,  # 40min
        'long': 1800,       # 30min
        'medium': 900,      # 15min
        'short': 300        # 5min
    }
    
    # Minimum absolu pour éviter la sur-analyse
    MIN_SAMPLE_INTERVAL = 2.0


class AdDetectionConfig:
    """Configuration pour la détection des publicités."""
    
    # Durées minimum et maximum
    MIN_AD_DURATION = 60           # 60 secondes minimum pour une pub normale
    MAX_EXPECTED_AD_INTERVAL = 1800  # 30 minutes maximum entre pubs
    MIN_AD_INTERVAL = 300          # 5 minutes minimum entre pubs
    
    # Seuils pour l'analyse des patterns
    ANOMALY_THRESHOLD_MULTIPLIER = 2.0
    
    # Fusion des intervalles proches
    MERGE_GAP_THRESHOLD = 10.0  # Fusionner si écart < 10 secondes
    
    # Définition des segments intro/outro
    INTRO_OUTRO_THRESHOLD = 300  # 5 minutes pour considérer comme intro/outro
    
    # Précision pour l'affinement des bornes
    BOUNDARY_REFINEMENT_PRECISION = 0.3  # 0.3 seconde de précision
    
    # Intervalles pour l'affinement interne
    REFINEMENT_INTERVALS = {
        'short': 60,    # Moins de 5 minutes
        'medium': 90,   # Plus de 5 minutes
        'long': 120     # Plus de 15 minutes
    }
    
    REFINEMENT_DURATION_THRESHOLDS = {
        'short': 300,   # 5 minutes
        'long': 900     # 15 minutes
    }


class LoggingConfig:
    """Configuration pour les logs."""
    
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s:%(funcName)s - %(message)s'
    LOG_LEVEL = 'INFO'
    
    # Fréquence d'affichage des progressions
    PROGRESS_REPORT_FREQUENCY = {
        'first_pass': 100,      # Tous les 100 frames
        'second_pass': 50,      # Tous les 50 points
        'template_loading': 100  # Tous les 100 templates
    }


class DefaultConfig:
    """Configuration par défaut pour l'ensemble du système."""
    
    # Coordonnées par défaut du logo (x, y, width, height)
    DEFAULT_LOGO_COORDS: Tuple[int, int, int, int] = (120, 905, 163, 103)
    
    # Extensions de fichiers supportées
    SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    SUPPORTED_LOGO_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    
    # Performance
    MAX_THREADS = 4  # Pour le traitement parallèle si implémenté
    
    # Validation
    MIN_LOGO_SIZE = (10, 10)
    MAX_LOGO_SIZE = (500, 500)


# Instance globale pour un accès facile
detection_config = DetectionConfig()
video_config = VideoAnalysisConfig()
ad_config = AdDetectionConfig()
logging_config = LoggingConfig()
default_config = DefaultConfig()


def get_sample_interval(duration: float) -> float:
    """
    Retourne l'intervalle d'échantillonnage approprié selon la durée de la vidéo.
    
    Args:
        duration: Durée de la vidéo en secondes
        
    Returns:
        Intervalle d'échantillonnage en secondes
    """
    if duration > video_config.DURATION_THRESHOLDS['very_long']:
        return video_config.SAMPLE_INTERVALS['very_long']
    elif duration > video_config.DURATION_THRESHOLDS['long']:
        return video_config.SAMPLE_INTERVALS['long']
    elif duration > video_config.DURATION_THRESHOLDS['medium']:
        return video_config.SAMPLE_INTERVALS['medium']
    else:
        return video_config.SAMPLE_INTERVALS['short']


def get_second_pass_interval(duration: float, density_factor: float = 1.0) -> float:
    """
    Retourne l'intervalle d'échantillonnage pour la deuxième passe.
    
    Args:
        duration: Durée de l'intervalle à analyser
        density_factor: Facteur de densité d'analyse
        
    Returns:
        Intervalle d'échantillonnage en secondes
    """
    if duration > video_config.SECOND_PASS_DURATION_THRESHOLDS['very_long']:
        base_interval = video_config.SECOND_PASS_INTERVALS['very_long']
    elif duration > video_config.SECOND_PASS_DURATION_THRESHOLDS['long']:
        base_interval = video_config.SECOND_PASS_INTERVALS['long']
    elif duration > video_config.SECOND_PASS_DURATION_THRESHOLDS['medium']:
        base_interval = video_config.SECOND_PASS_INTERVALS['medium']
    elif duration > video_config.SECOND_PASS_DURATION_THRESHOLDS['short']:
        base_interval = video_config.SECOND_PASS_INTERVALS['short']
    else:
        base_interval = video_config.SECOND_PASS_INTERVALS['very_short']
    
    # Appliquer le facteur de densité
    interval = base_interval / density_factor
    
    # Respecter le minimum
    return max(interval, video_config.MIN_SAMPLE_INTERVAL)


def get_density_factor(priority: float) -> float:
    """
    Retourne le facteur de densité selon la priorité de l'intervalle.
    
    Args:
        priority: Score de priorité de l'intervalle
        
    Returns:
        Facteur de densité à appliquer
    """
    if priority > 8:
        return video_config.SECOND_PASS_DENSITY_FACTORS['ultra_high']
    elif priority > 5:
        return video_config.SECOND_PASS_DENSITY_FACTORS['high']
    elif priority > 3:
        return video_config.SECOND_PASS_DENSITY_FACTORS['elevated']
    else:
        return video_config.SECOND_PASS_DENSITY_FACTORS['standard']


def validate_logo_coords(coords: Tuple[int, int, int, int]) -> bool:
    """
    Valide les coordonnées du logo.
    
    Args:
        coords: Tuple (x, y, width, height)
        
    Returns:
        True si les coordonnées sont valides
    """
    x, y, width, height = coords
    
    # Vérifier que les valeurs sont positives
    if any(val < 0 for val in coords):
        return False
    
    # Vérifier la taille minimum et maximum
    min_w, min_h = default_config.MIN_LOGO_SIZE
    max_w, max_h = default_config.MAX_LOGO_SIZE
    
    if width < min_w or height < min_h:
        return False
    
    if width > max_w or height > max_h:
        return False
    
    return True


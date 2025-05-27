#!/usr/bin/env python3

"""
Utilitaires pour le détecteur de publicités.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging


def seconds_to_hms(seconds: float) -> str:
    """
    Convertit des secondes en format HH:MM:SS.
    
    Args:
        seconds: Nombre de secondes (peut être flottant)
        
    Returns:
        Chaîne au format HH:MM:SS
    """
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def hms_to_seconds(hms: str) -> float:
    """
    Convertit un format HH:MM:SS en secondes.
    
    Args:
        hms: Chaîne au format HH:MM:SS
        
    Returns:
        Nombre de secondes
    """
    try:
        parts = hms.split(':')
        if len(parts) != 3:
            raise ValueError("Format invalide, attendu HH:MM:SS")
        
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError) as e:
        raise ValueError(f"Impossible de convertir '{hms}' en secondes: {e}")


def format_duration(seconds: float, precision: int = 1) -> str:
    """
    Formate une durée en secondes de manière lisible.
    
    Args:
        seconds: Durée en secondes
        precision: Nombre de décimales pour les minutes/heures
        
    Returns:
        Chaîne formatée (ex: "5.2min", "1.3h", "45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.{precision}f}min"
    else:
        return f"{seconds/3600:.{precision}f}h"


def validate_video_path(video_path: Union[str, Path]) -> Path:
    """
    Valide et normalise le chemin vers une vidéo.
    
    Args:
        video_path: Chemin vers la vidéo
        
    Returns:
        Path object validé
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si l'extension n'est pas supportée
    """
    path = Path(video_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Le fichier vidéo n'existe pas: {path}")
    
    if not path.is_file():
        raise ValueError(f"Le chemin ne pointe pas vers un fichier: {path}")
    
    # Import config here to avoid circular import
    from .config import default_config
    
    supported_extensions = default_config.SUPPORTED_VIDEO_EXTENSIONS
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(f"Extension non supportée: {path.suffix}. "
                        f"Extensions supportées: {', '.join(supported_extensions)}")
    
    return path


def validate_logo_dataset(dataset_path: Union[str, Path]) -> Path:
    """
    Valide le chemin vers le dataset de logos.
    
    Args:
        dataset_path: Chemin vers le dossier de logos
        
    Returns:
        Path object validé
        
    Raises:
        FileNotFoundError: Si le dossier n'existe pas
        ValueError: Si le dossier est vide ou ne contient pas de logos
    """
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Le dossier de logos n'existe pas: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Le chemin ne pointe pas vers un dossier: {path}")
    
    # Import config here to avoid circular import
    from .config import default_config
    
    supported_extensions = default_config.SUPPORTED_LOGO_EXTENSIONS
    logo_files = []
    
    for ext in supported_extensions:
        logo_files.extend(path.glob(f"*{ext}"))
        logo_files.extend(path.glob(f"*{ext.upper()}"))
    
    if not logo_files:
        raise ValueError(f"Aucun fichier de logo trouvé dans {path}. "
                        f"Extensions supportées: {', '.join(supported_extensions)}")
    
    return path


def create_csv_output(intervals: List[Tuple[str, str]], output_path: Optional[str] = None) -> List[str]:
    """
    Crée une sortie CSV à partir des intervalles de publicité.
    
    Args:
        intervals: Liste de tuples (début, fin) au format HH:MM:SS
        output_path: Chemin optionnel pour sauvegarder le CSV
        
    Returns:
        Liste des lignes CSV
    """
    csv_lines = []
    
    # En-tête optionnel
    # csv_lines.append("debut,fin")
    
    for start, end in intervals:
        csv_lines.append(f"{start},{end}")
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in csv_lines:
                    f.write(line + '\n')
            logging.info(f"Fichier CSV sauvegardé: {output_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde CSV: {e}")
    
    return csv_lines


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure le système de logging.
    
    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin optionnel vers un fichier de log
        
    Returns:
        Logger configuré
    """
    from .config import logging_config
    
    # Configuration de base
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Formateurs
    formatter = logging.Formatter(logging_config.LOG_FORMAT)
    
    # Logger principal
    logger = logging.getLogger('ad_detector')
    logger.setLevel(log_level)
    
    # Éviter les doublons si déjà configuré
    if logger.handlers:
        return logger
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier si spécifié
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Impossible de créer le fichier de log {log_file}: {e}")
    
    return logger


class Timer:
    """Utilitaire pour mesurer le temps d'exécution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Temps écoulé en secondes."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def __str__(self) -> str:
        return f"{self.name}: {format_duration(self.elapsed)}"


class ProgressReporter:
    """Utilitaire pour rapporter la progression d'une opération."""
    
    def __init__(self, total: int, name: str = "Progress", report_frequency: int = 10):
        self.total = total
        self.name = name
        self.report_frequency = report_frequency
        self.current = 0
        self.start_time = time.time()
        self.logger = logging.getLogger('ad_detector')
    
    def update(self, increment: int = 1) -> None:
        """Met à jour la progression."""
        self.current += increment
        
        if self.current % self.report_frequency == 0 or self.current >= self.total:
            self._report()
    
    def _report(self) -> None:
        """Rapporte la progression actuelle."""
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0 and elapsed > 0:
                rate = self.current / elapsed
                if rate > 0:
                    eta = (self.total - self.current) / rate
                    self.logger.info(f"{self.name}: {percentage:.1f}% "
                                   f"({self.current}/{self.total}) - "
                                   f"ETA: {format_duration(eta)}")
                else:
                    self.logger.info(f"{self.name}: {percentage:.1f}% "
                                   f"({self.current}/{self.total})")
            else:
                self.logger.info(f"{self.name}: {percentage:.1f}% "
                               f"({self.current}/{self.total})")


def calculate_file_size(file_path: Union[str, Path]) -> str:
    """
    Calcule et formate la taille d'un fichier.
    
    Args:
        file_path: Chemin vers le fichier
        
    Returns:
        Taille formatée (ex: "1.2 GB", "500 MB")
    """
    try:
        size_bytes = os.path.getsize(file_path)
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} PB"
    except (OSError, TypeError):
        return "Taille inconnue"


def create_output_filename(video_path: Union[str, Path], suffix: str = "_ads", 
                          extension: str = ".csv") -> str:
    """
    Crée un nom de fichier de sortie basé sur le nom de la vidéo.
    
    Args:
        video_path: Chemin vers la vidéo source
        suffix: Suffixe à ajouter
        extension: Extension du fichier de sortie
        
    Returns:
        Nom du fichier de sortie
    """
    path = Path(video_path)
    return f"{path.stem}{suffix}{extension}"


def merge_overlapping_intervals(intervals: List[Tuple[float, float]], 
                               tolerance: float = 0.0) -> List[Tuple[float, float]]:
    """
    Fusionne les intervalles qui se chevauchent ou sont très proches.
    
    Args:
        intervals: Liste d'intervalles (début, fin)
        tolerance: Tolérance pour considérer deux intervalles comme proches
        
    Returns:
        Liste d'intervalles fusionnés
    """
    if not intervals:
        return intervals
    
    # Trier par temps de début
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        # Vérifier si les intervalles se chevauchent ou sont proches
        if current_start <= last_end + tolerance:
            # Fusionner en prenant la fin la plus tardive
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    return merged


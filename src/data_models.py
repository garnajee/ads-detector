#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

@dataclass
class DetectionPoint:
    """Représente un point de détection avec ses métadonnées."""
    timestamp: float
    frame_number: int
    logo_present: bool
    confidence: float = 0.0
    analysis_pass: int = 1  # 1 pour première passe, 2 pour deuxième passe
    
    def __str__(self) -> str:
        return (f"DetectionPoint(t={self.timestamp:.1f}s, "
                f"frame={self.frame_number}, "
                f"logo={'✓' if self.logo_present else '✗'}, "
                f"conf={self.confidence:.3f}, "
                f"pass={self.analysis_pass})")


@dataclass
class SuspiciousInterval:
    """Représente un intervalle suspect qui nécessite une analyse plus poussée."""
    start_time: float
    end_time: float
    expected_ads: int
    detected_ads: int
    gap_duration: float
    priority: float  # Score de priorité pour l'analyse
    
    @property
    def duration_minutes(self) -> float:
        """Durée de l'intervalle en minutes."""
        return self.gap_duration / 60.0
    
    def __str__(self) -> str:
        return (f"SuspiciousInterval({self.start_time:.1f}s -> {self.end_time:.1f}s, "
                f"gap={self.duration_minutes:.1f}min, "
                f"expected={self.expected_ads}, "
                f"priority={self.priority:.1f})")


@dataclass
class AdSegment:
    """Représente un segment publicitaire détecté."""
    start_time: float
    end_time: float
    duration: float
    segment_type: str = "normal"  # "intro", "outro", "normal"
    confidence: float = 1.0
    refined: bool = False  # Indique si les bornes ont été affinées
    
    @property
    def duration_minutes(self) -> float:
        """Durée du segment en minutes."""
        return self.duration / 60.0
    
    @property
    def is_intro(self) -> bool:
        """Vérifie si c'est un segment d'intro."""
        return self.segment_type == "intro"
    
    @property
    def is_outro(self) -> bool:
        """Vérifie si c'est un segment d'outro."""
        return self.segment_type == "outro"
    
    def __str__(self) -> str:
        type_marker = f" ({self.segment_type.upper()})" if self.segment_type != "normal" else ""
        refined_marker = " [REFINED]" if self.refined else ""
        return (f"AdSegment({self.start_time:.1f}s -> {self.end_time:.1f}s, "
                f"dur={self.duration_minutes:.1f}min{type_marker}{refined_marker})")


@dataclass
class VideoInfo:
    """Informations sur la vidéo analysée."""
    path: str
    duration: float
    fps: float
    total_frames: int
    width: int = 0
    height: int = 0
    
    @property
    def duration_hours(self) -> float:
        """Durée en heures."""
        return self.duration / 3600.0
    
    @property
    def duration_minutes(self) -> float:
        """Durée en minutes."""
        return self.duration / 60.0
    
    def __str__(self) -> str:
        return (f"VideoInfo(dur={self.duration_hours:.2f}h, "
                f"fps={self.fps:.1f}, "
                f"frames={self.total_frames}, "
                f"res={self.width}x{self.height})")


@dataclass
class AnalysisStats:
    """Statistiques de l'analyse effectuée."""
    video_info: VideoInfo
    total_detections: int
    first_pass_detections: int
    second_pass_detections: int
    suspicious_intervals_count: int
    ad_segments_found: int
    total_ad_duration: float
    analysis_duration: float  # Temps d'exécution en secondes
    
    @property
    def ad_percentage(self) -> float:
        """Pourcentage de publicité dans la vidéo."""
        return (self.total_ad_duration / self.video_info.duration) * 100 if self.video_info.duration > 0 else 0
    
    def __str__(self) -> str:
        return (f"AnalysisStats(segments={self.ad_segments_found}, "
                f"ad_dur={self.total_ad_duration/60:.1f}min, "
                f"ad_pct={self.ad_percentage:.1f}%, "
                f"analysis_time={self.analysis_duration:.1f}s)")


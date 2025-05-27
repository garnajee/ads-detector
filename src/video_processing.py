#!/usr/bin/env python3

"""
Module de traitement vidéo pour l'affinement précis des bornes de segments publicitaires.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Callable
from pathlib import Path

from .data_models import VideoInfo, AdSegment, DetectionPoint
from .config import ad_config
from .utils import Timer, format_duration


class VideoProcessor:
    """Classe pour le traitement vidéo et l'affinement des bornes."""
    
    def __init__(self):
        self.logger = logging.getLogger('ad_detector.video_processing')
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Extrait les informations de base d'une vidéo.
        
        Args:
            video_path: Chemin vers la vidéo
            
        Returns:
            Informations sur la vidéo
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoInfo(
                path=video_path,
                duration=duration,
                fps=fps,
                total_frames=total_frames,
                width=width,
                height=height
            )
        finally:
            cap.release()
    
    def get_frame_at_timestamp(self, cap: cv2.VideoCapture, timestamp: float) -> Optional[np.ndarray]:
        """
        Récupère une frame à un timestamp donné avec gestion d'erreur améliorée.
        
        Args:
            cap: Capture vidéo OpenCV
            timestamp: Timestamp en secondes
            
        Returns:
            Frame ou None si échec
        """
        try:
            # Conversion timestamp vers position en millisecondes
            pos_msec = timestamp * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
            ret, frame = cap.read()
            return frame if ret else None
        except Exception as e:
            self.logger.warning(f"Erreur lors de la récupération de la frame à {timestamp}s: {e}")
            return None
    
    def get_frame_at_frame_number(self, cap: cv2.VideoCapture, frame_number: int) -> Optional[np.ndarray]:
        """
        Récupère une frame à un numéro de frame donné.
        
        Args:
            cap: Capture vidéo OpenCV
            frame_number: Numéro de la frame
            
        Returns:
            Frame ou None si échec
        """
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            return frame if ret else None
        except Exception as e:
            self.logger.warning(f"Erreur lors de la récupération de la frame #{frame_number}: {e}")
            return None
    
    def binary_search_logo_transition(self, cap: cv2.VideoCapture, start_time: float, 
                                    end_time: float, logo_present_at_start: bool,
                                    logo_detector_func: Callable[[np.ndarray], Tuple[bool, float]],
                                    precision: float = 0.5) -> float:
        """
        Recherche binaire pour trouver le moment exact de transition du logo.
        Version optimisée avec précision configurable.
        
        Args:
            cap: Capture vidéo
            start_time: Timestamp de début en secondes
            end_time: Timestamp de fin en secondes
            logo_present_at_start: État du logo au début
            logo_detector_func: Fonction de détection du logo
            precision: Précision de la recherche en secondes
            
        Returns:
            Timestamp exact de la transition
        """
        self.logger.debug(f"Recherche binaire: {start_time:.1f}s -> {end_time:.1f}s "
                         f"(logo au début: {logo_present_at_start})")
        
        search_count = 0
        max_searches = int(np.log2((end_time - start_time) / precision)) + 5
        
        while (end_time - start_time) > precision and search_count < max_searches:
            mid_time = (start_time + end_time) / 2
            frame = self.get_frame_at_timestamp(cap, mid_time)
            
            if frame is None:
                self.logger.warning(f"Frame introuvable à {mid_time:.1f}s, arrêt de la recherche")
                break
            
            logo_present, confidence = logo_detector_func(frame)
            
            self.logger.debug(f"  Test à {mid_time:.1f}s: logo={logo_present}, conf={confidence:.3f}")
            
            if logo_present == logo_present_at_start:
                start_time = mid_time
            else:
                end_time = mid_time
            
            search_count += 1
        
        result_time = end_time
        self.logger.debug(f"Transition trouvée à {result_time:.1f}s après {search_count} recherches")
        
        return result_time
    
    def refine_segment_boundaries(self, cap: cv2.VideoCapture, rough_start: float, rough_end: float,
                                logo_detector_func: Callable[[np.ndarray], Tuple[bool, float]],
                                is_intro_outro: bool = False) -> Tuple[float, float]:
        """
        Affine les bornes d'un segment publicitaire avec précision accrue.
        
        Args:
            cap: Capture vidéo
            rough_start: Début approximatif du segment
            rough_end: Fin approximative du segment
            logo_detector_func: Fonction de détection du logo
            is_intro_outro: True si c'est un segment d'intro/outro
            
        Returns:
            Tuple (start_précis, end_précis)
        """
        self.logger.info(f"Affinement du segment {rough_start:.1f}s -> {rough_end:.1f}s "
                        f"{'(intro/outro)' if is_intro_outro else ''}")
        
        # Précision plus élevée pour intro/outro
        precision = 0.2 if is_intro_outro else 0.5
        
        # Étendre légèrement la zone de recherche pour capturer les vraies bornes
        search_margin = 5.0 if not is_intro_outro else 2.0
        search_start = max(0, rough_start - search_margin)
        search_end = rough_end + search_margin
        
        # Déterminer l'état du logo avant le début du segment
        pre_start_frame = self.get_frame_at_timestamp(cap, search_start)
        if pre_start_frame is not None:
            logo_before, _ = logo_detector_func(pre_start_frame)
        else:
            logo_before = True  # Assumption par défaut
        
        # Affiner le début du segment
        precise_start = self.binary_search_logo_transition(
            cap, search_start, rough_start + search_margin, 
            logo_before, logo_detector_func, precision
        )
        
        # Déterminer l'état du logo après la fin du segment
        post_end_frame = self.get_frame_at_timestamp(cap, search_end)
        if post_end_frame is not None:
            logo_after, _ = logo_detector_func(post_end_frame)
        else:
            logo_after = True  # Assumption par défaut
        
        # Affiner la fin du segment
        precise_end = self.binary_search_logo_transition(
            cap, rough_end - search_margin, search_end,
            not logo_after, logo_detector_func, precision
        )
        
        self.logger.info(f"Segment affiné: {precise_start:.1f}s -> {precise_end:.1f}s "
                        f"(ajustement début: {precise_start - rough_start:+.1f}s, "
                        f"fin: {precise_end - rough_end:+.1f}s)")
        
        return precise_start, precise_end
    
    def detect_internal_transitions(self, cap: cv2.VideoCapture, start_time: float, end_time: float,
                                  logo_detector_func: Callable[[np.ndarray], Tuple[bool, float]],
                                  check_interval: float = 30.0) -> List[Tuple[float, bool]]:
        """
        Détecte les transitions internes dans un long segment (pour détecter des coupures pub).
        
        Args:
            cap: Capture vidéo
            start_time: Début du segment
            end_time: Fin du segment
            logo_detector_func: Fonction de détection du logo
            check_interval: Intervalle de vérification en secondes
            
        Returns:
            Liste des transitions (timestamp, logo_present)
        """
        duration = end_time - start_time
        
        if duration < 300:  # Moins de 5 minutes, pas de vérification interne
            return [(start_time, False), (end_time, False)]
        
        self.logger.debug(f"Détection de transitions internes sur {duration/60:.1f} min")
        
        transitions = []
        current_time = start_time
        
        # État initial
        initial_frame = self.get_frame_at_timestamp(cap, start_time)
        if initial_frame is not None:
            initial_state, _ = logo_detector_func(initial_frame)
            transitions.append((start_time, initial_state))
            last_state = initial_state
        else:
            last_state = False
            transitions.append((start_time, False))
        
        # Parcourir le segment avec l'intervalle donné
        current_time += check_interval
        
        while current_time < end_time:
            frame = self.get_frame_at_timestamp(cap, current_time)
            if frame is not None:
                logo_present, confidence = logo_detector_func(frame)
                
                if logo_present != last_state:
                    # Transition détectée, affiner la position
                    precise_transition = self.binary_search_logo_transition(
                        cap, current_time - check_interval, current_time,
                        last_state, logo_detector_func, 0.5
                    )
                    transitions.append((precise_transition, logo_present))
                    last_state = logo_present
                    
                    self.logger.debug(f"Transition interne à {precise_transition:.1f}s: "
                                    f"logo={'présent' if logo_present else 'absent'}")
            
            current_time += check_interval
        
        # État final
        final_frame = self.get_frame_at_timestamp(cap, end_time)
        if final_frame is not None:
            final_state, _ = logo_detector_func(final_frame)
            if len(transitions) == 0 or final_state != transitions[-1][1]:
                # Affiner la transition finale si nécessaire
                if len(transitions) > 0 and final_state != transitions[-1][1]:
                    precise_end = self.binary_search_logo_transition(
                        cap, transitions[-1][0], end_time,
                        transitions[-1][1], logo_detector_func, 0.5
                    )
                    transitions.append((precise_end, final_state))
                else:
                    transitions.append((end_time, final_state))
        else:
            transitions.append((end_time, last_state))
        
        return transitions
    
    def process_complex_segment(self, cap: cv2.VideoCapture, start_time: float, end_time: float,
                              logo_detector_func: Callable[[np.ndarray], Tuple[bool, float]],
                              is_intro_outro: bool = False) -> List[AdSegment]:
        """
        Traite un segment complexe qui peut contenir plusieurs coupures publicitaires.
        
        Args:
            cap: Capture vidéo
            start_time: Début du segment
            end_time: Fin du segment
            logo_detector_func: Fonction de détection du logo
            is_intro_outro: True si c'est un segment d'intro/outro
            
        Returns:
            Liste des segments publicitaires raffinés
        """
        self.logger.info(f"Traitement du segment complexe {start_time:.1f}s -> {end_time:.1f}s")
        
        # Détecter les transitions internes
        check_interval = 20.0 if is_intro_outro else 30.0
        transitions = self.detect_internal_transitions(
            cap, start_time, end_time, logo_detector_func, check_interval
        )
        
        # Construire les segments à partir des transitions
        ad_segments = []
        
        for i in range(len(transitions) - 1):
            curr_time, curr_state = transitions[i]
            next_time, next_state = transitions[i + 1]
            
            # Si le logo est absent, c'est un segment publicitaire
            if not curr_state:
                # Affiner les bornes de ce segment
                refined_start, refined_end = self.refine_segment_boundaries(
                    cap, curr_time, next_time, logo_detector_func, is_intro_outro
                )
                
                duration = refined_end - refined_start
                
                # Vérifier la durée minimum (sauf pour intro/outro)
                if is_intro_outro or duration >= ad_config.MIN_AD_DURATION:
                    segment = AdSegment(
                        start_time=refined_start,
                        end_time=refined_end,
                        confidence=0.8,  # Confiance élevée après affinement
                        is_intro=is_intro_outro and refined_start < 300,
                        is_outro=is_intro_outro and refined_end > (end_time - 300)
                    )
                    ad_segments.append(segment)
                    
                    self.logger.info(f"Segment publicitaire créé: {segment}")
                else:
                    self.logger.info(f"Segment trop court rejeté: {duration:.1f}s < {ad_config.MIN_AD_DURATION}s")
        
        return ad_segments
    
    def optimize_second_pass_sampling(self, suspicious_intervals, total_video_duration: float) -> List[DetectionPoint]:
        """
        Optimise l'échantillonnage pour la deuxième passe en se concentrant sur les zones suspectes.
        
        Args:
            suspicious_intervals: Intervalles suspects à analyser
            total_video_duration: Durée totale de la vidéo
            
        Returns:
            Liste des points de détection optimisés
        """
        optimized_points = []
        
        for interval in suspicious_intervals:
            duration = interval.gap_duration
            
            # Calcul de la densité d'échantillonnage basée sur la priorité
            base_interval = 15.0  # Intervalle de base en secondes
            
            if interval.priority > 8:
                sample_interval = base_interval / 4  # Très haute densité
            elif interval.priority > 5:
                sample_interval = base_interval / 3  # Haute densité
            elif interval.priority > 3:
                sample_interval = base_interval / 2  # Densité moyenne
            else:
                sample_interval = base_interval / 1.5  # Densité normale
            
            # Générer les points d'échantillonnage
            current_time = interval.start_time
            while current_time < interval.end_time:
                point = DetectionPoint(
                    timestamp=current_time,
                    frame_number=0,  # Sera calculé lors de l'analyse
                    logo_present=False,  # Sera déterminé lors de l'analyse
                    confidence=0.0,
                    analysis_pass=2
                )
                optimized_points.append(point)
                current_time += sample_interval
        
        self.logger.info(f"Points d'échantillonnage optimisés pour la 2e passe: {len(optimized_points)}")
        
        return optimized_points
    
    def extract_video_segment(self, cap: cv2.VideoCapture, start_time: float, end_time: float,
                            output_path: Optional[str] = None) -> bool:
        """
        Extrait un segment vidéo (utile pour le debug ou l'export).
        
        Args:
            cap: Capture vidéo
            start_time: Début en secondes
            end_time: Fin en secondes
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            True si succès
        """
        if output_path is None:
            return False
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Codec pour l'écriture
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            current_time = start_time
            frame_interval = 1.0 / fps
            
            while current_time <= end_time:
                frame = self.get_frame_at_timestamp(cap, current_time)
                if frame is not None:
                    out.write(frame)
                current_time += frame_interval
            
            out.release()
            self.logger.info(f"Segment extrait: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction: {e}")
            return False
    
    def analyze_segment_quality(self, cap: cv2.VideoCapture, start_time: float, end_time: float,
                              logo_detector_func: Callable[[np.ndarray], Tuple[bool, float]]) -> dict:
        """
        Analyse la qualité d'un segment détecté pour validation.
        
        Args:
            cap: Capture vidéo
            start_time: Début du segment
            end_time: Fin du segment
            logo_detector_func: Fonction de détection du logo
            
        Returns:
            Dictionnaire avec les métriques de qualité
        """
        duration = end_time - start_time
        sample_count = min(20, int(duration / 10))  # Maximum 20 échantillons
        
        if sample_count < 3:
            sample_count = 3
        
        sample_interval = duration / sample_count
        logo_absent_count = 0
        confidence_scores = []
        
        for i in range(sample_count):
            sample_time = start_time + (i * sample_interval)
            frame = self.get_frame_at_timestamp(cap, sample_time)
            
            if frame is not None:
                logo_present, confidence = logo_detector_func(frame)
                confidence_scores.append(confidence)
                
                if not logo_present:
                    logo_absent_count += 1
        
        absence_ratio = logo_absent_count / sample_count if sample_count > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        quality_metrics = {
            'duration': duration,
            'samples_analyzed': sample_count,
            'logo_absence_ratio': absence_ratio,
            'average_confidence': avg_confidence,
            'quality_score': absence_ratio * avg_confidence,
            'is_reliable': absence_ratio > 0.7 and avg_confidence > 0.3
        }
        
        self.logger.debug(f"Analyse qualité segment {start_time:.1f}s: "
                         f"absence={absence_ratio:.2f}, conf={avg_confidence:.3f}")
        
        return quality_metrics


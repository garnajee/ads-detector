#!/usr/bin/env python3

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
import logging
import statistics

# Assuming data_models.py and utils.py are in the same 'src' package
from .data_models import DetectionPoint, SuspiciousInterval
from .utils import seconds_to_hms

class LogoDetector:
    """
    Détecteur de logo optimisé pour analyser des vidéos longues.
    Version v6 avec deuxième passe performante, validation de durée minimum,
    et affinement précis des transitions.
    """

    def __init__(self, logo_dataset_path: str, logo_coords: Tuple[int, int, int, int]):
        """
        Initialise le détecteur de logo.

        Args:
            logo_dataset_path: Chemin vers le dossier contenant les logos PNG
            logo_coords: Tuple (x, y, width, height) de la position du logo dans la vidéo
        """
        self.logo_dataset_path = Path(logo_dataset_path)
        self.logo_x, self.logo_y, self.logo_width, self.logo_height = logo_coords
        self.logo_templates = []
        self.logo_masks = []
        self.threshold = 0.65

        # Paramètres pour la gestion des cas limites
        self.merge_gap_threshold = 10.0  # Temps en secondes pour fusionner des segments proches
        self.black_screen_threshold = 15 # Moyenne d'intensité pour écran noir
        self.white_screen_threshold = 240 # Moyenne d'intensité pour écran blanc

        # Paramètres pour l'analyse intelligente
        self.detection_history: List[DetectionPoint] = []
        self.ad_intervals_stats: Dict[str, float] = {}
        self.anomaly_threshold_multiplier = 2.0  # Seuil d'anomalie (× écart-type)
        self.second_pass_density_multiplier = 4
        self.min_ad_interval = 300  # Intervalle minimum attendu entre pubs (5 min) - pour stats
        self.max_expected_ad_interval = 1800  # Intervalle maximum normal (30 min) - pour stats

        # Durée minimum pour un segment publicitaire
        self.min_ad_duration = 60  # 60 secondes minimum par défaut

        # Optimisations pour la deuxième passe
        self.adaptive_threshold_enabled = True
        self.confidence_boost_threshold = 0.55

        # Configuration de logging
        # Le logger est généralement configuré dans main.py ou au niveau de l'application.
        # Ici, on s'assure d'utiliser un logger nommé.
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO) # Le niveau est mieux géré par la config globale.

        self.video_duration = 0.0 # Sera défini lors de l'analyse

        # using cached templates
        self.cache_path = Path("cache_templates.npz")
        if self.cache_path.exists():
            self._load_from_cache()
        else:
            self._load_logo_templates()
            self._save_to_cache()

    def _load_logo_templates(self):
        """Charge tous les templates de logos depuis le dataset."""
        self.logger.info(f"Chargement des templates de logos depuis {self.logo_dataset_path}")
        logo_files = list(self.logo_dataset_path.glob("*.png"))

        if not logo_files:
            self.logger.error(f"Aucun fichier PNG trouvé dans {self.logo_dataset_path}")
            raise ValueError(f"Aucun fichier PNG trouvé dans {self.logo_dataset_path}")

        max_templates = min(300, len(logo_files)) # Limiter le nombre de templates chargés
        for i, logo_file in enumerate(logo_files[:max_templates]):
            try:
                template = cv2.imread(str(logo_file), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    if template.shape != (self.logo_height, self.logo_width):
                        template = cv2.resize(template, (self.logo_width, self.logo_height))
                    self.logo_templates.append(template)
                    mask = np.where(template > 20, 255, 0).astype(np.uint8)
                    self.logo_masks.append(mask)
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement de {logo_file}: {e}")
                continue
            if (i + 1) % 100 == 0 or (i + 1) == max_templates :
                 self.logger.info(f"Chargé {i+1}/{max_templates} templates")


        self.logger.info(f"Templates chargés: {len(self.logo_templates)}")
        if not self.logo_templates:
            self.logger.error("Aucun template de logo valide n'a pu être chargé.")
            raise ValueError("Aucun template de logo valide n'a pu être chargé")

    def _save_to_cache(self):
        # Sauvegarde tous les templates et masques en .npz
        np.savez_compressed(
            self.cache_path,
            *self.logo_templates,
            # on sépare templates et masks en deux groupes de variables arr_0…arr_N
        )
        np.savez_compressed(
            Path("cache_masks.npz"),
            *self.logo_masks
        )

    def _load_from_cache(self):
        # Charge le fichier .npz
        data_t = np.load(self.cache_path)
        self.logo_templates = [data_t[f] for f in data_t.files]
        data_m = np.load("cache_masks.npz")
        self.logo_masks = [data_m[f] for f in data_m.files]

    def _extract_logo_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrait la région du logo depuis une frame."""
        if frame is None:
            return None
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        # Vérifier si les coordonnées sont valides pour la taille de l'image
        frame_h, frame_w = frame_gray.shape[:2]
        if self.logo_y + self.logo_height > frame_h or \
           self.logo_x + self.logo_width > frame_w:
            self.logger.warning(f"Les coordonnées du logo ({self.logo_x}, {self.logo_y}, {self.logo_width}, {self.logo_height}) sont en dehors des dimensions de la frame ({frame_w}x{frame_h}).")
            return None # Ou gérer autrement, e.g., en ajustant

        logo_region = frame_gray[
            self.logo_y:self.logo_y + self.logo_height,
            self.logo_x:self.logo_x + self.logo_width
        ]
        return logo_region

    def _is_extreme_screen(self, frame: np.ndarray) -> str:
        """Détecte si l'écran est complètement noir, blanc ou normal."""
        if frame is None: return 'normal'
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        mean_intensity = np.mean(frame_gray)
        variance = np.var(frame_gray)

        if mean_intensity < self.black_screen_threshold and variance < 100: # Variance faible pour écrans unis
            return 'black'
        elif mean_intensity > self.white_screen_threshold and variance < 100:
            return 'white'
        else:
            return 'normal'

    def _is_logo_present(self, logo_region: Optional[np.ndarray], full_frame: Optional[np.ndarray] = None,
                        analysis_pass: int = 1) -> Tuple[bool, float]:
        """Détermine si le logo est présent dans la région extraite."""
        if logo_region is None or logo_region.size == 0:
            return False, 0.0

        if full_frame is not None:
            screen_type = self._is_extreme_screen(full_frame)
            if screen_type in ['black', 'white']:
                return False, 0.0
        
        if np.mean(logo_region) < 5: # Région du logo trop sombre
            return False, 0.0

        best_match = 0.0
        positive_matches = 0
        confidence_scores = []
        
        max_templates_to_test = len(self.logo_templates)
        if analysis_pass == 2:
            max_templates_to_test = min(len(self.logo_templates), 200)
        else:
            max_templates_to_test = min(len(self.logo_templates), 100)

        for i, (template, mask) in enumerate(zip(self.logo_templates[:max_templates_to_test],
                                               self.logo_masks[:max_templates_to_test])):
            try:
                result = cv2.matchTemplate(logo_region, template, cv2.TM_CCOEFF_NORMED, mask=mask)
                _, max_val, _, _ = cv2.minMaxLoc(result) # Obtenir la valeur max de similarité
                confidence_scores.append(max_val)
                if max_val > best_match:
                    best_match = max_val
                if max_val > 0.5: # Seuil arbitraire pour compter un "match positif"
                    positive_matches +=1
            except cv2.error as e:
                # self.logger.warning(f"Erreur cv2.matchTemplate (template {i}): {e}. Région: {logo_region.shape}, Template: {template.shape}")
                # Cela peut arriver si la région du logo est plus petite que le template à cause d'un redimensionnement ou crop.
                # S'assurer que logo_region a au moins la taille de template.
                # Pour l'instant, on ignore ce template pour ce frame.
                continue


            if analysis_pass == 1 and best_match > 0.85 : break # Arrêt anticipé en passe 1
            if best_match > 0.9 : break # Arrêt anticipé si très bon match


        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        confidence = max(best_match, avg_confidence * 1.2) # Score de confiance composite

        if analysis_pass == 2 and self.adaptive_threshold_enabled:
            if confidence > self.confidence_boost_threshold:
                confidence = min(1.0, confidence * 1.1)

        effective_threshold = self.threshold
        if analysis_pass == 2:
            effective_threshold *= 0.95 # Un peu plus permissif en 2e passe

        if best_match > effective_threshold:
            return True, confidence
        
        # Logique de décision alternative basée sur le ratio de matchs positifs
        match_ratio = positive_matches / len(self.logo_templates[:max_templates_to_test]) if max_templates_to_test > 0 else 0
        
        # Critères plus souples pour la deuxième passe
        if analysis_pass == 2:
            if match_ratio > 0.12 and best_match > 0.52: # Ajuster ces seuils si nécessaire
                return True, confidence
        else: # Première passe
            if match_ratio > 0.15 and best_match > 0.55: # Ajuster ces seuils si nécessaire
                return True, confidence

        return False, confidence

    def _analyze_detection_patterns(self) -> Dict[str, float]:
        """Analyse les patterns de détection pour identifier les anomalies."""
        self.logger.info("Analyse des patterns de détection...")
        ad_timestamps = [dp.timestamp for dp in self.detection_history if not dp.logo_present and dp.analysis_pass == 1]

        if len(ad_timestamps) < 2:
            self.logger.warning("Pas assez de publicités détectées (passe 1) pour l'analyse statistique des intervalles.")
            # Fournir des valeurs par défaut pour éviter les erreurs si stats est utilisé plus tard
            return {
                'mean_interval': self.min_ad_interval * 2, # Estimation grossière
                'std_interval': self.min_ad_interval,
                'median_interval': self.min_ad_interval * 2,
                'min_interval': self.min_ad_interval,
                'max_interval': self.max_expected_ad_interval,
                'total_ads': len(ad_timestamps),
                'intervals_count': 0
            }

        intervals = np.diff(sorted(ad_timestamps))
        if len(intervals) == 0: # Moins de 2 pubs
            self.logger.warning("Pas assez d'intervalles entre publicités pour statistiques.")
            return {
                'mean_interval': self.min_ad_interval * 2,
                'std_interval': self.min_ad_interval,
                'median_interval': self.min_ad_interval * 2,
                'min_interval': min(intervals) if len(intervals)>0 else self.min_ad_interval,
                'max_interval': max(intervals) if len(intervals)>0 else self.max_expected_ad_interval,
                'total_ads': len(ad_timestamps),
                'intervals_count': len(intervals)
            }

        stats = {
            'mean_interval': statistics.mean(intervals) if intervals.size > 0 else self.min_ad_interval * 2,
            'std_interval': statistics.stdev(intervals) if len(intervals) > 1 else self.min_ad_interval,
            'median_interval': statistics.median(intervals) if intervals.size > 0 else self.min_ad_interval * 2,
            'min_interval': min(intervals) if intervals.size > 0 else self.min_ad_interval,
            'max_interval': max(intervals) if intervals.size > 0 else self.max_expected_ad_interval,
            'total_ads': len(ad_timestamps),
            'intervals_count': len(intervals)
        }
        self.logger.info(f"Statistiques des intervalles publicitaires (basées sur passe 1):")
        self.logger.info(f"  - Moyenne: {stats['mean_interval']/60:.1f} min, Médiane: {stats['median_interval']/60:.1f} min")
        self.logger.info(f"  - Écart-type: {stats['std_interval']/60:.1f} min, Min-Max: {stats['min_interval']/60:.1f} - {stats['max_interval']/60:.1f} min")
        return stats

    def _identify_suspicious_intervals(self, stats: Dict[str, float]) -> List[SuspiciousInterval]:
        """Identifie les intervalles suspects où des publicités ont pu être manquées."""
        self.logger.info("Identification des intervalles suspects...")
        suspicious_intervals = []
        
        # Utiliser la médiane comme base plus robuste que la moyenne si possible, sinon la moyenne.
        # Si la std est très grande, la moyenne peut être trompeuse.
        base_interval_for_expectation = stats.get('median_interval', stats.get('mean_interval', self.min_ad_interval * 1.5))
        if base_interval_for_expectation <=0: # Sanity check
             base_interval_for_expectation = self.min_ad_interval * 1.5


        anomaly_threshold = base_interval_for_expectation + (stats.get('std_interval', self.min_ad_interval) * self.anomaly_threshold_multiplier)
        anomaly_threshold = max(anomaly_threshold, self.max_expected_ad_interval, base_interval_for_expectation * 1.5) # Assurer un seuil minimum raisonnable

        self.logger.info(f"Seuil d'anomalie pour les gaps: > {anomaly_threshold/60:.1f} min (basé sur intervalle de {base_interval_for_expectation/60:.1f} min)")

        # Considérer tous les points de détection (logo présent ou non) pour définir les "gaps" de non-publicité
        # Les "ads" sont où logo_present == False.
        # Un "gap suspect" est un long segment où logo_present == True entre deux pubs,
        # ou avant la première pub / après la dernière pub.

        ad_detections = sorted([dp for dp in self.detection_history if not dp.logo_present and dp.analysis_pass == 1], key=lambda x: x.timestamp)

        # Points de référence incluant le début et la fin de la vidéo
        reference_points = [0.0] + [dp.timestamp for dp in ad_detections] + [self.video_duration]
        reference_points = sorted(list(set(reference_points))) # Uniques et triés

        for i in range(len(reference_points) - 1):
            start_gap = reference_points[i]
            end_gap = reference_points[i+1]
            gap_duration = end_gap - start_gap

            # Est-ce un gap de "programme" (entre deux pubs, ou avant/après pub)?
            # Si start_gap est 0 (début video) et end_gap est la 1ere pub, c'est un gap de programme potentiel.
            # Si start_gap est une pub et end_gap est la pub suivante, c'est un gap de programme.
            # Si start_gap est la dernière pub et end_gap est la fin video, c'est un gap de programme.
            
            # On s'intéresse aux longs segments SANS pub.
            # Ad_detections sont les moments où il y a pub.
            # Un intervalle [ad_detections[j].timestamp, ad_detections[j+1].timestamp] est un intervalle entre débuts de pub.
            # Ce n'est pas ce qu'on veut.
            # On veut les intervalles où le LOGO EST PRÉSENT pendant longtemps.

            # Repensons:
            # On a des detection_points. Certains sont logo_present=True, d'autres False.
            # Un suspicious interval est une longue période où logo_present=True, qui excède l'attente.

            # Itérons sur les transitions logo_absent -> logo_present et logo_present -> logo_absent
            # Ou plus simple: itérer sur les segments où le logo EST PRÉSENT.
            
            last_ad_end_time = 0.0
            if not ad_detections: # Pas de pub du tout, toute la vidéo est suspecte si elle est longue
                if self.video_duration > anomaly_threshold :
                     expected_ads = max(1, int(self.video_duration / base_interval_for_expectation))
                     priority = (self.video_duration / base_interval_for_expectation) * expected_ads
                     suspicious_intervals.append(SuspiciousInterval(0, self.video_duration, expected_ads, 0, self.video_duration, priority))
                     self.logger.info(f"Aucune pub détectée, toute la vidéo ({self.video_duration/60:.1f} min) est suspecte.")
                # else :
                #     self.logger.info("Aucune pub détectée, vidéo courte, pas d'analyse suspecte approfondie.")

            for idx, ad in enumerate(ad_detections):
                # Segment de programme avant la première pub
                if idx == 0 and ad.timestamp > anomaly_threshold:
                    duration_prog = ad.timestamp
                    expected_ads_missed = max(1, int(duration_prog / base_interval_for_expectation) -1) # -1 car on s'attend à une pub à la fin
                    if expected_ads_missed > 0:
                        priority = (duration_prog / base_interval_for_expectation) * expected_ads_missed
                        suspicious_intervals.append(SuspiciousInterval(0, ad.timestamp, expected_ads_missed, 0, duration_prog, priority))
                        self.logger.info(f"Intervalle suspect (début vidéo): {seconds_to_hms(0)} -> {seconds_to_hms(ad.timestamp)} (durée: {duration_prog/60:.1f}min, pubs attendues: {expected_ads_missed})")
                
                # Segment de programme entre deux pubs
                if idx < len(ad_detections) - 1:
                    current_ad_timestamp = ad.timestamp # Début de la pub actuelle
                    # Pour trouver la fin de la pub actuelle, il faudrait avoir les segments précis.
                    # Pour l'instant, on va se baser sur le début de la pub suivante.
                    # Ceci est une approximation pour identifier les gaps de programme.
                    # L'idéal serait d'avoir les *fins* des pubs de la passe 1.
                    # Pour simplifier, on considère l'intervalle entre les *débuts* de détection de non-logo.
                    # Si le temps entre le début de pub N et début de pub N+1 est très grand...
                    
                    # Alternative: utiliser tous les points de détection_history
                    # Trouver les segments où logo_present == True
                    # Si un tel segment est > anomaly_threshold, il est suspect.
                    # Cette logique est déjà plus ou moins dans la formation des intervalles finaux.

                    # Restons sur la logique originale : gaps entre pubs détectées.
                    # La "fin" d'une pub est le timestamp du point de détection où le logo est revenu.
                    # Mais ad_detections ne contient que les points où logo_present = False.

                    # On va considérer le "gap" comme le temps entre la fin implicite d'une pub et le début de la suivante.
                    # Pour la passe 1, les "pubs" sont des points.
                    # Le gap est entre dp_pub1 et dp_pub2.
                    next_ad = ad_detections[idx+1]
                    gap_prog_duration = next_ad.timestamp - ad.timestamp # Ceci est en fait la durée (pub + programme)
                    
                    # On cherche les LONGUES périodes de PROGRAMME (logo présent)
                    # On doit identifier les blocs de programme.
                    # Les `ad_detections` sont des points où le logo est absent.
                    # On cherche des intervalles [t1, t2] où `logo_present` est `True` pendant longtemps.
                    # Le code original se basait sur les `DetectionPoint` où `not dp.logo_present`.
                    # C'est correct : un long intervalle entre deux `not dp.logo_present` signifie
                    # que le logo a été présent (ou non analysé) pendant cette durée.

                    gap_duration_between_ads = next_ad.timestamp - ad.timestamp # Temps entre début pub N et début pub N+1
                    if gap_duration_between_ads > anomaly_threshold:
                        # On s'attend à `gap / base_interval` pubs au total dans ce segment.
                        # On en a détecté 2 (les bornes). Donc `(gap / base_interval) - 2` pubs manquées.
                        # Si base_interval est la durée typique prog + pub.
                        expected_ads_in_gap = max(1, int(gap_duration_between_ads / base_interval_for_expectation) -1) # -1 car on a la pub au début de next_ad
                        if expected_ads_in_gap > 0:
                            priority = (gap_duration_between_ads / base_interval_for_expectation) * expected_ads_in_gap
                            # Le segment suspect est entre la fin de la pub actuelle et le début de la pub suivante.
                            # On n'a pas la "fin" de la pub 'ad'. On a son 'timestamp'.
                            # Pour la deuxième passe, on analyse [ad.timestamp, next_ad.timestamp]
                            suspicious_intervals.append(SuspiciousInterval(
                                ad.timestamp, # On pourrait tenter d'affiner le début
                                next_ad.timestamp, # On pourrait tenter d'affiner la fin
                                expected_ads_in_gap, 0, gap_duration_between_ads, priority))
                            self.logger.info(f"Intervalle suspect (entre pubs): {seconds_to_hms(ad.timestamp)} -> {seconds_to_hms(next_ad.timestamp)} (durée: {gap_duration_between_ads/60:.1f}min, pubs attendues: {expected_ads_in_gap}, prio: {priority:.1f})")

                last_ad_end_time = ad.timestamp # Approximation: le point de détection de la pub

            # Segment de programme après la dernière pub
            if ad_detections:
                last_ad = ad_detections[-1]
                duration_prog_end = self.video_duration - last_ad.timestamp
                if duration_prog_end > anomaly_threshold:
                    expected_ads_missed = max(1, int(duration_prog_end / base_interval_for_expectation) -1)
                    if expected_ads_missed > 0:
                        priority = (duration_prog_end / base_interval_for_expectation) * expected_ads_missed
                        suspicious_intervals.append(SuspiciousInterval(last_ad.timestamp, self.video_duration, expected_ads_missed, 0, duration_prog_end, priority))
                        self.logger.info(f"Intervalle suspect (fin vidéo): {seconds_to_hms(last_ad.timestamp)} -> {seconds_to_hms(self.video_duration)} (durée: {duration_prog_end/60:.1f}min, pubs attendues: {expected_ads_missed})")
        
        suspicious_intervals.sort(key=lambda x: x.priority, reverse=True)
        self.logger.info(f"Total d'intervalles suspects identifiés pour 2e passe: {len(suspicious_intervals)}")
        return suspicious_intervals

    def _perform_targeted_analysis(self, cap: cv2.VideoCapture, suspicious_intervals: List[SuspiciousInterval]) -> List[DetectionPoint]:
        """Effectue une analyse ciblée des intervalles suspects avec une densité accrue."""
        self.logger.info("Début de la deuxième passe d'analyse ciblée optimisée...")
        new_detections = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        for idx, interval in enumerate(suspicious_intervals):
            self.logger.info(f"Analyse ciblée {idx+1}/{len(suspicious_intervals)}: "
                           f"{seconds_to_hms(interval.start_time)} -> "
                           f"{seconds_to_hms(interval.end_time)} "
                           f"(priorité: {interval.priority:.1f})")

            duration = interval.end_time - interval.start_time
            base_density = self.second_pass_density_multiplier
            density_factor = base_density
            
            if interval.priority > 8: density_factor = base_density * 3
            elif interval.priority > 5: density_factor = base_density * 2.5
            elif interval.priority > 3: density_factor = base_density * 2
            else: density_factor = base_density * 1.5

            if duration > 2400: sample_interval_sec = 15 / density_factor
            elif duration > 1800: sample_interval_sec = 12 / density_factor
            elif duration > 900: sample_interval_sec = 10 / density_factor
            elif duration > 300: sample_interval_sec = 8 / density_factor
            else: sample_interval_sec = 6 / density_factor
            
            sample_interval_sec = max(sample_interval_sec, 1.0) # Au moins 1 sec pour la 2e passe dense
            sample_interval_frames = int(sample_interval_sec * fps)

            self.logger.info(f"  Densité d'échantillonnage 2e passe: {sample_interval_sec:.1f}s ({sample_interval_frames} frames)")

            current_time = interval.start_time
            points_analyzed_in_interval = 0
            while current_time < interval.end_time:
                frame_num = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret: break

                logo_region = self._extract_logo_region(frame)
                logo_present, confidence = self._is_logo_present(logo_region, frame, analysis_pass=2)
                
                new_detections.append(DetectionPoint(
                    timestamp=current_time, frame_number=frame_num,
                    logo_present=logo_present, confidence=confidence, analysis_pass=2
                ))
                points_analyzed_in_interval+=1
                if not logo_present:
                    self.logger.debug(f"  Nouvelle absence de logo détectée en 2e passe à {seconds_to_hms(current_time)} (conf: {confidence:.2f})")

                current_time += sample_interval_sec
                if points_analyzed_in_interval % 50 == 0:
                    self.logger.info(f"  Progression analyse intervalle: {((current_time - interval.start_time)/duration)*100 if duration > 0 else 0:.1f}%")
            self.logger.info(f"  Analyse intervalle terminée: {points_analyzed_in_interval} points.")


        self.logger.info(f"Deuxième passe terminée. Nouveaux points de détection: {len(new_detections)}")
        return new_detections

    def _get_frame_at_timestamp(self, cap: cv2.VideoCapture, timestamp: float) -> Optional[np.ndarray]:
        """Récupère une frame à un timestamp donné."""
        # Pourrait être déplacé dans video_processing.py
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        return frame if ret else None

    def _binary_search_logo_transition(self, cap: cv2.VideoCapture,
                                     start_time: float, end_time: float,
                                     logo_present_at_start_of_search_range: bool,
                                     precision: float = 0.3) -> float:
        """
        Recherche binaire pour trouver le moment exact de transition du logo.
        Args:
            start_time: Début de la plage de recherche.
            end_time: Fin de la plage de recherche.
            logo_present_at_start_of_search_range: État du logo au début de la plage `start_time`.
                                                Si True, on cherche quand le logo disparaît.
                                                Si False, on cherche quand le logo réapparaît.
            precision: Précision souhaitée en secondes.
        Returns:
            Timestamp exact de la transition (début du NOUVEL état).
        """
        # Pourrait être déplacé dans video_processing.py
        low = start_time
        high = end_time
        transition_point = end_time # Par défaut, si aucune transition n'est trouvée dans la plage.

        # Sanity check: si start et end sont trop proches ou invalides
        if (high - low) <= precision:
            # On vérifie l'état à 'high' pour voir s'il est différent de l'état de départ.
            # Ceci est important si la plage est déjà plus petite que la précision.
            frame = self._get_frame_at_timestamp(cap, high)
            if frame is not None:
                logo_region = self._extract_logo_region(frame)
                # Utiliser la passe 2 pour _is_logo_present pour la précision
                logo_present_at_high, _ = self._is_logo_present(logo_region, frame, analysis_pass=2)
                if logo_present_at_high != logo_present_at_start_of_search_range:
                    return high # La transition est à 'high' ou juste avant.
            return low if logo_present_at_start_of_search_range else high # Pas de changement détectable, retourne le début de l'état

        while (high - low) > precision:
            mid_time = (low + high) / 2
            if mid_time <= low or mid_time >=high: # Eviter boucle infinie si précision trop fine
                break
            frame = self._get_frame_at_timestamp(cap, mid_time)
            if frame is None: # Ne peut pas lire la frame, on ne peut pas diviser la recherche ici.
                              # On pourrait essayer de se décaler un peu, ou arrêter.
                              # Pour l'instant, on arrête la recherche pour cette branche.
                self.logger.warning(f"Binary search: Impossible de lire la frame à {mid_time:.2f}s. Arrêt de l'affinement pour cette branche.")
                break 

            logo_region = self._extract_logo_region(frame)
            # Utiliser la passe 2 pour _is_logo_present pour la précision lors de l'affinement
            current_logo_state, _ = self._is_logo_present(logo_region, frame, analysis_pass=2)

            if current_logo_state == logo_present_at_start_of_search_range:
                # Le logo est toujours dans l'état initial à mid_time, donc la transition est plus loin.
                low = mid_time
            else:
                # Le logo a changé d'état à mid_time ou avant. La transition est dans [low, mid_time].
                high = mid_time
                # transition_point = mid_time # high devient notre meilleur candidat pour la transition

        # 'high' est maintenant le premier point où l'état du logo est différent de celui à start_time.
        # Ou si aucune transition trouvée, 'high' sera resté 'end_time'.
        # On veut le début du *nouvel* état.
        # Si on cherchait la disparition (True -> False), 'high' est le début de False.
        # Si on cherchait la réapparition (False -> True), 'high' est le début de True.
        return high


    def _merge_close_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Fusionne les intervalles d'absence de logo qui sont séparés par moins de merge_gap_threshold secondes."""
        if not intervals: return []
        
        # Trier par temps de début
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(sorted_intervals[0])] # Utiliser une liste pour la modification

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start - last_end <= self.merge_gap_threshold: # Si le début actuel est proche de la fin précédente
                merged[-1][1] = max(last_end, current_end) # Fusionner: étendre la fin du segment précédent
                self.logger.info(f"Fusion d'intervalles proches: {seconds_to_hms(last_end)} et {seconds_to_hms(current_start)} (écart: {current_start - last_end:.1f}s)")
            else:
                merged.append([current_start, current_end])
        
        return [tuple(item) for item in merged]


    def _filter_minimum_duration_intervals(self, intervals: List[Tuple[float, float]],
                                         video_duration: float) -> List[Tuple[float, float]]:
        """Filtre les intervalles pour ne conserver que ceux qui respectent la durée minimum,
           sauf pour intro/outro."""
        if not intervals: return []
        
        filtered_intervals = []
        # Trier les intervalles par temps de début pour identifier correctement intro/outro
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        for i, (start, end) in enumerate(sorted_intervals):
            duration = end - start
            is_first_segment = (i == 0 and start < 300) # Premier segment dans les 5 premières minutes
            is_last_segment = (i == len(sorted_intervals) - 1 and (video_duration - end) < 300) # Dernier dans les 5 dernières minutes

            if duration <=0: # Ignorer les segments de durée nulle ou négative
                self.logger.warning(f"Intervalle de durée nulle ou négative ignoré: {seconds_to_hms(start)} -> {seconds_to_hms(end)}")
                continue

            if is_first_segment or is_last_segment:
                filtered_intervals.append((start, end))
                segment_type = "intro" if is_first_segment else "outro"
                self.logger.info(f"Segment {segment_type} conservé (pas de durée min): {seconds_to_hms(start)} -> {seconds_to_hms(end)} (durée: {duration:.1f}s)")
            elif duration >= self.min_ad_duration:
                filtered_intervals.append((start, end))
            else:
                self.logger.info(f"Intervalle rejeté (trop court): {seconds_to_hms(start)} -> {seconds_to_hms(end)} (durée: {duration:.1f}s < {self.min_ad_duration}s)")
        
        return filtered_intervals

    def analyze_video(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Analyse une vidéo pour détecter les moments où le logo n'est pas présent.
        Retourne une liste de tuples (start_seconds, end_seconds).
        """
        self.logger.info(f"Analyse de la vidéo: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Impossible d'ouvrir la vidéo: {video_path}")
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = total_frames / fps if fps > 0 else 0

        self.logger.info(f"Durée vidéo: {seconds_to_hms(self.video_duration)} ({self.video_duration:.0f}s), FPS: {fps:.2f}, Frames: {total_frames}")

        self.detection_history = [] # Réinitialiser pour une nouvelle analyse

        # ========== PREMIÈRE PASSE ==========
        self.logger.info("🔍 PREMIÈRE PASSE - Échantillonnage général")
        if self.video_duration > 7200: sample_interval_sec_p1 = 45
        elif self.video_duration > 3600: sample_interval_sec_p1 = 30
        elif self.video_duration > 1800: sample_interval_sec_p1 = 20
        else: sample_interval_sec_p1 = 15
        
        # Assurer un sample_interval_frames > 0
        sample_interval_frames_p1 = max(1, int(sample_interval_sec_p1 * fps if fps > 0 else sample_interval_sec_p1))
        self.logger.info(f"Intervalle d'échantillonnage première passe: {sample_interval_sec_p1}s (~{sample_interval_frames_p1} frames)")

        current_frame_num = 0
        frames_analyzed_p1 = 0
        while current_frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = cap.read()
            if not ret: break

            timestamp = current_frame_num / fps if fps > 0 else 0
            logo_region = self._extract_logo_region(frame)
            logo_present, confidence = self._is_logo_present(logo_region, frame, analysis_pass=1)
            self.detection_history.append(DetectionPoint(timestamp, current_frame_num, logo_present, confidence, 1))
            
            frames_analyzed_p1 += 1
            current_frame_num += sample_interval_frames_p1
            if frames_analyzed_p1 % 100 == 0:
                progress = (current_frame_num / total_frames) * 100 if total_frames > 0 else 0
                self.logger.info(f"Progression P1: {progress:.1f}% ({frames_analyzed_p1} frames analysées)")
        self.logger.info(f"Première passe terminée: {frames_analyzed_p1} points analysés.")

        # ========== ANALYSE STATISTIQUE & IDENTIFICATION ZONES SUSPECTES (basée sur Passe 1) ==========
        self.ad_intervals_stats = self._analyze_detection_patterns() # Utilise les détections de passe 1
        suspicious_intervals_for_p2 = self._identify_suspicious_intervals(self.ad_intervals_stats)

        # ========== DEUXIÈME PASSE (Ciblée) ==========
        if suspicious_intervals_for_p2:
            self.logger.info("🔎 DEUXIÈME PASSE - Analyse ciblée sur intervalles suspects")
            second_pass_detections = self._perform_targeted_analysis(cap, suspicious_intervals_for_p2)
            self.detection_history.extend(second_pass_detections)
            self.logger.info(f"Points ajoutés en deuxième passe: {len(second_pass_detections)}")
        else:
            self.logger.info("✅ Aucun intervalle suspect majeur détecté - Pas de deuxième passe nécessaire ou zones trop petites.")

        # ========== TRAITEMENT FINAL DES RÉSULTATS ==========
        self.logger.info("📊 Traitement final des résultats avec affinement des transitions")
        self.detection_history.sort(key=lambda x: x.timestamp)

        # S'assurer qu'il y a au moins un point de détection pour éviter les erreurs d'index
        if not self.detection_history:
            self.logger.warning("Aucun point de détection après les passes d'analyse. Aucune pub ne sera détectée.")
            cap.release()
            return []

        precise_ad_intervals = []
        in_ad_segment = False
        current_ad_refined_start = 0.0

        # Ajouter un point de détection synthétique au début et à la fin si nécessaire
        # pour faciliter la gestion des transitions aux extrémités de la vidéo.
        # L'état initial est supposé être "logo présent" sauf si la première détection dit autre chose.
        
        # État du logo avant la première détection réelle. Supposons True (logo présent).
        # Si la première détection est très tôt et dit False, cela sera corrigé.
        last_known_logo_state = True
        last_known_logo_time = 0.0

        # Si la première détection est une absence de logo dès le début
        if not self.detection_history[0].logo_present and self.detection_history[0].timestamp < 1.0 : # Consider <1s as "début"
            last_known_logo_state = False # Commencer en segment pub
            in_ad_segment = True
            current_ad_refined_start = 0.0 # Pub commence à t=0
            last_known_logo_time = self.detection_history[0].timestamp # Pour la prochaine transition

        # Traiter les points de détection pour trouver les transitions et les affiner
        for i in range(len(self.detection_history)):
            detection = self.detection_history[i]
            
            # Transition: Logo Présent -> Logo Absent (Début de pub)
            if last_known_logo_state and not detection.logo_present:
                refined_start = self._binary_search_logo_transition(
                    cap, last_known_logo_time, detection.timestamp, True # True = logo_present_at_start_of_search
                )
                current_ad_refined_start = refined_start
                in_ad_segment = True
                self.logger.debug(f"Transition P->A: Coarse {seconds_to_hms(detection.timestamp)}, Refined Start {seconds_to_hms(current_ad_refined_start)}")

            # Transition: Logo Absent -> Logo Présent (Fin de pub)
            elif not last_known_logo_state and detection.logo_present:
                if in_ad_segment: # S'assurer qu'on était bien dans un segment pub
                    refined_end = self._binary_search_logo_transition(
                        cap, last_known_logo_time, detection.timestamp, False # False = logo_absent_at_start_of_search
                    )
                    # S'assurer que refined_end n'est pas avant current_ad_refined_start
                    if refined_end > current_ad_refined_start:
                         precise_ad_intervals.append((current_ad_refined_start, refined_end))
                         self.logger.debug(f"Transition A->P: Coarse {seconds_to_hms(detection.timestamp)}, Refined End {seconds_to_hms(refined_end)}. Interval: [{seconds_to_hms(current_ad_refined_start)}-{seconds_to_hms(refined_end)}]")
                    else:
                        self.logger.warning(f"Fin de pub affinée ({seconds_to_hms(refined_end)}) antérieure au début ({seconds_to_hms(current_ad_refined_start)}). Interval ignoré.")
                    in_ad_segment = False
            
            last_known_logo_state = detection.logo_present
            last_known_logo_time = detection.timestamp

        # Gérer si la vidéo se termine pendant une publicité
        if in_ad_segment:
            # La pub continue jusqu'à la fin de la vidéo.
            # On peut affiner la fin si le dernier point de détection n'est pas exactement self.video_duration
            refined_end = self._binary_search_logo_transition(
                cap, last_known_logo_time, self.video_duration, False # False = logo_absent_at_start
            )
            if refined_end > current_ad_refined_start:
                precise_ad_intervals.append((current_ad_refined_start, refined_end))
                self.logger.debug(f"Fin de vidéo en pub. Interval: [{seconds_to_hms(current_ad_refined_start)}-{seconds_to_hms(refined_end)}]")
            else:
                 self.logger.warning(f"Fin de pub (vidéo) affinée ({seconds_to_hms(refined_end)}) antérieure au début ({seconds_to_hms(current_ad_refined_start)}). Interval ignoré.")


        self.logger.info(f"Intervalles d'absence de logo (précis, avant fusion/filtrage): {len(precise_ad_intervals)}")
        for start, end in precise_ad_intervals:
             self.logger.debug(f"  Brut précis: {seconds_to_hms(start)} -> {seconds_to_hms(end)}")


        # Fusionner les intervalles précis qui sont proches
        merged_intervals = self._merge_close_intervals(precise_ad_intervals)
        self.logger.info(f"Après fusion des intervalles proches: {len(merged_intervals)}")

        # Filtrer par durée minimum (avec exception intro/outro)
        final_intervals = self._filter_minimum_duration_intervals(merged_intervals, self.video_duration)
        self.logger.info(f"Après filtrage durée min ({self.min_ad_duration}s, sauf intro/outro): {len(final_intervals)}")

        cap.release()

        # Affichage des statistiques finales dans les logs
        total_ad_duration_final = sum(end - start for start, end in final_intervals)
        self.logger.info("="*60)
        self.logger.info("📈 STATISTIQUES FINALES (via LogoDetector)")
        self.logger.info(f"Segments publicitaires finaux détectés: {len(final_intervals)}")
        if self.video_duration > 0:
            self.logger.info(f"Durée totale des publicités: {seconds_to_hms(total_ad_duration_final)} ({total_ad_duration_final/self.video_duration*100:.1f}% de la vidéo)")
        else:
            self.logger.info(f"Durée totale des publicités: {seconds_to_hms(total_ad_duration_final)}")
        self.logger.info(f"Durée totale de la vidéo: {seconds_to_hms(self.video_duration)}")

        return final_intervals

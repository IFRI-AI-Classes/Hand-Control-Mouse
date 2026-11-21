"""Avertissements de qualité sur une vidéo traitée."""
from .config import MIN_DETECTION_RATE, MIN_FPS, MIN_SIDE_SHARE
from .video_labels import VideoLabel
from .video_stats import VideoStats


def collect_warnings(stats: VideoStats, label: VideoLabel) -> list:
    warnings = list(label.problems)
    if stats.detection_rate < MIN_DETECTION_RATE:
        warnings.append(f"taux de détection faible ({stats.detection_rate:.0%})")
    if stats.frames_detected and stats.majority_side_share < MIN_SIDE_SHARE:
        warnings.append(f"côté de main incohérent ({stats.majority_side_share:.0%} {stats.majority_side})")
    if stats.height > stats.width:
        warnings.append("vidéo en portrait")
    if stats.fps_estimated:
        warnings.append("FPS illisible, 30 supposé")
    elif stats.fps < MIN_FPS:
        warnings.append(f"FPS bas ({stats.fps:.1f})")
    if not stats.trim_end_applied:
        warnings.append("trim de fin non appliqué (durée inconnue)")
    return warnings

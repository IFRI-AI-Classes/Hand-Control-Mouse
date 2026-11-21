"""Objets de données échangés entre les modules d'extraction."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HandObservation:
    """Une main détectée : 21 points (x, y, z) bruts MediaPipe."""
    confidence: float
    handedness: str
    landmarks: tuple


@dataclass(frozen=True)
class FrameRecord:
    """Position d'une image dans sa vidéo."""
    frame_idx: int
    t_ms: int
    width: int
    height: int

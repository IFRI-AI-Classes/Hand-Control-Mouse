"""Détection de la main : interface commune et implémentation MediaPipe (mêmes réglages que l'application)."""
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import (
    MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE, NUM_HANDS,
)
from .data_models import HandObservation


class HandExtractor(Protocol):
    def extract(self, rgb_frame, timestamp_ms: int) -> HandObservation | None: ...
    def close(self) -> None: ...


class MediaPipeHandExtractor:
    """Un détecteur par vidéo : aucun état de suivi n'est partagé entre vidéos."""

    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"modèle MediaPipe introuvable : {model_path}")
        import mediapipe as mp  # import tardif : les tests n'en ont pas besoin
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, rgb_frame, timestamp_ms: int) -> HandObservation | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.hand_landmarks:
            return None
        points = tuple((p.x, p.y, p.z) for p in result.hand_landmarks[0])
        side = result.handedness[0][0] if result.handedness and result.handedness[0] else None
        return HandObservation(
            confidence=float(side.score) if side else 0.0,
            handedness=side.category_name if side else "",
            landmarks=points,
        )

    def close(self) -> None:
        self._landmarker.close()


def create_mediapipe_extractor(model_path: Path) -> MediaPipeHandExtractor:
    return MediaPipeHandExtractor(model_path)

"""Lecture d'une vidéo image par image (OpenCV)."""
import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import FALLBACK_FPS


class VideoOpenError(Exception):
    """La vidéo ne peut pas être ouverte."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    fps_estimated: bool  # True si le FPS est illisible et remplacé par une valeur par défaut
    frame_count: int     # 0 si inconnu


class VideoSource:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._capture = None
        self.info = None

    def __enter__(self):
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoOpenError(f"impossible d'ouvrir la vidéo : {self._path.name}")
        self.info = self._read_info()
        return self

    def _read_info(self) -> VideoInfo:
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        fps_valid = bool(fps) and math.isfinite(fps) and 0 < fps < 1000
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoInfo(fps if fps_valid else FALLBACK_FPS, not fps_valid, max(frame_count, 0))

    def frames(self):
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return
            yield index, frame
            index += 1

    def __exit__(self, exc_type, exc, traceback):
        self._capture.release()
        return False

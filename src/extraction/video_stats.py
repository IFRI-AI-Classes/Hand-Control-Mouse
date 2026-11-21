"""Compteurs accumulés pendant le traitement d'une vidéo."""
from collections import Counter
from dataclasses import dataclass, field

from .data_models import FrameRecord, HandObservation


@dataclass
class VideoStats:
    fps: float = 0.0
    fps_estimated: bool = False
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    frames_read: int = 0
    frames_kept: int = 0
    frames_detected: int = 0
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0
    trim_end_applied: bool = True
    side_counts: Counter = field(default_factory=Counter)

    def add_frame(self, record: FrameRecord, observation: HandObservation | None) -> None:
        if self.frames_kept == 0:
            self.width, self.height = record.width, record.height
        self.frames_kept += 1
        if observation is not None:
            self.frames_detected += 1
            self.side_counts[observation.handedness] += 1

    @property
    def detection_rate(self) -> float:
        return self.frames_detected / self.frames_kept if self.frames_kept else 0.0

    @property
    def majority_side(self) -> str:
        return self.side_counts.most_common(1)[0][0] if self.side_counts else ""

    @property
    def majority_side_share(self) -> float:
        if not self.side_counts:
            return 0.0
        return self.side_counts.most_common(1)[0][1] / self.frames_detected

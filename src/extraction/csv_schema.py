"""Colonnes du CSV de landmarks et construction d'une ligne."""
from .config import NUM_LANDMARKS
from .data_models import FrameRecord, HandObservation
from .video_labels import VideoLabel

META_COLUMNS = [
    "person", "gesture", "lighting", "orientation", "source", "video",
    "frame_idx", "t_ms", "width", "height",
    "hand_detected", "confidence", "handedness",
]
LANDMARK_COLUMNS = [f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
COLUMNS = META_COLUMNS + LANDMARK_COLUMNS


def build_row(label: VideoLabel, record: FrameRecord, observation: HandObservation | None) -> list:
    row = [
        label.person, label.gesture, label.lighting, label.orientation, label.source, label.video,
        record.frame_idx, record.t_ms, record.width, record.height,
    ]
    if observation is None:
        return row + [0, "", ""] + [""] * len(LANDMARK_COLUMNS)
    flat = [f"{value:.6f}" for point in observation.landmarks for value in point]
    if len(flat) != len(LANDMARK_COLUMNS):
        raise ValueError(f"{len(flat)} valeurs de landmarks reçues, {len(LANDMARK_COLUMNS)} attendues")
    return row + [1, f"{observation.confidence:.4f}", observation.handedness] + flat

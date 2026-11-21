"""Vidéo de contrôle : la vidéo traitée avec les 21 points de la main dessinés dessus."""
from pathlib import Path

import cv2

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
_GREEN = (0, 220, 0)
_RED = (0, 0, 255)


class PreviewRenderer:
    def __init__(self, output_path: Path):
        self._path = Path(output_path)
        self._fps = 30.0
        self._writer = None

    def start(self, info) -> None:
        self._fps = info.fps

    def on_frame(self, frame_bgr, label, record, observation) -> None:
        canvas = frame_bgr.copy()
        if observation is not None:
            _draw_hand(canvas, observation.landmarks)
        _draw_caption(canvas, label, record, observation is not None)
        self._ensure_writer(canvas.shape[1], canvas.shape[0]).write(canvas)

    def _ensure_writer(self, width: int, height: int):
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (width, height))
        return self._writer

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


def _draw_hand(canvas, landmarks) -> None:
    height, width = canvas.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(canvas, points[start], points[end], _GREEN, 2)
    for point in points:
        cv2.circle(canvas, point, 3, _RED, -1)


def _draw_caption(canvas, label, record, hand_found: bool) -> None:
    status = "main OK" if hand_found else "PAS de main"
    text = f"frame {record.frame_idx} | {label.gesture} | {status}"
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

"""Calculs de temps : timestamp d'une image et fenêtre conservée après trim."""
from dataclasses import dataclass


def frame_timestamp_ms(frame_idx: int, fps: float) -> int:
    """Timestamp lu dans la vidéo (jamais l'horloge de l'ordinateur) : résultat reproductible."""
    return int(round(frame_idx * 1000.0 / fps))


@dataclass(frozen=True)
class KeepWindow:
    start_ms: int
    end_ms: int | None  # None : fin non bornée (durée de la vidéo inconnue)

    def contains(self, t_ms: int) -> bool:
        return t_ms >= self.start_ms and (self.end_ms is None or t_ms <= self.end_ms)


def build_keep_window(frame_count: int, fps: float, trim_start_s: float, trim_end_s: float) -> KeepWindow:
    start_ms = int(round(max(trim_start_s, 0.0) * 1000))
    if frame_count <= 0:
        return KeepWindow(start_ms, None)
    duration_ms = int(round(frame_count / fps * 1000))
    return KeepWindow(start_ms, duration_ms - int(round(max(trim_end_s, 0.0) * 1000)))

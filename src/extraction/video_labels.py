"""Déduit les étiquettes (personne, geste, conditions) du chemin d'une vidéo."""
from dataclasses import dataclass
from pathlib import Path

from .config import GESTURES, LIGHTINGS, ORIENTATIONS, SOURCES, UNKNOWN


class LabelError(Exception):
    """Le chemin ne permet pas d'étiqueter la vidéo : elle doit être ignorée."""


@dataclass(frozen=True)
class VideoLabel:
    person: str
    gesture: str
    lighting: str
    orientation: str
    source: str
    video: str
    filename_ok: bool
    problems: tuple = ()


def parse_video_label(video_path: Path, input_root: Path) -> VideoLabel:
    relative = _relative_parts(video_path, input_root)
    if len(relative.parts) != 3:
        raise LabelError(f"structure attendue personne/geste/fichier, trouvé : {relative.as_posix()}")
    person, gesture_folder, filename = relative.parts
    gesture = gesture_folder.lower()
    if gesture not in GESTURES:
        raise LabelError(f"dossier de geste inconnu : {gesture_folder}")
    lighting, orientation, source = _parse_conditions(Path(filename).stem)
    ok = UNKNOWN not in (lighting, orientation, source)
    problems = () if ok else (f"nom de fichier non conforme : {filename}",)
    return VideoLabel(person, gesture, lighting, orientation, source, relative.as_posix(), ok, problems)


def _relative_parts(video_path: Path, input_root: Path) -> Path:
    try:
        return Path(video_path).resolve().relative_to(Path(input_root).resolve())
    except ValueError:
        raise LabelError("vidéo située hors du dossier d'entrée") from None


def _parse_conditions(stem: str) -> tuple:
    parts = stem.lower().split("_")
    valid = (
        len(parts) == 3
        and parts[0] in LIGHTINGS
        and parts[1] in ORIENTATIONS
        and parts[2] in SOURCES
    )
    return tuple(parts) if valid else (UNKNOWN, UNKNOWN, UNKNOWN)

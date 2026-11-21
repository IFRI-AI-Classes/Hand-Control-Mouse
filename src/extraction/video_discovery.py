"""Recherche des vidéos dans le dossier d'entrée."""
from pathlib import Path

from .config import VIDEO_EXTENSIONS


def find_videos(input_dir: Path) -> list:
    return sorted(
        path for path in Path(input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

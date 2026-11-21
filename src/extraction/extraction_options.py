"""Réglages de traitement appliqués à chaque vidéo."""
from dataclasses import dataclass

from .config import DEFAULT_TRIM_S, MIRROR_DEFAULT


@dataclass(frozen=True)
class ExtractionOptions:
    trim_start_s: float = DEFAULT_TRIM_S
    trim_end_s: float = DEFAULT_TRIM_S
    mirror: bool = MIRROR_DEFAULT

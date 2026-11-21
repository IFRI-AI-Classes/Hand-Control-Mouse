"""Constantes de l'extraction : conventions de nommage, réglages MediaPipe, seuils d'alerte."""
from pathlib import Path

# --- Conventions de nommage du Drive : <personne>/<geste>/<eclairage>_<orientation>_<source>.mp4
GESTURES = ("clic_droit", "clic_gauche", "deplacement", "double_clic", "drag", "neutre", "scroll")
LIGHTINGS = ("jour", "neon", "faible", "contrejour")
ORIENTATIONS = ("face", "incline")
SOURCES = ("webcam", "tel")
UNKNOWN = "inconnu"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# --- Réglages MediaPipe : à garder identiques à ceux de src/detection.py (application)
NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_PRESENCE_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
NUM_LANDMARKS = 21
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "hand_landmarker.task"

# --- Pré-traitement : à garder identique à MIRROR dans src/main.py
MIRROR_DEFAULT = True
DEFAULT_TRIM_S = 0.5
FALLBACK_FPS = 30.0

# --- Seuils des avertissements du rapport
MIN_DETECTION_RATE = 0.80
MIN_SIDE_SHARE = 0.90
MIN_FPS = 20.0

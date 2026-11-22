"""Chemins, liste des classes et hyperparamètres du MLP."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = ROOT / "data" / "landmarks" / "dataset.csv"   # vidéos, produit par extract_landmarks --merge
PHOTOS_DIR = ROOT / "data" / "videos"                       # les photos sont rangées comme les vidéos
PHOTOS_CSV = ROOT / "data" / "landmarks" / "photos.csv"     # produit par python -m mlp.extract_photos
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "mlp.joblib"

# Colonnes du CSV produit par src/extraction (csv_schema.py)
PERSON_COL = "person"
LABEL_COL = "gesture"
DETECTED_COL = "hand_detected"
VIDEO_COL = "video"
FRAME_COL = "frame_idx"
NUM_LANDMARKS = 21
LANDMARK_COLS = [f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in "xyz"]

# Le double-clic n'est plus une posture (deux clics gauches rapprochés) :
# les données "double_clic" (pouce + auriculaire) deviennent des clics gauches,
# comme dans les règles où l'auriculaire est ignoré.
RELABEL = {"double_clic": "clic_gauche"}
GESTURES = ["deplacement", "scroll", "clic_gauche", "clic_droit", "drag", "neutre"]

# Découpage dans le temps : dans chaque vidéo, les dernières images servent au test
TEST_RATIO = 0.3

# Augmentation (idée reprise du module de robustesse de l'équipe) : miroir + rotations légères
AUGMENT_ROTATIONS = 2
AUGMENT_MAX_DEGREES = 15.0
SEED = 0

# Indices MediaPipe utilisés pour la normalisation
WRIST = 0
MIDDLE_MCP = 9

# MLP : petit réseau, suffisant pour 63 entrées et 6 classes
MLP_PARAMS = dict(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=0)

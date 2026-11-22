"""Décision de posture des règles de src/gestures.py, image par image (utilisée par train.py et detection.py)."""
import sys

import numpy as np

from .config import NUM_LANDMARKS, ROOT

sys.path.insert(0, str(ROOT / "src"))
from gestures import GestureRecognizer  # noqa: E402


def rules_predict(X):
    """Décision de posture des règles, image par image.

    Depuis la nouvelle version de gestures.py :
    - les doigts sont jugés par angle en 3D, donc on passe x, y et z ;
    - le double-clic n'est plus une posture mais deux clics gauches rapprochés,
      les règles ne peuvent donc jamais prédire "double_clic" image par image ;
    - le pouce a une hystérésis : on la remet à zéro à chaque image
      (seuil d'ouverture strict), faute d'historique fiable entre images.
    """
    rec = GestureRecognizer()
    preds = []
    for row in X:
        points = row.reshape(NUM_LANDMARKS, 3)
        rec._thumb_open = False
        f = rec._finger_ups(points)
        # même logique que GestureRecognizer.recognize : le pointeur bloque les clics
        if rec.is_drag_pose(f):
            preds.append("drag")
        elif rec.is_pointer_pose(f):
            preds.append("deplacement")
        elif rec.is_left_click_pose(f):
            preds.append("clic_gauche")
        elif rec.is_right_click_pose(f):
            preds.append("clic_droit")
        elif rec.is_scroll_pose(f):
            preds.append("scroll")
        else:
            preds.append("neutre")
    return np.array(preds)

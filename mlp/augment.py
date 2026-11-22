"""Augmentation des données d'entraînement : miroir et rotations légères autour du poignet.

Idée reprise du module de robustesse écrit par l'équipe. Le but : qu'un modèle entraîné sur peu de
personnes voie des mains un peu plus variées (main gauche, main inclinée).
Ne s'applique qu'aux données d'entraînement, jamais au test.
"""
import numpy as np

from .config import AUGMENT_MAX_DEGREES, AUGMENT_ROTATIONS, NUM_LANDMARKS, SEED, WRIST


def augment(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    pts = np.asarray(X, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3)
    wrist = pts[:, WRIST:WRIST + 1, :]
    batches, labels = [pts], [y]

    mirrored = pts.copy()                                   # main droite <-> main gauche
    mirrored[:, :, 0] = 2 * wrist[:, :, 0] - pts[:, :, 0]
    batches.append(mirrored)
    labels.append(y)

    for _ in range(AUGMENT_ROTATIONS):                      # inclinaison dans le plan de l'image
        angle = np.deg2rad(rng.uniform(-AUGMENT_MAX_DEGREES, AUGMENT_MAX_DEGREES, len(pts)))[:, None]
        dx, dy = pts[:, :, 0] - wrist[:, :, 0], pts[:, :, 1] - wrist[:, :, 1]
        rotated = pts.copy()
        rotated[:, :, 0] = wrist[:, :, 0] + dx * np.cos(angle) - dy * np.sin(angle)
        rotated[:, :, 1] = wrist[:, :, 1] + dx * np.sin(angle) + dy * np.cos(angle)
        batches.append(rotated)
        labels.append(y)

    return np.concatenate(batches).reshape(-1, NUM_LANDMARKS * 3), np.concatenate(labels)

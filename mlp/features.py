"""Normalisation des landmarks : centrage sur le poignet et mise à l'échelle par la taille de la main.

Une même posture doit donner les mêmes chiffres, que la main soit à gauche ou à droite
de l'image, proche ou loin de la caméra.
"""
import numpy as np

from .config import MIDDLE_MCP, NUM_LANDMARKS, WRIST


def normalize(X):
    pts = np.asarray(X, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3)
    pts = pts - pts[:, WRIST:WRIST + 1, :]                          # poignet en (0, 0, 0)
    scale = np.linalg.norm(pts[:, MIDDLE_MCP, :2], axis=1)           # poignet -> base du majeur
    pts = pts / (scale[:, None, None] + 1e-6)
    return pts.reshape(len(pts), -1)



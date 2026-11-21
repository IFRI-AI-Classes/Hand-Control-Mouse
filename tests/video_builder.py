"""Génère de petites vidéos de test (MJPG, lisibles partout)."""
import cv2
import numpy as np


def make_video(path, n_frames=30, fps=10, size=(64, 48)):
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for i in range(n_frames):
        writer.write(np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8))
    writer.release()
    return path

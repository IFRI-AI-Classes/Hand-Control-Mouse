"""
Ce fichier allume la webcam et récupère les images, une par une.
Rien d'autre. Pas de détection de main, pas de souris ici.
"""

import cv2

def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")
    return cap

def get_frame(cap):
    success, frame = cap.read()
    if not success:
        return None
    return frame

def release_camera(cap):
    cap.release()
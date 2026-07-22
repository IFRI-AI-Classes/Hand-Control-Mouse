"""
Ce fichier regarde l'image et trouve la main dedans.
Il renvoie la position des points de la main (doigts, paume...).
Il ne s'occupe pas de l'écran ni de la souris.

Détection en mode synchrone (VIDEO) : chaque frame est traitée
immédiatement, et le résultat renvoyé correspond exactement à cette
frame. Contrairement au mode LIVE_STREAM (asynchrone), il n'y a pas de
décalage entre l'image affichée et le résultat de détection utilisé.
"""
import time
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Modèle téléchargé.")


class HandDetector:
    def __init__(self):
        download_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def detect(self, rgb_frame):
        """
        Détecte la main dans la frame RGB donnée et renvoie directement
        le résultat : une liste de mains (chaque main = liste de 21
        landmarks), ou None si aucune main détectée.
        Appel bloquant (synchrone), mais rapide pour num_hands=1.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        return result.hand_landmarks if result.hand_landmarks else None

    def close(self):
        self._detector.close()
"""
Ce fichier regarde l'image et trouve la main dedans.
Il renvoie la position des points de la main (doigts, paume...).
Il ne s'occupe pas de l'écran ni de la souris.

"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import urllib.request
import os

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
        self._landmarks = None
        self._lock = threading.Lock()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=self._on_result
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._timestamp = 0

    def _on_result(self, result, output_image, timestamp_ms):
        with self._lock:
            self._landmarks = result.hand_landmarks if result.hand_landmarks else None

    def detect(self, rgb_frame):
        """Envoie une frame RGB pour détection async."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._timestamp += 1
        self._detector.detect_async(mp_image, self._timestamp)

    def get_landmarks(self):
        """Retourne les derniers landmarks détectés (thread-safe)."""
        with self._lock:
            return self._landmarks

    def close(self):
        self._detector.close()
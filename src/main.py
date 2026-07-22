"""
C'est le fichier qui lance tout et qui fait le lien entre les autres.
Il ne fait aucun calcul lui-même : il appelle juste, dans l'ordre,
les fonctions des autres fichiers.

Boucle principale : orchestre capture -> detection -> mapping -> gestures -> controller -> ui.
"""

import cv2
import argparse
import pyautogui

from capture import open_camera, get_frame, release_camera
from detection import HandDetector
from gestures import GestureRecognizer, Gesture
from controller import MouseController
from mapping import CoordinateMapper
from ui import ActionOverlay, show_debug_window, close_debug_window

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Affiche la fenêtre de visualisation")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialisation
    cap = open_camera()
    detector = HandDetector()
    recognizer = GestureRecognizer()
    mouse = MouseController()
    screen_w, screen_h = pyautogui.size()
    mapper = CoordinateMapper(screen_w, screen_h)
    overlay = ActionOverlay()  # toujours actif, debug ou non

    try:
        while True:
            # 1. Capture
            frame = get_frame(cap)
            if frame is None:
                continue

            # 2. Détection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector.detect(rgb_frame)
            landmarks = detector.get_landmarks()

            if landmarks:
                hand = landmarks[0]  # on prend la première main détectée

                # 3. Mapping (zone active + lissage) via mapping.py
                index_tip = hand[8]
                x, y = mapper.to_screen(index_tip.x, index_tip.y)

                # 4. Déplacement du curseur
                mouse.move(x, y)

                # 5. Reconnaissance du geste
                gesture = recognizer.recognize(hand)

                # 6. Action souris
                mouse.handle(gesture)

            else:
                gesture = Gesture.NONE
                mapper.reset()  # évite un saut du curseur au retour de la main

            # 7. Affichage de l'action en cours (toujours actif)
            overlay.update(gesture.name)

            # 8. Fenêtre webcam de debug (optionnelle, --debug uniquement)
            if args.debug:
                if not show_debug_window(frame, gesture.name):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        release_camera(cap)
        detector.close()
        overlay.close()
        if args.debug:
            close_debug_window()


if __name__ == "__main__":
    main()
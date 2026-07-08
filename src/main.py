"""
C'est le fichier qui lance tout et qui fait le lien entre les autres.
Il ne fait aucun calcul lui-même : il appelle juste, dans l'ordre,
les fonctions des autres fichiers.

Boucle principale : orchestre capture -> detection -> mapping -> gestures -> controller.
"""

"""
C'est le fichier qui lance tout et qui fait le lien entre les autres.
Il ne fait aucun calcul lui-même : il appelle juste, dans l'ordre,
les fonctions des autres fichiers.
"""

import cv2
import argparse
import pyautogui

from capture import open_camera, get_frame, release_camera
from detection import HandDetector
from gestures import GestureRecognizer, Gesture
from controller import MouseController

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

                # 3. Mapping brut (sans mapping.py pour l'instant)
                index_tip = hand[8]
                x = index_tip.x * screen_w
                y = index_tip.y * screen_h

                # 4. Déplacement du curseur
                mouse.move(x, y)

                # 5. Reconnaissance du geste
                gesture = recognizer.recognize(hand)

                # 6. Action souris
                mouse.handle(gesture)

            else:
                gesture = Gesture.NONE

            # 7. Affichage debug (optionnel)
            if args.debug:
                label = gesture.name if landmarks else "Aucune main"
                cv2.putText(frame, label, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Hand Control Mouse - DEBUG", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        release_camera(cap)
        detector.close()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
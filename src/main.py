"""
C'est le fichier qui lance tout et qui fait le lien entre les autres.
Il ne fait aucun calcul lui-même : il appelle juste, dans l'ordre,
les fonctions des autres fichiers.

Boucle principale : orchestre capture -> detection -> mapping -> gestures -> controller -> ui.
"""

import cv2
import time
import argparse
import pyautogui

from capture import open_camera, get_frame, release_camera
from detection import HandDetector
from gestures import GestureRecognizer, Gesture
from controller import MouseController
from mapping import CoordinateMapper
from ui import ActionOverlay, show_debug_window, close_debug_window

# Certaines webcams (souvent selon l'OS/le driver) renvoient déjà une
# image pré-inversée. Si le curseur va dans le mauvais sens, passe cette
# valeur à False (ou l'inverse) et reteste.
MIRROR = True


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

    # Suivi du FPS (lissé pour ne pas sauter dans tous les sens à l'affichage)
    prev_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            # 1. Capture
            frame = get_frame(cap)
            if frame is None:
                continue

            # Effet miroir : la webcam nous voit "en face", donc sans ce
            # flip, un mouvement vers la droite du doigt peut partir vers
            # la gauche dans l'image. Réglable via MIRROR en haut du fichier.
            if MIRROR:
                frame = cv2.flip(frame, 1)

            # 2. Détection (synchrone, résultat immédiat pour cette frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb_frame)

            # Calcul du FPS (lissage exponentiel pour un affichage stable)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps_smooth = instant_fps if fps_smooth == 0 else fps_smooth * 0.9 + instant_fps * 0.1

            if landmarks:
                hand = landmarks[0]  # on prend la première main détectée

                # 3. Reconnaissance du geste (dit aussi si on peut bouger
                #    le curseur, et à quelle vitesse scroller le cas échéant)
                gesture, moving_allowed, scroll_speed = recognizer.recognize(hand)

                # 4. Déplacement du curseur : suit le bout de l'index en
                #    posture pointeur, ou le bout du majeur en posture
                #    drag (c'est ce doigt-là qui est tendu dans ce cas).
                if moving_allowed:
                    tracked_tip = hand[12] if gesture == Gesture.DRAG else hand[8]
                    dx, dy = mapper.get_relative_move(tracked_tip.x, tracked_tip.y)
                    mouse.move_relative(dx, dy)
                else:
                    mapper.reset()

                # 5. Action souris (clic, scroll à vitesse variable...)
                mouse.handle(gesture, scroll_delta=scroll_speed)

            else:
                gesture = Gesture.NONE
                mapper.reset()  # évite un saut du curseur au retour de la main
                recognizer.reset()  # évite un faux scroll au retour de la main

            # 7. Affichage de l'action en cours (toujours actif)
            overlay.update(gesture.name)

            # 8. Fenêtre webcam de debug (optionnelle, --debug uniquement)
            if args.debug:
                displayed_hand = landmarks[0] if landmarks else None
                if not show_debug_window(frame, gesture.name,
                                          landmarks=displayed_hand, fps=fps_smooth):
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
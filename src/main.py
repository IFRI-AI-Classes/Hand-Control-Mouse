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

# Réglages du drag (glisser-déposer)
DRAG_SPEED = 0.6    # 1.0 = même vitesse qu'en déplacement normal
DRAG_SMOOTH = 0.5   # entre 0 et 1 : plus bas = plus doux, mais plus de retard


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

    # Lissage du déplacement pendant un drag
    smooth_dx = smooth_dy = 0.0

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

                if gesture in (Gesture.LEFT_CLICK, Gesture.RIGHT_CLICK, Gesture.DOUBLE_CLICK):
                    print(gesture.name, {k: int(v) for k, v in recognizer.last_fingers.items()},
                        round(recognizer.last_thumb_ratio, 2))

                # 4. Déplacement du curseur : suit la base de l'index en
                #    posture pointeur, la base du majeur en posture drag
                #    (y compris pendant la confirmation du drag).
                if moving_allowed:
                    in_drag = gesture == Gesture.DRAG or recognizer.drag_pending
                    tracked_point = hand[9] if in_drag else hand[5]
                    dx, dy = mapper.get_relative_move(tracked_point.x, tracked_point.y)
                    frozen = recognizer.drag_pending or recognizer.drag_releasing

                    if gesture == Gesture.DRAG:
                        dx *= DRAG_SPEED
                        dy *= DRAG_SPEED
                        smooth_dx += DRAG_SMOOTH * (dx - smooth_dx)
                        smooth_dy += DRAG_SMOOTH * (dy - smooth_dy)
                        dx, dy = smooth_dx, smooth_dy
                    else:
                        smooth_dx = smooth_dy = 0.0

                    if not frozen:
                        mouse.move_relative(dx, dy)
                else:
                    smooth_dx = smooth_dy = 0.0
                    mapper.reset()

                # 5. Action souris (clic, drag, scroll à vitesse variable...)
                mouse.handle(gesture, scroll_delta=scroll_speed)

                # Nom affiché à l'écran
                shown_name = gesture.name
                if gesture == Gesture.NONE and moving_allowed:
                    shown_name = "MOVE"
                if gesture == Gesture.NONE and recognizer.scroll_ready:
                    shown_name = "SCROLL_READY"
                if recognizer.drag_pending:
                    shown_name = "DRAG_PENDING"
                elif recognizer.drag_releasing:
                    shown_name = "DRAG_RELEASE"

            else:
                gesture = Gesture.NONE
                shown_name = "NONE"
                smooth_dx = smooth_dy = 0.0
                mapper.reset()      # évite un saut du curseur au retour de la main
                recognizer.reset()  # évite un faux scroll au retour de la main
                mouse.handle(Gesture.NONE)  # relâche le bouton si un drag était en cours

            # 7. Affichage de l'action en cours (toujours actif)
            overlay.update(shown_name)

            # 8. Fenêtre webcam de debug (optionnelle, --debug uniquement)
            if args.debug:
                displayed_hand = landmarks[0] if landmarks else None
                if not show_debug_window(frame, shown_name,
                                          landmarks=displayed_hand, fps=fps_smooth):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        mouse.release()
        release_camera(cap)
        detector.close()
        overlay.close()
        if args.debug:
            close_debug_window()


if __name__ == "__main__":
    main()
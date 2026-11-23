"""
C'est le fichier qui lance tout et qui fait le lien entre les autres.
Il ne fait aucun calcul lui-même : il appelle juste, dans l'ordre,
les fonctions des autres fichiers.

Boucle principale : orchestre capture -> detection -> mapping -> gestures -> controller -> ui.
"""

import csv
import cv2
import math
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

# Accélération du curseur en déplacement normal : lent = précis, rapide = grande
# portée. La vitesse est mesurée en pixels par image avant accélération.
ACCEL_V_LOW = 3.0        # en dessous : gain minimal
ACCEL_V_HIGH = 40.0      # au dessus : gain maximal
ACCEL_GAIN_LOW = 0.6
ACCEL_GAIN_HIGH = 1.6
SENS_Y = 1.25            # la main bouge moins en vertical qu'en horizontal

# Passe à True pour afficher chaque clic détecté avec l'état des doigts
# (1 = tendu, 0 = replié) et la mesure du pouce. Utile pour régler les seuils.
DEBUG_CLICKS = False


def accelerate(dx, dy):
    v = math.hypot(dx, dy)
    t = min(max((v - ACCEL_V_LOW) / (ACCEL_V_HIGH - ACCEL_V_LOW), 0.0), 1.0)
    gain = ACCEL_GAIN_LOW + (ACCEL_GAIN_HIGH - ACCEL_GAIN_LOW) * t
    return dx * gain, dy * gain * SENS_Y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Affiche la fenêtre de visualisation")
    parser.add_argument("--log", metavar="FICHIER.csv",
                        help="Enregistre une ligne par image (mesures objectif 1)")
    parser.add_argument("--filter", choices=["none", "ema", "oneeuro"], default="oneeuro",
                        help="Lissage du curseur (défaut : oneeuro)")
    parser.add_argument("--min-cutoff", type=float, default=0.4,
                        help="One Euro : plus bas = plus de lissage (défaut 0.4)")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="One Euro : réactivité aux mouvements rapides (défaut 0.3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialisation
    cap = open_camera()
    detector = HandDetector()
    recognizer = GestureRecognizer()
    mouse = MouseController()
    screen_w, screen_h = pyautogui.size()
    mapper = CoordinateMapper(screen_w, screen_h,
                              min_cutoff=args.min_cutoff, beta=args.beta,
                              filter_mode=args.filter)
    overlay = ActionOverlay()  # toujours actif, debug ou non

    # Suivi du FPS (lissé pour ne pas sauter dans tous les sens à l'affichage)
    prev_time = time.time()
    fps_smooth = 0.0

    # Lissage du déplacement pendant un drag
    smooth_dx = smooth_dy = 0.0

    # Journal de mesures (une ligne par image), voir --log
    log_file = log_writer = None
    t_start = time.perf_counter()
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["t_s", "dt_ms", "fps", "hand", "gesture",
                             "raw_x", "raw_y", "sm_x", "sm_y", "dx", "dy",
                             "cursor_x", "cursor_y", "detect_ms", "total_ms"])

    try:
        while True:
            # 1. Capture
            frame = get_frame(cap)
            if frame is None:
                continue
            t_frame = time.perf_counter()
            hand_present = 0
            raw_x = raw_y = sm_x = sm_y = ""
            log_dx = log_dy = 0.0

            # Effet miroir (voir MIRROR en haut du fichier)
            if MIRROR:
                frame = cv2.flip(frame, 1)

            # 2. Détection (synchrone, résultat immédiat pour cette frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb_frame)
            t_detect = time.perf_counter()

            # Calcul du FPS (lissage exponentiel pour un affichage stable)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps_smooth = instant_fps if fps_smooth == 0 else fps_smooth * 0.9 + instant_fps * 0.1

            if landmarks:
                hand_present = 1
                hand = landmarks[0]  # on prend la première main détectée

                # 3. Reconnaissance du geste
                gesture, moving_allowed, scroll_speed = recognizer.recognize(hand)

                if DEBUG_CLICKS and gesture in (Gesture.LEFT_CLICK, Gesture.RIGHT_CLICK,
                                                Gesture.DOUBLE_CLICK):
                    print(gesture.name,
                          {k: int(v) for k, v in recognizer.last_fingers.items()},
                          round(recognizer.last_thumb_ratio, 2))

                # 4. Déplacement du curseur : suit la base de l'index en
                #    posture pointeur, la base du majeur en posture drag
                #    (y compris pendant la confirmation du drag).
                if moving_allowed:
                    in_drag = gesture == Gesture.DRAG or recognizer.drag_pending
                    tracked_point = hand[9] if in_drag else hand[5]
                    dx, dy = mapper.get_relative_move(tracked_point.x, tracked_point.y)
                    frozen = recognizer.drag_pending or recognizer.drag_releasing
                    raw_x, raw_y = mapper.last_raw
                    sm_x, sm_y = mapper.last_smooth

                    if gesture == Gesture.DRAG:
                        dx *= DRAG_SPEED
                        dy *= DRAG_SPEED * SENS_Y
                        smooth_dx += DRAG_SMOOTH * (dx - smooth_dx)
                        smooth_dy += DRAG_SMOOTH * (dy - smooth_dy)
                        dx, dy = smooth_dx, smooth_dy
                    else:
                        dx, dy = accelerate(dx, dy)
                        smooth_dx = smooth_dy = 0.0

                    if not frozen:
                        log_dx, log_dy = dx, dy
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
                if gesture == Gesture.NONE and recognizer.open_hand:
                    shown_name = "CLUTCH"
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

            # Fin du traitement : on mesure ici, avant l'affichage
            t_action = time.perf_counter()
            if log_writer:
                cx, cy = pyautogui.position()
                fmt = lambda v: "" if v == "" or v is None else round(v, 6)
                log_writer.writerow([
                    round(t_frame - t_start, 4), round(dt * 1000, 2),
                    round(fps_smooth, 2), hand_present, shown_name,
                    fmt(raw_x), fmt(raw_y), fmt(sm_x), fmt(sm_y),
                    round(log_dx, 2), round(log_dy, 2), cx, cy,
                    round((t_detect - t_frame) * 1000, 2),
                    round((t_action - t_frame) * 1000, 2)])

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
        if log_file:
            log_file.close()
        mouse.release()
        release_camera(cap)
        detector.close()
        overlay.close()
        if args.debug:
            close_debug_window()


if __name__ == "__main__":
    main()
"""
Ce fichier regarde la forme de la main et décide quel geste c'est.
Il dit juste "c'est un clic", il ne clique pas lui-même.

Tous les gestes sont basés sur le COMPTAGE DE DOIGTS TENDUS/REPLIÉS :

| Geste            | Doigts tendus (les autres repliés) |
|------------------|-------------------------------------|
| Déplacement      | Index seul                          |
| Scroll           | Index + Majeur (en V)               |
| Clic gauche      | Pouce seul (auriculaire ignoré)     |
| Clic droit       | Auriculaire seul                    |
| Double-clic      | Pouce seul, deux fois de suite      |
| Drag (glisser)   | Majeur seul                         |
"""

from enum import Enum
import time
import numpy as np


class Gesture(Enum):
    NONE = 0
    LEFT_CLICK = 1
    RIGHT_CLICK = 2
    SCROLL_UP = 3
    SCROLL_DOWN = 4
    DRAG = 5
    DOUBLE_CLICK = 6


# Indices des landmarks MediaPipe
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

# Un doigt est tendu si l'angle MCP-PIP-TIP dépasse cette valeur (degrés)
FINGER_EXTENDED_ANGLE = 140

# Pendant le déplacement (index), la posture est tenue avec des seuils
# tolérants : en pointant vers le bas, le doigt s'incline vers la caméra et
# le majeur "apparaît" tendu (faux scroll). Tant qu'on pointe, l'index reste
# tendu au-dessus de 120 degrés, et le majeur ne compte comme tendu (donc
# scroll) qu'au-dessus de 160 degrés.
POINTER_KEEP_INDEX_ANGLE = 120
POINTER_KEEP_MIDDLE_ANGLE = 160

# Pendant un drag, la posture est tenue avec des seuils plus tolérants
# (hystérésis) : majeur encore "assez tendu", index encore "assez replié".
# Évite que le curseur s'arrête quand la main s'incline en bougeant.
DRAG_KEEP_MIDDLE_ANGLE = 120
DRAG_KEEP_INDEX_ANGLE = 165

# Pouce : distance bout du pouce -> base de l'index, divisée par la taille
# de la main. Au-dessus de OPEN : tendu. En dessous de CLOSE : replié.
# Entre les deux : on garde l'état précédent (évite le clignotement).
THUMB_OPEN_RATIO = 0.40
THUMB_CLOSE_RATIO = 0.32

# Frames consécutives où une posture doit être détectée avant validation
CONFIRM_FRAMES = 3
CLICK_CONFIRM_FRAMES = 7

# Double-clic = deux clics gauches (pouce) rapprochés.
# Écart inférieur à DOUBLE_TAP_MIN : Windows reconnaît déjà le double-clic
# avec deux clics simples. Entre MIN et MAX : on envoie un vrai double-clic.
# Au-delà de MAX : c'est un nouveau clic simple.
DOUBLE_TAP_MIN = 0.55
DOUBLE_TAP_MAX = 1.5

# Après une main ouverte, on bloque les clics pendant ce délai (secondes)
OPEN_HAND_SUPPRESS = 0.5

# Drag
DRAG_CONFIRM_FRAMES = 6      # images pour valider le début du drag
DRAG_RELEASE_FRAMES = 5      # images sans posture tolérées avant de relâcher
DRAG_SUPPRESS_AFTER_CLICK = 0.4

# Scroll "autoscroll" (voir _recognize_scroll)
SCROLL_ARM_FRAMES = 5
SCROLL_DEADZONE = 0.03
SCROLL_RANGE = 0.15
SCROLL_BASE_SPEED = 4
SCROLL_MAX_SPEED = 30
SCROLL_NATUREL = True        # True : au-dessus du neutre, la page descend


class GestureRecognizer:
    def __init__(self):
        self._scroll_anchor_y = None
        self._scroll_arm = 0
        self.scroll_ready = False

        self._left_streak = 0
        self._right_streak = 0
        self._drag_streak = 0

        self._left_armed = True
        self._right_armed = True
        # Verrou : après un clic, aucun autre clic tant que la main n'est
        # pas sortie de toutes les postures de clic (évite l'écho
        # "double-clic puis clic gauche" quand on relâche le pouce).
        self._click_locked = False

        self._last_click_time = 0.0
        self._last_left_time = 0.0
        self._last_open_hand_time = 0.0

        self._dragging = False
        self._drag_lost = 0

        self._thumb_open = False
        self.last_fingers = {}
        self.last_angles = {}
        self.last_thumb_ratio = 0.0
        self.open_hand = False
        self._pointing = False

    def reset(self):
        """À appeler quand la main disparaît de l'image."""
        self._scroll_anchor_y = None
        self._scroll_arm = 0
        self.scroll_ready = False
        self._left_streak = 0
        self._right_streak = 0
        self._drag_streak = 0
        self._left_armed = True
        self._right_armed = True
        self._click_locked = False
        self._last_click_time = 0.0
        self._last_left_time = 0.0
        self._last_open_hand_time = 0.0
        self._dragging = False
        self._drag_lost = 0
        self._thumb_open = False
        self.open_hand = False
        self._pointing = False

    # --- états du drag, lus par main.py ---
    @property
    def drag_pending(self):
        """Posture drag en cours de confirmation."""
        return (not self._dragging) and self._drag_streak > 0

    @property
    def drag_releasing(self):
        """Drag actif mais posture perdue depuis quelques images."""
        return self._dragging and self._drag_lost > 0

    # --- géométrie ---
    def _to_points(self, landmarks):
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

    def _angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _finger_angle(self, points, mcp_idx, pip_idx, tip_idx):
        return self._angle(points[mcp_idx], points[pip_idx], points[tip_idx])

    def _thumb_extended(self, points):
        scale = np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]) + 1e-6
        ratio = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_MCP]) / scale)
        self.last_thumb_ratio = ratio
        if self._thumb_open:
            self._thumb_open = ratio > THUMB_CLOSE_RATIO
        else:
            self._thumb_open = ratio > THUMB_OPEN_RATIO
        return self._thumb_open

    def _finger_ups(self, points):
        self.last_angles = {
            'index': self._finger_angle(points, INDEX_MCP, INDEX_PIP, INDEX_TIP),
            'middle': self._finger_angle(points, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            'ring': self._finger_angle(points, RING_MCP, RING_PIP, RING_TIP),
            'pinky': self._finger_angle(points, PINKY_MCP, PINKY_PIP, PINKY_TIP),
        }
        a = self.last_angles
        return {
            'thumb': self._thumb_extended(points),
            'index': a['index'] > FINGER_EXTENDED_ANGLE,
            'middle': a['middle'] > FINGER_EXTENDED_ANGLE,
            'ring': a['ring'] > FINGER_EXTENDED_ANGLE,
            'pinky': a['pinky'] > FINGER_EXTENDED_ANGLE,
        }

    def _exactly(self, f, up_names):
        up_names = set(up_names)
        for name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            if f[name] != (name in up_names):
                return False
        return True

    # --- postures ---
    def is_pointer_pose(self, f):
        return f['index'] and not f['middle']

    def is_scroll_pose(self, f):
        return f['index'] and f['middle'] and not f['ring'] and not f['pinky']

    def is_left_click_pose(self, f):
        # Pouce tendu, index/majeur/annulaire repliés. L'auriculaire est
        # ignoré : il se lève souvent tout seul avec le pouce.
        return f['thumb'] and not f['index'] and not f['middle'] and not f['ring']

    def is_right_click_pose(self, f):
        return self._exactly(f, {'pinky'})

    def is_drag_pose(self, f):
        return f['middle'] and not f['index']

    def is_drag_pose_relaxed(self):
        """Version tolérante, utilisée seulement pendant un drag."""
        a = self.last_angles
        return (a['middle'] > DRAG_KEEP_MIDDLE_ANGLE
                and a['index'] < DRAG_KEEP_INDEX_ANGLE)

    # --- drag ---
    def _update_drag(self, drag_pose):
        just_clicked = (time.time() - self._last_click_time) < DRAG_SUPPRESS_AFTER_CLICK
        if not self._dragging:
            if self._drag_streak >= DRAG_CONFIRM_FRAMES and not just_clicked:
                self._dragging = True
                self._drag_lost = 0
        else:
            self._drag_lost = 0 if drag_pose else self._drag_lost + 1
            if self._drag_lost > DRAG_RELEASE_FRAMES:
                self._dragging = False

    # --- scroll ---
    def _recognize_scroll(self, points, f):
        """Autoscroll : un point neutre est fixé après SCROLL_ARM_FRAMES
        images de posture stable ; plus la main s'en éloigne, plus ça
        scrolle vite ; revenir au neutre arrête le scroll."""
        if not self.is_scroll_pose(f):
            self._scroll_anchor_y = None
            self._scroll_arm = 0
            self.scroll_ready = False
            return Gesture.NONE, 0.0

        self.scroll_ready = True
        y = (points[INDEX_TIP][1] + points[MIDDLE_TIP][1]) / 2

        if self._scroll_arm < SCROLL_ARM_FRAMES:
            self._scroll_arm += 1
            self._scroll_anchor_y = y
            return Gesture.NONE, 0.0

        d = self._scroll_anchor_y - y  # positif : main montée
        dist = abs(d) - SCROLL_DEADZONE
        if dist <= 0:
            return Gesture.NONE, 0.0

        t = min(dist / (SCROLL_RANGE - SCROLL_DEADZONE), 1.0)
        speed = SCROLL_BASE_SPEED + (SCROLL_MAX_SPEED - SCROLL_BASE_SPEED) * t * t

        main_monte = d > 0
        page_monte = (not main_monte) if SCROLL_NATUREL else main_monte
        gesture = Gesture.SCROLL_UP if page_monte else Gesture.SCROLL_DOWN
        return gesture, speed

    # --- reconnaissance ---
    def recognize(self, landmarks):
        """
        Renvoie (gesture, moving_allowed, scroll_speed).
        moving_allowed : True en posture déplacement (index) ou drag (majeur).
        """
        if landmarks is None:
            self.reset()
            return Gesture.NONE, False, 0.0

        now = time.time()
        points = self._to_points(landmarks)
        f = self._finger_ups(points)
        self.last_fingers = f

        # Version de f utilisée pour le pointeur et le scroll : tolérante
        # tant qu'on est déjà en train de pointer (voir POINTER_KEEP_*).
        f_ptr = dict(f)
        if self._pointing:
            a = self.last_angles
            f_ptr['index'] = a['index'] > POINTER_KEEP_INDEX_ANGLE
            f_ptr['middle'] = a['middle'] > POINTER_KEEP_MIDDLE_ANGLE

        pointer_pose = self.is_pointer_pose(f_ptr)
        self._pointing = pointer_pose
        drag_pose = self.is_drag_pose(f)
        if self._dragging and not drag_pose:
            drag_pose = self.is_drag_pose_relaxed()
        left_pose = self.is_left_click_pose(f)
        right_pose = self.is_right_click_pose(f)

        # Tant qu'on pointe (index encore "tendu" avec les seuils tolérants),
        # aucun clic ne peut démarrer : un index qui s'incline ne doit pas
        # être pris pour un pouce seul.
        if pointer_pose:
            left_pose = False
            right_pose = False

        moving_allowed = pointer_pose or drag_pose

        # Main ouverte : on note l'instant (bloque les clics un court moment)
        self.open_hand = bool(f['index'] and f['middle'] and f['ring'] and f['pinky'])
        if self.open_hand:
            self._last_open_hand_time = now

        # Confirmation sur plusieurs images (anti-bruit)
        self._left_streak = self._left_streak + 1 if left_pose else 0
        self._right_streak = self._right_streak + 1 if right_pose else 0
        self._drag_streak = self._drag_streak + 1 if drag_pose else 0

        # Ré-armement dès que la posture est relâchée
        if not left_pose:
            self._left_armed = True
        if not right_pose:
            self._right_armed = True

        # Levée du verrou quand plus aucune posture de clic n'est tenue
        if not (left_pose or right_pose):
            self._click_locked = False

        # Main ouverte récente : on désarme les clics
        if (now - self._last_open_hand_time) < OPEN_HAND_SUPPRESS:
            self._left_armed = False
            self._right_armed = False

        # Drag : prioritaire, aucun clic possible pendant un drag
        self._update_drag(drag_pose)
        if self._dragging:
            self.scroll_ready = False
            return Gesture.DRAG, moving_allowed, 0.0

        # Clics
        if not self._click_locked:
            if self._left_streak >= CLICK_CONFIRM_FRAMES and self._left_armed:
                self._left_armed = False
                self._click_locked = True
                self._last_click_time = now
                gap = now - self._last_left_time
                if DOUBLE_TAP_MIN <= gap <= DOUBLE_TAP_MAX:
                    self._last_left_time = 0.0
                    return Gesture.DOUBLE_CLICK, moving_allowed, 0.0
                self._last_left_time = now
                return Gesture.LEFT_CLICK, moving_allowed, 0.0

            if self._right_streak >= CLICK_CONFIRM_FRAMES and self._right_armed:
                self._right_armed = False
                self._click_locked = True
                self._last_click_time = now
                return Gesture.RIGHT_CLICK, moving_allowed, 0.0

        scroll_gesture, scroll_speed = self._recognize_scroll(points, f_ptr)
        if scroll_gesture != Gesture.NONE:
            return scroll_gesture, moving_allowed, scroll_speed

        return Gesture.NONE, moving_allowed, 0.0
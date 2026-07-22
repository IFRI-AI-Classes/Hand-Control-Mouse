"""
Ce fichier regarde la forme de la main et décide quel geste c'est.
Il dit juste "c'est un clic", il ne clique pas lui-même.

Tous les gestes sont basés sur le COMPTAGE DE DOIGTS TENDUS/REPLIÉS
(aucun pincement/distance entre doigts, trop fragile en 2D) :

| Geste            | Doigts tendus (les autres repliés) |
|------------------|-------------------------------------|
| Déplacement      | Index seul                          |
| Scroll           | Index + Majeur (en V)               |
| Clic gauche      | Pouce seul                          |
| Clic droit       | Auriculaire seul                    |
| Double-clic      | Pouce + Auriculaire                 |
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


# Indices des landmarks MediaPipe utilisés ici
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# Nombre de frames consécutives où une posture doit être détectée avant
# d'être validée. Filtre le bruit/tremblement de la détection (une seule
# frame "limite" ne suffit plus à déclencher un geste).
CONFIRM_FRAMES = 3

# Le drag exige une confirmation un peu plus longue que les clics : ça
# laisse le temps à la posture de la main de bien se stabiliser en
# passant d'une main neutre à "majeur seul tendu", plutôt que de capter
# une configuration intermédiaire qui ressemble à un autre geste.
DRAG_CONFIRM_FRAMES = 6

# Si un clic (simple, droit ou double) vient de se déclencher, on bloque
# le démarrage d'un drag pendant ce délai : un vrai drag part d'une main
# neutre, pas juste après un clic - ça évite qu'un double-clic parasite,
# capté pendant la transition vers la posture drag, n'ouvre le fichier/
# dossier juste avant que le drag ne prenne le relais.
DRAG_SUPPRESS_AFTER_CLICK = 0.4

# Vitesse de scroll, façon "manette/joystick" (voir _recognize_scroll)
SCROLL_BASE_SPEED = 4        # vitesse minimale, main immobile sur le point de référence
SCROLL_MAX_SPEED = 30        # vitesse plafond
SCROLL_DISTANCE_GAIN = 250   # sensibilité : distance au point de référence -> vitesse
SCROLL_SMOOTHING = 0.3       # lissage (0-1) : plus haut = plus réactif, plus bas = plus doux


class GestureRecognizer:
    def __init__(self):
        # État du "joystick" de scroll : point de référence et vitesse lissée
        self._scroll_anchor_y = None
        self._scroll_speed_smooth = SCROLL_BASE_SPEED

        # Compteurs de confirmation (anti-bruit) pour chaque geste "one-shot"
        self._left_streak = 0
        self._right_streak = 0
        self._double_streak = 0
        self._drag_streak = 0

        # Empêche un clic de se répéter tant que la posture reste tenue :
        # il faut relâcher avant de pouvoir redéclencher.
        self._left_armed = True
        self._right_armed = True
        self._double_armed = True

        # Horodatage du dernier clic déclenché, pour bloquer temporairement
        # le drag juste après (voir DRAG_SUPPRESS_AFTER_CLICK).
        self._last_click_time = 0.0

    def reset(self):
        """À appeler quand la main disparaît de l'image, pour ne pas
        comparer avec une ancienne position au retour de la main."""
        self._scroll_anchor_y = None
        self._scroll_speed_smooth = SCROLL_BASE_SPEED
        self._left_streak = 0
        self._right_streak = 0
        self._double_streak = 0
        self._drag_streak = 0
        self._left_armed = True
        self._right_armed = True
        self._double_armed = True
        self._last_click_time = 0.0

    def _to_points(self, landmarks):
        return np.array([(lm.x, lm.y) for lm in landmarks])

    def _hand_scale(self, points):
        """Taille de référence de la main (poignet -> base du majeur)."""
        return float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST])) + 1e-6

    def _finger_extended(self, points, tip_idx, pip_idx):
        wrist = points[WRIST]
        tip_dist = np.linalg.norm(points[tip_idx] - wrist)
        pip_dist = np.linalg.norm(points[pip_idx] - wrist)
        return tip_dist > pip_dist

    def _finger_ups(self, points):
        """État tendu/replié de chaque doigt, pouce inclus."""
        return {
            'thumb': self._finger_extended(points, THUMB_TIP, THUMB_IP),
            'index': self._finger_extended(points, INDEX_TIP, INDEX_PIP),
            'middle': self._finger_extended(points, MIDDLE_TIP, MIDDLE_PIP),
            'ring': self._finger_extended(points, RING_TIP, RING_PIP),
            'pinky': self._finger_extended(points, PINKY_TIP, PINKY_PIP),
        }

    def _exactly(self, f, up_names):
        """Vrai si UNIQUEMENT les doigts de up_names sont tendus (parmi
        les 5, pouce inclus), tous les autres repliés. Utilisé pour les
        clics, où le pouce est le doigt actif ou doit être explicitement
        replié pour éviter toute confusion."""
        up_names = set(up_names)
        for name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            if f[name] != (name in up_names):
                return False
        return True

    def is_pointer_pose(self, f):
        """Index tendu, majeur replié (annulaire/auriculaire/pouce
        ignorés) : déplace la souris. Ne dépend que d'index/majeur pour
        rester fiable même si l'annulaire/auriculaire ne sont pas
        parfaitement repliés (geste peu naturel à isoler)."""
        return f['index'] and not f['middle']

    def is_scroll_pose(self, f):
        """Index ET majeur tendus (en V), peu importe annulaire/
        auriculaire/pouce : scroll."""
        return f['index'] and f['middle']

    def is_left_click_pose(self, f):
        """Pouce seul tendu : clic gauche."""
        return self._exactly(f, {'thumb'})

    def is_right_click_pose(self, f):
        """Auriculaire seul tendu, pouce replié : clic droit."""
        return self._exactly(f, {'pinky'})

    def is_double_click_pose(self, f):
        """Pouce + auriculaire tendus : double-clic."""
        return self._exactly(f, {'thumb', 'pinky'})

    def is_drag_pose(self, f):
        """Majeur tendu, index replié (annulaire/auriculaire/pouce
        ignorés) : glisser (drag)."""
        return f['middle'] and not f['index']

    def _is_oriented_up(self, points):
        """True si le "V" du scroll pointe vers le haut de l'écran
        (posture normale), False s'il pointe vers le bas (main inversée)."""
        wrist_y = points[WRIST][1]
        fingertip_y = (points[INDEX_TIP][1] + points[MIDDLE_TIP][1]) / 2.0
        return fingertip_y < wrist_y  # y plus petit = plus haut à l'écran

    def _recognize_scroll(self, points, f):
        """
        Renvoie (gesture, speed) :
          - gesture : SCROLL_UP si le V pointe vers le haut, SCROLL_DOWN
            s'il pointe vers le bas (recalculé à chaque frame), NONE hors
            posture.
          - speed : vitesse façon joystick : un point de référence est
            fixé à l'entrée en posture ; plus tu t'éloignes verticalement
            de ce point, plus la vitesse augmente ; revenir dessus la
            ramène à la vitesse de base.
        """
        if not self.is_scroll_pose(f):
            self._scroll_anchor_y = None
            self._scroll_speed_smooth = SCROLL_BASE_SPEED
            return Gesture.NONE, 0.0

        current_y = points[MIDDLE_TIP][1]

        if self._scroll_anchor_y is None:
            self._scroll_anchor_y = current_y

        distance = abs(current_y - self._scroll_anchor_y)

        raw_speed = SCROLL_BASE_SPEED + SCROLL_DISTANCE_GAIN * distance
        raw_speed = min(raw_speed, SCROLL_MAX_SPEED)

        self._scroll_speed_smooth += SCROLL_SMOOTHING * (raw_speed - self._scroll_speed_smooth)

        gesture = Gesture.SCROLL_UP if self._is_oriented_up(points) else Gesture.SCROLL_DOWN
        return gesture, self._scroll_speed_smooth

    def recognize(self, landmarks):
        """
        Renvoie un tuple (gesture, moving_allowed, scroll_speed) :
          - gesture : le geste détecté (Gesture.NONE si rien)
          - moving_allowed : True si la main est en posture "déplacement"
            (index seul) OU "drag" (majeur seul) : ce sont les deux seuls
            gestes qui déplacent le curseur.
          - scroll_speed : vitesse à appliquer si gesture est SCROLL_UP
            ou SCROLL_DOWN (0.0 sinon).
        """
        if landmarks is None:
            self.reset()
            return Gesture.NONE, False, 0.0

        points = self._to_points(landmarks)
        f = self._finger_ups(points)

        pointer_pose = self.is_pointer_pose(f)
        drag_pose = self.is_drag_pose(f)
        left_pose = self.is_left_click_pose(f)
        right_pose = self.is_right_click_pose(f)
        double_pose = self.is_double_click_pose(f)

        moving_allowed = pointer_pose or drag_pose

        # --- Confirmation sur plusieurs frames (anti-bruit) ---
        self._left_streak = self._left_streak + 1 if left_pose else 0
        self._right_streak = self._right_streak + 1 if right_pose else 0
        self._double_streak = self._double_streak + 1 if double_pose else 0
        self._drag_streak = self._drag_streak + 1 if drag_pose else 0

        # Ré-arme chaque clic dès que la posture correspondante est relâchée
        if not left_pose:
            self._left_armed = True
        if not right_pose:
            self._right_armed = True
        if not double_pose:
            self._double_armed = True

        # Double-clic vérifié en premier (posture la plus spécifique)
        if self._double_streak >= CONFIRM_FRAMES and self._double_armed:
            self._double_armed = False
            self._last_click_time = time.time()
            return Gesture.DOUBLE_CLICK, moving_allowed, 0.0

        if self._left_streak >= CONFIRM_FRAMES and self._left_armed:
            self._left_armed = False
            self._last_click_time = time.time()
            return Gesture.LEFT_CLICK, moving_allowed, 0.0

        if self._right_streak >= CONFIRM_FRAMES and self._right_armed:
            self._right_armed = False
            self._last_click_time = time.time()
            return Gesture.RIGHT_CLICK, moving_allowed, 0.0

        # Drag : continu tant que la posture est tenue (pas de "armed",
        # on doit pouvoir glisser aussi longtemps qu'on garde le geste).
        # Confirmation plus longue (DRAG_CONFIRM_FRAMES) pour laisser la
        # posture se stabiliser, et bloqué juste après un clic (transition
        # ambiguë entre postures) pour éviter un double-clic parasite qui
        # ouvrirait le fichier/dossier juste avant que le drag démarre.
        just_clicked = (time.time() - self._last_click_time) < DRAG_SUPPRESS_AFTER_CLICK
        if self._drag_streak >= DRAG_CONFIRM_FRAMES and not just_clicked:
            return Gesture.DRAG, moving_allowed, 0.0

        scroll_gesture, scroll_speed = self._recognize_scroll(points, f)
        if scroll_gesture != Gesture.NONE:
            return scroll_gesture, moving_allowed, scroll_speed

        return Gesture.NONE, moving_allowed, 0.0
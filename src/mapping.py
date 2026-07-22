"""
Ce fichier transforme la position de la main en position sur l'écran.
Il rend aussi le mouvement plus fluide (moins de tremblement).
Il ne bouge jamais la souris lui-même, ça c'est le rôle de controller.py.
"""

import math
import time


class OneEuroFilter:
    """
    Filtre "One Euro" : lisse une coordonnée bruitée (x ou y) tout en
    restant réactif sur les mouvements rapides.

    Le principe : plus la main bouge vite, moins on lisse (pour ne pas
    avoir de latence) ; plus la main est presque immobile, plus on lisse
    fort (pour supprimer les tremblements). C'est le filtre standard
    utilisé pour le suivi de curseur (bien plus efficace qu'une simple
    moyenne mobile).
    """

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        """Réinitialise le filtre (utile si la main disparaît puis revient)."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x, timestamp=None):
        """Renvoie la valeur lissée pour la nouvelle mesure x."""
        if timestamp is None:
            timestamp = time.time()

        if self._t_prev is None:
            dt = 1.0 / self.freq
        else:
            dt = timestamp - self._t_prev
            if dt <= 0:
                dt = 1.0 / self.freq

        freq = 1.0 / dt
        self._t_prev = timestamp

        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        # Vitesse filtrée (dérivée du signal)
        a_d = self._alpha(self.d_cutoff, freq)
        dx = (x - self._x_prev) * freq
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Le cutoff s'adapte à la vitesse : plus ça bouge vite, moins on lisse
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, freq)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat

        return x_hat


class CoordinateMapper:
    """
    Convertit la position normalisée (0-1) d'un landmark MediaPipe en
    position pixel sur l'écran, avec :
      - une "zone active" réduite dans l'image webcam, pour pouvoir
        atteindre les bords de l'écran sans avoir à mettre la main
        tout au bord du champ de la caméra
      - un lissage (One Euro Filter) appliqué séparément en x et en y
    """

    def __init__(self, screen_w, screen_h, margin=0.15,
                 min_cutoff=0.8, beta=0.4, freq=30.0):
        """
        screen_w, screen_h : résolution de l'écran (pixels)
        margin : fraction de l'image webcam ignorée sur chaque bord
                 (0.15 -> on utilise la zone [0.15, 0.85] de l'image ;
                 0 = pas de zone active, toute l'image est utilisée)
        min_cutoff, beta : réglages du lissage
            - min_cutoff plus bas => plus de lissage sur les mouvements lents
            - beta plus élevé => réactivité plus forte sur les mouvements rapides
        freq : fréquence approximative de la boucle (fps)
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.margin = margin

        self.filter_x = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)

    def _apply_active_region(self, value):
        """Étend la zone [margin, 1-margin] vers [0, 1], puis clamp."""
        span = 1.0 - 2 * self.margin
        if span <= 0:
            return min(max(value, 0.0), 1.0)
        value = (value - self.margin) / span
        return min(max(value, 0.0), 1.0)

    def to_screen(self, x_norm, y_norm):
        """
        x_norm, y_norm : coordonnées normalisées (0-1) d'un landmark
        (typiquement le bout de l'index, landmark 8).

        Renvoie (x, y) en pixels écran, lissés et prêts à être passés
        à controller.py.
        """
        x_active = self._apply_active_region(x_norm)
        y_active = self._apply_active_region(y_norm)

        x_screen = x_active * self.screen_w
        y_screen = y_active * self.screen_h

        now = time.time()
        x_smooth = self.filter_x.filter(x_screen, now)
        y_smooth = self.filter_y.filter(y_screen, now)

        return x_smooth, y_smooth

    def reset(self):
        """À appeler quand la main disparaît, pour éviter un 'saut' du
        curseur au prochain mouvement détecté (le filtre repart de zéro
        plutôt que d'interpoler depuis une ancienne position)."""
        self.filter_x.reset()
        self.filter_y.reset()
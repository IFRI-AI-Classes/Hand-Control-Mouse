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
    mouvement de curseur, façon PAVÉ TACTILE :
      - le curseur avance de la même quantité que le doigt (mouvement
        RELATIF), au lieu de sauter vers une position absolue à l'écran
      - une zone morte ignore les micro-tremblements quand la main est
        quasi immobile
      - un lissage (One Euro Filter) est appliqué avant de calculer le
        déplacement, pour un mouvement fluide
    """

    def __init__(self, screen_w, screen_h,
                 min_cutoff=0.4, beta=0.3, freq=30.0,
                 sensitivity=2.5, deadzone=0.0008):
        """
        screen_w, screen_h : résolution de l'écran (pixels)
        min_cutoff : plus bas => plus de lissage (moins de tremblement,
                     un peu plus de latence). Monte-le si tu trouves le
                     curseur trop "mou".
        beta : réactivité sur les mouvements rapides. Descends-le si le
               curseur est encore trop nerveux/sensible.
        sensitivity : multiplicateur du déplacement (comme la sensibilité
                      d'un pavé tactile). >1 = un petit geste déplace
                      beaucoup le curseur ; <1 = il faut bouger plus la
                      main pour un même déplacement à l'écran.
        deadzone : en dessous de ce seuil (unité normalisée 0-1), un
                   micro-mouvement est ignoré. Augmente-le si le curseur
                   dérive tout seul quand ta main est immobile.
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sensitivity = sensitivity
        self.deadzone = deadzone

        self.filter_x = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)

        self._prev_x = None
        self._prev_y = None

    def get_relative_move(self, x_norm, y_norm):
        """
        x_norm, y_norm : coordonnées normalisées (0-1) d'un landmark
        (typiquement le bout de l'index, landmark 8).

        Renvoie (dx, dy) en pixels : le déplacement à appliquer au
        curseur depuis sa position actuelle (comme un pavé tactile),
        prêt à passer à controller.move_relative(dx, dy).
        """
        now = time.time()
        x_smooth = self.filter_x.filter(x_norm, now)
        y_smooth = self.filter_y.filter(y_norm, now)

        if self._prev_x is None:
            self._prev_x, self._prev_y = x_smooth, y_smooth
            return 0, 0

        dx_norm = x_smooth - self._prev_x
        dy_norm = y_smooth - self._prev_y

        self._prev_x, self._prev_y = x_smooth, y_smooth

        # Zone morte : ignore les micro-mouvements résiduels
        if abs(dx_norm) < self.deadzone:
            dx_norm = 0.0
        if abs(dy_norm) < self.deadzone:
            dy_norm = 0.0

        dx = dx_norm * self.screen_w * self.sensitivity
        dy = dy_norm * self.screen_h * self.sensitivity

        return dx, dy

    def reset(self):
        """À appeler quand la main disparaît, pour repartir de zéro au
        lieu de créer un grand saut au retour de la main."""
        self.filter_x.reset()
        self.filter_y.reset()
        self._prev_x = None
        self._prev_y = None
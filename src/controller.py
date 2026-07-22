"""
C'est le seul fichier qui touche vraiment la souris de l'ordinateur.
Il bouge le curseur et fait les clics.
Tous les autres fichiers font juste des calculs, celui-ci agit pour de vrai.
"""
import pyautogui
import time
from gestures import Gesture

# Par défaut, pyautogui attend 0.1s après CHAQUE appel (moveTo, click...).
# Comme move() est appelé à chaque frame, ça plafonnait la boucle à ~10
# images/seconde. On désactive cette pause pour un mouvement fluide.
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # garde la sécurité coin d'écran

COOLDOWN = 0.5  # secondes entre deux actions
SAFE_MARGIN = 3  # pixels de marge pour ne jamais toucher pile un coin

class MouseController:
    def __init__(self):
        self.last_action_time = 0
        self.screen_w, self.screen_h = pyautogui.size()

    def _cooldown_ok(self):
        return time.time() - self.last_action_time > COOLDOWN

    def _reset_cooldown(self):
        self.last_action_time = time.time()

    def move(self, x, y):
        # Déplacement absolu (curseur envoyé à une position précise).
        x, y = self._clamp(x, y)
        pyautogui.moveTo(x, y)

    def move_relative(self, dx, dy):
        # Déplacement relatif, façon pavé tactile : le curseur avance de
        # (dx, dy) par rapport à sa position actuelle.
        if dx == 0 and dy == 0:
            return

        try:
            current_x, current_y = pyautogui.position()
        except Exception:
            return

        target_x, target_y = self._clamp(current_x + dx, current_y + dy)

        try:
            pyautogui.moveTo(target_x, target_y)
        except pyautogui.FailSafeException:
            # Sécurité : on ne plante jamais, on ignore juste ce déplacement.
            pass

    def _clamp(self, x, y):
        """Empêche le curseur d'atteindre pile un coin de l'écran
        (ce qui déclencherait le fail-safe de pyautogui)."""
        x = max(SAFE_MARGIN, min(self.screen_w - SAFE_MARGIN, x))
        y = max(SAFE_MARGIN, min(self.screen_h - SAFE_MARGIN, y))
        return x, y

    def handle(self, gesture, scroll_delta=None):
        # Reçoit un Gesture depuis gestures.py et exécute l'action
        if gesture == Gesture.DRAG:
            pyautogui.mouseDown()
            return  # rien d'autre à faire pendant un drag

        # Dès qu'on n'est plus en train de glisser, on s'assure que le
        # bouton est relâché - peu importe le geste qui suit (et pas
        # seulement Gesture.NONE), pour ne jamais rester "collé".
        pyautogui.mouseUp()

        if gesture == Gesture.LEFT_CLICK and self._cooldown_ok():
            pyautogui.click()
            self._reset_cooldown()

        elif gesture == Gesture.DOUBLE_CLICK and self._cooldown_ok():
            pyautogui.doubleClick()
            self._reset_cooldown()

        elif gesture == Gesture.RIGHT_CLICK and self._cooldown_ok():
            pyautogui.click(button='right')
            self._reset_cooldown()

        elif gesture == Gesture.SCROLL_UP:
            # Pas de cooldown ici : le scroll doit être continu, avec une
            # vitesse variable (scroll_delta), pas une impulsion isolée.
            speed = scroll_delta if scroll_delta else 3
            pyautogui.scroll(int(round(speed)))

        elif gesture == Gesture.SCROLL_DOWN:
            speed = scroll_delta if scroll_delta else 3
            pyautogui.scroll(-int(round(speed)))
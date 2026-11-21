"""
C'est le seul fichier qui touche vraiment la souris de l'ordinateur.
Il bouge le curseur et fait les clics.
Tous les autres fichiers font juste des calculs, celui-ci agit pour de vrai.
"""
import pyautogui
import time
import sys
import ctypes
from gestures import Gesture

# Par défaut, pyautogui attend 0.1s après CHAQUE appel (moveTo, click...).
# Comme move() est appelé à chaque frame, ça plafonnait la boucle à ~10
# images/seconde. On désactive cette pause pour un mouvement fluide.
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # garde la sécurité coin d'écran

COOLDOWN = 0.5  # secondes entre deux actions
SAFE_MARGIN = 3  # pixels de marge pour ne jamais toucher pile un coin
# Windows : pyautogui compte le scroll en 1/120 de cran de molette.
# Linux/Mac : en crans entiers. On convertit pour avoir un scroll utilisable.
SCROLL_UNIT = 10 if sys.platform == "win32" else 0.1

class MouseController:
    def __init__(self):
        self.last_action_time = 0
        self.screen_w, self.screen_h = pyautogui.size()
        self._button_down = False  # True si on est en train de glisser (drag)
        self._rem_x = 0.0
        self._rem_y = 0.0

    def _cooldown_ok(self):
        return time.time() - self.last_action_time > COOLDOWN

    def _reset_cooldown(self):
        self.last_action_time = time.time()

    def move(self, x, y):
        # Déplacement absolu (curseur envoyé à une position précise).
        x, y = self._clamp(x, y)
        self._move_to(x, y)

    def _move_to(self, x, y):
        # Pendant un drag sous Windows, il faut de vrais événements de
        # mouvement : SetCursorPos (utilisé par pyautogui.moveTo) n'en
        # génère pas, donc Windows ne voit pas le glisser avancer.
        if sys.platform == "win32" and self._button_down:
            ax = int(x * 65535 / (self.screen_w - 1))
            ay = int(y * 65535 / (self.screen_h - 1))
            ctypes.windll.user32.mouse_event(0x8001, ax, ay, 0, 0)  # MOVE | ABSOLUTE
        else:
            pyautogui.moveTo(x, y)

    def move_relative(self, dx, dy):
        # On garde la fraction de pixel non utilisée pour l'image suivante :
        # sans ça, les petits mouvements (< 1 px) seraient perdus.
        dx += self._rem_x
        dy += self._rem_y
        ix, iy = round(dx), round(dy)
        self._rem_x, self._rem_y = dx - ix, dy - iy
        if ix == 0 and iy == 0:
            return

        try:
            current_x, current_y = pyautogui.position()
        except Exception:
            return

        target_x, target_y = self._clamp(current_x + ix, current_y + iy)

        try:
            self._move_to(target_x, target_y)
        except pyautogui.FailSafeException:
            pass

    def handle(self, gesture, scroll_delta=None):
        if gesture == Gesture.DRAG:
            if not self._button_down:
                pyautogui.mouseDown()
                self._button_down = True
            return

        if self._button_down:
            pyautogui.mouseUp()
            self._button_down = False

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
            speed = scroll_delta if scroll_delta else 3
            pyautogui.scroll(int(round(speed * SCROLL_UNIT)))

        elif gesture == Gesture.SCROLL_DOWN:
            speed = scroll_delta if scroll_delta else 3
            pyautogui.scroll(-int(round(speed * SCROLL_UNIT)))


    def release(self):
        if self._button_down:
            pyautogui.mouseUp()
            self._button_down = False
    
    def _clamp(self, x, y):
            """Empêche le curseur d'atteindre pile un coin de l'écran
            (ce qui déclencherait le fail-safe de pyautogui)."""
            x = max(SAFE_MARGIN, min(self.screen_w - SAFE_MARGIN, x))
            y = max(SAFE_MARGIN, min(self.screen_h - SAFE_MARGIN, y))
            return x, y  
"""
C'est le seul fichier qui touche vraiment la souris de l'ordinateur.
Il bouge le curseur et fait les clics.
Tous les autres fichiers font juste des calculs, celui-ci agit pour de vrai.
"""
import pyautogui
import time
from gestures import Gesture

COOLDOWN = 0.5  # secondes entre deux actions

class MouseController:
    def __init__(self):
        self.last_action_time = 0

    def _cooldown_ok(self):
        return time.time() - self.last_action_time > COOLDOWN

    def _reset_cooldown(self):
        self.last_action_time = time.time()

    def move(self, x, y):
        # Pas de cooldown sur le déplacement, doit être fluide
        pyautogui.moveTo(x, y)

    def handle(self, gesture, scroll_delta=None):
        # Reçoit un Gesture depuis gestures.py et exécute l'action
        if gesture == Gesture.LEFT_CLICK and self._cooldown_ok():
            pyautogui.click()
            self._reset_cooldown()

        elif gesture == Gesture.RIGHT_CLICK and self._cooldown_ok():
            pyautogui.click(button='right')
            self._reset_cooldown()

        elif gesture == Gesture.SCROLL_UP and self._cooldown_ok():
            pyautogui.scroll(3)
            self._reset_cooldown()

        elif gesture == Gesture.SCROLL_DOWN and self._cooldown_ok():
            pyautogui.scroll(-3)
            self._reset_cooldown()

        elif gesture == Gesture.DRAG:
            pyautogui.mouseDown()

        elif gesture == Gesture.NONE:
            pyautogui.mouseUp()  # relâche si drag en cours
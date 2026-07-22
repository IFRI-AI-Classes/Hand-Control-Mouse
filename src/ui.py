"""
Ce fichier gère tout ce qui est affiché à l'utilisateur :
  - ActionOverlay : une petite fenêtre transparente, toujours au premier
    plan sur l'écran, qui affiche l'action/geste en cours. Active en
    permanence, que le mode --debug soit activé ou non.
  - show_debug_window : la fenêtre avec le flux webcam + le texte du
    geste, réservée au mode --debug.
Ne fait aucun calcul de position ni de geste.
"""

import threading
import queue
import tkinter as tk
import cv2

# Libellés affichés à l'utilisateur pour chaque geste (plus lisible que
# les noms internes de l'enum Gesture)
GESTURE_LABELS = {
    "NONE": "Aucune action",
    "LEFT_CLICK": "Clic gauche",
    "RIGHT_CLICK": "Clic droit",
    "SCROLL_UP": "Scroll haut",
    "SCROLL_DOWN": "Scroll bas",
    "DRAG": "Glisser (drag)",
}


def label_for(gesture_name):
    """Renvoie le texte à afficher pour un nom de geste donné."""
    return GESTURE_LABELS.get(gesture_name, gesture_name)


class ActionOverlay:
    """
    Fenêtre transparente sans bordure, toujours au premier plan, posée
    sur le bureau (indépendante de la fenêtre webcam). Affiche le nom de
    l'action en cours en temps réel.

    Tourne dans son propre thread pour ne jamais bloquer la boucle
    principale (capture/détection/souris).
    """

    def __init__(self, x=20, y=20):
        self._queue = queue.Queue()
        self._x = x
        self._y = y
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2)  # attend que la fenêtre soit créée

    def _run(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)        # pas de barre de titre
        self.root.attributes("-topmost", True)   # toujours au-dessus
        try:
            self.root.attributes("-alpha", 0.85)  # légère transparence
        except tk.TclError:
            pass  # pas supporté sur toutes les plateformes
        self.root.configure(bg="black")
        self.root.geometry(f"+{self._x}+{self._y}")

        self.label = tk.Label(
            self.root, text="Aucune action",
            fg="white", bg="black",
            font=("Segoe UI", 14, "bold"),
            padx=12, pady=6,
        )
        self.label.pack()

        self._ready.set()
        self._poll()
        self.root.mainloop()

    def _poll(self):
        try:
            while True:
                text = self._queue.get_nowait()
                self.label.config(text=text)
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    def update(self, gesture_name):
        """À appeler depuis la boucle principale à chaque frame, avec le
        nom du geste courant (ex: 'LEFT_CLICK' ou 'NONE')."""
        self._queue.put(label_for(gesture_name))

    def close(self):
        try:
            self.root.after(0, self.root.destroy)
        except Exception:
            pass


def show_debug_window(frame, gesture_name):
    """Affiche la fenêtre webcam de debug avec le geste en cours dessus.
    Renvoie False si l'utilisateur a appuyé sur 'q' (signal d'arrêt)."""
    cv2.putText(frame, label_for(gesture_name), (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Control Mouse - DEBUG", frame)
    return (cv2.waitKey(1) & 0xFF) != ord('q')


def close_debug_window():
    cv2.destroyAllWindows()
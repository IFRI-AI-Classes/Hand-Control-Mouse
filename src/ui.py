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
    "DOUBLE_CLICK": "Double-clic",
    "RIGHT_CLICK": "Clic droit",
    "SCROLL_UP": "Scroll haut",
    "SCROLL_DOWN": "Scroll bas",
    "DRAG": "Glisser (drag)",
}

# Squelette standard des 21 landmarks MediaPipe (paires d'indices reliées
# par un trait). Codé en dur ici pour ne pas dépendre de mediapipe dans
# ce fichier, qui ne s'occupe que d'affichage.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # majeur
    (9, 13), (13, 14), (14, 15), (15, 16),   # annulaire
    (13, 17), (17, 18), (18, 19), (19, 20),  # auriculaire
    (0, 17),                                  # base de la paume
]


def label_for(gesture_name):
    """Renvoie le texte à afficher pour un nom de geste donné."""
    return GESTURE_LABELS.get(gesture_name, gesture_name)


def draw_landmarks(frame, hand_landmarks):
    """Dessine les 21 points de la main et le squelette qui les relie,
    directement sur l'image (en pixels, à partir des coordonnées
    normalisées 0-1 de MediaPipe)."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


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


def show_debug_window(frame, gesture_name, landmarks=None, fps=None):
    """
    Affiche la fenêtre webcam de debug avec :
      - les landmarks de la main dessinés dessus (si détectée)
      - un bandeau d'état : geste courant, main détectée ou non, FPS
    Renvoie False si l'utilisateur a appuyé sur 'q' (signal d'arrêt).

    landmarks : liste des 21 landmarks de la main (ou None si aucune main
        détectée sur cette frame).
    fps : nombre d'images par seconde à afficher (ou None pour ne pas
        l'afficher).
    """
    if landmarks is not None:
        draw_landmarks(frame, landmarks)

    lines = [
        label_for(gesture_name),
        "Main détectée" if landmarks is not None else "Aucune main",
    ]
    if fps is not None:
        lines.append(f"FPS: {fps:.0f}")

    y = 35
    for line in lines:
        cv2.putText(frame, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y += 35

    cv2.imshow("Hand Control Mouse - DEBUG", frame)
    return (cv2.waitKey(1) & 0xFF) != ord('q')


def close_debug_window():
    cv2.destroyAllWindows()
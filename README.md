# Hand-Control-Mouse

Contrôler la souris de l'ordinateur avec la main, devant une simple webcam.

La webcam filme la main, [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) en repère 21 points, et des règles traduisent la posture des doigts en action : déplacer le curseur, cliquer, faire défiler, glisser-déposer.

## Les gestes

| Action | Posture |
|---|---|
| Déplacer le curseur | Index seul tendu. Le curseur suit la main comme sur un pavé tactile |
| Clic gauche | Pouce seul tendu (l'auriculaire est ignoré) |
| Double-clic | Deux clics gauches rapprochés (entre 0,55 et 1,5 s) |
| Clic droit | Auriculaire seul tendu |
| Glisser-déposer | Majeur seul tendu, ou signe OK (pouce et index en rond), tenu pendant le déplacement |
| Faire défiler | Index et majeur en V. Le point de départ sert de neutre : plus la main s'en éloigne vers le haut ou le bas, plus la page défile vite |
| Repositionner la main | Main ouverte : le curseur ne bouge pas |

Les règles sont dans `src/gestures.py`. Un doigt est considéré comme tendu d'après l'angle de son articulation. Le pouce est jugé par sa distance à la base de l'index. Chaque clic doit être confirmé sur 7 images d'affilée pour éviter les déclenchements involontaires.

## Installation

Python 3.12 recommandé.

```
python -m venv .venv
.venv\Scripts\activate          # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'application

Depuis la racine du dépôt :

```
python src/main.py              # contrôle de la souris
python src/main.py --debug      # avec la fenêtre webcam, les points de la main et le FPS
```

Pour arrêter : `q` dans la fenêtre de debug, ou `Ctrl+C` dans le terminal.

Sous Linux, pyautogui a besoin d'une session X11 (le curseur ne bouge pas sous Wayland).

## Évaluation : règles contre modèles appris

Le dépôt contient aussi la chaîne qui compare les règles à des modèles appris (régression logistique, k plus proches voisins, random forest, MLP) sur des vidéos de gestes.

```
python src/extract_landmarks.py --input data/videos --merge    # vidéos -> points de la main
python -m mlp.extract_photos                                   # photos -> points de la main
python -m mlp.train                                            # comparaison des méthodes
python -m mlp.detection                                        # détection et précision de bout en bout
python -m mlp.export                                           # modèle final dans models/
```


**Résultats en bref** (2 personnes filmées, détails dans [`results/README.md`](results/README.md)) :

| | Règles | Meilleur modèle appris |
|---|---|---|
| Personnes déjà vues à l'entraînement (F1) | 0,85 | 0,98 (random forest) |
| Personne jamais vue (F1) | 0,77 à 0,84 | 0,46 à 0,52 (random forest + augmentation) |

Les modèles appris font mieux sur des mains qu'ils connaissent. Les règles généralisent beaucoup mieux à une nouvelle personne. Leur point faible est le clic droit, souvent lu comme un clic gauche.

## Organisation du dépôt

```
src/                    application
├── main.py             boucle principale : capture -> détection -> gestes -> souris
├── capture.py          webcam
├── detection.py        MediaPipe HandLandmarker
├── gestures.py         règles : posture des doigts -> geste
├── mapping.py          position de la main -> déplacement du curseur (One Euro Filter)
├── controller.py       actions souris (pyautogui)
├── ui.py               overlay et fenêtre de debug
├── extract_landmarks.py
└── extraction/         vidéos -> CSV de points de la main
mlp/                    comparaison règles / modèles appris
data/                   vidéos (non versionnées) et points extraits
results/                résultats de l'évaluation
models/                 modèle entraîné
tests/                  tests de l'extraction (python -m pytest tests)
```

## Licence

Voir [LICENSE](LICENSE).

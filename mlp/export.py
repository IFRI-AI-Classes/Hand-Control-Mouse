"""Entraîne le MLP sur tous les participants et le sauvegarde pour l'application.

Lancer depuis la racine du dépôt :  python -m mlp.export
"""
import joblib

from .config import MODEL_PATH
from .data import load_dataset
from .models import build_models


def main():
    X, y, _ = load_dataset()
    model = build_models()["mlp"].fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH} ({len(X)} images, classes {[str(c) for c in model.classes_]})")


if __name__ == "__main__":
    main()

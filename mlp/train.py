"""Compare les règles et les modèles appris sur trois évaluations.

1. temporel  : dans chaque vidéo, 70 % des images pour apprendre, les 30 % suivantes pour tester
               (résultat principal : les deux personnes sont vues à l'entraînement)
2. personne  : on apprend sur une personne, on teste sur l'autre (le cas le plus dur)
Chaque modèle est évalué sans et avec augmentation (miroir + rotations, voir augment.py).

3. photos    : on apprend sur toutes les vidéos, on teste sur les photos d'une 3e personne
               (nombre de bonnes réponses, la plupart des photos n'ont pas de main détectée)

Lancer depuis la racine du dépôt :  python -m mlp.train
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score

from .augment import augment
from .config import GESTURES, PHOTOS_CSV, RESULTS_DIR
from .data import describe, load_frame, temporal_split, to_xy
from .eval_rules import rules_predict
from .models import build_models
from .plots import save_confusion

warnings.filterwarnings("ignore", category=UserWarning)      # classes absentes d'une personne

# Matrices de confusion enregistrées : seulement celles dont on parle dans results/README.md
PLOTTED = {"regles", "random_forest", "mlp"}


def f1(y, y_pred):
    return f1_score(y, y_pred, average="macro", labels=GESTURES, zero_division=0)


def predictors():
    """(nom, modèle, augmentation) : les règles, puis chaque modèle sans et avec augmentation."""
    yield "regles", None, False
    for name, model in build_models().items():
        yield name, model, False
        yield f"{name}+aug", model, True


def run(model, X_train, y_train, X_test, aug=False):
    if model is None:
        return rules_predict(X_test)
    if aug:
        X_train, y_train = augment(X_train, y_train)      # entraînement seulement
    return clone(model).fit(X_train, y_train).predict(X_test)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_frame()
    X, y, groups = to_xy(df)
    test = temporal_split(df)
    photos = load_frame(PHOTOS_CSV) if PHOTOS_CSV.is_file() else None
    print(describe(y, groups), "\n")

    rows, reports = [], []
    for name, model, aug in predictors():
        row = {"methode": name}

        # 1. Découpage dans le temps
        y_pred = run(model, X[~test], y[~test], X[test], aug)
        row["f1_temporel"] = f1(y[test], y_pred)
        if name in PLOTTED:
            save_confusion(name, f"{name} : découpage dans le temps", y[test], y_pred)
        reports.append(f"### {name} (découpage dans le temps)\n\n```\n"
                       f"{classification_report(y[test], y_pred, labels=GESTURES, digits=3, zero_division=0)}```\n")

        # 2. Une personne de côté
        for person in sorted(set(groups)):
            out = groups == person
            row[f"f1_test_sur_{person}"] = f1(y[out], run(model, X[~out], y[~out], X[out], aug))

        # 3. Photos d'une 3e personne
        if photos is not None and len(photos):
            Xp, yp, _ = to_xy(photos)
            y_pred = run(model, X, y, Xp, aug)
            # très peu de photos exploitables : on compte les bonnes réponses plutôt qu'un F1
            row["photos_correctes"] = f"{int((y_pred == yp).sum())}/{len(yp)}"

        rows.append(row)
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{v:14s}" for k, v in row.items()))

    pd.DataFrame(rows).round(3).to_csv(RESULTS_DIR / "scores.csv", index=False)
    (RESULTS_DIR / "rapports_par_classe.md").write_text("\n".join(reports), encoding="utf-8")
    print(f"\nRésultats dans {RESULTS_DIR}")


if __name__ == "__main__":
    main()

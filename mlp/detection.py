"""Détection et précision de bout en bout, par personne et par geste.

Idée reprise du module de robustesse écrit par l'équipe.: une main que MediaPipe ne trouve pas
est une erreur pour l'utilisateur, même si les règles auraient bien reconnu le geste.
Lancer depuis la racine du dépôt :  python -m mlp.detection
"""
import numpy as np
import pandas as pd

from .config import DATASET_CSV, LABEL_COL, PERSON_COL, PHOTOS_CSV, RESULTS_DIR
from .data import load_all_frames, to_xy
from .eval_rules import rules_predict


def with_rules(df):
    df = df.copy()
    df["regles"] = None
    detected = df["detected"].to_numpy()
    if detected.any():
        X, _, _ = to_xy(df[detected])
        df.loc[detected, "regles"] = rules_predict(X)
    df["correct_bout_en_bout"] = df["detected"] & (df["regles"] == df[LABEL_COL])
    return df


def summarize(df, by):
    rows = []
    for key, part in df.groupby(by):
        det = part[part["detected"]]
        rows.append({
            by: key,
            "images": len(part),
            "detection": part["detected"].mean(),
            "regles_sur_mains_detectees": (det["regles"] == det[LABEL_COL]).mean() if len(det) else np.nan,
            "regles_bout_en_bout": part["correct_bout_en_bout"].mean(),
        })
    return pd.DataFrame(rows)


def build_tables():
    frames = [with_rules(load_all_frames(DATASET_CSV)).assign(source="video")]
    if PHOTOS_CSV.is_file():
        frames.append(with_rules(load_all_frames(PHOTOS_CSV)).assign(source="photo"))
    df = pd.concat(frames, ignore_index=True)
    return summarize(df, PERSON_COL), summarize(df[df["source"] == "video"], LABEL_COL)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    by_person, by_gesture = build_tables()
    by_person.round(3).to_csv(RESULTS_DIR / "detection_par_personne.csv", index=False)
    by_gesture.round(3).to_csv(RESULTS_DIR / "detection_par_geste.csv", index=False)
    print(by_person.round(3).to_string(index=False), "\n")
    print(by_gesture.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

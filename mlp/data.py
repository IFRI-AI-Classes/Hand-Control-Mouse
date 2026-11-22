"""Lit les CSV de landmarks et renvoie X (n, 63), y (labels), groups (participants)."""
import numpy as np
import pandas as pd

from .config import (DATASET_CSV, DETECTED_COL, FRAME_COL, GESTURES, LABEL_COL, LANDMARK_COLS,
                     PERSON_COL, RELABEL, TEST_RATIO, VIDEO_COL)


def load_frame(path=DATASET_CSV):
    """Le tableau nettoyé : images avec une main, étiquettes corrigées, classes retenues."""
    df = pd.read_csv(path)
    df = df[df[DETECTED_COL] == 1].dropna(subset=LANDMARK_COLS).copy()
    df[PERSON_COL] = df[PERSON_COL].str.lower()            # "Freud" et "freud" = même personne
    df[LABEL_COL] = df[LABEL_COL].replace(RELABEL)
    return df[df[LABEL_COL].isin(GESTURES)].reset_index(drop=True)


def load_all_frames(path=DATASET_CSV):
    """Toutes les images, y compris celles où MediaPipe n'a pas trouvé de main.

    Sert à mesurer la détection elle-même et la précision de bout en bout.
    """
    df = pd.read_csv(path)
    df[PERSON_COL] = df[PERSON_COL].str.lower()
    df[LABEL_COL] = df[LABEL_COL].replace(RELABEL)
    df = df[df[LABEL_COL].isin(GESTURES)].copy()
    df["detected"] = (df[DETECTED_COL] == 1) & df[LANDMARK_COLS].notna().all(axis=1)
    return df.reset_index(drop=True)


def to_xy(df):
    X = df[LANDMARK_COLS].to_numpy(dtype=np.float32)
    return X, df[LABEL_COL].to_numpy(), df[PERSON_COL].to_numpy()


def load_dataset(path=DATASET_CSV):
    return to_xy(load_frame(path))


def temporal_split(df, test_ratio=TEST_RATIO):
    """Masque booléen du test : les dernières images de chaque vidéo.

    Les images voisines d'une même vidéo sont presque identiques ; un mélange au hasard
    mettrait des quasi-doublons des deux côtés et gonflerait le score.
    """
    rank = df.groupby(VIDEO_COL)[FRAME_COL].rank(method="first", pct=True)
    return (rank > 1 - test_ratio).to_numpy()


def describe(y, groups):
    """Nombre d'images par personne et par geste : à regarder avant d'entraîner."""
    return pd.crosstab(pd.Series(groups, name="personne"), pd.Series(y, name="geste"), margins=True)

"""
Analyse des journaux enregistrés avec `python main.py --log fichier.csv`.
Affiche un tableau (FPS, latence, gigue au repos, dérive, retard du lissage)
pour un ou plusieurs fichiers, par exemple un fichier par filtre :

    python tools/analyze_log.py none.csv ema.csv oneeuro.csv

Pendant l'enregistrement, il faut tenir l'index immobile quelques secondes
(libellé "Déplacement" affiché). Le script repère tout seul ces moments
immobiles : des fenêtres d'une seconde où le point suivi bouge très peu.
"""
import argparse
import csv
import os
import numpy as np

CAM_W, CAM_H = 640, 480    # résolution de la webcam (voir capture.py)
WINDOW = 30                # images par fenêtre (environ 1 seconde)
STILL_RANGE_PX = 12.0      # étendue max du point suivi pour dire "immobile"
MAX_LAG = 12                # retard de lissage cherché, en images


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(rows, name):
    return np.array([float(r[name]) for r in rows])


def move_runs(rows):
    """Séries d'images consécutives en Déplacement avec une main détectée."""
    runs, cur = [], []
    for r in rows:
        if r["gesture"] == "MOVE" and r["hand"] == "1" and r["raw_x"] != "":
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def best_lag(raw, sm):
    """Retard (en images) qui fait le mieux coller le signal lissé au brut."""
    errs = []
    for lag in range(MAX_LAG + 1):
        a = raw[: len(raw) - lag] if lag else raw
        b = sm[lag:]
        errs.append(np.mean((a - b) ** 2))
    return int(np.argmin(errs))


def analyse(path):
    rows = load(path)
    if not rows:
        return None
    dt = num(rows, "dt_ms")
    dt = dt[dt > 0]
    fps = 1000.0 / dt
    present = [r for r in rows if r["hand"] == "1"]
    total = num(present, "total_ms") if present else np.array([0.0])
    detect = num(present, "detect_ms") if present else np.array([0.0])
    dt_mean = float(np.mean(dt))

    n_move = 0
    still_raw, still_sm, still_drift, lags = [], [], [], []
    for run in move_runs(rows):
        n_move += len(run)
        rx = num(run, "raw_x") * CAM_W
        ry = num(run, "raw_y") * CAM_H
        sx = num(run, "sm_x") * CAM_W
        sy = num(run, "sm_y") * CAM_H
        ddx, ddy = num(run, "dx"), num(run, "dy")
        for i in range(0, len(run) - WINDOW + 1, WINDOW):
            s = slice(i, i + WINDOW)
            if max(np.ptp(rx[s]), np.ptp(ry[s])) < STILL_RANGE_PX:
                still_raw.append(float(np.hypot(rx[s].std(), ry[s].std())))
                still_sm.append(float(np.hypot(sx[s].std(), sy[s].std())))
                still_drift.append(float((np.abs(ddx[s]) + np.abs(ddy[s])).sum()
                                         / (WINDOW * dt_mean / 1000.0)))
        # retard du lissage : seulement sur les séries où la main bouge vraiment
        if len(run) >= WINDOW and max(np.ptp(rx), np.ptp(ry)) > 40:
            lags.append(best_lag(rx, sx))

    nan = float("nan")
    jr = float(np.mean(still_raw)) if still_raw else nan
    js = float(np.mean(still_sm)) if still_sm else nan
    return {
        "fichier": os.path.basename(path),
        "images": len(rows),
        "main détectée (%)": 100.0 * len(present) / len(rows),
        "FPS médian": float(np.median(fps)),
        "FPS 5e centile": float(np.percentile(fps, 5)),
        "détection médiane (ms)": float(np.median(detect)),
        "latence logicielle médiane (ms)": float(np.median(total)),
        "latence logicielle 95e centile (ms)": float(np.percentile(total, 95)),
        "latence logicielle max (ms)": float(total.max()),
        "images en Déplacement": n_move,
        "fenêtres immobiles (1 s)": len(still_raw),
        "gigue brute au repos (px webcam)": jr,
        "gigue lissée au repos (px webcam)": js,
        "bruit supprimé par le lissage (%)": 100.0 * (1 - js / jr) if jr and jr > 0 else nan,
        "dérive du curseur au repos (px/s)": float(np.mean(still_drift)) if still_drift else nan,
        "retard du lissage estimé (ms)": float(np.mean(lags)) * dt_mean if lags else nan,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fichiers", nargs="+")
    args = p.parse_args()

    results = [r for r in (analyse(f) for f in args.fichiers) if r]
    if not results:
        print("Aucune donnée.")
        return
    keys = list(results[0].keys())
    print("| Mesure | " + " | ".join(r["fichier"] for r in results) + " |")
    print("|---|" + "---|" * len(results))
    for k in keys[1:]:
        vals = []
        for r in results:
            v = r[k]
            vals.append("n/d" if isinstance(v, float) and np.isnan(v)
                        else (f"{v:.1f}" if isinstance(v, float) else str(v)))
        print(f"| {k} | " + " | ".join(vals) + " |")
    print("\nLa latence est logicielle (de l'image reçue à l'ordre envoyé à la "
          "souris) : elle n'inclut ni la caméra ni l'affichage. Les mesures "
          "« au repos » ne portent que sur les fenêtres d'une seconde où la "
          "main est immobile ; n/d = aucune fenêtre de ce type dans le fichier.")


if __name__ == "__main__":
    main()
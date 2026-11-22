"""Matrices de confusion enregistrées en image dans results/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from .config import GESTURES, RESULTS_DIR


def save_confusion(name, title, y, y_pred):
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=GESTURES, normalize="true",
                                            values_format=".2f", xticks_rotation=45, ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)

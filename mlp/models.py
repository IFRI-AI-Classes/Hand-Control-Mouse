"""Les modèles comparés. Tous commencent par la même normalisation."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import MLP_PARAMS
from .features import normalize


def build_models():
    norm = lambda: FunctionTransformer(normalize)
    return {
        "logreg": make_pipeline(norm(), StandardScaler(), LogisticRegression(max_iter=2000)),
        "knn": make_pipeline(norm(), StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": make_pipeline(norm(), RandomForestClassifier(n_estimators=200, random_state=0)),
        "mlp": make_pipeline(norm(), StandardScaler(), MLPClassifier(**MLP_PARAMS)),
    }

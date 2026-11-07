"""Definición de modelos, espacios de búsqueda y utilidades de entrenamiento.

Es la única fuente de verdad de las familias de modelos: la usan tanto
notebooks/03_modelos.ipynb como src/train.py, para que ambos den lo mismo.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import config, features

MODEL_NAMES = ["Regresión logística", "Random forest", "Gradient boosting"]
BASELINE_NAME = "Regresión logística (baseline, C=1)"
DUMMY_NAME = "Sin habilidad (prior)"
SLUG = {
    "Regresión logística": "logreg",
    "Random forest": "random_forest",
    "Gradient boosting": "gradient_boosting",
}


def get_models():
    """Estimadores base (sin hiperparámetros ajustados) y si necesitan escalado."""
    seed = config.SEED
    return {
        "Regresión logística": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            "scale": True,
        },
        "Random forest": {
            "model": RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=seed),
            "scale": False,
        },
        "Gradient boosting": {
            "model": HistGradientBoostingClassifier(random_state=seed),
            "scale": False,
        },
    }


def get_spaces(quick=False):
    """Espacios de búsqueda: nombre -> (grilla, n_iter). n_iter=None => malla completa."""
    return {
        "Regresión logística": ({"C": [0.001, 0.01, 0.1, 1, 10]}, None),
        "Random forest": ({
            "n_estimators": [100, 200],
            "max_depth": [8, 10, 14],
            "min_samples_leaf": [50, 100, 200],
            "max_features": ["sqrt", 0.4],
        }, 2 if quick else 6),
        "Gradient boosting": ({
            "learning_rate": [0.03, 0.05, 0.1],
            "max_iter": [150, 300],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [50, 200],
            "l2_regularization": [0.0, 1.0],
        }, 3 if quick else 10),
    }


def get_xy(df, grey=False):
    """(X, y) con las columnas de features; grey=True agrega la zona gris."""
    num, cat = features.get_feature_lists(grey)
    return df[num + cat], df["target"].values


def make_pipeline(model, grey=False, scale=True):
    """Preprocesador + modelo en un solo Pipeline (todo se ajusta solo con los datos de entrenamiento)."""
    prep = features.build_preprocessor(include_grey_zone=grey, scale=scale)
    return Pipeline([("prep", prep), ("clf", model)])


def fit_pipeline(name, params, X, y, grey=False):
    """Entrena la familia `name` con los hiperparámetros `params` ({} = valores por defecto)."""
    cfg = get_models()[name]
    model = clone(cfg["model"]).set_params(**(params or {}))
    return make_pipeline(model, grey=grey, scale=cfg["scale"]).fit(X, y)


def no_skill_scores(y_train, n):
    """Baseline sin habilidad: siempre predice la tasa de default de train."""
    return np.full(n, float(np.mean(y_train)))


def temporal_folds(df):
    """Folds expansivos por año: entrena con años < y y valida en el año y."""
    years = sorted(df["issue_year"].unique())
    idx = np.arange(len(df))
    folds = []
    for y in years[1:]:
        tr = idx[(df["issue_year"] < y).values]
        va = idx[(df["issue_year"] == y).values]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def tune(name, X, y, folds, quick=False):
    """Búsqueda de hiperparámetros con folds temporales. Devuelve (mejores params, tabla CV)."""
    cfg = get_models()[name]
    space, n_iter = get_spaces(quick)[name]
    pipe = make_pipeline(cfg["model"], grey=False, scale=cfg["scale"])
    grid = {f"clf__{k}": v for k, v in space.items()}
    if n_iter is None:
        search = GridSearchCV(pipe, grid, scoring="average_precision", cv=folds, n_jobs=1, refit=False)
    else:
        search = RandomizedSearchCV(
            pipe, grid, n_iter=n_iter, scoring="average_precision", cv=folds,
            random_state=config.SEED, n_jobs=1, refit=False,
        )
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    best = {k.replace("clf__", ""): v for k, v in res.iloc[0]["params"].items()}
    return best, res

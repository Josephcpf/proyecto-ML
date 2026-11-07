"""Limpieza por reglas de dominio, ingeniería de variables y preprocesador de sklearn.

Separación deliberada para controlar leakage:
- clean() y engineer() son transformaciones SIN estado (fila a fila): se pueden
  aplicar al dataset completo antes del split sin filtrar información.
- Todo lo que aprende de los datos (cuantiles de winsorización, medianas de
  imputación, escalado, categorías) vive en build_preprocessor() y se ajusta
  solo con train.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

DATE_FMT = "%b-%Y"
TOP_PURPOSES = ["debt_consolidation", "credit_card"]


# --------------------------------------------------------------------------
# 1. Limpieza por reglas de dominio (sin estado)
# --------------------------------------------------------------------------
def clean(df, report=False):
    """Marca como faltantes (NaN) valores imposibles o sospechosos.

    No se eliminan filas: el valor inválido pasa a NaN y se imputa después en el
    pipeline (ajustado solo con train).
    """
    out = df.copy()
    rules = {
        "dti < 0 o > 100": (out["dti"] < 0) | (out["dti"] > 100),
        "revol_util > 150": out["revol_util"] > 150,
        "annual_inc <= 0": out["annual_inc"] <= 0,
    }
    counts = {name: int(mask.sum()) for name, mask in rules.items()}

    out.loc[rules["dti < 0 o > 100"], "dti"] = np.nan
    out.loc[rules["revol_util > 150"], "revol_util"] = np.nan
    out.loc[rules["annual_inc <= 0"], "annual_inc"] = np.nan

    return (out, counts) if report else out


# --------------------------------------------------------------------------
# 2. Ingeniería de variables (sin estado)
# --------------------------------------------------------------------------
def _parse_emp_length(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    if text.startswith("<"):
        return 0.0
    match = re.search(r"\d+", text)
    return float(match.group()) if match else np.nan


def engineer(df):
    """Crea las variables derivadas usadas por el modelo."""
    out = df.copy()

    out["term_months"] = out["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    out["emp_length_years"] = out["emp_length"].map(_parse_emp_length)
    out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2

    issue = pd.to_datetime(out["issue_d"], format=DATE_FMT)
    earliest = pd.to_datetime(out["earliest_cr_line"], format=DATE_FMT)
    out["credit_history_years"] = (issue - earliest).dt.days / 365.25

    income = out["annual_inc"].where(out["annual_inc"] > 0)
    out["loan_to_income"] = out["loan_amnt"] / income

    out["purpose_grouped"] = np.where(
        out["purpose"].isin(TOP_PURPOSES), out["purpose"], "other"
    )
    return out


def get_feature_lists(include_grey_zone=False):
    """Devuelve (numéricas, categóricas). La zona gris (grade, int_rate...) es opcional."""
    num = list(config.NUM_FEATURES)
    cat = list(config.CAT_FEATURES)
    if include_grey_zone:
        num += config.GREY_ZONE_NUM
        cat += config.GREY_ZONE_CAT
    return num, cat


# --------------------------------------------------------------------------
# 3. Transformadores con estado (se ajustan SOLO con train)
# --------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta valores extremos a los cuantiles aprendidos en train."""

    def __init__(self, columns=None, lower=0.005, upper=0.995):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        wanted = self.columns if self.columns is not None else list(X.columns)
        self.cols_ = [c for c in wanted if c in X.columns]
        self.bounds_ = {
            c: (X[c].quantile(self.lower), X[c].quantile(self.upper)) for c in self.cols_
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (lo, hi) in self.bounds_.items():
            X[col] = X[col].clip(lo, hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor(include_grey_zone=False, scale=True, min_frequency=0.005):
    """ColumnTransformer: winsorizar -> imputar (+ indicadores) -> escalar; one-hot para categóricas.

    - scale=False para modelos de árboles (no lo necesitan).
    - min_frequency agrupa categorías raras (<0.5%) y maneja categorías nuevas en inferencia.
    """
    num, cat = get_feature_lists(include_grey_zone)

    num_steps = [
        ("winsor", Winsorizer(columns=config.WINSOR_COLS)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
            dtype=np.float32,
        )),
    ])

    return ColumnTransformer(
        [("num", Pipeline(num_steps), num), ("cat", cat_pipe, cat)],
        verbose_feature_names_out=False,
    )

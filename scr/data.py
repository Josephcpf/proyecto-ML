"""Carga de datos, definición del target, diagnóstico de censura y split temporal."""
import pandas as pd

from src import config, features

DATE_FMT = "%b-%Y"


def load_raw(path=None, usecols=None, nrows=None):
    """Lee el CSV de Lending Club solo con las columnas necesarias."""
    path = path or config.DATA_RAW
    usecols = usecols or config.USECOLS
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)


def build_target(df):
    """Filtra a préstamos con desenlace definitivo y crea target (1 = default/charged off)."""
    mask = df["loan_status"].isin(config.BAD_STATUS + config.GOOD_STATUS)
    out = df.loc[mask].copy()
    out["target"] = out["loan_status"].isin(config.BAD_STATUS).astype(int)
    return out


def add_issue_year(df):
    """Agrega issue_date (datetime) e issue_year a partir de issue_d."""
    out = df.copy()
    out["issue_date"] = pd.to_datetime(out["issue_d"], format=DATE_FMT)
    out["issue_year"] = out["issue_date"].dt.year
    return out


def censoring_table(df):
    """Por año de originación y plazo: volumen, % con desenlace definitivo y tasa de default.

    Sirve para decidir el split temporal: un año es "confiable" si casi todos sus
    préstamos ya tienen desenlace. Se calcula sobre el dataset crudo (con loan_status).
    """
    d = add_issue_year(df)
    d["resuelto"] = d["loan_status"].isin(config.GOOD_STATUS + config.BAD_STATUS)
    d["malo"] = d["loan_status"].isin(config.BAD_STATUS)
    keys = ["issue_year", "term"]
    tabla = d.groupby(keys).agg(
        n_prestamos=("resuelto", "size"),
        pct_resueltos=("resuelto", "mean"),
    )
    tabla["tasa_default_resueltos"] = d[d["resuelto"]].groupby(keys)["malo"].mean()
    return tabla.round(3)


def temporal_split(df, train_years=None, valid_year=None, test_year=None, only_term=None):
    """Split out-of-time por issue_year. Los parámetros por defecto vienen de config."""
    train_years = train_years or config.TRAIN_YEARS
    valid_year = valid_year or config.VALID_YEAR
    test_year = test_year or config.TEST_YEAR
    only_term = only_term or config.ONLY_TERM

    if "issue_year" not in df.columns:
        df = add_issue_year(df)
    if only_term:
        df = df[df["term"].str.strip() == only_term.strip()]

    lo, hi = train_years
    train = df[df["issue_year"].between(lo, hi)].copy()
    valid = df[df["issue_year"] == valid_year].copy()
    test = df[df["issue_year"] == test_year].copy()
    return train, valid, test


def split_summary(train, valid, test, show_test_target=False):
    """Resumen de tamaños y tasa de default por split.

    Por defecto NO muestra la tasa de default del test, para no tomar decisiones
    de modelado mirando el conjunto de evaluación final.
    """
    rows = {}
    for name, d in {"train": train, "valid": valid, "test": test}.items():
        rate = d["target"].mean() if (name != "test" or show_test_target) else float("nan")
        rows[name] = {
            "filas": len(d),
            "años": f"{d['issue_year'].min()}-{d['issue_year'].max()}" if len(d) else "-",
            "tasa_default": round(rate, 4),
        }
    return pd.DataFrame(rows).T


def prepare_dataset(df_raw):
    """Cadena completa de preparación: target -> año -> limpieza -> features."""
    df = build_target(df_raw)
    df = add_issue_year(df)
    df = features.clean(df)
    df = features.engineer(df)
    return df

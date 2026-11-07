import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
 
 
def ks_statistic(y_true, y_score):
    """KS: máxima separación entre las distribuciones de score de malos y buenos."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)
 
 
def recall_at_precision(y_true, y_score, min_precision=0.4):
    """Mayor recall alcanzable con precisión >= min_precision. Devuelve (recall, umbral).
 
    Si ninguna precisión alcanza el mínimo, devuelve (0.0, nan).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ok = np.where(precision[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 0.0, float("nan")
    best = ok[np.argmax(recall[:-1][ok])]
    return float(recall[best]), float(thresholds[best])
 
 
def lift_at(y_true, y_score, frac=0.10):
    """Lift: tasa de default en el `frac` de préstamos con mayor score / tasa global."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.all(y_score == y_score[0]):   # scores constantes (modelo sin habilidad): sin ordenamiento posible
        return 1.0
    k = max(1, int(len(y_true) * frac))
    top = np.argsort(-y_score, kind="stable")[:k]
    return float(y_true[top].mean() / y_true.mean())
 
 
def evaluate_scores(y_true, y_score, min_precision=0.4):
    """Diccionario con todas las métricas a partir de probabilidades/scores."""
    recall, threshold = recall_at_precision(y_true, y_score, min_precision)
    tag = int(round(min_precision * 100))
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        f"recall@p{tag}": recall,
        f"umbral@p{tag}": threshold,
        "lift_top10": lift_at(y_true, y_score, 0.10),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        "tasa_default": float(np.mean(y_true)),
    }
 
 
def metrics_table(results):
    """results: {nombre: dict de evaluate_scores} -> DataFrame ordenado por AUC-PR."""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
 
 
def best_f1_threshold(y_true, y_score):
    """Umbral que maximiza F1 (se usa como respaldo si ninguno alcanza la precisión mínima)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])
 
 
def choose_threshold(y_true, y_score, min_precision=0.4):
    """Elige el umbral operativo EN VALIDACIÓN. Devuelve (umbral, regla usada)."""
    _, thr = recall_at_precision(y_true, y_score, min_precision)
    if np.isnan(thr):
        return best_f1_threshold(y_true, y_score), (
            f"máximo F1 (ninguna precisión alcanzó {min_precision:.0%} en validación)"
        )
    return thr, f"mayor recall con precisión >= {min_precision:.0%} en validación"
 
 
def metrics_at_threshold(y_true, y_score, threshold):
    """Precisión, recall, F1 y matriz de confusión para un umbral FIJO (elegido en validación)."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "umbral": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pct_marcados_como_malos": float((tp + fp) / len(y_true)),
    }
 
 
# --------------------------------------------------------------------------
# Análisis de errores por segmentos y equidad
# --------------------------------------------------------------------------
def _quantile_bins(series, q=5, decimals=0):
    """Cuantiles con etiquetas ordenadas ('1) 662–696', '2) ...') y una categoría final 'desconocido'."""
    cats = pd.qcut(series, q, duplicates="drop")
    fmt = f"{{:,.{decimals}f}}"
    labels = [f"{i + 1}) {fmt.format(iv.left)}–{fmt.format(iv.right)}"
              for i, iv in enumerate(cats.cat.categories)]
    out = cats.cat.rename_categories(labels)
    return out.cat.add_categories("desconocido").fillna("desconocido")
 
 
def add_analysis_segments(df, top_states=8):
    """Agrega columnas de segmento (cuantiles ordenados y grupos legibles) para el análisis de errores."""
    out = df.copy()
    out["fico_bin"] = _quantile_bins(out["fico_avg"], 5, 0)
    out["dti_bin"] = _quantile_bins(out["dti"], 5, 1)
    out["income_bin"] = _quantile_bins(out["annual_inc"], 5, 0)
    out["loan_bin"] = _quantile_bins(out["loan_amnt"], 5, 0)
    emp = pd.cut(
        out["emp_length_years"], [-1, 0, 2, 5, 9, 10],
        labels=["<1 año", "1-2 años", "3-5 años", "6-9 años", "10+ años"],
    )
    out["emp_bin"] = emp.cat.add_categories("desconocido").fillna("desconocido")
    top = out["addr_state"].value_counts().head(top_states).index
    out["state_group"] = np.where(out["addr_state"].isin(top), out["addr_state"], "otros")
    return out
 
 
def segment_metrics(df, by, threshold, y_col="target", score_col="score", min_rows=500):
    """Métricas por valor de la columna `by`. Descarta grupos con menos de `min_rows` filas."""
    rows = []
    col = df[by]
    if isinstance(col.dtype, pd.CategoricalDtype):
        keys = col                      # conserva el orden natural de los rangos
    else:
        keys = col.astype(object).where(col.notna(), "desconocido")
    for value, g in df.groupby(keys, observed=True):
        if len(g) < min_rows:
            continue
        y = g[y_col].to_numpy()
        s = g[score_col].to_numpy()
        pred = s >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        two_classes = len(np.unique(y)) == 2
        rows.append({
            "segmento": by,
            "valor": str(value),
            "n": len(g),
            "tasa_default": float(y.mean()),
            "score_medio": float(s.mean()),
            "pct_marcados": float(pred.mean()),
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "roc_auc": float(roc_auc_score(y, s)) if two_classes else np.nan,
            "pr_auc": float(average_precision_score(y, s)) if two_classes else np.nan,
        })
    return pd.DataFrame(rows)
 
 
def disparity_table(seg_df):
    """Compara cada grupo contra el grupo de referencia (el más grande) de su misma variable.
 
    - ratio_marcados: tasa de marcados del grupo / la de referencia (regla informal de los 4/5: entre 0.8 y 1.25).
    - dif_recall: diferencia en recall (igualdad de oportunidad).
    - dif_fpr: diferencia en falsos positivos (buenos pagadores marcados como malos).
    """
    parts = []
    for _, g in seg_df.groupby("segmento"):
        ref = g.loc[g["n"].idxmax()]
        t = g.copy()
        t["referencia"] = ref["valor"]
        t["ratio_marcados"] = t["pct_marcados"] / ref["pct_marcados"] if ref["pct_marcados"] else np.nan
        t["dif_recall"] = t["recall"] - ref["recall"]
        t["dif_fpr"] = t["fpr"] - ref["fpr"]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)
 
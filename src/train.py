"""Pipeline principal, reproducible desde la terminal (sin abrir notebooks).

Ejecutar SIEMPRE desde la raíz del repositorio:

    python -m src.train --stage all --force-test    # todo, de punta a punta
    python -m src.train --stage prepare             # CSV crudo -> splits en data/processed/
    python -m src.train --stage tune                # búsqueda de hiperparámetros (CV temporal en train)
    python -m src.train --stage compare             # compara familias en validación
    python -m src.train --stage final               # evaluación final en TEST (una sola vez)

Añadir --quick para una pasada rápida de depuración: usa muestras e iteraciones
reducidas, escribe en outputs/quick/ y NUNCA toca el test (el stage final se
ensaya sobre validación).

Protección del test: si outputs/metrics.json ya contiene una evaluación en test,
el stage final se niega a repetirla salvo que se pase --force-test. Cada
evaluación queda contada en metrics.json ("test_evaluations").
"""
import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from src import config, data, evaluate, models

log = logging.getLogger("train")
STAGES = ["prepare", "tune", "compare", "final"]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
class Paths:
    """Carpetas de salida. En modo --quick todo va a subcarpetas 'quick' para no mezclar."""

    def __init__(self, quick):
        self.outputs = config.OUTPUTS / "quick" if quick else config.OUTPUTS
        self.figures = config.FIGURES / "quick" if quick else config.FIGURES
        self.artifacts = config.DATA_PROCESSED / ("artifacts_quick" if quick else "artifacts")
        for p in (self.outputs, self.figures, self.artifacts, config.DATA_PROCESSED):
            p.mkdir(parents=True, exist_ok=True)
        self.best_params = self.outputs / "best_params.json"
        self.metrics = self.outputs / "metrics.json"


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    log.info("Guardado %s", path)


def split_info():
    return {
        "train_years": list(config.TRAIN_YEARS),
        "valid_year": config.VALID_YEAR,
        "test_year": config.TEST_YEAR,
        "only_term": config.ONLY_TERM,
    }


def maybe_sample(df, n):
    return df if (n is None or n >= len(df)) else df.sample(n, random_state=config.SEED)


# --------------------------------------------------------------------------
# Stage 1: prepare (equivale a notebooks/02_limpieza_features.ipynb)
# --------------------------------------------------------------------------
def stage_prepare(args, paths):
    log.info("Cargando %s", config.DATA_RAW)
    df_raw = data.load_raw()
    log.info("Filas crudas: %s", f"{len(df_raw):,}")

    data.censoring_table(df_raw).to_csv(paths.outputs / "censoring_table.csv")

    df = data.prepare_dataset(df_raw)
    train, valid, test = data.temporal_split(df)
    log.info("Split:\n%s", data.split_summary(train, valid, test).to_string())

    for name, d in {"train": train, "valid": valid, "test": test}.items():
        d.to_parquet(config.DATA_PROCESSED / f"{name}.parquet", index=False)
    log.info("Splits guardados en %s", config.DATA_PROCESSED)


# --------------------------------------------------------------------------
# Stage 2: tune (CV temporal dentro de train)
# --------------------------------------------------------------------------
def stage_tune(args, paths):
    (train,) = data.load_splits(("train",))
    n = 20_000 if args.quick else args.search_sample
    search_df = train.sample(min(n, len(train)), random_state=config.SEED).reset_index(drop=True)
    folds = models.temporal_folds(search_df)
    X, y = models.get_xy(search_df)
    log.info("Búsqueda sobre %s filas con %d folds temporales", f"{len(search_df):,}", len(folds))

    best, cv = {}, {}
    for name in models.MODEL_NAMES:
        t0 = time.time()
        best[name], res = models.tune(name, X, y, folds, quick=args.quick)
        cv[name] = {
            "cv_pr_auc_mean": float(res.iloc[0]["mean_test_score"]),
            "cv_pr_auc_std": float(res.iloc[0]["std_test_score"]),
        }
        res[["params", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
            paths.outputs / f"cv_results_{models.SLUG[name]}.csv", index=False
        )
        log.info("%s: AUC-PR CV=%.4f | %s (%.0fs)", name, cv[name]["cv_pr_auc_mean"], best[name], time.time() - t0)

    write_json(paths.best_params, {
        "seed": config.SEED,
        "split": split_info(),
        "best_params": best,
        "cv": cv,
        "search": {"sample_rows": len(search_df), "n_temporal_folds": len(folds), "quick_run": args.quick},
        "note": "Hiperparámetros elegidos con CV temporal dentro de train; el test NO se usó.",
    })


# --------------------------------------------------------------------------
# Stage 3: compare (familias y zona gris, en validación)
# --------------------------------------------------------------------------
def stage_compare(args, paths):
    payload = read_json(paths.best_params)
    if "best_params" not in payload:
        raise SystemExit("Falta best_params.json. Ejecuta primero: python -m src.train --stage tune")
    best = payload["best_params"]

    train, valid = data.load_splits(("train", "valid"))
    train_fit = maybe_sample(train, 50_000 if args.quick else args.train_sample)
    X_tr, y_tr = models.get_xy(train_fit)
    X_va, y_va = models.get_xy(valid)

    results = {}
    results[(models.DUMMY_NAME, "sin zona gris")] = evaluate.evaluate_scores(
        y_va, models.no_skill_scores(y_tr, len(y_va)), args.min_precision)
    base = models.fit_pipeline("Regresión logística", {}, X_tr, y_tr)
    results[(models.BASELINE_NAME, "sin zona gris")] = evaluate.evaluate_scores(
        y_va, base.predict_proba(X_va)[:, 1], args.min_precision)

    for grey in (False, True):
        variante = "con zona gris" if grey else "sin zona gris"
        X_tr, y_tr = models.get_xy(train_fit, grey)
        X_va, y_va = models.get_xy(valid, grey)
        for name in models.MODEL_NAMES:
            t0 = time.time()
            pipe = models.fit_pipeline(name, best[name], X_tr, y_tr, grey)
            p = pipe.predict_proba(X_va)[:, 1]
            results[(name, variante)] = evaluate.evaluate_scores(y_va, p, args.min_precision)
            log.info("%-22s | %-13s | AUC-PR=%.4f (%.0fs)", name, variante, results[(name, variante)]["pr_auc"], time.time() - t0)

    tabla = pd.DataFrame(results).T
    tabla.index.names = ["modelo", "variante"]
    tabla = tabla.sort_values("pr_auc", ascending=False)
    tabla.round(6).to_csv(paths.outputs / "model_comparison_valid.csv")
    log.info("Comparación en validación:\n%s", tabla[["pr_auc", "roc_auc", "ks"]].round(4).to_string())

    candidatos = tabla.reset_index()
    candidatos = candidatos[(candidatos["variante"] == "sin zona gris") & (candidatos["modelo"].isin(models.MODEL_NAMES))]
    primary = candidatos.sort_values("pr_auc", ascending=False).iloc[0]["modelo"]
    log.info("Modelo principal (mayor AUC-PR en valid, sin zona gris): %s", primary)

    payload.update({
        "primary_model": primary,
        "include_grey_zone": False,
        "valid_metrics": {f"{m} | {v}": vals for (m, v), vals in results.items()},
    })
    write_json(paths.best_params, payload)


# --------------------------------------------------------------------------
# Stage 4: final (evaluación en TEST, una sola vez)
# --------------------------------------------------------------------------
def _plot_final(y_eval, scores, thr_metrics, label, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, s in scores.items():
        if name == models.DUMMY_NAME:
            continue
        pr, rc, _ = precision_recall_curve(y_eval, s)
        axes[0].plot(rc, pr, label=f"{name} (AUC-PR={evaluate.evaluate_scores(y_eval, s)['pr_auc']:.3f})")
    axes[0].axhline(y_eval.mean(), color="gray", ls="--", label=f"Sin habilidad ({y_eval.mean():.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precisión")
    axes[0].set_title(f"Curva Precisión-Recall en {label}")
    axes[0].legend(fontsize=8)

    cm = np.array([[thr_metrics["tn"], thr_metrics["fp"]], [thr_metrics["fn"], thr_metrics["tp"]]])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=["Pred. bueno", "Pred. malo"], yticklabels=["Real bueno", "Real malo"])
    axes[1].set_title(f"Matriz de confusión en {label}\n(umbral {thr_metrics['umbral']:.3f} elegido en validación)")
    plt.tight_layout()
    fig.savefig(figures_dir / "final_evaluacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def stage_final(args, paths):
    quick = args.quick
    payload = read_json(paths.best_params)
    if "primary_model" not in payload:
        raise SystemExit("Falta el modelo principal. Ejecuta primero: python -m src.train --stage compare")

    previous = read_json(paths.metrics)
    n_prev = int(previous.get("test_evaluations", 0))
    if not quick and n_prev >= 1 and not args.force_test:
        raise SystemExit(
            f"El test ya fue evaluado {n_prev} vez/veces (ver {paths.metrics}).\n"
            "Repetir la evaluación en test invalida su imparcialidad. Si solo quieres regenerar los resultados "
            "con el MISMO pipeline y las mismas semillas, vuelve a ejecutar con --force-test."
        )

    primary = payload["primary_model"]
    grey = bool(payload.get("include_grey_zone", False))
    params = payload["best_params"][primary]

    splits = ("train", "valid") if quick else ("train", "valid", "test")
    loaded = dict(zip(splits, data.load_splits(splits)))
    train, valid = loaded["train"], loaded["valid"]
    eval_df = valid if quick else loaded["test"]
    eval_label = "validación (ENSAYO, test intacto)" if quick else "test"
    eval_key = "dry_run_on_valid" if quick else "test"
    log.info("Evaluación final sobre: %s (%s filas)", eval_label, f"{len(eval_df):,}")

    train_fit = maybe_sample(train, 50_000 if quick else args.train_sample)
    X_tr, y_tr = models.get_xy(train_fit, grey)
    X_va, y_va = models.get_xy(valid, grey)
    X_ev, y_ev = models.get_xy(eval_df, grey)

    # Modelo final: se entrena SOLO con train (el mismo que se comparó en validación)
    final = models.fit_pipeline(primary, params, X_tr, y_tr, grey)
    p_valid = final.predict_proba(X_va)[:, 1]
    p_eval = final.predict_proba(X_ev)[:, 1]

    # Umbral operativo elegido en VALIDACIÓN y aplicado tal cual al conjunto de evaluación
    thr, rule = evaluate.choose_threshold(y_va, p_valid, args.min_precision)
    final_metrics = evaluate.evaluate_scores(y_ev, p_eval, args.min_precision)
    final_metrics["a_umbral_fijo"] = evaluate.metrics_at_threshold(y_ev, p_eval, thr)

    # Baselines, evaluados con el mismo protocolo
    Xb_tr, yb_tr = models.get_xy(train_fit)
    Xb_ev, _ = models.get_xy(eval_df)
    base = models.fit_pipeline("Regresión logística", {}, Xb_tr, yb_tr)
    p_base = base.predict_proba(Xb_ev)[:, 1]
    base_metrics = evaluate.evaluate_scores(y_ev, p_base, args.min_precision)
    base_metrics["a_umbral_fijo"] = evaluate.metrics_at_threshold(y_ev, p_base, thr)
    p_noskill = models.no_skill_scores(yb_tr, len(y_ev))
    noskill_metrics = evaluate.evaluate_scores(y_ev, p_noskill, args.min_precision)

    comparison = {models.DUMMY_NAME: noskill_metrics, models.BASELINE_NAME: base_metrics, primary: final_metrics}
    pd.DataFrame(comparison).T.drop(columns=["a_umbral_fijo"], errors="ignore").round(6).to_csv(
        paths.outputs / ("test_comparison.csv" if not quick else "dry_run_comparison.csv"))

    gain_abs = final_metrics["pr_auc"] - base_metrics["pr_auc"]
    write_json(paths.metrics, {
        "seed": config.SEED,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_run": quick,
        "evaluated_on": eval_label,
        "test_evaluations": n_prev if quick else n_prev + 1,
        "split": split_info(),
        "model": {"name": primary, "include_grey_zone": grey, "params": params},
        "threshold": {"value": thr, "rule": rule, "chosen_on": "validación"},
        "valid": evaluate.evaluate_scores(y_va, p_valid, args.min_precision),
        eval_key: {
            "final_model": final_metrics,
            "baseline_logreg": base_metrics,
            "no_skill": noskill_metrics,
            "mejora_vs_baseline": {
                "pr_auc_abs": gain_abs,
                "pr_auc_rel_pct": 100 * gain_abs / base_metrics["pr_auc"],
                "ks_abs": final_metrics["ks"] - base_metrics["ks"],
            },
        },
    })

    _plot_final(y_ev, {primary: p_eval, models.BASELINE_NAME: p_base, models.DUMMY_NAME: p_noskill},
                final_metrics["a_umbral_fijo"], eval_label.split(" (")[0], paths.figures)

    # Artefactos para el análisis de errores (carpeta ignorada por git)
    for name, df, s in (("valid", valid, p_valid), (eval_key.replace("dry_run_on_", ""), eval_df, p_eval)):
        out = df.copy()
        out["score"] = s
        out["pred_malo"] = (s >= thr).astype(int)
        out.to_parquet(paths.artifacts / f"predictions_{name}.parquet", index=False)
    joblib.dump(final, paths.artifacts / "model_final.joblib")
    log.info("Artefactos (predicciones y modelo) en %s", paths.artifacts)

    log.info("RESULTADO en %s: AUC-PR %.4f (baseline %.4f, sin habilidad %.4f) | KS %.4f",
             eval_label, final_metrics["pr_auc"], base_metrics["pr_auc"], noskill_metrics["pr_auc"], final_metrics["ks"])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pipeline reproducible del proyecto (Lending Club).")
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--quick", action="store_true", help="pasada rápida de depuración (no toca el test)")
    ap.add_argument("--force-test", action="store_true", help="permite repetir la evaluación en test")
    ap.add_argument("--search-sample", type=int, default=150_000, help="filas de train para la búsqueda")
    ap.add_argument("--train-sample", type=int, default=None, help="submuestra de train para el entrenamiento final")
    ap.add_argument("--min-precision", type=float, default=0.40, help="precisión mínima para recall@precisión")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=UserWarning)
    np.random.seed(config.SEED)

    paths = Paths(args.quick)
    runners = {"prepare": stage_prepare, "tune": stage_tune, "compare": stage_compare, "final": stage_final}
    for stage in (STAGES if args.stage == "all" else [args.stage]):
        log.info("===== STAGE: %s%s =====", stage, " (quick)" if args.quick else "")
        t0 = time.time()
        runners[stage](args, paths)
        log.info("Stage %s terminado en %.0fs", stage, time.time() - t0)


if __name__ == "__main__":
    main()

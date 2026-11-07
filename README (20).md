# Proyecto Final de Machine Learning — Grupo 11

Predicción de riesgo de default en préstamos personales, sobre el dataset **Lending Club Loan Data**.
El planteamiento del problema, la variable objetivo, los riesgos de leakage y el plan de validación están en [`proposal.md`](./proposal.md).

## Estructura del repositorio

```
proyecto-ML/
├── README.md
├── proposal.md                        planteamiento del problema
├── requirements.txt                   dependencias de Python
├── .gitignore
├── data/
│   ├── README.md                      cómo descargar el dataset
│   ├── raw/                           CSV original de Kaggle (no versionado)
│   └── processed/                     splits y artefactos generados (no versionado)
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb   exploración inicial (entrega previa)
│   ├── 02_limpieza_features.ipynb     split temporal, limpieza y features
│   ├── 03_modelos.ipynb               baselines, 3 familias de modelos, hiperparámetros
│   └── 04_analisis_errores.ipynb      errores por segmentos, interpretabilidad y sesgos
├── src/
│   ├── config.py                      semilla, rutas, columnas y años del split
│   ├── data.py                        carga, target, tabla de censura, split temporal
│   ├── features.py                    limpieza, variables derivadas, preprocesador
│   ├── models.py                      familias de modelos, búsqueda de hiperparámetros
│   ├── evaluate.py                    métricas (AUC-PR, KS, recall@precisión, lift...)
│   └── train.py                       script principal reproducible
├── outputs/                           resultados (metrics.json, comparaciones, CV)
└── reports/
    ├── figures/                       figuras
    └── informe_final.md               informe final (pendiente)
```

## Cómo reproducir

### 1. Clonar e instalar

```powershell
git clone https://github.com/Josephcpf/proyecto-ML.git
cd proyecto-ML
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

En Mac/Linux: `source .venv/bin/activate` en lugar de la línea de activación.

### 2. Descargar los datos

El dataset (~1.6 GB) no está en el repositorio. Seguir [`data/README.md`](./data/README.md) y dejar el CSV en `data/raw/`
(por defecto se espera `data/raw/accepted_2007_to_2018Q4.csv`; se puede cambiar con la variable de entorno `LC_DATA_PATH`).

### 3. Ejecutar el pipeline (desde la raíz del repositorio)

```powershell
python -m src.train --stage all --force-test
```

Este comando regenera todos los resultados: prepara los splits, ajusta hiperparámetros, compara modelos en validación
y hace la evaluación final en test. Todo usa semilla fija (`SEED = 42` en `src/config.py`).

También se puede correr por etapas:

| Etapa | Comando | Qué hace | Salidas |
|---|---|---|---|
| prepare | `python -m src.train --stage prepare` | CSV → target, limpieza, features, split temporal | `data/processed/*.parquet`, `outputs/censoring_table.csv` |
| tune | `python -m src.train --stage tune` | Búsqueda de hiperparámetros con CV temporal dentro de train | `outputs/best_params.json`, `outputs/cv_results_*.csv` |
| compare | `python -m src.train --stage compare` | Compara 3 familias (con y sin zona gris) en validación y elige el modelo principal | `outputs/model_comparison_valid.csv` |
| final | `python -m src.train --stage final` | Evalúa el modelo elegido en test, una sola vez | `outputs/metrics.json`, `outputs/test_comparison.csv`, `reports/figures/final_evaluacion.png` |

**Pasada rápida de depuración:** agregar `--quick` (muestras e iteraciones reducidas). Escribe en `outputs/quick/` y **nunca toca el test**: la etapa `final` se ensaya sobre validación.

Los notebooks 02 y 03 hacen lo mismo paso a paso, con explicaciones y gráficos, usando las mismas funciones de `src/`.
El notebook 04 analiza los errores del modelo elegido sobre **validación**; su sección opcional sobre test (`ANALYZE_TEST = True`) solo debe activarse después de la etapa `final`.

## Protocolo de evaluación

- **Split temporal por año de originación** (no aleatorio): train 2012–2014, validación 2015, test 2016. La justificación (censura por la derecha) está en `proposal.md`, sección 10.
- **El test se evalúa una sola vez**, después de elegir modelo e hiperparámetros con train/validación. La etapa `final` se niega a repetirlo si `outputs/metrics.json` ya contiene una evaluación en test, salvo con `--force-test`; el número de evaluaciones queda registrado en `test_evaluations`.
- El umbral de decisión se elige en validación y se aplica sin cambios al test.
- Todo el preprocesamiento (winsorización, imputación, escalado, categorías) está dentro de un `Pipeline` de scikit-learn y se ajusta solo con datos de entrenamiento.
- Métrica principal: **AUC-PR**. Secundarias: KS, recall a precisión fija, lift, ROC-AUC. No se reporta accuracy sola.

## Resultados

Se generan en `outputs/metrics.json` y `outputs/test_comparison.csv`. *(Completar aquí un resumen con los números finales cuando se ejecute el pipeline.)*

> Nota: el baseline de la entrega previa (AUC-PR 0.273) usó otro split (train 2015–2017, validación 2018) y no es comparable con los resultados actuales.

## Estado del proyecto

- [x] Entrega previa: problema definido, exploración inicial, baseline.
- [x] Limpieza, features y split temporal (`02`, `src/data.py`, `src/features.py`).
- [x] Modelos e hiperparámetros (`03`, `src/models.py`).
- [x] Pipeline reproducible (`src/train.py`, `src/evaluate.py`).
- [ ] Evaluación final en test.
- [x] Análisis de errores, interpretabilidad y sesgos (`04`, sobre validación).
- [ ] Informe final y presentación.

## Equipo

Grupo 11 — integrantes en [`proposal.md`](./proposal.md).

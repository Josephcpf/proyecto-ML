# Proyecto Final de Machine Learning — Grupo 11

Predicción de riesgo de default en préstamos personales, sobre el dataset **Lending Club Loan Data**.

Ver el planteamiento completo del problema, la variable objetivo, los riesgos de leakage y el plan de trabajo en [`proposal.md`](./proposal.md).

## Estructura del repositorio

```
proyecto-ML/
├── README.md                          (este archivo)
├── proposal.md                        planteamiento del problema (entrega previa)
├── requirements.txt                   dependencias de Python
├── .gitignore
├── data/
│   ├── README.md                      instrucciones para descargar el dataset
│   └── raw/                           datos crudos (no versionado, ver data/README.md)
└── notebooks/
    └── 01_exploracion_inicial.ipynb   exploración inicial + baseline
```

## Cómo reproducir la exploración

### 1. Clonar el repositorio

```bash
git clone https://github.com/Josephcpf/proyecto-ML.git
cd proyecto-ML
```

### 2. Crear un entorno virtual e instalar dependencias

En Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

En Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 3. Descargar los datos

El dataset no se incluye en el repositorio por su tamaño (~1.6 GB). Seguir las instrucciones en [`data/README.md`](./data/README.md) para descargarlo de Kaggle y colocarlo en `data/raw/`.

### 4. Correr el notebook

```bash
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

O abrirlo directamente en VS Code con la extensión de Jupyter y correr todas las celdas (`Run All`).

## Resultados de la exploración inicial

- Dataset filtrado a préstamos con desenlace definitivo (`Fully Paid`, `Charged Off`, `Default`): **1,345,350 filas**.
- Tasa de default global: **~20%** (fuerte desbalance de clases, justifica el uso de AUC-PR sobre accuracy).
- Baseline (regresión logística, split temporal 2015-2017 train / 2018 valid):
  - AUC-ROC: **0.682**
  - AUC-PR: **0.272**
  - Recall clase "Malo" (default): **0.57**
- Se observó *drift* temporal entre años: tasa de default de 21.9% en train (2015-2017) vs. 15.8% en valid (2018), relevante para la estrategia de validación de las siguientes etapas.


## Equipo



Grupo 11 — ver integrantes en [`proposal.md`](./proposal.md).

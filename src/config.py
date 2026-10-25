"""Configuración central del proyecto: rutas, semilla, columnas y split temporal.

Todo lo que pueda cambiar entre experimentos vive aquí, para que notebooks y
scripts usen exactamente las mismas decisiones.
"""
import os
from pathlib import Path

SEED = 42

# --- Rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
# Se puede sobreescribir con la variable de entorno LC_DATA_PATH
DATA_RAW = Path(
    os.environ.get("LC_DATA_PATH", ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv")
)
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"

# --- Variable objetivo -----------------------------------------------------
BAD_STATUS = ["Charged Off", "Default"]   # target = 1
GOOD_STATUS = ["Fully Paid"]              # target = 0
# Cualquier otro estado (Current, Late, In Grace Period...) se excluye: desenlace no definido.

# --- Split temporal (out-of-time) por año de originación (issue_d) ----------
# Se eligió más atrás en el tiempo que 2018 para evitar censura por la derecha:
# los préstamos recientes solo tienen desenlace si se pagaron antes de tiempo o
# cayeron en default rápido. Verificar con data.censoring_table() (notebook 02).
TRAIN_YEARS = (2012, 2014)   # inclusive
VALID_YEAR = 2015            # selección de modelo e hiperparámetros
TEST_YEAR = 2016             # una sola evaluación final
ONLY_TERM = None             # None = ambos plazos; o " 36 months" para filtrar

# --- Columnas ----------------------------------------------------------------
USECOLS = [
    "id", "loan_status", "issue_d",
    "loan_amnt", "term", "purpose", "annual_inc", "dti", "emp_length",
    "home_ownership", "verification_status", "fico_range_low", "fico_range_high",
    "open_acc", "total_acc", "revol_bal", "revol_util", "delinq_2yrs",
    "earliest_cr_line", "pub_rec", "inq_last_6mths", "mort_acc",
    "application_type", "addr_state",
    # zona gris (no entran al baseline)
    "grade", "sub_grade", "int_rate",
]

# Features finales (después de features.engineer)
NUM_FEATURES = [
    "loan_amnt", "term_months", "annual_inc", "dti", "fico_avg", "open_acc",
    "total_acc", "revol_bal", "revol_util", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "mort_acc", "emp_length_years", "credit_history_years",
    "loan_to_income",
]
CAT_FEATURES = ["purpose_grouped", "home_ownership", "verification_status", "application_type"]

# Zona gris: se evalúan aparte (con y sin)
GREY_ZONE_NUM = ["int_rate"]
GREY_ZONE_CAT = ["grade", "sub_grade"]

# Columnas con colas pesadas: se recortan (winsorización) con cuantiles ajustados SOLO en train
WINSOR_COLS = [
    "annual_inc", "revol_bal", "dti", "revol_util", "loan_to_income",
    "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec", "mort_acc",
]

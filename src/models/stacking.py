"""
src/models/stacking.py
========================
Fase 4: Stacking ensemble.

Combina las probabilidades 1X2 de Dixon-Coles (Fase 2), XGBoost y CatBoost
(Fase 3) -- más la diferencia de Elo cruda -- en un meta-modelo (regresión
logística multinomial) que aprende cómo pesar a cada uno.

Por qué no usar simplemente el promedio de los tres:
--------------------------------------------------------
Un promedio fijo asume que los tres modelos aciertan/fallan por igual en
todas las situaciones. La regresión logística, en cambio, puede aprender
matices -- por ejemplo, si Dixon-Coles tiende a ser mejor en partidos muy
parejos y XGBoost en partidos con gran diferencia de Elo, el meta-modelo
puede aprender a pesar más a cada uno según el contexto (porque también le
damos `elo_diff` como feature de entrada, no solo las probabilidades).

Validación sin fuga (walk-forward):
-------------------------------------
Para entrenar el meta-modelo SIN hacer trampa, no se puede simplemente
reusar las probabilidades de los modelos base ya ajustados con todo el
histórico (eso sería como dejar que el meta-modelo "vea" el futuro). En vez
de eso, se generan folds hacia atrás en el tiempo: para cada fold se
reentrena Dixon-Coles + XGBoost + CatBoost usando SOLO datos anteriores al
fold, se predice ese fold, y esas predicciones (con su resultado real) son
las que alimentan el entrenamiento del meta-modelo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from models.ml_models import ModelosML, etiquetar_resultado
from models.poisson_dixon_coles import DixonColesModel

COLUMNAS_META = [
    "elo_diff",
    "prob_local_dc", "prob_empate_dc", "prob_visita_dc",
    "prob_local_xgb", "prob_empate_xgb", "prob_visita_xgb",
    "prob_local_cb", "prob_empate_cb", "prob_visita_cb",
]


def _predicciones_dc(modelo_dc: DixonColesModel, df: pd.DataFrame) -> np.ndarray:
    filas = []
    for f in df.itertuples(index=False):
        p = modelo_dc.predecir_partido(f.home_team, f.away_team, bool(f.neutral))
        filas.append([p["prob_local"], p["prob_empate"], p["prob_visita"]])
    return np.array(filas)


def generar_meta_features(df_train_base: pd.DataFrame, df_target: pd.DataFrame,
                           historico_crudo: pd.DataFrame, categorias_torneo: list[str],
                           fecha_corte_dc: str) -> pd.DataFrame:
    """Entrena Dixon-Coles + ModelosML usando SOLO `df_train_base` (pasado),
    y devuelve las meta-features para `df_target` (presente/futuro)."""
    dc = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    dc.fit(historico_crudo, fecha_corte=fecha_corte_dc)
    probs_dc = _predicciones_dc(dc, df_target)

    ml = ModelosML(categorias_torneo)
    ml.fit(df_train_base)
    probs_ml = ml.predict_proba(df_target)

    meta = pd.DataFrame({
        "elo_diff": df_target["elo_diff"].values,
        "prob_local_dc": probs_dc[:, 0], "prob_empate_dc": probs_dc[:, 1], "prob_visita_dc": probs_dc[:, 2],
        "prob_local_xgb": probs_ml["xgboost"][:, 0], "prob_empate_xgb": probs_ml["xgboost"][:, 1],
        "prob_visita_xgb": probs_ml["xgboost"][:, 2],
        "prob_local_cb": probs_ml["catboost"][:, 0], "prob_empate_cb": probs_ml["catboost"][:, 1],
        "prob_visita_cb": probs_ml["catboost"][:, 2],
    }, index=df_target.index)
    return meta


def generar_folds_walk_forward(historico_feat: pd.DataFrame, anio_inicio: int, fecha_corte_holdout: pd.Timestamp):
    """Devuelve lista de (fecha_inicio_fold, fecha_fin_fold) anuales, desde
    `anio_inicio` hasta justo antes del holdout final."""
    folds = []
    cursor = pd.Timestamp(f"{anio_inicio}-01-01")
    while cursor < fecha_corte_holdout:
        siguiente = min(cursor + pd.DateOffset(years=1), fecha_corte_holdout)
        folds.append((cursor, siguiente))
        cursor = siguiente
    return folds


class StackingEnsemble:
    def __init__(self):
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )

    def fit(self, meta_train: pd.DataFrame, y_train: np.ndarray) -> "StackingEnsemble":
        self.pipeline.fit(meta_train[COLUMNAS_META], y_train)
        return self

    def predict_proba(self, meta: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(meta[COLUMNAS_META])

    def pesos(self) -> pd.DataFrame:
        modelo = self.pipeline.named_steps["logisticregression"]
        return pd.DataFrame(modelo.coef_, columns=COLUMNAS_META, index=["local", "empate", "visita"])

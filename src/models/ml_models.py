"""
src/models/ml_models.py
=========================
Fase 3: XGBoost y CatBoost para predecir el resultado (local/empate/visita)
usando las features de src/features.py (Elo + forma reciente + head-to-head).

A diferencia de Elo/Dixon-Coles (que modelan goles), estos modelos clasifican
directamente el resultado 1X2. No producen un marcador exacto por sí solos
-- ese sigue siendo el trabajo de Dixon-Coles -- pero aportan una vista
distinta (basada en patrones no lineales e interacciones entre features) que
en la Fase 4 se combina con Poisson/Dixon-Coles en el stacking ensemble.

resultado: 0 = gana local, 1 = empate, 2 = gana visitante
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from features import FEATURE_COLUMNS_NUMERICAS, FEATURE_COLUMNAS_CATEGORICAS

COLUMNAS_MODELO = FEATURE_COLUMNS_NUMERICAS + FEATURE_COLUMNAS_CATEGORICAS


def etiquetar_resultado(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(df["home_score"] > df["away_score"], 0,
                 np.where(df["home_score"] == df["away_score"], 1, 2)),
        index=df.index, name="resultado",
    )


class ModelosML:
    """Envuelve XGBoost + CatBoost entrenados sobre las mismas features,
    con categorías de torneo fijadas de antemano (para que un torneo nuevo
    en el set de predicción no rompa el encoding)."""

    def __init__(self, categorias_torneo: list[str]):
        self.categorias_torneo = categorias_torneo
        self.xgb: XGBClassifier | None = None
        self.cb: CatBoostClassifier | None = None

    def _preparar_X(self, df: pd.DataFrame, para_catboost: bool) -> pd.DataFrame:
        X = df[COLUMNAS_MODELO].copy()
        if para_catboost:
            X["tournament"] = X["tournament"].astype(str)
        else:
            X["tournament"] = pd.Categorical(X["tournament"], categories=self.categorias_torneo)
        return X

    def fit(self, df_train: pd.DataFrame) -> "ModelosML":
        y = etiquetar_resultado(df_train)

        X_xgb = self._preparar_X(df_train, para_catboost=False)
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            enable_categorical=True, tree_method="hist",
            eval_metric="mlogloss", random_state=42,
        )
        self.xgb.fit(X_xgb, y)

        X_cb = self._preparar_X(df_train, para_catboost=True)
        idx_cat = [COLUMNAS_MODELO.index(c) for c in FEATURE_COLUMNAS_CATEGORICAS]
        self.cb = CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.05,
            loss_function="MultiClass", cat_features=idx_cat,
            random_state=42, verbose=False,
        )
        self.cb.fit(X_cb, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Devuelve probabilidades [P_local, P_empate, P_visita] de cada
        modelo, más el promedio simple (preview del stacking de Fase 4)."""
        probs_xgb = self.xgb.predict_proba(self._preparar_X(df, para_catboost=False))
        probs_cb = self.cb.predict_proba(self._preparar_X(df, para_catboost=True))
        probs_cb = probs_cb / probs_cb.sum(axis=1, keepdims=True)  # normaliza por seguridad
        probs_blend = (probs_xgb + probs_cb) / 2
        return {"xgboost": probs_xgb, "catboost": probs_cb, "blend": probs_blend}

    def importancia_features(self, top: int = 12) -> pd.Series:
        imp = pd.Series(self.xgb.feature_importances_, index=COLUMNAS_MODELO)
        return imp.sort_values(ascending=False).head(top)

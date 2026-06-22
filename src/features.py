"""
src/features.py
================
Feature engineering para XGBoost/CatBoost (Fase 3).

Principio rector: CERO fuga de información. Toda métrica de "forma reciente"
o "historial head-to-head" de un partido se calcula usando exclusivamente
partidos ANTERIORES a esa fecha (vía `.shift(1)` antes de cualquier rolling
o expanding, dentro de cada grupo ordenado cronológicamente). Si la fila del
partido N incluyera su propio resultado, el backtest quedaría inflado
artificialmente y el modelo sería inútil en producción.

Detalle importante para los 40 partidos del Mundial 2026: 32 de las 48
selecciones juegan dos de esos partidos en este lote. Para el segundo
partido de esas selecciones, su "forma reciente" se calcula salteando el
primer partido del lote (porque todavía no tiene resultado real) y usando el
último partido jugado de verdad -- pandas `.rolling()/.expanding()` ignoran
NaN automáticamente, así que esto sale gratis si los goles/puntos de los
partidos sin jugar se dejan como NaN (nunca como 0).

Features generadas:
- elo_local, elo_visita, elo_diff               (ya vienen de la Fase 1)
- forma_*_local / forma_*_visita                 (últimos 10 partidos de cada
                                                   selección, sin importar rival)
- h2h_*                                          (historial específico entre
                                                   ese local y ese visitante)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VENTANA_FORMA = 10  # partidos


def _long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo por partido (perspectiva propia). `df.index` debe
    ser un RangeIndex limpio: se usa como match_id para poder reconstruir
    el formato ancho después."""
    home = pd.DataFrame({
        "date": df["date"].values, "match_id": df.index, "lado": "local",
        "team": df["home_team"].values, "opponent": df["away_team"].values,
        "goles_favor": df["home_score"].values, "goles_contra": df["away_score"].values,
    })
    away = pd.DataFrame({
        "date": df["date"].values, "match_id": df.index, "lado": "visita",
        "team": df["away_team"].values, "opponent": df["home_team"].values,
        "goles_favor": df["away_score"].values, "goles_contra": df["home_score"].values,
    })
    long_df = pd.concat([home, away], ignore_index=True)

    sin_resultado = long_df["goles_favor"].isna() | long_df["goles_contra"].isna()
    puntos = np.where(long_df["goles_favor"] > long_df["goles_contra"], 3.0,
                       np.where(long_df["goles_favor"] == long_df["goles_contra"], 1.0, 0.0))
    long_df["puntos"] = np.where(sin_resultado, np.nan, puntos)
    long_df["dif_goles"] = long_df["goles_favor"] - long_df["goles_contra"]  # NaN se propaga sola
    return long_df


def _forma_reciente(long_df: pd.DataFrame, ventana: int = VENTANA_FORMA) -> pd.DataFrame:
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    g = long_df.groupby("team", group_keys=False)

    long_df["forma_goles_favor"] = g["goles_favor"].transform(
        lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
    long_df["forma_goles_contra"] = g["goles_contra"].transform(
        lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
    long_df["forma_puntos"] = g["puntos"].transform(
        lambda s: s.shift(1).rolling(ventana, min_periods=1).mean())
    long_df["forma_partidos_jugados"] = g["puntos"].transform(
        lambda s: s.shift(1).expanding().count())
    return long_df


def _head_to_head(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.sort_values(["team", "opponent", "date"]).reset_index(drop=True)
    g = long_df.groupby(["team", "opponent"], group_keys=False)

    long_df["h2h_partidos_jugados"] = g["puntos"].transform(lambda s: s.shift(1).expanding().count())
    long_df["h2h_dif_goles_prom"] = g["dif_goles"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["h2h_puntos_prom"] = g["puntos"].transform(lambda s: s.shift(1).expanding().mean())
    return long_df


COLS_FORMA = ["forma_goles_favor", "forma_goles_contra", "forma_puntos", "forma_partidos_jugados"]
COLS_H2H = ["h2h_partidos_jugados", "h2h_dif_goles_prom", "h2h_puntos_prom"]

RELLENOS_LOCAL = {
    "forma_goles_favor": 1.2, "forma_goles_contra": 1.2,
    "forma_puntos": 1.0, "forma_partidos_jugados": 0,
}
RELLENOS_H2H = {"h2h_partidos_jugados": 0, "h2h_dif_goles_prom": 0.0, "h2h_puntos_prom": 1.0}


def construir_features(df_entrada: pd.DataFrame) -> pd.DataFrame:
    """Recibe histórico + (opcionalmente) partidos por predecir concatenados
    y ordenados por fecha -- igual que `elo.calcular_elo_historico` -- y
    devuelve el mismo dataframe con columnas nuevas de forma/h2h. El índice
    de salida coincide 1:1 con el índice de `df_entrada` (se resetea primero
    para garantizar esa correspondencia)."""
    df = df_entrada.reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    long_df = _long_format(df)
    long_df = _forma_reciente(long_df)
    long_df = _head_to_head(long_df)

    cols = ["match_id", "lado"] + COLS_FORMA + COLS_H2H
    locales = long_df.loc[long_df["lado"] == "local", cols].set_index("match_id")
    visitas = long_df.loc[long_df["lado"] == "visita", cols].set_index("match_id")

    df = df.join(locales[COLS_FORMA].add_suffix("_local"))
    df = df.join(locales[COLS_H2H])  # h2h ya es direccional (local -> visita)
    df = df.join(visitas[COLS_FORMA].add_suffix("_visita"))

    rellenos = {f"{k}_local": v for k, v in RELLENOS_LOCAL.items()}
    rellenos.update({f"{k}_visita": v for k, v in RELLENOS_LOCAL.items()})
    rellenos.update(RELLENOS_H2H)
    df = df.fillna(rellenos)

    df["elo_diff"] = df["elo_local_antes"] - df["elo_visita_antes"]
    return df


FEATURE_COLUMNS_NUMERICAS = (
    ["elo_local_antes", "elo_visita_antes", "elo_diff"]
    + [f"{c}_local" for c in COLS_FORMA] + [f"{c}_visita" for c in COLS_FORMA]
    + COLS_H2H
)
FEATURE_COLUMNAS_CATEGORICAS = ["tournament"]

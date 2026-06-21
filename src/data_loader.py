"""
data_loader.py
==============
Carga y limpieza de los datasets de martj42/international_results.

Issues encontrados en la exploración inicial (ver pseudocodigo.py original) y
cómo se resuelven aquí:

1. `home_score`/`away_score` con 40 nulos
   -> NO son datos corruptos. Son los 40 partidos del Mundial 2026 (matchday 1,
      2026-06-20 a 2026-06-27) que todavía no se han jugado. Son exactamente el
      "target" que queremos predecir, así que NUNCA se eliminan: se separan en
      un dataframe propio (`partidos_a_predecir`).
2. Filtro "futuros = fecha > hoy"
   -> Si se eliminan, se borra el objetivo del proyecto. Se reemplaza por un
      split explícito: histórico (resultado conocido) vs. por_predecir
      (resultado nulo).
3. Nombres de selecciones que cambiaron de nombre
   -> Se usa former_names.csv para unificar la identidad histórica (ej.
      Macedonia -> North Macedonia, Swaziland -> Eswatini) SOLO cuando hay
      sucesor 1:1. Casos de ruptura múltiple (Czechoslovakia, Yugoslavia,
      German DR) se dejan como entidades separadas a propósito: no tienen
      un único sucesor legítimo y mezclarlas distorsionaría el rating.
4. Penales (shootouts.csv)
   -> Un empate decidido por penales sigue siendo 1-1 (o el marcador que sea)
      a efectos de goles. Se agrega una columna booleana `decided_by_shootout`
      para que los modelos de goles (Poisson/Dixon-Coles) usen el marcador
      real y el Elo trate el resultado como empate (convención estándar).
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _cargar_csv(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / nombre)


def estandarizar_nombres_equipos(df: pd.DataFrame, df_former_names: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza nombres históricos por el nombre actual, respetando el rango
    de fechas en el que aplicaba cada nombre (former_names.csv).

    Solo se aplican los mapeos 1:1 (un solo `current` por `former`). No se
    fusionan entidades con múltiples sucesores.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for _, fila in df_former_names.iterrows():
        former = fila["former"]
        current = fila["current"]
        inicio = pd.to_datetime(fila["start_date"])
        fin = pd.to_datetime(fila["end_date"])

        en_rango = (df["date"] >= inicio) & (df["date"] <= fin)
        df.loc[en_rango & (df["home_team"] == former), "home_team"] = current
        df.loc[en_rango & (df["away_team"] == former), "away_team"] = current

    return df


def cargar_datos(verbose: bool = True):
    """Carga los 4 CSV, limpia resultados y separa histórico vs. partidos
    por predecir (Mundial 2026).

    Returns
    -------
    dict con:
        'historico'         : partidos con resultado conocido, ordenados por fecha
        'por_predecir'       : partidos sin resultado (target del proyecto)
        'shootouts'          : df de penales
        'goalscorers'        : df de goleadores
        'former_names'       : df de nombres históricos (ya usado internamente)
    """
    df_results = _cargar_csv("results.csv")
    df_shootouts = _cargar_csv("shootouts.csv")
    df_goalscorers = _cargar_csv("goalscorers.csv")
    df_former_names = _cargar_csv("former_names.csv")

    df_results["date"] = pd.to_datetime(df_results["date"])

    # 1. Estandarizar nombres de equipos (mapeos 1:1 con vigencia por fecha)
    df_results = estandarizar_nombres_equipos(df_results, df_former_names)

    # 2. Marcar partidos decididos por penales (siguen siendo el marcador real)
    df_shootouts["date"] = pd.to_datetime(df_shootouts["date"])
    llave = ["date", "home_team", "away_team"]
    shootout_keys = set(map(tuple, df_shootouts[llave].values))
    df_results["decided_by_shootout"] = (
        df_results[llave].apply(tuple, axis=1).isin(shootout_keys)
    )

    # 3. Separar histórico vs. por predecir (no se descarta nada)
    mascara_sin_resultado = df_results["home_score"].isna() | df_results["away_score"].isna()
    df_por_predecir = df_results[mascara_sin_resultado].copy()
    df_historico = df_results[~mascara_sin_resultado].copy()

    # 4. Orden cronológico estricto (requisito para Elo y cualquier feature temporal)
    df_historico = df_historico.sort_values("date").reset_index(drop=True)
    df_por_predecir = df_por_predecir.sort_values("date").reset_index(drop=True)

    if verbose:
        print(f"Histórico (con resultado):     {len(df_historico):>6} partidos "
              f"({df_historico['date'].min().date()} -> {df_historico['date'].max().date()})")
        print(f"Por predecir (sin resultado):  {len(df_por_predecir):>6} partidos "
              f"({df_por_predecir['date'].min().date()} -> {df_por_predecir['date'].max().date()})")
        print(f"Decididos por penales:         {df_historico['decided_by_shootout'].sum():>6}")
        n_equipos = pd.concat([df_historico["home_team"], df_historico["away_team"]]).nunique()
        print(f"Selecciones distintas:         {n_equipos:>6}")

    return {
        "historico": df_historico,
        "por_predecir": df_por_predecir,
        "shootouts": df_shootouts,
        "goalscorers": df_goalscorers,
        "former_names": df_former_names,
    }


if __name__ == "__main__":
    cargar_datos()

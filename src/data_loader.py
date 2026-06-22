from __future__ import annotations

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _cargar_csv(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / nombre)


def estandarizar_nombres_equipos(df: pd.DataFrame, df_former_names: pd.DataFrame) -> pd.DataFrame:
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
    df_results = _cargar_csv("results.csv")
    df_shootouts = _cargar_csv("shootouts.csv")
    df_goalscorers = _cargar_csv("goalscorers.csv")
    df_former_names = _cargar_csv("former_names.csv")

    df_results["date"] = pd.to_datetime(df_results["date"])


    df_results = estandarizar_nombres_equipos(df_results, df_former_names)


    df_shootouts["date"] = pd.to_datetime(df_shootouts["date"])
    llave = ["date", "home_team", "away_team"]
    shootout_keys = set(map(tuple, df_shootouts[llave].values))
    df_results["decided_by_shootout"] = (
        df_results[llave].apply(tuple, axis=1).isin(shootout_keys)
    )


    mascara_sin_resultado = df_results["home_score"].isna() | df_results["away_score"].isna()
    df_por_predecir = df_results[mascara_sin_resultado].copy()
    df_historico = df_results[~mascara_sin_resultado].copy()


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

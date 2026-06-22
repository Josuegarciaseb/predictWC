"""
scripts/01_pipeline_elo.py
===========================
Fase 1 del proyecto: pipeline de datos + Elo dinámico.

Qué hace:
1. Carga y limpia los 4 CSV (data_loader.py)
2. Concatena histórico + partidos por predecir y corre el Elo dinámico sobre
   toda la serie temporal (así los partidos del Mundial 2026 quedan con su
   Elo "antes del partido" calculado correctamente, sin tocar resultados
   reales).
3. Guarda:
   - data/processed/historico_con_elo.csv      (dataset con features Elo,
     listo para alimentar Poisson/Dixon-Coles/XGBoost en fases siguientes)
   - outputs/ranking_elo_actual.csv             (top 25 selecciones por Elo)
   - outputs/predicciones_fase1_mundial2026.csv (los 40 partidos a predecir,
     con elo_local, elo_visita y la probabilidad esperada según Elo puro,
     como baseline -- el "marcador más probable" llegará en la Fase 2 con
     Poisson/Dixon-Coles)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from data_loader import cargar_datos
from elo import calcular_elo_historico, tabla_ranking_actual

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def E_local_pura(elo_local: float, elo_visita: float, neutral: bool, ventaja: int = 100) -> float:
    ventaja_aplicada = 0 if neutral else ventaja
    diferencia = (elo_local + ventaja_aplicada) - elo_visita
    return 1 / (1 + 10 ** (-diferencia / 400))


def main():
    datos = cargar_datos()
    historico = datos["historico"]
    por_predecir = datos["por_predecir"]

    # Concatenar para que el Elo de los partidos a predecir arranque
    # exactamente donde quedó el histórico.
    columnas_comunes = ["date", "home_team", "away_team", "home_score", "away_score",
                        "tournament", "city", "country", "neutral", "decided_by_shootout"]
    todo = pd.concat([historico[columnas_comunes], por_predecir[columnas_comunes]],
                      ignore_index=True)

    todo_con_elo = calcular_elo_historico(todo)
    elo_final_pre_mundial = todo_con_elo[todo_con_elo["home_score"].notna()].attrs.get("elo_final")

    # Volver a separar
    historico_con_elo = todo_con_elo[todo_con_elo["home_score"].notna()].copy()
    predicciones = todo_con_elo[todo_con_elo["home_score"].isna()].copy()

    # --- Guardar dataset procesado (insumo de las fases 2-5) ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    historico_con_elo.to_csv(PROCESSED_DIR / "historico_con_elo.csv", index=False)
    predicciones.to_csv(PROCESSED_DIR / "partidos_a_predecir.csv", index=False)

    # --- Ranking actual ---
    elo_final = calcular_elo_historico(historico)
    ranking = tabla_ranking_actual(elo_final.attrs["elo_final"], top=25)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUTS_DIR / "ranking_elo_actual.csv")

    print("=== Top 25 Elo (con todo el histórico hasta el Mundial 2026) ===")
    print(ranking.to_string())

    # --- Tabla de predicciones (Fase 1: baseline solo-Elo) ---
    predicciones = predicciones.copy()
    predicciones["prob_local_elo"] = predicciones.apply(
        lambda r: round(E_local_pura(r["elo_local_antes"], r["elo_visita_antes"], r["neutral"]), 3),
        axis=1,
    )
    predicciones["prob_visita_elo"] = (1 - predicciones["prob_local_elo"]).round(3)
    tabla_final = predicciones[[
        "date", "home_team", "away_team", "elo_local_antes", "elo_visita_antes",
        "prob_local_elo", "prob_visita_elo",
    ]].rename(columns={
        "date": "fecha", "home_team": "local", "away_team": "visitante",
        "elo_local_antes": "elo_local", "elo_visita_antes": "elo_visitante",
    })
    tabla_final["elo_local"] = tabla_final["elo_local"].round(1)
    tabla_final["elo_visitante"] = tabla_final["elo_visitante"].round(1)
    tabla_final.to_csv(OUTPUTS_DIR / "predicciones_fase1_mundial2026.csv", index=False)

    print("\n=== Fase 1 -- baseline Elo para los 40 partidos del Mundial 2026 ===")
    print(tabla_final.to_string(index=False))

    print(f"\nGuardado: {PROCESSED_DIR / 'historico_con_elo.csv'} "
          f"({len(historico_con_elo)} filas, {historico_con_elo.shape[1]} columnas)")
    print(f"Guardado: {PROCESSED_DIR / 'partidos_a_predecir.csv'}")
    print(f"Guardado: {OUTPUTS_DIR / 'ranking_elo_actual.csv'}")
    print(f"Guardado: {OUTPUTS_DIR / 'predicciones_fase1_mundial2026.csv'}")


if __name__ == "__main__":
    main()

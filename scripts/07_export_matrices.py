"""
FASE 2 (export web) -- Matrices de marcador a JSON para el dashboard.

Exporta, para cada partido del Mundial 2026, la matriz de probabilidad de
marcador exacto (goles de local x goles de visita) de Dixon-Coles a
outputs/matrices_marcador.json, que el sitio (web/) consume para dibujar
el heatmap de distribución de probabilidades al hacer clic en un partido.

Uso:
    python scripts/07_export_matrices.py

Salida (JSON):
    {
      "Germany|Ivory Coast": {
        "max": 6,
        "matriz": [[P(loc=0,vis=0), ..., P(loc=0,vis=6)], ...],  # 7x7
        "prob_local": 0.53, "prob_empate": 0.26, "prob_visita": 0.21,
        "marcador": "1-0", "prob_marcador": 0.127
      },
      ...
    }
matriz[i][j] = P(goles_local = i, goles_visita = j)
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from models.poisson_dixon_coles import DixonColesModel

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

MAX_GOLES = 6  # heatmap 0..6 en cada eje (como el ejemplo de StatisKicks)


def main():
    historico = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])
    a_predecir = pd.read_csv(PROCESSED_DIR / "partidos_a_predecir.csv", parse_dates=["date"])

    modelo = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    modelo.fit(historico)

    salida = {}
    n = MAX_GOLES + 1
    for fila in a_predecir.itertuples(index=False):
        local, visitante = fila.home_team, fila.away_team
        M = modelo.matriz_marcador(local, visitante, neutral=bool(fila.neutral))[:n, :n]
        pred = modelo.predecir_partido(local, visitante, neutral=bool(fila.neutral))
        salida[f"{local}|{visitante}"] = {
            "max": MAX_GOLES,
            "matriz": [[round(float(M[i, j]), 5) for j in range(n)] for i in range(n)],
            "prob_local": round(pred["prob_local"], 3),
            "prob_empate": round(pred["prob_empate"], 3),
            "prob_visita": round(pred["prob_visita"], 3),
            "marcador": pred["marcador_mas_probable"],
            "prob_marcador": round(pred["prob_marcador_exacto"], 3),
        }

    ruta = OUTPUTS_DIR / "matrices_marcador.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {ruta}  ({len(salida)} partidos)")


if __name__ == "__main__":
    main()

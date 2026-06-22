"""
scripts/02_pipeline_poisson_dixon_coles.py
============================================
Fase 2: ajusta el modelo Poisson + Dixon-Coles, lo valida con un backtest
temporal (para no engañarnos con métricas infladas) y genera la tabla final
con el marcador más probable de cada uno de los 40 partidos del Mundial 2026.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from models.poisson_dixon_coles import DixonColesModel

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def log_loss_3clases(y_true_idx: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(probs[np.arange(len(y_true_idx)), y_true_idx], eps, 1 - eps)
    return float(-np.mean(np.log(p)))


def backtest(historico: pd.DataFrame, meses_holdout: int = 18):
    """Entrena con todo lo anterior al período de holdout y evalúa en los
    últimos `meses_holdout` meses de partidos con resultado conocido."""
    historico = historico.copy()
    historico["date"] = pd.to_datetime(historico["date"])
    fecha_max = historico["date"].max()
    fecha_corte = fecha_max - pd.DateOffset(months=meses_holdout)

    holdout = historico[historico["date"] > fecha_corte]
    print(f"\nBacktest: entrenando con partidos <= {fecha_corte.date()}, "
          f"evaluando {len(holdout)} partidos posteriores ({fecha_corte.date()} -> {fecha_max.date()})")

    modelo_bt = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    modelo_bt.fit(historico, fecha_corte=str(fecha_corte.date()))

    y_true_idx = []  # 0=local gana, 1=empate, 2=visita gana
    probs = []
    for fila in holdout.itertuples(index=False):
        pred = modelo_bt.predecir_partido(fila.home_team, fila.away_team, bool(fila.neutral))
        probs.append([pred["prob_local"], pred["prob_empate"], pred["prob_visita"]])
        if fila.home_score > fila.away_score:
            y_true_idx.append(0)
        elif fila.home_score == fila.away_score:
            y_true_idx.append(1)
        else:
            y_true_idx.append(2)

    y_true_idx = np.array(y_true_idx)
    probs = np.array(probs)
    pred_idx = probs.argmax(axis=1)

    accuracy_modelo = float((pred_idx == y_true_idx).mean())
    logloss_modelo = log_loss_3clases(y_true_idx, probs)

    # Baselines de referencia
    distrib_real = np.bincount(y_true_idx, minlength=3) / len(y_true_idx)
    clase_mayoritaria = distrib_real.argmax()
    accuracy_mayoritaria = float((y_true_idx == clase_mayoritaria).mean())
    logloss_mayoritaria = log_loss_3clases(
        y_true_idx, np.tile(distrib_real, (len(y_true_idx), 1))
    )

    print(f"  Distribución real (L/E/V):     {distrib_real.round(3)}")
    print(f"  Modelo Dixon-Coles  -> accuracy: {accuracy_modelo:.3f}  |  log-loss: {logloss_modelo:.3f}")
    print(f"  Baseline ingenuo*   -> accuracy: {accuracy_mayoritaria:.3f}  |  log-loss: {logloss_mayoritaria:.3f}")
    print("  *Baseline ingenuo = siempre predecir la distribución histórica real (L/E/V) sin mirar los equipos.")


def main():
    historico = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])
    a_predecir = pd.read_csv(PROCESSED_DIR / "partidos_a_predecir.csv", parse_dates=["date"])

    # --- Validación honesta antes de confiar en el modelo final ---
    backtest(historico, meses_holdout=18)

    # --- Ajuste final con todo el histórico disponible ---
    modelo = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    modelo.fit(historico)

    print(f"\nVentaja de local: equipo local anota {np.exp(modelo.home_adv_):.2f}x más que en cancha neutral")
    print(f"Rho (corrección Dixon-Coles marcadores bajos): {modelo.rho_:.4f}")
    print("\n=== Top 15 selecciones por índice ofensivo (ataque - defensa) ===")
    print(modelo.ranking_ataque_defensa(15).round(3).to_string())

    # --- Predicciones de los 40 partidos del Mundial 2026 ---
    filas = []
    for fila in a_predecir.itertuples(index=False):
        pred = modelo.predecir_partido(fila.home_team, fila.away_team, bool(fila.neutral))
        top3_str = " | ".join(
            f"{g1}-{g2} ({p*100:.1f}%)" for g1, g2, p in pred["top3_marcadores"]
        )
        filas.append({
            "fecha": fila.date.date(),
            "local": fila.home_team,
            "visitante": fila.away_team,
            "elo_local": round(fila.elo_local_antes, 1),
            "elo_visitante": round(fila.elo_visita_antes, 1),
            "marcador_mas_probable": pred["marcador_mas_probable"],
            "prob_marcador_exacto": round(pred["prob_marcador_exacto"], 3),
            "prob_local": round(pred["prob_local"], 3),
            "prob_empate": round(pred["prob_empate"], 3),
            "prob_visita": round(pred["prob_visita"], 3),
            "goles_esperados_local": round(pred["goles_esperados_local"], 2),
            "goles_esperados_visita": round(pred["goles_esperados_visita"], 2),
            "top3_marcadores": top3_str,
        })

    tabla = pd.DataFrame(filas)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUTS_DIR / "predicciones_fase2_poisson_dixon_coles.csv", index=False)

    print("\n=== Fase 2 -- marcador más probable, Mundial 2026 (matchday 1) ===")
    cols_mostrar = ["fecha", "local", "visitante", "marcador_mas_probable",
                     "prob_local", "prob_empate", "prob_visita"]
    print(tabla[cols_mostrar].to_string(index=False))

    print(f"\nGuardado: {OUTPUTS_DIR / 'predicciones_fase2_poisson_dixon_coles.csv'}")


if __name__ == "__main__":
    main()

"""Fase D — Pipeline de tiros a puerta por jugador (modelo jerárquico).

1. Validación walk-forward: ¿conocer al jugador (efecto aleatorio) y al rival
   mejora la predicción de tiros a puerta frente a la tasa global? Se evalúa con
   los minutos reales de cada jugador (aísla el modelo de tasa, no la alineación).
   Métricas: log-loss del conteo y O/U 0.5 (¿registra al menos un tiro a puerta?).
2. Ranking de los mayores tiradores (sustituye a la tabla de fixtures: sin once
   probable del Mundial 2026 no se puede tabular por partido).

Uso:
    python scripts/D01_pipeline_tiros_jugador.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.tiros_jugador import TirosJugadorModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
EPS = 1e-12
LINEA = 0.5


def _ll(p) -> float:
    return float(-np.log(np.clip(p, EPS, 1.0)))


def walk_forward(df: pd.DataFrame, min_train_dates_matches: int = 200) -> None:
    df = df[df["date"].dt.year >= 2018].sort_values("date").reset_index(drop=True)
    df = df[df["minutes"] > 0]
    fechas = sorted(df["date"].unique())

    res = {"full": {"ll": [], "ou_ll": [], "ou_acc": []},
           "solo_jug": {"ll": [], "ou_ll": [], "ou_acc": []},
           "baseline": {"ll": [], "ou_ll": [], "ou_acc": []}}
    n, n_con_datos = 0, 0
    res_con = {"full": [], "baseline": []}  # O/U acc en jugadores con historial

    for d in fechas:
        train = df[df["date"] < d]
        if len(train) < min_train_dates_matches:
            continue
        test = df[df["date"] == d]
        modelo = TirosJugadorModel().fit(train, fecha_corte=str(pd.Timestamp(d).date()))
        lam0 = modelo.lambda0_

        for f in test.itertuples(index=False):
            obs = int(f.sot)
            e = f.minutes / 90.0
            over_real = obs > LINEA

            pred = modelo.predecir(f.player, f.opponent, f.minutes)
            pmf = pred["pmf"]
            p_full = pmf[obs] if obs < len(pmf) else EPS
            po_full = pred["prob_over"][LINEA]

            # Solo jugador (sin rival): phi=1.
            pred_pj = modelo.predecir(f.player, "___sin_rival___", f.minutes)
            pmf_pj = pred_pj["pmf"]
            p_pj = pmf_pj[obs] if obs < len(pmf_pj) else EPS
            po_pj = pred_pj["prob_over"][LINEA]

            # Baseline: Poisson(lam0 * e), sin info de jugador ni rival.
            mu_b = lam0 * e
            p_b = float(poisson.pmf(obs, mu_b))
            po_b = float(1 - poisson.pmf(0, mu_b))

            res["full"]["ll"].append(_ll(p_full))
            res["full"]["ou_ll"].append(_ll(po_full if over_real else 1 - po_full))
            res["full"]["ou_acc"].append((po_full > 0.5) == over_real)
            res["solo_jug"]["ll"].append(_ll(p_pj))
            res["solo_jug"]["ou_ll"].append(_ll(po_pj if over_real else 1 - po_pj))
            res["solo_jug"]["ou_acc"].append((po_pj > 0.5) == over_real)
            res["baseline"]["ll"].append(_ll(p_b))
            res["baseline"]["ou_ll"].append(_ll(po_b if over_real else 1 - po_b))
            res["baseline"]["ou_acc"].append((po_b > 0.5) == over_real)

            if pred["tiene_datos"]:
                n_con_datos += 1
                res_con["full"].append((po_full > 0.5) == over_real)
                res_con["baseline"].append((po_b > 0.5) == over_real)
            n += 1

    print(f"\n=== Walk-forward tiros a puerta por jugador (2018+): {n} jugador-partidos ===")
    print(f"  ({n_con_datos} con historial previo del jugador; el resto cae al promedio)\n")
    print(f"  {'Variante':<16}{'Conteo LL':>11}{'O/U0.5 LL':>11}{'O/U0.5 acc':>12}")
    print(f"  {'-'*50}")
    for k, lab in [("full", "Jugador+Rival"), ("solo_jug", "Solo jugador"), ("baseline", "Baseline global")]:
        print(f"  {lab:<16}{np.mean(res[k]['ll']):>11.3f}{np.mean(res[k]['ou_ll']):>11.3f}"
              f"{np.mean(res[k]['ou_acc']):>12.3f}")
    print(f"\n  Solo en jugadores CON historial ({n_con_datos}): O/U 0.5 accuracy")
    print(f"    Jugador+Rival : {np.mean(res_con['full']):.3f}")
    print(f"    Baseline      : {np.mean(res_con['baseline']):.3f}")
    print("  Lectura: el efecto jugador aporta señal real; el efecto rival es débil.")
    print("  Recordatorio: esto asume minutos conocidos. Sin once probable la")
    print("  incertidumbre de alineación domina la predicción real para el Mundial.")


def ranking_final(df: pd.DataFrame) -> None:
    modelo = TirosJugadorModel().fit(df)
    print(f"\nTasa global de tiros a puerta por 90': {modelo.lambda0_:.3f}")
    print(f"Encogimiento (forma del prior): k_jugador={modelo.k_player_:.2f}, "
          f"k_rival={modelo.k_opp_:.2f}")
    rank = modelo.ranking_tiradores(top=20, min_exp90=3.0)
    rank["sot_por_90"] = rank["sot_por_90"].round(3)
    rank["theta"] = rank["theta"].round(2)
    rank["exp90"] = rank["exp90"].round(1)
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / "ranking_tiros_jugador.csv"
    rank.to_csv(ruta, index=False, encoding="utf-8")
    print("\n=== Top 20 tiradores a puerta por 90' (>=3 partidos-90, encogido) ===")
    print(rank.to_string(index=False))
    print(f"\nGuardado: {ruta}")


def main() -> None:
    df = pd.read_csv(PROC / "statsbomb_jugador_minutos.csv", parse_dates=["date"])
    walk_forward(df)
    ranking_final(df)


if __name__ == "__main__":
    main()

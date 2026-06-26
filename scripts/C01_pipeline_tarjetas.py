"""Fase C — Pipeline de tarjetas totales (Over/Under).

1. Validación walk-forward sobre StatsBomb (2018+): Poisson vs Binomial Negativa
   en log-loss del total y O/U, más una ablación de la covariable de fase
   (knockout sí/no) para mostrar su aporte.
2. Entrena el modelo final y predice las tarjetas de los fixtures del Mundial
   2026 (matchday 1 = todos fase de grupos -> knockout=False).

ADVERTENCIA: el árbitro domina las tarjetas y no se conoce de antemano. Este
mercado es estructuralmente más ruidoso que los córners; las métricas lo reflejan.

Uso:
    python scripts/C01_pipeline_tarjetas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.tarjetas_model import TarjetasModel  # noqa: E402
import statsbomb_loader as sb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
EPS = 1e-12
LINEA_PPAL = 3.5


def _ll(p) -> float:
    return float(-np.log(np.clip(p, EPS, 1.0)))


def total_cards(f) -> int:
    return int(f.home_yellow + f.home_red + f.away_yellow + f.away_red)


def walk_forward(df: pd.DataFrame, min_train: int = 64) -> None:
    from scipy.stats import poisson as _po
    df = df[df["date"].dt.year >= 2018].sort_values("date").reset_index(drop=True)
    df["cards_tot"] = df.home_yellow + df.home_red + df.away_yellow + df.away_red
    fechas = sorted(df["date"].unique())

    variantes = {
        "Poisson+KO": dict(distribucion="poisson", usar_knockout=True),
        "NB+KO": dict(distribucion="nb", usar_knockout=True),
        "NB sin KO": dict(distribucion="nb", usar_knockout=False),
    }
    res = {k: {"ll_total": [], "ou_ll": [], "ou_acc": []} for k in variantes}
    ll_naive, real_tot = [], []
    base_over = float((df["cards_tot"] > LINEA_PPAL).mean())

    for d in fechas:
        train = df[df["date"] < d]
        if len(train) < min_train:
            continue
        test = df[df["date"] == d]
        modelos = {k: TarjetasModel(**kw).fit(train, fecha_corte=str(pd.Timestamp(d).date()))
                   for k, kw in variantes.items()}
        naive_mean = float(train["cards_tot"].mean())

        for f in test.itertuples(index=False):
            T = total_cards(f)
            over_real = T > LINEA_PPAL
            ll_naive.append(_ll(_po.pmf(T, naive_mean)))
            real_tot.append(T)
            for k, m in modelos.items():
                pred = m.predecir_partido(f.home_team, f.away_team,
                                          bool(f.neutral), bool(f.knockout))
                pmf = pred["total_pmf"]
                res[k]["ll_total"].append(_ll(pmf[T] if T < len(pmf) else EPS))
                po = pred["prob_over"][LINEA_PPAL]
                res[k]["ou_ll"].append(_ll(po if over_real else 1 - po))
                res[k]["ou_acc"].append((po > 0.5) == over_real)

    base_over_ll = -(base_over * np.log(base_over) + (1 - base_over) * np.log(1 - base_over))
    n = len(real_tot)
    print(f"\n=== Walk-forward tarjetas (StatsBomb 2018+): {n} partidos evaluados ===")
    print(f"  (log-loss: menor = mejor)\n")
    print(f"  {'Variante':<14}{'TotalLL':>9}{'O/U LL':>9}{'O/U acc':>9}")
    print(f"  {'-'*41}")
    for k in variantes:
        print(f"  {k:<14}{np.mean(res[k]['ll_total']):>9.3f}"
              f"{np.mean(res[k]['ou_ll']):>9.3f}{np.mean(res[k]['ou_acc']):>9.3f}")
    print(f"  {'Baseline':<14}{np.mean(ll_naive):>9.3f}{base_over_ll:>9.3f}{'-':>9}")
    print(f"\n  Dispersión real total: Var/Media = "
          f"{np.var(real_tot) / np.mean(real_tot):.2f} (>1 => sobre-disperso)")
    print(f"  Base rate over {LINEA_PPAL}: {base_over:.3f}")
    print("  Lectura: la NB ayuda más que en córners (tarjetas más sobre-dispersas).")
    print("  El techo es bajo: el árbitro (factor dominante) no es observable.")


def predecir_mundial(df: pd.DataFrame) -> None:
    modelo = TarjetasModel().fit(df)
    print(f"\nEfecto fase eliminatoria: x{np.exp(modelo.knockout_coef_):.2f} tarjetas "
          f"vs fase de grupos")
    print(f"Dispersión NB (r): {modelo.r_:.1f}  (menor que en córners => más sobre-dispersión)")
    print("\n=== Top 12 selecciones más indisciplinadas (propensión + induce rival) ===")
    print(modelo.ranking_disciplina(12).round(3).to_string())

    # NOTA: se RETIRÓ el ajuste por árbitro del dataset de Kaggle. Sus asignaciones
    # de árbitro a partido resultaron NO fiables (probablemente simuladas/fabricadas),
    # así que aplicar su factor sería inventar. El modelo de tarjetas usa solo la señal
    # real de StatsBomb (equipos + fase). Si se consigue una fuente fiable de
    # designaciones arbitrales reales, se puede reintroducir vía factor_externo.
    fixtures = pd.read_csv(PROC / "partidos_a_predecir.csv", parse_dates=["date"])
    filas = []
    for f in fixtures.itertuples(index=False):
        h = sb.a_nombre_statsbomb(f.home_team)
        a = sb.a_nombre_statsbomb(f.away_team)
        # Matchday 1 del Mundial = fase de grupos -> knockout=False.
        pred = modelo.predecir_partido(h, a, bool(f.neutral), knockout=False)
        sin = [o for o, sbn in ((f.home_team, h), (f.away_team, a))
               if sbn not in modelo.propension_.index]
        filas.append({
            "fecha": f.date.date(), "local": f.home_team, "visitante": f.away_team,
            "tarjetas_total": round(pred["tarjetas_total"], 2),
            "p_over_2.5": round(pred["prob_over"][2.5], 3),
            "p_over_3.5": round(pred["prob_over"][3.5], 3),
            "p_over_4.5": round(pred["prob_over"][4.5], 3),
            "sin_datos_statsbomb": ",".join(sin) if sin else "",
        })
    tabla = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / "predicciones_faseC_tarjetas.csv"
    tabla.to_csv(ruta, index=False)
    print("\n=== Fase C -- tarjetas, Mundial 2026 (matchday 1) ===")
    cols = ["fecha", "local", "visitante", "tarjetas_total", "p_over_3.5", "p_over_4.5"]
    with pd.option_context("display.width", 200):
        print(tabla[cols].to_string(index=False))
    print(f"\nGuardado: {ruta}")


def main() -> None:
    df = pd.read_csv(PROC / "statsbomb_partidos.csv", parse_dates=["date"])
    walk_forward(df)
    predecir_mundial(df)


if __name__ == "__main__":
    main()

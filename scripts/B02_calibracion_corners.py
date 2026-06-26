"""Fase B (calibración) — Afinar las líneas Over de córners.

Diagnóstico de calibración del total de córners y elección de un multiplicador de
localización 'c' (córners esperados -> lam*c, mu*c) que corrija el sesgo
sistemático medido out-of-sample, sin tocar la dispersión.

Procedimiento honesto:
  1. Walk-forward una sola vez: por partido se guarda (lam, mu, r, total real).
  2. Se diagnostica el sesgo de localización y se separa deriva temporal vs
     estructural (sesgo por torneo de test).
  3. Tabla de fiabilidad por línea: P(over) media del modelo vs frecuencia real.
  4. Se barre c analíticamente (recomputando la pmf, sin re-entrenar) y se elige
     el que minimiza el log-loss medio de O/U en las líneas centrales.

Uso:
    python scripts/B02_calibracion_corners.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.corners_dixon_coles import CornersModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EPS = 1e-12
LINEAS = (7.5, 8.5, 9.5, 10.5, 11.5)
LINEAS_CENTRALES = (8.5, 9.5, 10.5)
MAX_C = 25


def _ll(p) -> float:
    return float(-np.log(np.clip(p, EPS, 1.0)))


def _pmf(mean, r, dist):
    k = np.arange(0, MAX_C + 1)
    p = poisson.pmf(k, mean) if dist == "poisson" else nbinom.pmf(k, r, r / (r + mean))
    return p / p.sum()


def total_pmf(lam, mu, r, c, dist="nb"):
    return np.convolve(_pmf(lam * c, r, dist), _pmf(mu * c, r, dist))


def recolectar(df: pd.DataFrame, min_train: int = 64) -> pd.DataFrame:
    df = df[df["date"].dt.year >= 2018].sort_values("date").reset_index(drop=True)
    fechas = sorted(df["date"].unique())
    filas = []
    for d in fechas:
        train = df[df["date"] < d]
        if len(train) < min_train:
            continue
        test = df[df["date"] == d]
        # calibracion=1.0 explícito: este script ESTIMA c, no debe partir de él.
        m = CornersModel(calibracion=1.0).fit(train, fecha_corte=str(pd.Timestamp(d).date()))
        for f in test.itertuples(index=False):
            lam, mu = m.lambda_mu(f.home_team, f.away_team, bool(f.neutral))
            filas.append({"lam": lam, "mu": mu, "r": m.r_,
                          "total": int(f.home_corners + f.away_corners),
                          "tournament": f.tournament, "season": f.season})
    return pd.DataFrame(filas)


def ou_logloss(rec: pd.DataFrame, c: float, lineas) -> float:
    lls = []
    for row in rec.itertuples(index=False):
        pmf = total_pmf(row.lam, row.mu, row.r, c)
        sop = np.arange(len(pmf))
        for ln in lineas:
            po = float(pmf[sop > ln].sum())
            over = row.total > ln
            lls.append(_ll(po if over else 1 - po))
    return float(np.mean(lls))


def main() -> None:
    df = pd.read_csv(PROC / "statsbomb_partidos.csv", parse_dates=["date"])
    rec = recolectar(df)
    n = len(rec)
    pred_mean = (rec["lam"] + rec["mu"]).mean()
    real_mean = rec["total"].mean()
    print(f"\n=== Calibración de córners (walk-forward, {n} partidos) ===")
    print(f"  Total predicho (c=1): {pred_mean:.2f}  |  real: {real_mean:.2f}  "
          f"|  ratio real/pred = {real_mean / pred_mean:.3f}")

    print("\n  -- Sesgo por torneo de test (¿deriva temporal o estructural?) --")
    g = rec.assign(pred=rec.lam + rec.mu).groupby(["tournament", "season"]).agg(
        n=("total", "size"), pred=("pred", "mean"), real=("total", "mean"))
    g["gap"] = (g["real"] - g["pred"]).round(2)
    print(g.round(2).to_string())

    print("\n  -- Fiabilidad por línea (c=1) --")
    print(f"  {'línea':>6}{'P(over) modelo':>16}{'frec. real':>12}{'gap':>8}")
    for ln in LINEAS:
        pos = [float(np.convolve(_pmf(row.lam, row.r, 'nb'), _pmf(row.mu, row.r, 'nb'))
                     [np.arange(2 * MAX_C + 1) > ln].sum()) for row in rec.itertuples(index=False)]
        emp = float((rec["total"] > ln).mean())
        print(f"  {ln:>6}{np.mean(pos):>16.3f}{emp:>12.3f}{emp - np.mean(pos):>8.3f}")

    print("\n  -- Barrido del multiplicador c (log-loss O/U en líneas centrales) --")
    base = ou_logloss(rec, 1.0, LINEAS_CENTRALES)
    mejor_c, mejor_ll = 1.0, base
    for c in np.round(np.arange(1.00, 1.121, 0.02), 3):
        ll = ou_logloss(rec, float(c), LINEAS_CENTRALES)
        marca = "  <-- mejor" if ll < mejor_ll else ""
        if ll < mejor_ll:
            mejor_c, mejor_ll = float(c), ll
        print(f"     c={c:.2f}  ->  O/U log-loss = {ll:.4f}{marca}")
    print(f"\n  c=1.00 (sin corrección): {base:.4f}")
    print(f"  Mejor c = {mejor_c:.2f}  ->  {mejor_ll:.4f}  "
          f"(mejora {(base - mejor_ll):.4f})")
    print(f"\n  Fiabilidad por línea con c={mejor_c:.2f}:")
    print(f"  {'línea':>6}{'P(over) modelo':>16}{'frec. real':>12}{'gap':>8}")
    for ln in LINEAS:
        pos = [float(total_pmf(row.lam, row.mu, row.r, mejor_c)
                     [np.arange(2 * MAX_C + 1) > ln].sum()) for row in rec.itertuples(index=False)]
        emp = float((rec["total"] > ln).mean())
        print(f"  {ln:>6}{np.mean(pos):>16.3f}{emp:>12.3f}{emp - np.mean(pos):>8.3f}")
    print(f"\n  Recomendación: CornersModel(calibracion={mejor_c:.2f}).")


if __name__ == "__main__":
    main()

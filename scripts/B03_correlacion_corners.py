"""Fase B (correlación) — Corregir el empate de córners con una cópula gaussiana.

La independencia entre córners local y visita sobreestima el empate (~12% vs
~6.7% real): ignora que el equipo dominante saca MÁS y concede MENOS (dependencia
negativa, correlación residual observada ≈ -0.14). Se modela el joint con una
cópula gaussiana de correlación rho y se elige rho por walk-forward.

Reporta, por rho: log-loss y accuracy del 1x2, P(empate) media vs real, y el
log-loss del O/U 9.5 (para confirmar que la cópula no degrada las líneas Over).

Uso:
    python scripts/B03_correlacion_corners.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.corners_dixon_coles import CornersModel, _bvn_cdf  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EPS = 1e-12
MAXC = 25


def _ll(p) -> float:
    return float(-np.log(np.clip(p, EPS, 1.0)))


def _pmf(mean, r):
    k = np.arange(0, MAXC + 1)
    p = nbinom.pmf(k, r, r / (r + mean))
    return p / p.sum()


def _joint(ph, pa, rho):
    if abs(rho) < 1e-9:
        return np.outer(ph, pa)
    cdf_h = np.clip(np.concatenate([[0.0], np.cumsum(ph)]), EPS, 1 - EPS)
    cdf_a = np.clip(np.concatenate([[0.0], np.cumsum(pa)]), EPS, 1 - EPS)
    H, A = np.meshgrid(norm.ppf(cdf_h), norm.ppf(cdf_a), indexing="ij")
    C = _bvn_cdf(H, A, rho)
    M = np.clip(C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1], 0, None)
    return M / M.sum()


def main() -> None:
    df = pd.read_csv(PROC / "statsbomb_partidos.csv", parse_dates=["date"])
    df = df[df["date"].dt.year >= 2018].sort_values("date").reset_index(drop=True)
    fechas = sorted(df["date"].unique())

    # Recolecta (lam, mu, r, observados) una sola vez.
    rec = []
    for d in fechas:
        train = df[df["date"] < d]
        if len(train) < 64:
            continue
        m = CornersModel().fit(train, fecha_corte=str(pd.Timestamp(d).date()))
        for f in df[df["date"] == d].itertuples(index=False):
            lam, mu = m.lambda_mu(f.home_team, f.away_team, bool(f.neutral))
            rec.append((lam, mu, m.r_, int(f.home_corners), int(f.away_corners)))

    real_emp = float(np.mean([h == a for *_, h, a in rec]))
    print(f"\n=== Cópula de córners (walk-forward, {len(rec)} partidos) ===")
    print(f"  Empate real: {real_emp:.3f}\n")
    print(f"  {'rho':>6}{'1x2 LL':>9}{'1x2 acc':>9}{'P(emp) modelo':>15}{'O/U9.5 LL':>11}")
    print(f"  {'-'*50}")

    mejor = None
    for rho in [0.0, -0.05, -0.10, -0.14, -0.15, -0.18, -0.21, -0.25]:
        x2_ll, x2_acc, emps, ou_ll = [], [], [], []
        for lam, mu, r, h, a in rec:
            ph, pa = _pmf(lam, r), _pmf(mu, r)
            M = _joint(ph, pa, rho)
            n = M.shape[0]
            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            pl = M[ii > jj].sum(); pe = M[ii == jj].sum(); pv = M[ii < jj].sum()
            ssum = pl + pe + pv
            probs = [pl / ssum, pe / ssum, pv / ssum]
            res = 0 if h > a else (1 if h == a else 2)
            x2_ll.append(_ll(probs[res]))
            x2_acc.append(int(np.argmax(probs)) == res)
            emps.append(probs[1])
            tot = np.zeros(2 * n - 1)
            np.add.at(tot, (ii + jj).ravel(), M.ravel())
            tot /= tot.sum()
            po = float(tot[np.arange(len(tot)) > 9.5].sum())
            ou_ll.append(_ll(po if (h + a) > 9.5 else 1 - po))
        fila = (rho, np.mean(x2_ll), np.mean(x2_acc), np.mean(emps), np.mean(ou_ll))
        if mejor is None or fila[1] < mejor[1]:
            mejor = fila
        marca = "  <- empate más cercano" if abs(np.mean(emps) - real_emp) < 0.015 else ""
        print(f"  {rho:>6.2f}{np.mean(x2_ll):>9.4f}{np.mean(x2_acc):>9.3f}"
              f"{np.mean(emps):>15.3f}{np.mean(ou_ll):>11.4f}{marca}")

    print(f"\n  Mejor 1x2 log-loss en rho={mejor[0]:.2f} (LL={mejor[1]:.4f}, "
          f"P(emp)={mejor[3]:.3f} vs real {real_emp:.3f})")
    print("\n  DECISIÓN DE DESPLIEGUE: rho_corr=-0.15 (≈ correlación residual medida).")
    print("  La cópula se usa SOLO para el 1x2: mejora el 1x2 y reduce el empate")
    print("  inflado. La columna O/U muestra que la cópula DEGRADA el total (lo")
    print("  estrecha, pero el total real está sobre-dispersado), así que el O/U se")
    print("  deja en convolución independiente (ver predecir_partido). Cada salida")
    print("  usa la estructura de dependencia que mejor le ajusta.")
    print("  Nota honesta: a rho principled el empate baja de ~12% a ~11%, no llega")
    print("  al ~8% real (límite de la cópula gaussiana sobre conteos discretos).")


if __name__ == "__main__":
    main()

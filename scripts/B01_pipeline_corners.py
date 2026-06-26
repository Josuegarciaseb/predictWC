"""Fase B — Pipeline de córners (Dixon-Coles adaptado).

1. Validación walk-forward sobre los datos de StatsBomb (2018+): para cada
   partido se entrena solo con los anteriores (sin fuga) y se compara Poisson
   vs Binomial Negativa en:
     - log-loss del total de córners (¿la NB modela mejor la sobre-dispersión?)
     - Over/Under (accuracy + log-loss en líneas estándar)
     - 1x2 de córners (accuracy + log-loss vs baseline ingenuo)
2. Entrena el modelo final con todo y predice los córners de los fixtures del
   Mundial 2026 (mapeando nombres a StatsBomb).

Uso:
    python scripts/B01_pipeline_corners.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.corners_dixon_coles import CornersModel  # noqa: E402
import statsbomb_loader as sb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

EPS = 1e-12
LINEA_PPAL = 9.5


def _ll(p) -> float:
    return float(-np.log(np.clip(p, EPS, 1.0)))


def resultado_1x2(hc: int, ac: int) -> int:
    return 0 if hc > ac else (1 if hc == ac else 2)


def walk_forward(df: pd.DataFrame, min_train: int = 64) -> None:
    from scipy.stats import poisson as _po
    df = df[df["date"].dt.year >= 2018].sort_values("date").reset_index(drop=True)
    fechas = sorted(df["date"].unique())

    # Métricas por distribución: total log-loss, OU (acc/ll), 1x2 (acc/ll).
    M = {dist: {"ll_total": [], "ou_acc": [], "ou_ll": [], "x2_acc": [], "x2_ll": []}
         for dist in ("poisson", "nb")}
    ll_total_naive, pred_total, real_total = [], [], []

    base_1x2 = np.bincount([resultado_1x2(h, a) for h, a in
                            zip(df["home_corners"], df["away_corners"])], minlength=3) / len(df)
    base_over = float(((df["home_corners"] + df["away_corners"]) > LINEA_PPAL).mean())
    x2_ll_naive = []

    n = 0
    for d in fechas:
        train = df[df["date"] < d]
        if len(train) < min_train:
            continue
        test = df[df["date"] == d]

        modelo = CornersModel()  # defaults calibrados; expone ambas distribuciones
        modelo.fit(train, fecha_corte=str(pd.Timestamp(d).date()))
        naive_mean = float((train["home_corners"] + train["away_corners"]).mean())

        for fila in test.itertuples(index=False):
            T = int(fila.home_corners + fila.away_corners)
            res = resultado_1x2(fila.home_corners, fila.away_corners)
            over_real = T > LINEA_PPAL
            ll_total_naive.append(_ll(_po.pmf(T, naive_mean)))
            x2_ll_naive.append(_ll(base_1x2[res]))

            for dist in ("poisson", "nb"):
                pred = modelo.predecir_partido(fila.home_team, fila.away_team,
                                               bool(fila.neutral), distribucion=dist)
                pmf = pred["total_pmf"]
                M[dist]["ll_total"].append(_ll(pmf[T] if T < len(pmf) else EPS))
                p_over = pred["prob_over"][LINEA_PPAL]
                M[dist]["ou_acc"].append((p_over > 0.5) == over_real)
                M[dist]["ou_ll"].append(_ll(p_over if over_real else 1 - p_over))
                probs = [pred["prob_mas_corners_local"], pred["prob_empate_corners"],
                         pred["prob_mas_corners_visita"]]
                M[dist]["x2_acc"].append(int(np.argmax(probs)) == res)
                M[dist]["x2_ll"].append(_ll(probs[res]))
                if dist == "poisson":
                    pred_total.append(pred["corners_total"])
            real_total.append(T)
            n += 1

    base_over_ll = -(base_over * np.log(base_over) + (1 - base_over) * np.log(1 - base_over))
    print(f"\n=== Walk-forward córners (StatsBomb 2018+): {n} partidos evaluados ===")
    print(f"  Entrenamiento mínimo: {min_train} partidos previos (arranca tras WC2018).")
    print(f"  (log-loss: menor = mejor)\n")
    print(f"  {'Métrica':<28}{'Poisson':>10}{'Bin.Neg.':>10}{'Baseline':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Total córners (log-loss)':<28}{np.mean(M['poisson']['ll_total']):>10.3f}"
          f"{np.mean(M['nb']['ll_total']):>10.3f}{np.mean(ll_total_naive):>10.3f}")
    print(f"  {'O/U '+str(LINEA_PPAL)+' (log-loss)':<28}{np.mean(M['poisson']['ou_ll']):>10.3f}"
          f"{np.mean(M['nb']['ou_ll']):>10.3f}{base_over_ll:>10.3f}")
    print(f"  {'O/U '+str(LINEA_PPAL)+' (accuracy)':<28}{np.mean(M['poisson']['ou_acc']):>10.3f}"
          f"{np.mean(M['nb']['ou_acc']):>10.3f}{'-':>10}")
    print(f"  {'1x2 córners (log-loss)':<28}{np.mean(M['poisson']['x2_ll']):>10.3f}"
          f"{np.mean(M['nb']['x2_ll']):>10.3f}{np.mean(x2_ll_naive):>10.3f}")
    print(f"  {'1x2 córners (accuracy)':<28}{np.mean(M['poisson']['x2_acc']):>10.3f}"
          f"{np.mean(M['nb']['x2_acc']):>10.3f}{'-':>10}")
    print(f"\n  Calibración total -> media predicha {np.mean(pred_total):.2f} vs real "
          f"{np.mean(real_total):.2f}  |  Var/Media real = "
          f"{np.var(real_total) / np.mean(real_total):.2f} (>1 => sobre-disperso)")
    print("  Lectura: la NB iguala/supera ligeramente a Poisson en la cola del total;")
    print("  el valor del modelo está en el 1x2 (bate baseline), no en el O/U (~= base rate).")


def predecir_mundial(df: pd.DataFrame) -> None:
    modelo = CornersModel()  # NB, defaults calibrados (shr=0.65, ridge=0.05)
    modelo.fit(df)
    print(f"\nVentaja de local en córners: {np.exp(modelo.home_adv_):.2f}x "
          f"(poco identificable: casi todos los partidos son en sede neutral; "
          f"en los fixtures WC2026 neutral=True, asi que no aplica)")
    print(f"Dispersión NB estimada (r): {modelo.r_:.0f}  "
          f"(sobre-dispersion leve condicional; NB gana por poco a Poisson)")
    print("\n=== Top 12 selecciones por índice de córners (gana - concede) ===")
    print(modelo.ranking_corners(12).round(3).to_string())

    fixtures = pd.read_csv(PROC / "partidos_a_predecir.csv", parse_dates=["date"])
    filas = []
    for fila in fixtures.itertuples(index=False):
        h = sb.a_nombre_statsbomb(fila.home_team)
        a = sb.a_nombre_statsbomb(fila.away_team)
        pred = modelo.predecir_partido(h, a, bool(fila.neutral))
        sin_datos = []
        if h not in modelo.attack_.index:
            sin_datos.append(fila.home_team)
        if a not in modelo.attack_.index:
            sin_datos.append(fila.away_team)
        filas.append({
            "fecha": fila.date.date(),
            "local": fila.home_team,
            "visitante": fila.away_team,
            "corners_local": round(pred["corners_local"], 2),
            "corners_visita": round(pred["corners_visita"], 2),
            "corners_total": round(pred["corners_total"], 2),
            "p_over_8.5": round(pred["prob_over"][8.5], 3),
            "p_over_9.5": round(pred["prob_over"][9.5], 3),
            "p_over_10.5": round(pred["prob_over"][10.5], 3),
            "p_mas_corners_local": round(pred["prob_mas_corners_local"], 3),
            "p_empate_corners": round(pred["prob_empate_corners"], 3),
            "p_mas_corners_visita": round(pred["prob_mas_corners_visita"], 3),
            "sin_datos_statsbomb": ",".join(sin_datos) if sin_datos else "",
        })
    tabla = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / "predicciones_faseB_corners.csv"
    tabla.to_csv(ruta, index=False)

    print("\n=== Fase B -- córners, Mundial 2026 (matchday 1) ===")
    cols = ["fecha", "local", "visitante", "corners_total", "p_over_9.5",
            "p_mas_corners_local", "p_empate_corners", "p_mas_corners_visita"]
    with pd.option_context("display.width", 200):
        print(tabla[cols].to_string(index=False))
    n_sin = (tabla["sin_datos_statsbomb"] != "").sum()
    print(f"\n[{n_sin} partidos con al menos una selección sin historial -> usa promedio del campo]")
    print(f"Guardado: {ruta}")


def main() -> None:
    df = pd.read_csv(PROC / "statsbomb_partidos.csv", parse_dates=["date"])
    walk_forward(df)
    predecir_mundial(df)


if __name__ == "__main__":
    main()

"""
FASE 0 (diagnóstico) -- Backtest walk-forward de calibración de marcadores.

Mide si la cola alta de la distribución de marcadores exactos de Dixon-Coles
está bien calibrada, ANTES de tocar el modelo. Métricas:

  1. PIT aleatorizado por marginal (goles local / visita): si el histograma
     tiene forma de ∩ o acumula masa en el último bin, el modelo tiene colas
     más cortas que la realidad.
  2. Fiabilidad (reliability) de binarios de cola: over 3.5 / over 4.5 goles
     totales, margen >= 3, algún equipo >= 3. Brier global y por bucket.
  3. Ratio goleadas esperadas/observadas por tramo de |Δelo| pre-partido:
     el KPI de cola. < 1.0 => el modelo subestima las goleadas.
  4. Log-loss 1X2 y RPS: métricas de no-regresión (no deben empeorar cuando
     se ajuste la cola).

Protocolo walk-forward anual: para cada año Y se ajusta Dixon-Coles SOLO con
partidos hasta el 1 de enero de Y y se evalúa sobre los partidos jugados en Y.
Sin fuga de información.

Uso:
    python scripts/09_backtest_calibracion.py                     # config actual
    python scripts/09_backtest_calibracion.py --shrinkage 1.0     # sin encogimiento
    python scripts/09_backtest_calibracion.py --anio-inicio 2015 --anio-fin 2025

Salidas en outputs/backtest_calibracion/shrinkage_<s>/:
    resumen.csv, pit_histograma.csv, pit_histograma.png,
    reliability_colas.csv, reliability_colas.png,
    goleadas_por_tramo_elo.csv, partidos_evaluados.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chisquare

from models.poisson_dixon_coles import DixonColesModel

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "backtest_calibracion"

N_BINS_PIT = 10
N_BUCKETS_RELIABILITY = 10
TRAMOS_ELO = [0, 100, 200, 300, np.inf]

# Binarios de cola: nombre -> máscara sobre (goles_local, goles_visita)
MERCADOS_COLA = {
    "over_3.5": lambda gl, gv: gl + gv >= 4,
    "over_4.5": lambda gl, gv: gl + gv >= 5,
    "margen_>=3": lambda gl, gv: np.abs(gl - gv) >= 3,
    "alguno_>=3": lambda gl, gv: (gl >= 3) | (gv >= 3),
}


def evaluar_anio(historico: pd.DataFrame, anio: int, shrinkage: float,
                 dispersion: float = 0.0) -> pd.DataFrame:
    """Ajusta DC con datos hasta el 1-ene del año y evalúa los partidos de ese año."""
    corte = pd.Timestamp(f"{anio}-01-01")
    modelo = DixonColesModel(cutoff_years=11, half_life_years=2.5,
                             strength_shrinkage=shrinkage, dispersion=dispersion)
    modelo.fit(historico, fecha_corte=str(corte.date()))

    df_eval = historico[
        (historico["date"] > corte) & (historico["date"] < pd.Timestamp(f"{anio + 1}-01-01"))
        & historico["home_score"].notna() & historico["away_score"].notna()
    ].copy()

    # Equipos sin historial en la ventana de entrenamiento caerían al promedio
    # global: no miden la calibración del modelo real, se excluyen.
    conocidos = set(modelo.equipos_)
    df_eval = df_eval[df_eval["home_team"].isin(conocidos) & df_eval["away_team"].isin(conocidos)]

    g = np.arange(0, modelo.max_goals + 1)
    xs, ys = np.meshgrid(g, g, indexing="ij")
    mascaras = {nombre: fn(xs, ys) for nombre, fn in MERCADOS_COLA.items()}
    m_local, m_empate, m_visita = xs > ys, xs == ys, xs < ys

    filas = []
    for f in df_eval.itertuples(index=False):
        M = modelo.matriz_marcador(f.home_team, f.away_team, neutral=bool(f.neutral))
        gl, gv = int(f.home_score), int(f.away_score)
        gl_c = min(gl, modelo.max_goals)  # clip solo para el PIT (cola >10 es despreciable)
        gv_c = min(gv, modelo.max_goals)

        marg_local = M.sum(axis=1)
        marg_visita = M.sum(axis=0)

        fila = {
            "anio": anio,
            "date": f.date,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "goles_local": gl,
            "goles_visita": gv,
            "abs_elo_diff": abs(float(f.elo_local_antes) - float(f.elo_visita_antes)),
            # componentes del PIT aleatorizado: F(x-1) y P(X=x)
            "cdf_prev_local": float(marg_local[:gl_c].sum()),
            "pmf_local": float(marg_local[gl_c]),
            "cdf_prev_visita": float(marg_visita[:gv_c].sum()),
            "pmf_visita": float(marg_visita[gv_c]),
            "prob_local": float(M[m_local].sum()),
            "prob_empate": float(M[m_empate].sum()),
            "prob_visita": float(M[m_visita].sum()),
        }
        for nombre, mask in mascaras.items():
            fila[f"pred_{nombre}"] = float(M[mask].sum())
            fila[f"obs_{nombre}"] = int(bool(MERCADOS_COLA[nombre](np.array(gl), np.array(gv))))
        filas.append(fila)

    return pd.DataFrame(filas)


def calcular_pit(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PIT aleatorizado por marginal: u = F(x-1) + v*P(X=x), v ~ U(0,1)."""
    out = {}
    for lado in ["local", "visita"]:
        v = rng.uniform(size=len(df))
        u = df[f"cdf_prev_{lado}"].values + v * df[f"pmf_{lado}"].values
        counts, _ = np.histogram(u, bins=N_BINS_PIT, range=(0, 1))
        chi2, pval = chisquare(counts)
        out[lado] = {"counts": counts, "chi2": chi2, "pval": pval}

    bins = [f"[{i / N_BINS_PIT:.1f},{(i + 1) / N_BINS_PIT:.1f})" for i in range(N_BINS_PIT)]
    tabla = pd.DataFrame({
        "bin": bins,
        "freq_local": out["local"]["counts"] / out["local"]["counts"].sum(),
        "freq_visita": out["visita"]["counts"] / out["visita"]["counts"].sum(),
    })
    tabla.attrs["tests"] = {lado: (out[lado]["chi2"], out[lado]["pval"]) for lado in out}
    return tabla


def calcular_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Buckets por cuantil de prob. predicha para cada binario de cola."""
    filas = []
    for nombre in MERCADOS_COLA:
        pred, obs = df[f"pred_{nombre}"], df[f"obs_{nombre}"]
        bucket = pd.qcut(pred, N_BUCKETS_RELIABILITY, labels=False, duplicates="drop")
        agg = df.assign(_b=bucket).groupby("_b").agg(
            prob_media_pred=(f"pred_{nombre}", "mean"),
            frec_observada=(f"obs_{nombre}", "mean"),
            n=(f"obs_{nombre}", "size"),
        ).reset_index(drop=True)
        agg["mercado"] = nombre
        filas.append(agg)
    return pd.concat(filas, ignore_index=True)


def calcular_goleadas_por_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Goleadas (margen >= 3) esperadas vs observadas por tramo de |Δelo|."""
    etiquetas = [f"[{TRAMOS_ELO[i]:.0f},{TRAMOS_ELO[i+1]:.0f})" for i in range(len(TRAMOS_ELO) - 1)]
    tramo = pd.cut(df["abs_elo_diff"], bins=TRAMOS_ELO, labels=etiquetas, right=False)
    agg = df.assign(_t=tramo).groupby("_t", observed=True).agg(
        n_partidos=("obs_margen_>=3", "size"),
        goleadas_esperadas=("pred_margen_>=3", "sum"),
        goleadas_observadas=("obs_margen_>=3", "sum"),
    ).reset_index().rename(columns={"_t": "tramo_elo"})

    total = pd.DataFrame([{
        "tramo_elo": "TOTAL",
        "n_partidos": len(df),
        "goleadas_esperadas": df["pred_margen_>=3"].sum(),
        "goleadas_observadas": df["obs_margen_>=3"].sum(),
    }])
    agg = pd.concat([agg, total], ignore_index=True)
    agg["ratio_esp_obs"] = agg["goleadas_esperadas"] / agg["goleadas_observadas"].replace(0, np.nan)
    return agg


def calcular_metricas_1x2(df: pd.DataFrame) -> dict:
    probs = df[["prob_local", "prob_empate", "prob_visita"]].values.clip(1e-12, 1)
    y = np.select(
        [df["goles_local"] > df["goles_visita"], df["goles_local"] == df["goles_visita"]],
        [0, 1], default=2,
    )
    onehot = np.eye(3)[y]
    log_loss = float(-np.mean(np.log(probs[np.arange(len(y)), y])))
    rps = float(np.mean(0.5 * np.sum(
        (np.cumsum(probs, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]) ** 2, axis=1)))
    return {"log_loss_1x2": log_loss, "rps_1x2": rps}


def graficar_pit(tabla_pit: pd.DataFrame, ruta: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, lado in zip(axes, ["local", "visita"]):
        freq = tabla_pit[f"freq_{lado}"]
        ax.bar(range(N_BINS_PIT), freq, color="#4472c4", edgecolor="white")
        ax.axhline(1 / N_BINS_PIT, color="#c00000", linestyle="--", label="uniforme ideal")
        chi2, pval = tabla_pit.attrs["tests"][lado]
        ax.set_title(f"PIT goles {lado}  (χ²={chi2:.1f}, p={pval:.4f})")
        ax.set_xticks(range(N_BINS_PIT))
        ax.set_xticklabels(tabla_pit["bin"], rotation=45, fontsize=7)
        ax.set_ylabel("frecuencia")
        ax.legend()
    fig.suptitle("PIT aleatorizado por marginal (∩ o masa en el último bin = colas cortas)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def graficar_reliability(tabla_rel: pd.DataFrame, ruta: Path):
    mercados = list(MERCADOS_COLA)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, nombre in zip(axes.ravel(), mercados):
        sub = tabla_rel[tabla_rel["mercado"] == nombre]
        lim = max(sub["prob_media_pred"].max(), sub["frec_observada"].max()) * 1.1
        ax.plot([0, lim], [0, lim], "--", color="#c00000", label="calibración perfecta")
        ax.plot(sub["prob_media_pred"], sub["frec_observada"], "o-", color="#4472c4")
        ax.set_xlabel("prob. media predicha")
        ax.set_ylabel("frecuencia observada")
        ax.set_title(nombre)
        ax.legend()
    fig.suptitle("Reliability de binarios de cola (curva por debajo de la diagonal = subestima)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main():
    # La consola de Windows usa cp1252: sin esto, los prints con Δ/χ²/∩ revientan.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Backtest de calibración de marcadores (Dixon-Coles)")
    parser.add_argument("--anio-inicio", type=int, default=2018)
    parser.add_argument("--anio-fin", type=int, default=2025)
    parser.add_argument("--shrinkage", type=float, default=1.0,
                        help="strength_shrinkage del Dixon-Coles (1.0 = config de producción; "
                             "el 0.85 se retiró tras el backtest de calibración)")
    parser.add_argument("--dispersion", type=float, default=0.0,
                        help="phi de sobredispersión Negative Binomial "
                             "(0 = Poisson puro; producción usa 0.06)")
    args = parser.parse_args()

    historico = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])

    sufijo = f"_phi_{args.dispersion:g}" if args.dispersion > 0 else ""
    dir_salida = OUTPUTS_DIR / f"shrinkage_{args.shrinkage:g}{sufijo}"
    dir_salida.mkdir(parents=True, exist_ok=True)

    partes = []
    for anio in range(args.anio_inicio, args.anio_fin + 1):
        t0 = time.time()
        parte = evaluar_anio(historico, anio, args.shrinkage, args.dispersion)
        partes.append(parte)
        print(f"  {anio}: {len(parte)} partidos evaluados ({time.time() - t0:.0f}s)")
    df = pd.concat(partes, ignore_index=True)

    rng = np.random.default_rng(42)
    tabla_pit = calcular_pit(df, rng)
    tabla_rel = calcular_reliability(df)
    tabla_gol = calcular_goleadas_por_elo(df)
    met_1x2 = calcular_metricas_1x2(df)

    resumen = {
        "shrinkage": args.shrinkage,
        "dispersion": args.dispersion,
        "anios": f"{args.anio_inicio}-{args.anio_fin}",
        "n_partidos": len(df),
        **met_1x2,
    }
    for lado in ["local", "visita"]:
        chi2, pval = tabla_pit.attrs["tests"][lado]
        resumen[f"pit_chi2_{lado}"] = round(chi2, 2)
        resumen[f"pit_pval_{lado}"] = round(pval, 5)
    for nombre in MERCADOS_COLA:
        pred, obs = df[f"pred_{nombre}"], df[f"obs_{nombre}"]
        resumen[f"brier_{nombre}"] = round(float(np.mean((pred - obs) ** 2)), 5)
        resumen[f"ratio_esp_obs_{nombre}"] = round(float(pred.sum() / max(obs.sum(), 1)), 4)

    df.to_csv(dir_salida / "partidos_evaluados.csv", index=False)
    tabla_pit.to_csv(dir_salida / "pit_histograma.csv", index=False)
    tabla_rel.to_csv(dir_salida / "reliability_colas.csv", index=False)
    tabla_gol.to_csv(dir_salida / "goleadas_por_tramo_elo.csv", index=False)
    pd.DataFrame([resumen]).to_csv(dir_salida / "resumen.csv", index=False)
    graficar_pit(tabla_pit, dir_salida / "pit_histograma.png")
    graficar_reliability(tabla_rel, dir_salida / "reliability_colas.png")

    print(f"\n=== Backtest calibración | shrinkage={args.shrinkage} phi={args.dispersion} | "
          f"{args.anio_inicio}-{args.anio_fin} | {len(df)} partidos ===")
    print(f"\nLog-loss 1X2: {met_1x2['log_loss_1x2']:.4f}   RPS: {met_1x2['rps_1x2']:.4f}")
    print("\nPIT (p < 0.05 => se rechaza uniformidad, mala calibración marginal):")
    for lado in ["local", "visita"]:
        chi2, pval = tabla_pit.attrs["tests"][lado]
        print(f"  goles {lado:7s}: chi2={chi2:7.1f}  p={pval:.5f}")
    print("\nColas (ratio esperado/observado; < 1.0 => subestima la cola):")
    for nombre in MERCADOS_COLA:
        print(f"  {nombre:11s}: ratio={resumen[f'ratio_esp_obs_{nombre}']:.3f}  "
              f"brier={resumen[f'brier_{nombre}']:.4f}")
    print("\nGoleadas (margen >= 3) por tramo de |Δelo|:")
    print(tabla_gol.to_string(index=False))
    print(f"\nGuardado en: {dir_salida}")


if __name__ == "__main__":
    main()

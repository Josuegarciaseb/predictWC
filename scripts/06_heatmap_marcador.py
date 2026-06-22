"""
FASE 2 (visual) -- Heatmap de distribución de probabilidades del marcador.

Genera una tabla/heatmap como las de StatisKicks: probabilidad de cada
marcador exacto (goles de local x goles de visita) para un partido del
Mundial 2026, usando la matriz de Dixon-Coles ya implementada en
src/models/poisson_dixon_coles.py.

Uso:
    # Por nombres de equipo (local visitante)
    python scripts/06_heatmap_marcador.py "Argentina" "Brazil"

    # Por índice de un partido de partidos_a_predecir.csv (0 = primero)
    python scripts/06_heatmap_marcador.py --partido 0

    # Todos los partidos del Mundial 2026 (genera un PNG por partido)
    python scripts/06_heatmap_marcador.py --todos

Por defecto los partidos del Mundial son en sede neutral (neutral=True).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from models.poisson_dixon_coles import DixonColesModel

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "heatmaps"

MAX_GOLES_VIS = 6  # goles 0..6 mostrados en la tabla (como el ejemplo)


def cargar_modelo() -> DixonColesModel:
    historico = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])
    modelo = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    modelo.fit(historico)
    return modelo


def dibujar_heatmap(modelo: DixonColesModel, local: str, visitante: str,
                    neutral: bool = True, subtitulo: str | None = None) -> Path:
    M = modelo.matriz_marcador(local, visitante, neutral=neutral)
    # recortamos a 0..MAX_GOLES_VIS en cada eje (la cola es despreciable)
    n = MAX_GOLES_VIS + 1
    M = M[:n, :n]  # filas = goles local, columnas = goles visita
    # En el ejemplo: eje X = goles de local, eje Y = goles de visita.
    # M[i,j] = P(local=i, visita=j) -> para mostrar visita en filas, transponemos.
    G = M.T  # G[fila=visita, col=local]

    pred = modelo.predecir_partido(local, visitante, neutral=neutral)

    rojo = LinearSegmentedColormap.from_list("rojo", ["#ffffff", "#ff2b2b"])
    vmax = G.max()

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(G, cmap=rojo, vmin=0, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=15, fontweight="bold")
    ax.set_yticklabels(range(n), fontsize=15, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(f"Goles de Local  ({local})", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_ylabel(f"Goles de Visita  ({visitante})", fontsize=16, fontweight="bold", labelpad=12)

    for i in range(n):
        for j in range(n):
            p = G[i, j] * 100
            color = "white" if G[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, f"{p:.2f}%", ha="center", va="center",
                    fontsize=12, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)

    titulo = f"DISTRIBUCIÓN DE PROBABILIDADES DEL MARCADOR"
    sub = subtitulo or f"{local.upper()} (LOCAL) vs {visitante.upper()} (VISITA) | MUNDIAL 2026"
    fig.suptitle(titulo, fontsize=22, fontweight="bold", x=0.5, y=0.99)
    ax.set_title(sub, fontsize=14, color="#888888", fontweight="bold", pad=42)

    resumen = (f"Marcador más probable: {pred['marcador_mas_probable']} "
               f"({pred['prob_marcador_exacto']*100:.1f}%)   |   "
               f"Gana {local}: {pred['prob_local']*100:.0f}%   "
               f"Empate: {pred['prob_empate']*100:.0f}%   "
               f"Gana {visitante}: {pred['prob_visita']*100:.0f}%")
    fig.text(0.5, 0.04, resumen, ha="center", fontsize=12, color="#333333")
    fig.text(0.5, 0.01, "Modelo Dixon-Coles | Predictor Mundial FIFA 2026",
             ha="center", fontsize=9, color="#aaaaaa")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = lambda s: "".join(c if c.isalnum() else "_" for c in s)
    ruta = OUTPUTS_DIR / f"heatmap_{safe(local)}_vs_{safe(visitante)}.png"
    fig.savefig(ruta, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def main():
    parser = argparse.ArgumentParser(description="Heatmap de marcadores Mundial 2026")
    parser.add_argument("local", nargs="?", help="Equipo local")
    parser.add_argument("visitante", nargs="?", help="Equipo visitante")
    parser.add_argument("--partido", type=int, help="Índice en partidos_a_predecir.csv")
    parser.add_argument("--todos", action="store_true", help="Genera todos los partidos")
    parser.add_argument("--no-neutral", action="store_true", help="Tratar como NO neutral")
    args = parser.parse_args()

    modelo = cargar_modelo()
    neutral = not args.no_neutral

    if args.todos or args.partido is not None:
        a_predecir = pd.read_csv(PROCESSED_DIR / "partidos_a_predecir.csv", parse_dates=["date"])
        filas = a_predecir if args.todos else a_predecir.iloc[[args.partido]]
        for fila in filas.itertuples(index=False):
            sub = (f"{fila.home_team.upper()} vs {fila.away_team.upper()} | "
                   f"{fila.date.date()} | MUNDIAL 2026")
            ruta = dibujar_heatmap(modelo, fila.home_team, fila.away_team,
                                   neutral=bool(fila.neutral), subtitulo=sub)
            print(f"Guardado: {ruta}")
        return

    if not args.local or not args.visitante:
        parser.error("Indica 'local visitante', o usa --partido N, o --todos")

    ruta = dibujar_heatmap(modelo, args.local, args.visitante, neutral=neutral)
    print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()

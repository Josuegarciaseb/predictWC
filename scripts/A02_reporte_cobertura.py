"""Fase A — Reporte de cobertura real de datos por año, torneo y mercado.

Lee los CSV cacheados por A01 y responde, con honestidad: ¿cuántos partidos
*utilizables* hay por mercado nuevo (córners, tarjetas, tiros totales a puerta,
tiros a puerta por jugador)? Escribe el reporte a outputs/ y lo imprime.

Uso:
    python scripts/A02_reporte_cobertura.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"


def _check(ruta: Path) -> None:
    if not ruta.exists():
        sys.exit(f"Falta {ruta}. Ejecuta primero scripts/A01_ingesta_statsbomb.py")


def main() -> None:
    ruta_part = PROC_DIR / "statsbomb_partidos.csv"
    ruta_jug = PROC_DIR / "statsbomb_tiros_jugador.csv"
    _check(ruta_part)
    _check(ruta_jug)

    df = pd.read_csv(ruta_part, parse_dates=["date"])
    df["anio"] = df["date"].dt.year
    df["corners_tot"] = df["home_corners"] + df["away_corners"]
    df["cards_tot"] = (df["home_yellow"] + df["away_yellow"]
                       + df["home_red"] + df["away_red"])
    df["sot_tot"] = df["home_sot"] + df["away_sot"]
    df["shots_tot"] = df["home_shots"] + df["away_shots"]

    # Un partido es "utilizable" por mercado si la señal está realmente presente.
    df["util_corners"] = df["corners_tot"] > 0
    df["util_cards"] = (df["home_fouls"] + df["away_fouls"]) > 0  # hay disciplina parseada
    df["util_sot"] = df["shots_tot"] > 0

    # ---- Tabla por torneo + temporada ----
    g = df.groupby(["tournament", "season"], as_index=False).agg(
        partidos=("match_id", "count"),
        util_corners=("util_corners", "sum"),
        util_cards=("util_cards", "sum"),
        util_sot=("util_sot", "sum"),
        corners_medio=("corners_tot", "mean"),
        cards_medio=("cards_tot", "mean"),
        sot_medio=("sot_tot", "mean"),
    )
    g = g.sort_values(["tournament", "season"]).reset_index(drop=True)
    for col in ("corners_medio", "cards_medio", "sot_medio"):
        g[col] = g[col].round(1)

    # ---- Totales por mercado ----
    tot = {
        "partidos_total": len(df),
        "corners_utiles": int(df["util_corners"].sum()),
        "cards_utiles": int(df["util_cards"].sum()),
        "sot_utiles": int(df["util_sot"].sum()),
    }

    # ---- Jugadores (Fase D) ----
    dj = pd.read_csv(ruta_jug)
    por_jugador = dj.groupby("player").agg(
        partidos=("match_id", "nunique"),
        shots=("shots", "sum"),
        sot=("sot", "sum"),
    )
    jug = {
        "filas_jugador_partido": len(dj),
        "jugadores_distintos": dj["player"].nunique(),
        "jugadores_ge3_partidos": int((por_jugador["partidos"] >= 3).sum()),
        "jugadores_ge5_partidos": int((por_jugador["partidos"] >= 5).sum()),
    }

    # ---- Salida ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT_DIR / "reporte_cobertura_mercados.csv", index=False)

    lineas = []
    lineas.append("# Reporte de cobertura de datos — mercados nuevos (StatsBomb)\n")
    lineas.append(f"Rango temporal: {df['date'].min().date()} -> {df['date'].max().date()}")
    lineas.append(f"Partidos totales ingeridos: {tot['partidos_total']}\n")
    lineas.append("## Partidos utilizables por mercado\n")
    lineas.append(f"- **Córners** (O/U y 1x2): {tot['corners_utiles']} partidos")
    lineas.append(f"- **Tarjetas** (O/U): {tot['cards_utiles']} partidos")
    lineas.append(f"- **Tiros a puerta totales** (O/U): {tot['sot_utiles']} partidos")
    lineas.append("\n## Tiros a puerta por jugador (Fase D)\n")
    lineas.append(f"- Filas jugador-partido: {jug['filas_jugador_partido']}")
    lineas.append(f"- Jugadores distintos: {jug['jugadores_distintos']}")
    lineas.append(f"- Con >=3 partidos: {jug['jugadores_ge3_partidos']}")
    lineas.append(f"- Con >=5 partidos: {jug['jugadores_ge5_partidos']}")
    lineas.append("\n## Detalle por torneo y temporada\n")
    lineas.append(g.to_string(index=False))
    md = "\n".join(lineas)
    (OUT_DIR / "reporte_cobertura_mercados.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"\n[Guardado] {OUT_DIR / 'reporte_cobertura_mercados.md'}")
    print(f"[Guardado] {OUT_DIR / 'reporte_cobertura_mercados.csv'}")


if __name__ == "__main__":
    main()

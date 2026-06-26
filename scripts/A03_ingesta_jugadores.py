"""Fase D (ingesta) — Jugador-partido: minutos jugados + tiros a puerta.

Deriva de los eventos cacheados (sin re-descargar) la exposición real de cada
jugador (minutos en cancha) y sus tiros a puerta, incluyendo a quien jugó pero no
disparó (0). Es el insumo para modelar una tasa de tiros a puerta por 90'.

Salida: data/processed/statsbomb_jugador_minutos.csv

Uso:
    python scripts/A03_ingesta_jugadores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import statsbomb_loader as sb  # noqa: E402

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    temporadas = sb.temporadas_internacionales()
    temporadas.sort(key=lambda c: (c["competition_name"], c["season_name"]))
    filas = []
    for c in temporadas:
        cid, sid = c["competition_id"], c["season_id"]
        cname, sname = c["competition_name"], c["season_name"]
        partidos = sb.cargar_partidos(cid, sid)
        print(f"  {cname} {sname}: {len(partidos)} partidos ...", end="", flush=True)
        for m in partidos:
            mid = m["match_id"]
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            stage = m.get("competition_stage", {}).get("name", "")
            eventos = sb.cargar_eventos(mid)
            for fila in sb.parsear_minutos_jugador(eventos, mid, home, away):
                fila["date"] = m["match_date"]
                fila["tournament"] = cname
                fila["season"] = sname
                fila["knockout"] = "Group" not in stage
                filas.append(fila)
        print(" ok")

    df = pd.DataFrame(filas)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    ruta = PROC / "statsbomb_jugador_minutos.csv"
    df.to_csv(ruta, index=False, encoding="utf-8")

    print(f"\nGuardado: {ruta}  ({len(df)} filas jugador-partido)")
    print(f"Jugadores distintos: {df['player'].nunique()}")
    print(f"Minutos: media {df['minutes'].mean():.1f}, titulares {df['is_starter'].mean():.1%}")
    print(f"Tiros a puerta: media {df['sot'].mean():.2f}/partido-jugador, "
          f"tasa por 90' global {df['sot'].sum() / (df['minutes'].sum()/90):.3f}")
    print("\nMuestra (máximos tiradores por partido):")
    cols = ["date", "player", "team", "opponent", "minutes", "shots", "sot"]
    print(df.sort_values("sot", ascending=False)[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()

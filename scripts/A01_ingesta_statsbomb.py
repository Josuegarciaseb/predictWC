"""Fase A — Ingesta de StatsBomb Open Data (selecciones, internacional masculino).

Descarga (con caché en disco) los eventos de cada partido de los torneos de
selecciones disponibles y los materializa en dos CSV con el estilo del pipeline:

  data/processed/statsbomb_partidos.csv        -> 1 fila por partido (home/away)
  data/processed/statsbomb_tiros_jugador.csv   -> 1 fila por jugador-partido

Uso:
    python scripts/A01_ingesta_statsbomb.py

La primera ejecución descarga ~333 ficheros de eventos (varios MB cada uno) y
puede tardar unos minutos; las siguientes leen del caché en data/raw/statsbomb/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import statsbomb_loader as sb  # noqa: E402

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    temporadas = sb.temporadas_internacionales()
    temporadas.sort(key=lambda c: (c["competition_name"], c["season_name"]))
    print(f"Temporadas internacionales a ingerir: {len(temporadas)}\n")

    filas_partido: list[dict] = []
    filas_jugador: list[dict] = []

    for c in temporadas:
        cid, sid = c["competition_id"], c["season_id"]
        cname, sname = c["competition_name"], c["season_name"]
        try:
            partidos = sb.cargar_partidos(cid, sid)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] {cname} {sname}: sin matches ({e})")
            continue

        print(f"  {cname} {sname}: {len(partidos)} partidos ...", end="", flush=True)
        ok = 0
        for m in partidos:
            mid = m["match_id"]
            try:
                eventos = sb.cargar_eventos(mid)
            except Exception as e:  # noqa: BLE001
                print(f"\n    [WARN] eventos {mid} fallaron: {e}")
                continue
            stats = sb.parsear_eventos(eventos)
            fila = sb.fila_partido(m, stats, cname, sname, cid, sid)
            if fila is not None:
                filas_partido.append(fila)
                ok += 1
            filas_jugador.extend(sb.parsear_tiros_jugador(eventos, mid))
        print(f" {ok} ok")

    df_part = pd.DataFrame(filas_partido)
    df_part["date"] = pd.to_datetime(df_part["date"])
    df_part = df_part.sort_values("date").reset_index(drop=True)

    df_jug = pd.DataFrame(filas_jugador)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ruta_part = PROC_DIR / "statsbomb_partidos.csv"
    ruta_jug = PROC_DIR / "statsbomb_tiros_jugador.csv"
    df_part.to_csv(ruta_part, index=False)
    df_jug.to_csv(ruta_jug, index=False)

    print(f"\nGuardado: {ruta_part}  ({len(df_part)} partidos)")
    print(f"Guardado: {ruta_jug}  ({len(df_jug)} filas jugador-partido)")
    print("\nMuestra:")
    cols = ["date", "home_team", "away_team", "home_corners", "away_corners",
            "home_sot", "away_sot", "home_yellow", "away_yellow"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df_part[cols].tail(8).to_string(index=False))


if __name__ == "__main__":
    main()

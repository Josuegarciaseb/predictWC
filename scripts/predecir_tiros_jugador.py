"""CLI a demanda para tiros a puerta de un jugador (Fase D).

Uso:
    python scripts/predecir_tiros_jugador.py "Messi" "France"
    python scripts/predecir_tiros_jugador.py "Mbappe" "England" --minutos 75
    python scripts/predecir_tiros_jugador.py "Lautaro" "Brazil" --lineas 0.5 1.5 2.5

El nombre del jugador se busca por subcadena (los nombres de StatsBomb son
completos, p.ej. "Lionel Andrés Messi Cuccittini"). La predicción es CONDICIONAL a
los minutos esperados (por defecto 90): sin once probable del Mundial 2026 no se
puede saber si jugará ni cuánto.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.tiros_jugador import TirosJugadorModel, LINEAS_OU_DEFECTO  # noqa: E402
import statsbomb_loader as sb  # noqa: E402

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def buscar_jugador(modelo: TirosJugadorModel, query: str) -> str | None:
    q = _norm(query)
    matches = [p for p in modelo.player_ab_ if q in _norm(p)]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # Varios: elige el de más exposición (más datos).
    matches.sort(key=lambda p: modelo.player_ab_[p][1], reverse=True)
    print(f"[varios coinciden con '{query}'] uso: {matches[0]}")
    print("   otros:", ", ".join(matches[1:6]))
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiros a puerta esperados de un jugador")
    ap.add_argument("jugador", help="Nombre o subcadena (p.ej. 'Messi')")
    ap.add_argument("rival", help="Selección rival (estilo fixtures o StatsBomb)")
    ap.add_argument("--minutos", type=float, default=90.0, help="Minutos esperados (def. 90)")
    ap.add_argument("--lineas", nargs="+", type=float, default=list(LINEAS_OU_DEFECTO))
    args = ap.parse_args()

    df = pd.read_csv(PROC / "statsbomb_jugador_minutos.csv", parse_dates=["date"])
    modelo = TirosJugadorModel().fit(df)

    jugador = buscar_jugador(modelo, args.jugador)
    rival = sb.a_nombre_statsbomb(args.rival)

    if jugador is None:
        print(f"\n[AVISO] '{args.jugador}' no tiene historial en StatsBomb -> se "
              f"usaría el promedio del campo (predicción no informativa). Prueba "
              f"otra grafía del nombre.")
        jugador = args.jugador

    pred = modelo.predecir(jugador, rival, args.minutos, lineas=args.lineas)
    print(f"\n=== Tiros a puerta: {jugador} vs {rival} ({args.minutos:.0f} min) ===")
    print(f"Factor jugador={pred['theta_jugador']:.2f}  |  factor rival={pred['phi_rival']:.2f}")
    print(f"Tiros a puerta esperados: {pred['sot_esperados']:.2f}")
    print("\nOver/Under (tiros a puerta del jugador):")
    for ln in args.lineas:
        po = pred["prob_over"][ln]
        print(f"   línea {ln:>4}:  Over {po*100:5.1f}%   Under {(1-po)*100:5.1f}%")
    if not pred["tiene_datos"]:
        print("\n[AVISO] Sin historial de este jugador -> promedio del campo.")
    print("\n[Nota] Condicional a jugar esos minutos. La incertidumbre de "
          "alineación (no hay once probable del Mundial) domina la predicción real.")


if __name__ == "__main__":
    main()

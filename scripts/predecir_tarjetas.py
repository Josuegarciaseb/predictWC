"""CLI a demanda para tarjetas totales de un partido (análogo a predecir_corners.py).

Uso:
    python scripts/predecir_tarjetas.py "Mexico" "Czech Republic"
    python scripts/predecir_tarjetas.py "Argentina" "Brazil" --knockout
    python scripts/predecir_tarjetas.py "Spain" "Italy" --lineas 3.5 4.5 5.5

Por defecto asume fase de grupos y cancha neutral (como el matchday 1 del
Mundial). Usa --knockout para fase eliminatoria y --local para ventaja de local.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.tarjetas_model import TarjetasModel, LINEAS_OU_DEFECTO  # noqa: E402
import statsbomb_loader as sb  # noqa: E402

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Predicción de tarjetas totales de un partido")
    ap.add_argument("local")
    ap.add_argument("visitante")
    ap.add_argument("--knockout", action="store_true", help="Fase eliminatoria (más tarjetas)")
    ap.add_argument("--local", dest="ventaja_local", action="store_true",
                    help="Ventaja de local (por defecto: cancha neutral)")
    ap.add_argument("--lineas", nargs="+", type=float, default=list(LINEAS_OU_DEFECTO))
    args = ap.parse_args()

    df = pd.read_csv(PROC / "statsbomb_partidos.csv", parse_dates=["date"])
    modelo = TarjetasModel().fit(df)

    h = sb.a_nombre_statsbomb(args.local)
    a = sb.a_nombre_statsbomb(args.visitante)
    neutral = not args.ventaja_local
    faltan = [o for o, sbn in ((args.local, h), (args.visitante, a))
              if sbn not in modelo.propension_.index]

    pred = modelo.predecir_partido(h, a, neutral=neutral, knockout=args.knockout,
                                   lineas=args.lineas)
    fase = "eliminatoria" if args.knockout else "grupos"
    print(f"\n=== Tarjetas: {args.local} vs {args.visitante} "
          f"(fase {fase}, {'neutral' if neutral else 'local con ventaja'}) ===")
    print(f"Tarjetas esperadas:  {args.local} {pred['tarjetas_local']:.2f}  |  "
          f"{args.visitante} {pred['tarjetas_visita']:.2f}  |  "
          f"total {pred['tarjetas_total']:.2f}")
    print("\nOver/Under (total tarjetas):")
    for ln in args.lineas:
        po = pred["prob_over"][ln]
        print(f"   línea {ln:>5}:  Over {po*100:5.1f}%   Under {(1-po)*100:5.1f}%")
    print("\n[Nota] El árbitro es el factor dominante en tarjetas y no se conoce "
          "de antemano: este mercado es más ruidoso que córners.")
    if faltan:
        print(f"[AVISO] Sin historial StatsBomb: {', '.join(faltan)} -> promedio del campo.")


if __name__ == "__main__":
    main()

"""CLI a demanda para córners de un partido (análogo a predecir_partido.py).

Entrena el modelo de córners con todo el histórico de StatsBomb y predice un
enfrentamiento puntual: córners esperados, Over/Under en varias líneas y 1x2.

Uso:
    python scripts/predecir_corners.py "Spain" "Germany"
    python scripts/predecir_corners.py "Mexico" "Brazil" --local      # con ventaja de local
    python scripts/predecir_corners.py "Portugal" "DR Congo" --lineas 8.5 9.5 10.5

Los nombres aceptan el estilo de los fixtures (p.ej. "DR Congo", "Ivory Coast",
"Cape Verde"): se mapean automáticamente a los de StatsBomb. Por defecto se
asume cancha neutral (como en el Mundial); usa --local para dar ventaja al local.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.corners_dixon_coles import CornersModel, LINEAS_OU_DEFECTO  # noqa: E402
import statsbomb_loader as sb  # noqa: E402

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Predicción de córners de un partido")
    ap.add_argument("local", help="Selección local (estilo fixtures o StatsBomb)")
    ap.add_argument("visitante", help="Selección visitante")
    ap.add_argument("--local", dest="ventaja_local", action="store_true",
                    help="Aplica ventaja de local (por defecto: cancha neutral)")
    ap.add_argument("--lineas", nargs="+", type=float, default=list(LINEAS_OU_DEFECTO),
                    help="Líneas de Over/Under a evaluar")
    args = ap.parse_args()

    df = pd.read_csv(PROC / "statsbomb_partidos.csv", parse_dates=["date"])
    modelo = CornersModel()
    modelo.fit(df)

    h = sb.a_nombre_statsbomb(args.local)
    a = sb.a_nombre_statsbomb(args.visitante)
    neutral = not args.ventaja_local

    faltan = [orig for orig, sbname in ((args.local, h), (args.visitante, a))
              if sbname not in modelo.attack_.index]

    pred = modelo.predecir_partido(h, a, neutral=neutral, lineas=args.lineas)

    print(f"\n=== Córners: {args.local} vs {args.visitante} "
          f"({'neutral' if neutral else 'local con ventaja'}) ===")
    print(f"Córners esperados:  {args.local} {pred['corners_local']:.2f}  |  "
          f"{args.visitante} {pred['corners_visita']:.2f}  |  "
          f"total {pred['corners_total']:.2f}")
    print(f"\nOver/Under (total córners):")
    for ln in args.lineas:
        po = pred["prob_over"][ln]
        print(f"   línea {ln:>5}:  Over {po*100:5.1f}%   Under {(1-po)*100:5.1f}%")
    print(f"\n1x2 de córners (quién saca más):")
    print(f"   {args.local:<18} {pred['prob_mas_corners_local']*100:5.1f}%")
    print(f"   Empate             {pred['prob_empate_corners']*100:5.1f}%")
    print(f"   {args.visitante:<18} {pred['prob_mas_corners_visita']*100:5.1f}%")
    sk = pred["skellam_1x2"]
    print(f"   (Skellam exacta: {sk[0]*100:.1f}% / {sk[1]*100:.1f}% / {sk[2]*100:.1f}%)")
    if faltan:
        print(f"\n[AVISO] Sin historial StatsBomb: {', '.join(faltan)} -> usa el "
              f"promedio del campo; la predicción es poco informativa para ese lado.")


if __name__ == "__main__":
    main()

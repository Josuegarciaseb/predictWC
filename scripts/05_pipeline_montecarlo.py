"""
scripts/05_pipeline_montecarlo.py
====================================
Fase 5: simulación Monte Carlo del Mundial 2026 completo (grupos -> Octavos
de 32 -> Octavos de Final -> Cuartos -> Semis -> Final).

Ver src/montecarlo.py y src/torneo.py para el diseño y las limitaciones
reconocidas (asignación simplificada de "mejores terceros" a la llave,
Elo/ataque-defensa estáticos durante toda la simulación, árbol de Octavos32
-> Octavos16 asumido secuencial).
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from elo import calcular_elo_historico
from models.poisson_dixon_coles import DixonColesModel
from montecarlo import simular_torneo_montecarlo

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

N_SIMULACIONES = 5000


def main():
    historico = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])

    print("Ajustando Dixon-Coles con todo el histórico disponible...")
    dc = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    dc.fit(historico)

    cols_elo = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    historico_elo = calcular_elo_historico(historico[cols_elo])
    elo_final = historico_elo.attrs["elo_final"]

    print(f"Corriendo {N_SIMULACIONES} simulaciones del Mundial completo...")
    t0 = time.time()
    tabla = simular_torneo_montecarlo(dc, elo_final, historico, n_sims=N_SIMULACIONES, seed=42)
    print(f"Listo en {time.time() - t0:.0f}s")

    # chequeos de consistencia (deben dar exactamente 32 / 1.0 / 2.0)
    print(f"\nChequeo -> suma prob_octavos_32: {tabla['prob_octavos_32'].sum():.1f} (esperado 32.0)")
    print(f"Chequeo -> suma prob_campeon:     {tabla['prob_campeon'].sum():.3f} (esperado 1.0)")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUTS_DIR / "predicciones_fase5_montecarlo.csv", index=False)

    print("\n=== Fase 5 -- Top 15 favoritos a ganar el Mundial 2026 (Monte Carlo) ===")
    print(tabla.head(15).to_string(index=False))

    print(f"\nGuardado: {OUTPUTS_DIR / 'predicciones_fase5_montecarlo.csv'}")


if __name__ == "__main__":
    main()

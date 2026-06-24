"""
FASE 8 -- Archivo acumulado de predicciones.

Problema que resuelve:
    El dataset de martj42 se actualiza a diario. En cuanto un partido se juega,
    deja de estar en `partidos_a_predecir` (pasa al histórico) y, por tanto,
    desaparece de los outputs de las fases 2 y 4 y de las matrices. La web
    construye sus tarjetas SOLO desde esos outputs, así que el partido jugado
    -- y el pick que el modelo dio para él -- se pierde, y con él el historial
    de aciertos (la sección "Resultados").

Solución:
    Mantener un archivo acumulado que CONGELA la predicción de cada partido.
    Regla de fusión (el actual manda):
      - Partido que sigue por jugarse -> se refresca con la predicción nueva.
      - Partido que ya se jugó (ya no está en el output actual) -> se conserva
        tal cual quedó su última predicción antes de jugarse.
      - Partido nuevo -> se añade.

    La web lee estos *_archivo.* en lugar de los outputs "vivos", de modo que
    el historial nunca se pierde aunque el dataset avance.

Se ejecuta al final del pipeline (después de 02, 04 y 07).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

# (output vivo, archivo acumulado)
CSV_FILES = [
    ("predicciones_fase4_stacking_ensemble.csv", "predicciones_fase4_archivo.csv"),
    ("predicciones_fase2_poisson_dixon_coles.csv", "predicciones_fase2_archivo.csv"),
]
MATRICES_VIVO = "matrices_marcador.json"
MATRICES_ARCHIVO = "matrices_marcador_archivo.json"

# Clave de un partido: misma que usa la web para cruzar fases (local|visitante).
CLAVE = ["local", "visitante"]


def _clave(df: pd.DataFrame) -> pd.Series:
    return df["local"].astype(str) + "|" + df["visitante"].astype(str)


def archivar_csv(vivo: str, archivo: str) -> None:
    df_vivo = pd.read_csv(OUTPUTS / vivo)
    ruta_archivo = OUTPUTS / archivo

    if ruta_archivo.exists():
        df_arch = pd.read_csv(ruta_archivo)
        df_arch = df_arch[~_clave(df_arch).isin(set(_clave(df_vivo)))]
        df = pd.concat([df_arch, df_vivo], ignore_index=True)
    else:
        df = df_vivo

    if "fecha" in df.columns:
        df = df.sort_values("fecha").reset_index(drop=True)

    df.to_csv(ruta_archivo, index=False)
    print(f"Archivado {archivo:<42} {len(df):>4} partidos "
          f"({len(df) - len(df_vivo)} congelados, {len(df_vivo)} vivos)")


def archivar_matrices() -> None:
    with open(OUTPUTS / MATRICES_VIVO, encoding="utf-8") as f:
        vivo = json.load(f)

    ruta_archivo = OUTPUTS / MATRICES_ARCHIVO
    if ruta_archivo.exists():
        with open(ruta_archivo, encoding="utf-8") as f:
            archivo = json.load(f)
    else:
        archivo = {}

    archivo.update(vivo)  # el actual manda; las claves ya jugadas se conservan

    with open(ruta_archivo, "w", encoding="utf-8") as f:
        json.dump(archivo, f, ensure_ascii=False)
    print(f"Archivado {MATRICES_ARCHIVO:<42} {len(archivo):>4} matrices "
          f"({len(archivo) - len(vivo)} congeladas, {len(vivo)} vivas)")


def main() -> None:
    for vivo, archivo in CSV_FILES:
        archivar_csv(vivo, archivo)
    archivar_matrices()
    print("Archivo acumulado actualizado.")


if __name__ == "__main__":
    main()

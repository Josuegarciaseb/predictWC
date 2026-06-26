"""Fase A (FBref) — Ingesta de stats de partido de selecciones vía soccerdata.

Rol: HISTÓRICO de entreno (tiros, tiros a puerta, tarjetas, faltas y córners donde
FBref los traiga) + respaldo en vivo del Mundial 2026 (sus páginas se actualizan a
diario). Complementa a StatsBomb (que no tiene el Mundial en vivo).

⚠️ IMPORTANTE: FBref está protegido por Cloudflare y en el entorno de desarrollo de
Claude devuelve 403 / se cuelga. Este script está pensado para correr en TU máquina
(IP residencial), donde soccerdata sí suele pasar. Respeta el rate-limiting propio de
soccerdata (NO se desactiva). Es self-verifying: imprime las columnas REALES que
encuentra para no hardcodear nada a ciegas (si algún nombre de columna difiere, se
ve en consola y se ajusta el mapeo `LEAF`).

Ligas verificadas en soccerdata 1.9.0: 'INT-World Cup', 'INT-European Championship'.
(Copa América / AFCON / Nations League NO están en la config por defecto; se podrían
añadir en ~/soccerdata/config/league_dict.json.)

Uso:
    python scripts/A04_ingesta_fbref.py --seasons 2018 2022 2026
    python scripts/A04_ingesta_fbref.py --leagues "INT-World Cup" --seasons 2026
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import nombres_seleccion as ns  # noqa: E402

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# Nombre de hoja (leaf) en la tabla de cada stat_type de FBref -> métrica nuestra.
# Si en tu primera corrida el nombre real difiere, se imprime y se ajusta aquí.
LEAF = {
    "shooting":      {"shots": "Sh", "sot": "SoT"},
    "misc":          {"yellow": "CrdY", "red": "CrdR", "fouls": "Fls"},
    "passing_types": {"corners": "CK"},
}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana columnas MultiIndex a su nombre de hoja (último nivel)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _pick(df: pd.DataFrame, leaf: str):
    """Devuelve la columna cuyo nombre de hoja casa (exacto, luego laxo)."""
    if leaf in df.columns:
        return df[leaf]
    for c in df.columns:
        if str(c).strip().lower() == leaf.lower():
            return df[c]
    return None


def ingerir(leagues, seasons) -> pd.DataFrame:
    import soccerdata as sd
    print(f"FBref: leagues={leagues} seasons={seasons}")
    fb = sd.FBref(leagues=leagues, seasons=seasons)

    sched = fb.read_schedule().reset_index()
    print(f"\nCalendario: {len(sched)} partidos. Columnas: {list(sched.columns)[:15]}")

    # Identificador de partido común para cruzar stats con calendario.
    key_cols = [c for c in ("league", "season", "game") if c in sched.columns]
    sched_small = sched[key_cols + ["date", "home_team", "away_team"]].copy()

    # Acumula stats por (key, team).
    base = None
    for stat_type, leaves in LEAF.items():
        try:
            ts = fb.read_team_match_stats(stat_type=stat_type).reset_index()
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] stat_type={stat_type} falló: {type(e).__name__}: {e}")
            continue
        ts = _flatten(ts)
        print(f"\n[{stat_type}] columnas reales: {list(ts.columns)}")
        cols = {"team": ts["team"]}
        for kc in key_cols:
            if kc in ts.columns:
                cols[kc] = ts[kc]
        for metrica, leaf in leaves.items():
            serie = _pick(ts, leaf)
            if serie is None:
                print(f"  [AVISO] no encontré hoja '{leaf}' para {metrica} en {stat_type}")
            else:
                cols[metrica] = pd.to_numeric(serie, errors="coerce")
        parcial = pd.DataFrame(cols)
        base = parcial if base is None else base.merge(parcial, on=key_cols + ["team"], how="outer")

    if base is None:
        raise SystemExit("No se pudo leer ninguna tabla de stats (¿Cloudflare/403?).")

    # Une con el calendario y arma filas home/away.
    base["team_canon"] = ns.normalizar_columna(base["team"])
    filas = []
    desconocidos: set = set()
    for _, g in sched_small.iterrows():
        h = ns.canonical(g["home_team"]); a = ns.canonical(g["away_team"])
        sel = base
        for kc in key_cols:
            sel = sel[sel[kc] == g[kc]]
        rh = sel[sel["team_canon"] == h]
        ra = sel[sel["team_canon"] == a]
        if rh.empty or ra.empty:
            if h not in ns.SELECCIONES_WC2026: desconocidos.add(h)
            if a not in ns.SELECCIONES_WC2026: desconocidos.add(a)
            continue
        rh, ra = rh.iloc[0], ra.iloc[0]
        fila = {"date": g["date"], "home_team": h, "away_team": a, "source": "fbref"}
        for met in ("shots", "sot", "corners", "yellow", "red", "fouls"):
            fila[f"home_{met}"] = rh.get(met)
            fila[f"away_{met}"] = ra.get(met)
        filas.append(fila)

    if desconocidos:
        print(f"\n[Nombres sin cruzar -> revisar ALIAS en nombres_seleccion.py]: {sorted(desconocidos)}")
    df = pd.DataFrame(filas)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+",
                    default=["INT-World Cup", "INT-European Championship"])
    ap.add_argument("--seasons", nargs="+", default=["2018", "2022", "2026"])
    args = ap.parse_args()

    df = ingerir(args.leagues, args.seasons)
    PROC.mkdir(parents=True, exist_ok=True)
    ruta = PROC / "fbref_partidos.csv"
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"\nGuardado: {ruta}  ({len(df)} partidos)")
    if not df.empty:
        cov = {m: int(df[f"home_{m}"].notna().sum()) for m in
               ("shots", "sot", "corners", "yellow", "red", "fouls")}
        print(f"Cobertura por métrica (partidos con dato local): {cov}")


if __name__ == "__main__":
    main()

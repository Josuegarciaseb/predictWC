from __future__ import annotations

import math
import pandas as pd

ELO_INICIAL = 1500
VENTAJA_LOCAL = 100


def k_por_torneo(tournament: str) -> int:
    t = tournament.lower()

    if t == "fifa world cup":
        return 60
    if "world cup qualification" in t:
        return 35
    if any(x in t for x in [
        "uefa euro", "copa américa", "copa america", "african cup of nations",
        "afc asian cup", "gold cup", "concacaf nations league", "uefa nations league",
    ]) and "qualification" not in t:
        return 45
    if "qualification" in t:
        return 30
    if t == "friendly":
        return 20
    return 25


def _multiplicador_goles(diferencia_goles: int) -> float:
    dg = abs(diferencia_goles)
    if dg <= 1:
        return 1.0
    if dg == 2:
        return 1.5
    return (11 + dg) / 8


def calcular_elo_historico(
    df: pd.DataFrame,
    elo_inicial: int = ELO_INICIAL,
    ventaja_local: int = VENTAJA_LOCAL,
) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    elo_actual: dict[str, float] = {}
    lista_elo_local_antes = []
    lista_elo_visita_antes = []

    for fila in df.itertuples(index=False):
        local = fila.home_team
        visita = fila.away_team
        neutral = bool(getattr(fila, "neutral", False))

        elo_local = elo_actual.get(local, elo_inicial)
        elo_visita = elo_actual.get(visita, elo_inicial)

        lista_elo_local_antes.append(elo_local)
        lista_elo_visita_antes.append(elo_visita)

        hay_resultado = pd.notna(fila.home_score) and pd.notna(fila.away_score)
        if not hay_resultado:
            continue

        ventaja = 0 if neutral else ventaja_local
        diferencia_rating = (elo_local + ventaja) - elo_visita
        E_local = 1 / (1 + 10 ** (-diferencia_rating / 400))

        home_score, away_score = fila.home_score, fila.away_score
        if home_score > away_score:
            S_local = 1.0
        elif home_score == away_score:
            S_local = 0.5
        else:
            S_local = 0.0
        S_visita = 1.0 - S_local
        E_visita = 1.0 - E_local

        K = k_por_torneo(fila.tournament)
        G = _multiplicador_goles(int(home_score - away_score))

        nuevo_elo_local = elo_local + K * G * (S_local - E_local)
        nuevo_elo_visita = elo_visita + K * G * (S_visita - E_visita)

        elo_actual[local] = nuevo_elo_local
        elo_actual[visita] = nuevo_elo_visita

    df["elo_local_antes"] = lista_elo_local_antes
    df["elo_visita_antes"] = lista_elo_visita_antes
    df.attrs["elo_final"] = elo_actual
    return df


def tabla_ranking_actual(elo_final: dict[str, float], top: int = 20) -> pd.DataFrame:
    ranking = (
        pd.Series(elo_final, name="elo")
        .sort_values(ascending=False)
        .head(top)
        .reset_index()
        .rename(columns={"index": "seleccion"})
    )
    ranking.index = ranking.index + 1
    ranking.index.name = "rank"
    return ranking

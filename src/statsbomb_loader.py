"""Ingesta de StatsBomb Open Data a nivel selecciones (internacional masculino).

El dataset martj42 que alimenta el pipeline de goles NO trae córners, tarjetas
ni tiros: solo el marcador. Este módulo abre una fuente nueva —StatsBomb Open
Data— de la que se derivan, a partir de los eventos crudos de cada partido:

  - córners      (pase con pass.type == 'Corner')
  - tiros         (evento 'Shot')
  - tiros a puerta (Shot con outcome en {Goal, Saved, Saved To Post, Saved Off T})
  - tarjetas      (card en 'Bad Behaviour' o en 'Foul Committed')
  - faltas        (evento 'Foul Committed')

Todo se cachea en disco (data/raw/statsbomb/) para no re-descargar, y el
resultado se materializa como CSV a nivel partido con el mismo estilo
home/away que data/processed/historico_con_elo.csv.

Fuente: https://github.com/statsbomb/open-data  (licencia no comercial de
StatsBomb; ver su LICENSE). Es JSON estático en GitHub, sin rate-limit duro,
pero igual se hace una pausa de cortesía entre descargas.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "statsbomb"
PAUSA_SEG = 0.3  # cortesía entre descargas

# Torneos de selecciones masculinas relevantes para el Mundial 2026.
TORNEOS_OBJETIVO = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa America",
    "African Cup of Nations",
    "UEFA Nations League",
}

# País anfitrión por (competition_id, season_id). En torneos los partidos se
# juegan en sede neutral salvo los del anfitrión. Euro 2020 fue multi-sede
# (paneuropeo): se trata todo como neutral. Lo que no esté aquí -> neutral.
ANFITRION = {
    (43, 3): "Russia",          # World Cup 2018
    (43, 106): "Qatar",         # World Cup 2022
    (55, 282): "Germany",       # Euro 2024
    (223, 282): "United States",  # Copa America 2024
    (1267, 107): "Ivory Coast",  # AFCON 2023 (Côte d'Ivoire)
    # World Cups históricos: anfitrión conocido pero datos escasos.
    (43, 269): "Sweden",        # 1958
    (43, 270): "Chile",         # 1962
    (43, 272): "Mexico",        # 1970
    (43, 51): "Germany",        # 1974 (RFA)
    (43, 54): "Mexico",         # 1986
    (43, 55): "Italy",          # 1990
    # Euro 2020 (55, 43): multi-sede -> sin anfitrión -> todo neutral.
}

SOT_OUTCOMES = {"Goal", "Saved", "Saved To Post", "Saved Off Target"}

# Mapeo de nombres de selección: como aparecen en los fixtures del Mundial 2026
# (data/processed/partidos_a_predecir.csv, estilo martj42) -> como aparecen en
# StatsBomb. Solo se listan los que difieren. Las 8 selecciones de WC2026 que no
# tienen NINGÚN partido en StatsBomb (Bosnia, Curaçao, Haití, Irak, Jordania,
# Nueva Zelanda, Noruega, Uzbekistán) caen al promedio del campo en el modelo.
MAPEO_FIXTURE_A_SB = {
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
}


def a_nombre_statsbomb(nombre: str) -> str:
    """Traduce un nombre estilo fixtures/martj42 al nombre de StatsBomb."""
    return MAPEO_FIXTURE_A_SB.get(nombre, nombre)


# --------------------------------------------------------------------------- #
# Descarga con caché en disco
# --------------------------------------------------------------------------- #
def _fetch_json(url: str, cache_path: Path, pausa: bool = True):
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    if pausa:
        time.sleep(PAUSA_SEG)
    return data


def cargar_competiciones():
    return _fetch_json(f"{BASE_URL}/competitions.json",
                       RAW_DIR / "competitions.json", pausa=False)


def cargar_partidos(competition_id: int, season_id: int):
    return _fetch_json(
        f"{BASE_URL}/matches/{competition_id}/{season_id}.json",
        RAW_DIR / "matches" / f"{competition_id}_{season_id}.json",
        pausa=False,
    )


def cargar_eventos(match_id: int):
    return _fetch_json(
        f"{BASE_URL}/events/{match_id}.json",
        RAW_DIR / "events" / f"{match_id}.json",
    )


def temporadas_internacionales(generos=("male",)):
    """Lista de competiciones-temporada de selecciones a ingerir."""
    comps = cargar_competiciones()
    out = []
    for c in comps:
        if c["competition_name"] not in TORNEOS_OBJETIVO:
            continue
        if c.get("competition_gender", "male") not in generos:
            continue
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Parseo de eventos -> estadísticas por equipo
# --------------------------------------------------------------------------- #
def _stats_equipo_vacio() -> dict:
    return {"corners": 0, "shots": 0, "sot": 0,
            "yellow": 0, "red": 0, "fouls": 0}


def parsear_eventos(eventos: list) -> dict[str, dict]:
    """Devuelve {nombre_equipo: {corners, shots, sot, yellow, red, fouls}}."""
    stats: dict[str, dict] = {}

    def equipo(nombre: str) -> dict:
        return stats.setdefault(nombre, _stats_equipo_vacio())

    for e in eventos:
        team = e.get("team", {}).get("name")
        if team is None:
            continue
        t = e["type"]["name"]

        if t == "Pass":
            if e.get("pass", {}).get("type", {}).get("name") == "Corner":
                equipo(team)["corners"] += 1
            continue

        if t == "Shot":
            s = equipo(team)
            s["shots"] += 1
            out = e.get("shot", {}).get("outcome", {}).get("name")
            if out in SOT_OUTCOMES:
                s["sot"] += 1
            continue

        if t == "Foul Committed":
            s = equipo(team)
            s["fouls"] += 1
            card = e.get("foul_committed", {}).get("card", {}).get("name")
            if card:
                if card.startswith("Red") or card == "Second Yellow":
                    s["red"] += 1
                elif card == "Yellow Card":
                    s["yellow"] += 1
            continue

        if t == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card", {}).get("name")
            if card:
                s = equipo(team)
                if card.startswith("Red") or card == "Second Yellow":
                    s["red"] += 1
                elif card == "Yellow Card":
                    s["yellow"] += 1

    return stats


def parsear_tiros_jugador(eventos: list, match_id: int) -> list[dict]:
    """Filas a nivel jugador para Fase D: tiros y tiros a puerta por jugador."""
    acc: dict[int, dict] = {}
    for e in eventos:
        if e["type"]["name"] != "Shot":
            continue
        player = e.get("player")
        if not player:
            continue
        pid = player["id"]
        d = acc.setdefault(pid, {
            "match_id": match_id,
            "player_id": pid,
            "player": player["name"],
            "team": e.get("team", {}).get("name"),
            "shots": 0,
            "sot": 0,
        })
        d["shots"] += 1
        if e.get("shot", {}).get("outcome", {}).get("name") in SOT_OUTCOMES:
            d["sot"] += 1
    return list(acc.values())


def parsear_minutos_jugador(eventos: list, match_id: int,
                            home: str, away: str) -> list[dict]:
    """Minutos jugados + tiros a puerta por jugador en un partido.

    Exposición = minutos en cancha, derivada de 'Starting XI' (titulares, min 0) y
    'Substitution' (entradas/salidas). El minuto final del partido se toma del
    último 'Half End'. Se incluyen los jugadores con 0 tiros a puerta (clave para
    una tasa por 90' no sesgada). Aproximación: no se recorta por expulsión (raro).
    """
    # Minuto final (cubre prórroga: último Half End de cualquier periodo).
    half_ends = [e.get("minute", 0) for e in eventos if e["type"]["name"] == "Half End"]
    fin = max(half_ends) if half_ends else max((e.get("minute", 0) for e in eventos), default=90)

    reg: dict[int, dict] = {}  # player_id -> dict
    for e in eventos:
        t = e["type"]["name"]
        if t == "Starting XI":
            team = e.get("team", {}).get("name")
            for j in e.get("tactics", {}).get("lineup", []):
                p = j["player"]
                reg[p["id"]] = {"player": p["name"], "team": team,
                                "on": 0, "off": None, "is_starter": True}
        elif t == "Substitution":
            minute = e.get("minute", fin)
            team = e.get("team", {}).get("name")
            off = e.get("player")
            if off and off["id"] in reg:
                reg[off["id"]]["off"] = minute
            rep = e.get("substitution", {}).get("replacement")
            if rep:
                reg[rep["id"]] = {"player": rep["name"], "team": team,
                                  "on": minute, "off": None, "is_starter": False}

    # Tiros a puerta por jugador (los que no aparezcan -> 0).
    sot = {d["player_id"]: d for d in parsear_tiros_jugador(eventos, match_id)}

    filas = []
    for pid, d in reg.items():
        off = d["off"] if d["off"] is not None else fin
        minutos = max(0, off - d["on"])
        team = d["team"]
        opp = away if team == home else home
        s = sot.get(pid, {})
        filas.append({
            "match_id": match_id, "player_id": pid, "player": d["player"],
            "team": team, "opponent": opp, "minutes": minutos,
            "is_starter": d["is_starter"],
            "shots": s.get("shots", 0), "sot": s.get("sot", 0),
        })
    return filas


def _es_neutral(comp_id: int, season_id: int, home: str, away: str) -> bool:
    anfitrion = ANFITRION.get((comp_id, season_id))
    if anfitrion is None:
        return True  # multi-sede o desconocido -> neutral
    return home != anfitrion and away != anfitrion


def fila_partido(match: dict, stats: dict[str, dict],
                 comp_name: str, season_name: str,
                 comp_id: int, season_id: int) -> dict | None:
    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]
    sh = stats.get(home, _stats_equipo_vacio())
    sa = stats.get(away, _stats_equipo_vacio())
    stage = match.get("competition_stage", {}).get("name", "")
    return {
        "date": match["match_date"],
        "home_team": home,
        "away_team": away,
        "home_score": match.get("home_score"),
        "away_score": match.get("away_score"),
        "tournament": comp_name,
        "season": season_name,
        "stage": stage,
        "knockout": "Group" not in stage,  # eliminatoria => más tensión/tarjetas
        "neutral": _es_neutral(comp_id, season_id, home, away),
        "home_corners": sh["corners"], "away_corners": sa["corners"],
        "home_shots": sh["shots"], "away_shots": sa["shots"],
        "home_sot": sh["sot"], "away_sot": sa["sot"],
        "home_yellow": sh["yellow"], "away_yellow": sa["yellow"],
        "home_red": sh["red"], "away_red": sa["red"],
        "home_fouls": sh["fouls"], "away_fouls": sa["fouls"],
        "match_id": match["match_id"],
    }

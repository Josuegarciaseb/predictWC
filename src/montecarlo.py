from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd

from torneo import GRUPOS, CALENDARIO_GRUPOS, BRACKET_R32, BRACKET_R16, BRACKET_AVANCE

FASES = ["grupos", "octavos_32", "octavos_16", "cuartos", "semis", "final", "campeon"]


def cargar_resultados_reales(historico: pd.DataFrame) -> dict:
    wc = historico[(historico["tournament"] == "FIFA World Cup") & (historico["date"] >= "2026-06-01")]
    resultados = {}
    for _, r in wc.iterrows():
        key = (r["date"].strftime("%Y-%m-%d"), frozenset([r["home_team"], r["away_team"]]))
        resultados[key] = (r["home_team"], int(r["home_score"]), int(r["away_score"]))
    return resultados


# --- Anclaje a la llave real (cuando la fase de grupos ya terminó) -------------
#
# El simulador "clásico" re-simula el Mundial entero desde la primera jornada de
# grupos. Una vez que la fase de grupos terminó en la realidad, eso es tirar
# información: ya sabemos quién clasificó y contra quién juega cada quien en la
# primera ronda eliminatoria (R32). Estas funciones leen esa llave REAL del propio
# dataset (martj42 publica los cruces) y la usan como punto de arranque, en lugar
# de re-muestrear los grupos. Es data-driven: si la fase de grupos aún no termina,
# `construir_bracket_real` devuelve None y el simulador cae al modo clásico.

_GRUPOS_SETS = {frozenset([local, visita]) for _, _, local, visita in CALENDARIO_GRUPOS}


def _resultados_wc_por_set(historico: pd.DataFrame) -> dict:
    """Resultados reales del Mundial 2026 indexados por el par de equipos."""
    wc = historico[(historico["tournament"] == "FIFA World Cup") & (historico["date"] >= "2026-06-01")]
    out = {}
    for _, r in wc.iterrows():
        if pd.notna(r["home_score"]) and pd.notna(r["away_score"]):
            out[frozenset([r["home_team"], r["away_team"]])] = (
                r["home_team"], int(r["home_score"]), int(r["away_score"]),
            )
    return out


def _standings_reales(res_por_set: dict):
    """Tabla final real por grupo. Devuelve (tablas, stats_terceros) o (None, None)
    si todavía falta algún partido de grupos por jugarse."""
    if not all(s in res_por_set for s in _GRUPOS_SETS):
        return None, None

    tablas, stats_terceros = {}, {}
    for letra, equipos in GRUPOS.items():
        st = {e: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for e in equipos}
        for _, grupo, local, visita in CALENDARIO_GRUPOS:
            if grupo != letra:
                continue
            equipo_local, gl, gv = res_por_set[frozenset([local, visita])]
            if equipo_local != local:
                gl, gv = gv, gl
            st[local]["gf"] += gl; st[local]["ga"] += gv
            st[visita]["gf"] += gv; st[visita]["ga"] += gl
            if gl > gv:
                st[local]["pts"] += 3
            elif gl == gv:
                st[local]["pts"] += 1; st[visita]["pts"] += 1
            else:
                st[visita]["pts"] += 3
        for e in equipos:
            st[e]["gd"] = st[e]["gf"] - st[e]["ga"]
        orden = sorted(equipos, key=lambda e: (-st[e]["pts"], -st[e]["gd"], -st[e]["gf"]))
        tablas[letra] = orden
        stats_terceros[orden[2]] = {**st[orden[2]], "grupo": letra}
    return tablas, stats_terceros


def construir_bracket_real(historico: pd.DataFrame, fixtures_pendientes: pd.DataFrame | None):
    """Construye la llave R32 REAL a partir del dataset. Devuelve
    (bracket, clasificados, eliminados) o None si la fase de grupos no terminó.

    `bracket` es {id_BRACKET_R32: (local, visita)} con los cruces reales. Cada
    cruce se ancla leyendo el fixture real publicado (por martj42) que contiene al
    equipo resuelto por posición de grupo; así la asignación de "mejores terceros"
    sale de la realidad, no del repartidor simplificado. Si algún cruce todavía no
    está publicado, se deriva de la tabla real como respaldo."""
    res_por_set = _resultados_wc_por_set(historico)
    tablas, stats_terceros = _standings_reales(res_por_set)
    if tablas is None:
        return None

    terceros8 = sorted(
        stats_terceros,
        key=lambda e: (-stats_terceros[e]["pts"], -stats_terceros[e]["gd"], -stats_terceros[e]["gf"]),
    )[:8]
    clasificados = set(terceros8)
    for orden in tablas.values():
        clasificados.update(orden[:2])
    todos = {e for eq in GRUPOS.values() for e in eq}
    eliminados = todos - clasificados

    # Fixtures eliminatorios (par de equipos), ordenados por fecha: el primero que
    # contiene a un equipo es su partido de R32 (las rondas posteriores van después).
    ko_fixtures = []  # (fecha, local, visita)
    vistos = set()
    fuentes = []
    if fixtures_pendientes is not None and len(fixtures_pendientes):
        fuentes.append(fixtures_pendientes)
    fuentes.append(historico)
    for df in fuentes:
        wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01")]
        for _, r in wc.iterrows():
            s = frozenset([r["home_team"], r["away_team"]])
            if s in _GRUPOS_SETS or s in vistos:
                continue
            vistos.add(s)
            ko_fixtures.append((r["date"], r["home_team"], r["away_team"]))
    ko_fixtures.sort(key=lambda x: x[0])

    terceros_disponibles = list(terceros8)
    bracket, usados = {}, set()
    for match in BRACKET_R32:
        pos_teams = []
        for lado in ("slot_a", "slot_b"):
            tipo, *resto = match[lado]
            if tipo == "pos":
                letra, posicion = resto
                pos_teams.append(tablas[letra][posicion - 1])

        cruce = None
        for pos_team in pos_teams:
            for fecha, local, visita in ko_fixtures:
                s = frozenset([local, visita])
                if s in usados:
                    continue
                if pos_team in s:
                    cruce = (local, visita)
                    usados.add(s)
                    for t in (local, visita):
                        if t in terceros_disponibles:
                            terceros_disponibles.remove(t)
                    break
            if cruce:
                break

        if cruce is None:  # respaldo: derivar de la tabla real si el fixture no se publicó aún
            equipos = []
            for lado in ("slot_a", "slot_b"):
                tipo, *resto = match[lado]
                if tipo == "pos":
                    letra, posicion = resto
                    equipos.append(tablas[letra][posicion - 1])
                else:
                    candidatos = resto[0]
                    pick = next((t for t in terceros_disponibles if stats_terceros[t]["grupo"] in candidatos),
                                terceros_disponibles[0])
                    terceros_disponibles.remove(pick)
                    equipos.append(pick)
            cruce = tuple(equipos)
        bracket[match["id"]] = cruce

    return bracket, clasificados, eliminados


def cargar_resultados_ko(historico: pd.DataFrame, shootouts: pd.DataFrame | None) -> tuple[dict, dict]:
    """Resultados reales de partidos eliminatorios ya jugados, indexados por par de
    equipos (cada cruce eliminatorio es único). Permite que el simulador fije los
    resultados reales en vez de re-muestrearlos. Devuelve (marcadores, ganadores_penales)."""
    res_por_set = _resultados_wc_por_set(historico)
    marcadores = {s: v for s, v in res_por_set.items() if s not in _GRUPOS_SETS}

    ganadores_penales = {}
    if shootouts is not None and len(shootouts):
        so = shootouts[pd.to_datetime(shootouts["date"]) >= "2026-06-01"]
        for _, r in so.iterrows():
            ganadores_penales[frozenset([r["home_team"], r["away_team"]])] = r["winner"]
    return marcadores, ganadores_penales


class MotorSimulacion:
    def __init__(self, modelo_dc, elo_final: dict[str, float], resultados_reales: dict, rng: random.Random,
                 bracket_real: tuple | None = None, resultados_ko: dict | None = None,
                 ganadores_penales: dict | None = None):
        self.dc = modelo_dc
        self.elo_final = elo_final
        self.resultados_reales = resultados_reales
        self.rng = rng
        self.bracket_real = bracket_real
        self.resultados_ko = resultados_ko or {}
        self.ganadores_penales = ganadores_penales or {}
        self._cache_matrices: dict[tuple[str, str], np.ndarray] = {}


    def _matriz(self, local: str, visita: str) -> np.ndarray:
        key = (local, visita)
        if key not in self._cache_matrices:
            self._cache_matrices[key] = self.dc.matriz_marcador(local, visita, neutral=True)
        return self._cache_matrices[key]

    def _muestrear_marcador(self, local: str, visita: str) -> tuple[int, int]:
        M = self._matriz(local, visita)
        flat_idx = self.rng.choices(range(M.size), weights=M.ravel(), k=1)[0]
        return int(flat_idx // M.shape[1]), int(flat_idx % M.shape[1])

    def jugar_partido_grupo(self, fecha: str, local: str, visita: str) -> tuple[int, int]:
        key = (fecha, frozenset([local, visita]))
        real = self.resultados_reales.get(key)
        if real is not None:
            equipo_local_real, gl, gv = real
            return (gl, gv) if equipo_local_real == local else (gv, gl)
        return self._muestrear_marcador(local, visita)

    def jugar_eliminacion(self, local: str, visita: str) -> str:
        # Si este cruce ya se jugó en la realidad, usar el resultado real.
        key = frozenset([local, visita])
        real = self.resultados_ko.get(key)
        if real is not None:
            equipo_local_real, gl, gv = real
            if equipo_local_real != local:
                gl, gv = gv, gl
            if gl > gv:
                return local
            if gv > gl:
                return visita
            ganador_penales = self.ganadores_penales.get(key)
            if ganador_penales is not None:
                return ganador_penales
            # empate sin dato de penales: cae al desempate por probabilidad de abajo

        gl, gv = self._muestrear_marcador(local, visita)
        if gl > gv:
            return local
        if gv > gl:
            return visita
        M = self._matriz(local, visita)
        xs, ys = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]), indexing="ij")
        p_local = M[xs > ys].sum()
        p_visita = M[xs < ys].sum()
        p_local_norm = p_local / (p_local + p_visita)
        return local if self.rng.random() < p_local_norm else visita


    def _ordenar_tabla(self, stats: dict[str, dict], h2h_partidos: list[tuple[str, str, int, int]]) -> list[str]:
        equipos = list(stats.keys())

        def clave_principal(eq):
            return (-stats[eq]["pts"], )

        equipos.sort(key=lambda e: -stats[e]["pts"])
        grupos_de_empate = defaultdict(list)
        for e in equipos:
            grupos_de_empate[stats[e]["pts"]].append(e)

        orden_final = []
        for pts in sorted(grupos_de_empate.keys(), reverse=True):
            empatados = grupos_de_empate[pts]
            if len(empatados) == 1:
                orden_final.extend(empatados)
                continue


            mini = {e: {"pts": 0, "gd": 0, "gf": 0} for e in empatados}
            for a, b, ga, gb in h2h_partidos:
                if a in mini and b in mini:
                    mini[a]["gf"] += ga
                    mini[b]["gf"] += gb
                    mini[a]["gd"] += ga - gb
                    mini[b]["gd"] += gb - ga
                    if ga > gb:
                        mini[a]["pts"] += 3
                    elif ga == gb:
                        mini[a]["pts"] += 1
                        mini[b]["pts"] += 1
                    else:
                        mini[b]["pts"] += 3

            def clave_desempate(e):
                return (
                    -mini[e]["pts"], -mini[e]["gd"], -mini[e]["gf"],
                    -stats[e]["gd"], -stats[e]["gf"],
                    -self.elo_final.get(e, 1500),
                    self.rng.random(),
                )

            empatados_ordenados = sorted(empatados, key=clave_desempate)
            orden_final.extend(empatados_ordenados)

        return orden_final

    def jugar_grupo(self, letra: str) -> list[str]:
        equipos = GRUPOS[letra]
        stats = {e: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for e in equipos}
        partidos_jugados = []

        for fecha, grupo, local, visita in CALENDARIO_GRUPOS:
            if grupo != letra:
                continue
            gl, gv = self.jugar_partido_grupo(fecha, local, visita)
            partidos_jugados.append((local, visita, gl, gv))

            stats[local]["gf"] += gl
            stats[local]["ga"] += gv
            stats[visita]["gf"] += gv
            stats[visita]["ga"] += gl
            if gl > gv:
                stats[local]["pts"] += 3
            elif gl == gv:
                stats[local]["pts"] += 1
                stats[visita]["pts"] += 1
            else:
                stats[visita]["pts"] += 3

        for e in equipos:
            stats[e]["gd"] = stats[e]["gf"] - stats[e]["ga"]

        return self._ordenar_tabla(stats, partidos_jugados), stats

    def jugar_fase_de_grupos(self):
        tablas = {}
        stats_terceros = {}
        for letra in GRUPOS:
            orden, stats = self.jugar_grupo(letra)
            tablas[letra] = orden
            stats_terceros[orden[2]] = {**stats[orden[2]], "grupo": letra}
        return tablas, stats_terceros

    def mejores_terceros(self, stats_terceros: dict) -> list[str]:
        def clave(equipo):
            s = stats_terceros[equipo]
            return (-s["pts"], -s["gd"], -s["gf"], -self.elo_final.get(equipo, 1500), self.rng.random())
        return sorted(stats_terceros.keys(), key=clave)[:8]


    def asignar_llave_r32(self, tablas: dict[str, list[str]], terceros_clasificados: list[str],
                            stats_terceros: dict) -> dict[int, tuple[str, str]]:
        grupo_del_tercero = {eq: stats_terceros[eq]["grupo"] for eq in terceros_clasificados}
        disponibles = list(terceros_clasificados)
        resueltos: dict[tuple[int, str], str] = {}

        slots_terceros = [m for m in BRACKET_R32 if m["slot_a"][0] == "tercero" or m["slot_b"][0] == "tercero"]
        for match in slots_terceros:
            for lado in ["slot_a", "slot_b"]:
                tipo, *resto = match[lado]
                if tipo == "tercero":
                    grupos_candidatos = resto[0]
                    candidato = next((e for e in disponibles if grupo_del_tercero[e] in grupos_candidatos), None)
                    if candidato is None:
                        candidato = disponibles[0]
                    disponibles.remove(candidato)
                    resueltos[(match["id"], lado)] = candidato

        partidos = {}
        for match in BRACKET_R32:
            equipos_resueltos = []
            for lado in ["slot_a", "slot_b"]:
                if (match["id"], lado) in resueltos:
                    equipos_resueltos.append(resueltos[(match["id"], lado)])
                else:
                    tipo, letra, pos = match[lado]
                    equipos_resueltos.append(tablas[letra][pos - 1])
            partidos[match["id"]] = tuple(equipos_resueltos)
        return partidos

    def jugar_eliminacion_directa(self, partidos_r32: dict[int, tuple[str, str]]) -> dict[str, str]:
        avance = {}

        # Octavos de 32: se juega cada llave; el ganador queda indexado por su id.
        ganador_r32 = {}
        for idr, (local, visita) in partidos_r32.items():
            ganador = self.jugar_eliminacion(local, visita)
            avance[visita if ganador == local else local] = "octavos_32"
            ganador_r32[idr] = ganador

        # Octavos de final: adyacencia OFICIAL (no secuencial) desde BRACKET_R16.
        ganadores = []
        for id_a, id_b in BRACKET_R16:
            local, visita = ganador_r32[id_a], ganador_r32[id_b]
            ganador = self.jugar_eliminacion(local, visita)
            avance[visita if ganador == local else local] = "octavos_16"
            ganadores.append(ganador)

        # Cuartos -> semis -> final, siguiendo el árbol oficial.
        for nombre_fase, empates in zip(["cuartos", "semis", "final"], BRACKET_AVANCE):
            nuevos = []
            for idx_a, idx_b in empates:
                local, visita = ganadores[idx_a], ganadores[idx_b]
                ganador = self.jugar_eliminacion(local, visita)
                avance[visita if ganador == local else local] = nombre_fase
                nuevos.append(ganador)
            ganadores = nuevos

        avance[ganadores[0]] = "campeon"
        return avance


    def simular_una_vez(self) -> dict[str, str]:
        if self.bracket_real is not None:
            return self.simular_desde_bracket_real()
        return self.simular_torneo_completo()

    def simular_desde_bracket_real(self) -> dict[str, str]:
        """Arranca desde la llave R32 real: no re-simula los grupos. Los 32
        clasificados parten de octavos_32 y se juega la eliminatoria hacia adelante
        (usando resultados reales donde ya se jugó, muestreando donde no)."""
        bracket, clasificados, eliminados = self.bracket_real
        avance = {eq: "grupos" for eq in eliminados}
        for eq in clasificados:
            avance[eq] = "octavos_32"
        avance.update(self.jugar_eliminacion_directa(bracket))
        return avance

    def simular_torneo_completo(self) -> dict[str, str]:
        tablas, stats_terceros = self.jugar_fase_de_grupos()
        terceros_clasificados = self.mejores_terceros(stats_terceros)

        avance = {}
        for letra, orden in tablas.items():
            for eq in orden[:2]:
                avance[eq] = "octavos_32"
            for eq in orden[2:]:
                avance[eq] = "grupos"
        for eq in terceros_clasificados:
            avance[eq] = "octavos_32"

        partidos_r32 = self.asignar_llave_r32(tablas, terceros_clasificados, stats_terceros)
        avance_eliminacion = self.jugar_eliminacion_directa(partidos_r32)
        avance.update(avance_eliminacion)
        return avance


def simular_torneo_montecarlo(modelo_dc, elo_final: dict[str, float], historico: pd.DataFrame,
                                fixtures_pendientes: pd.DataFrame | None = None,
                                shootouts: pd.DataFrame | None = None,
                                n_sims: int = 3000, seed: int = 42) -> pd.DataFrame:
    resultados_reales = cargar_resultados_reales(historico)
    bracket_real = construir_bracket_real(historico, fixtures_pendientes)
    resultados_ko, ganadores_penales = cargar_resultados_ko(historico, shootouts)

    if bracket_real is not None:
        print("Modo ANCLADO: la fase de grupos terminó -> arranco desde la llave R32 real "
              "(no re-simulo grupos).")
    else:
        print("Modo CLÁSICO: la fase de grupos no ha terminado -> re-simulo el torneo completo.")

    todos_los_equipos = sorted({e for grupo in GRUPOS.values() for e in grupo})
    contador = {eq: defaultdict(int) for eq in todos_los_equipos}

    rng = random.Random(seed)
    for _ in range(n_sims):
        motor = MotorSimulacion(modelo_dc, elo_final, resultados_reales, rng,
                                bracket_real=bracket_real, resultados_ko=resultados_ko,
                                ganadores_penales=ganadores_penales)
        avance = motor.simular_una_vez()
        for eq in todos_los_equipos:
            contador[eq][avance.get(eq, "grupos")] += 1

    orden_fases = ["grupos", "octavos_32", "octavos_16", "cuartos", "semis", "final", "campeon"]

    def prob_llegar_al_menos(eq, fase_objetivo):
        idx_objetivo = orden_fases.index(fase_objetivo)
        return sum(c for fase, c in contador[eq].items() if orden_fases.index(fase) >= idx_objetivo) / n_sims

    filas = []
    for eq in todos_los_equipos:
        filas.append({
            "seleccion": eq,
            "prob_octavos_32": round(prob_llegar_al_menos(eq, "octavos_32"), 4),
            "prob_octavos_16": round(prob_llegar_al_menos(eq, "octavos_16"), 4),
            "prob_cuartos": round(prob_llegar_al_menos(eq, "cuartos"), 4),
            "prob_semis": round(prob_llegar_al_menos(eq, "semis"), 4),
            "prob_final": round(prob_llegar_al_menos(eq, "final"), 4),
            "prob_campeon": round(prob_llegar_al_menos(eq, "campeon"), 4),
        })

    tabla = pd.DataFrame(filas).sort_values("prob_campeon", ascending=False).reset_index(drop=True)
    return tabla

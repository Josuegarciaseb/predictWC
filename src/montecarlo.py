from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd

from torneo import GRUPOS, CALENDARIO_GRUPOS, BRACKET_R32

FASES = ["grupos", "octavos_32", "octavos_16", "cuartos", "semis", "final", "campeon"]


def cargar_resultados_reales(historico: pd.DataFrame) -> dict:
    wc = historico[(historico["tournament"] == "FIFA World Cup") & (historico["date"] >= "2026-06-01")]
    resultados = {}
    for _, r in wc.iterrows():
        key = (r["date"].strftime("%Y-%m-%d"), frozenset([r["home_team"], r["away_team"]]))
        resultados[key] = (r["home_team"], int(r["home_score"]), int(r["away_score"]))
    return resultados


class MotorSimulacion:
    def __init__(self, modelo_dc, elo_final: dict[str, float], resultados_reales: dict, rng: random.Random):
        self.dc = modelo_dc
        self.elo_final = elo_final
        self.resultados_reales = resultados_reales
        self.rng = rng
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
        ronda_actual = [partidos_r32[i] for i in sorted(partidos_r32.keys())]
        nombre_fase = "octavos_32"

        while True:
            ganadores = []
            for local, visita in ronda_actual:
                ganador = self.jugar_eliminacion(local, visita)
                perdedor = visita if ganador == local else local
                avance[perdedor] = nombre_fase
                ganadores.append(ganador)

            if len(ganadores) == 1:
                avance[ganadores[0]] = "campeon"
                break

            nombre_fase = {"octavos_32": "octavos_16", "octavos_16": "cuartos",
                            "cuartos": "semis", "semis": "final"}[nombre_fase]
            ronda_actual = [(ganadores[i], ganadores[i + 1]) for i in range(0, len(ganadores), 2)]

        return avance


    def simular_una_vez(self) -> dict[str, str]:
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
                                n_sims: int = 3000, seed: int = 42) -> pd.DataFrame:
    resultados_reales = cargar_resultados_reales(historico)
    todos_los_equipos = sorted({e for grupo in GRUPOS.values() for e in grupo})
    contador = {eq: defaultdict(int) for eq in todos_los_equipos}

    rng = random.Random(seed)
    for _ in range(n_sims):
        motor = MotorSimulacion(modelo_dc, elo_final, resultados_reales, rng)
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

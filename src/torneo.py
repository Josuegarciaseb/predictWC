"""
src/torneo.py
==============
Estructura oficial del Mundial 2026 (48 equipos, 12 grupos, formato de
Octavos de eliminación de 32 -- no Octavos de Final de 16 como en ediciones
anteriores). Datos obtenidos de la fuente oficial (FIFA/ESPN, sorteo del
5 de diciembre de 2025) vía búsqueda web, porque el dataset local
(`results.csv`) solo contiene 40 de los 72 partidos de fase de grupos -- los
otros 32 ya tenían resultado real al momento de armar este dataset y quedaron
en `historico_con_elo.csv`, no en `partidos_a_predecir.csv`.

Limitación reconocida (ver README): el reglamento oficial de FIFA define una
tabla de asignación exacta para decidir qué grupo específico aporta el
"mejor tercero" a cada cruce de Octavos de 32 (depende de CUÁLES 8 de los 12
grupos terminan aportando los mejores terceros -- hay cientos de combinaciones
posibles). Esa tabla completa no se reprodujo aquí; en su lugar se usa una
asignación determinista simplificada (ver `asignar_terceros_a_llave`) que
respeta los grupos candidatos de cada cruce pero no es la tabla oficial
exacta. Tampoco se publicó el árbol explícito de qué ganador de Octavos de 32
enfrenta a cuál en Octavos de Final -- se asume el emparejamiento secuencial
estándar (ganador M1 vs ganador M2, etc.), que es la convención más común en
este tipo de llaves pero podría no coincidir 100% con el árbol real de FIFA.
"""
from __future__ import annotations

GRUPOS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# (fecha, grupo, local, visita) -- los 72 partidos de fase de grupos.
# "Turkey"/"Czech Republic" se usan tal cual están en el dataset (ver
# data/raw/results.csv) para que el cruce con los modelos funcione directo.
CALENDARIO_GRUPOS: list[tuple[str, str, str, str]] = [
    # Grupo A
    ("2026-06-11", "A", "Mexico", "South Africa"),
    ("2026-06-11", "A", "South Korea", "Czech Republic"),
    ("2026-06-18", "A", "Czech Republic", "South Africa"),
    ("2026-06-18", "A", "Mexico", "South Korea"),
    ("2026-06-24", "A", "Czech Republic", "Mexico"),
    ("2026-06-24", "A", "South Africa", "South Korea"),
    # Grupo B
    ("2026-06-12", "B", "Canada", "Bosnia and Herzegovina"),
    ("2026-06-13", "B", "Qatar", "Switzerland"),
    ("2026-06-18", "B", "Switzerland", "Bosnia and Herzegovina"),
    ("2026-06-18", "B", "Canada", "Qatar"),
    ("2026-06-24", "B", "Switzerland", "Canada"),
    ("2026-06-24", "B", "Bosnia and Herzegovina", "Qatar"),
    # Grupo C
    ("2026-06-13", "C", "Brazil", "Morocco"),
    ("2026-06-13", "C", "Haiti", "Scotland"),
    ("2026-06-19", "C", "Scotland", "Morocco"),
    ("2026-06-19", "C", "Brazil", "Haiti"),
    ("2026-06-24", "C", "Scotland", "Brazil"),
    ("2026-06-24", "C", "Morocco", "Haiti"),
    # Grupo D
    ("2026-06-12", "D", "United States", "Paraguay"),
    ("2026-06-13", "D", "Australia", "Turkey"),
    ("2026-06-19", "D", "United States", "Australia"),
    ("2026-06-19", "D", "Turkey", "Paraguay"),
    ("2026-06-25", "D", "Turkey", "United States"),
    ("2026-06-25", "D", "Paraguay", "Australia"),
    # Grupo E
    ("2026-06-14", "E", "Germany", "Curaçao"),
    ("2026-06-14", "E", "Ivory Coast", "Ecuador"),
    ("2026-06-20", "E", "Germany", "Ivory Coast"),
    ("2026-06-20", "E", "Ecuador", "Curaçao"),
    ("2026-06-25", "E", "Ecuador", "Germany"),
    ("2026-06-25", "E", "Curaçao", "Ivory Coast"),
    # Grupo F
    ("2026-06-14", "F", "Netherlands", "Japan"),
    ("2026-06-14", "F", "Sweden", "Tunisia"),
    ("2026-06-20", "F", "Netherlands", "Sweden"),
    ("2026-06-20", "F", "Tunisia", "Japan"),
    ("2026-06-25", "F", "Japan", "Sweden"),
    ("2026-06-25", "F", "Tunisia", "Netherlands"),
    # Grupo G
    ("2026-06-15", "G", "Belgium", "Egypt"),
    ("2026-06-15", "G", "Iran", "New Zealand"),
    ("2026-06-21", "G", "Belgium", "Iran"),
    ("2026-06-21", "G", "New Zealand", "Egypt"),
    ("2026-06-26", "G", "New Zealand", "Belgium"),
    ("2026-06-26", "G", "Egypt", "Iran"),
    # Grupo H
    ("2026-06-15", "H", "Spain", "Cape Verde"),
    ("2026-06-15", "H", "Saudi Arabia", "Uruguay"),
    ("2026-06-21", "H", "Spain", "Saudi Arabia"),
    ("2026-06-21", "H", "Uruguay", "Cape Verde"),
    ("2026-06-26", "H", "Cape Verde", "Saudi Arabia"),
    ("2026-06-26", "H", "Uruguay", "Spain"),
    # Grupo I
    ("2026-06-16", "I", "France", "Senegal"),
    ("2026-06-16", "I", "Iraq", "Norway"),
    ("2026-06-22", "I", "France", "Iraq"),
    ("2026-06-22", "I", "Norway", "Senegal"),
    ("2026-06-26", "I", "Norway", "France"),
    ("2026-06-26", "I", "Senegal", "Iraq"),
    # Grupo J
    ("2026-06-16", "J", "Argentina", "Algeria"),
    ("2026-06-16", "J", "Austria", "Jordan"),
    ("2026-06-22", "J", "Argentina", "Austria"),
    ("2026-06-22", "J", "Jordan", "Algeria"),
    ("2026-06-27", "J", "Algeria", "Austria"),
    ("2026-06-27", "J", "Jordan", "Argentina"),
    # Grupo K
    ("2026-06-17", "K", "Portugal", "DR Congo"),
    ("2026-06-17", "K", "Uzbekistan", "Colombia"),
    ("2026-06-23", "K", "Portugal", "Uzbekistan"),
    ("2026-06-23", "K", "Colombia", "DR Congo"),
    ("2026-06-27", "K", "Colombia", "Portugal"),
    ("2026-06-27", "K", "DR Congo", "Uzbekistan"),
    # Grupo L
    ("2026-06-17", "L", "England", "Croatia"),
    ("2026-06-17", "L", "Ghana", "Panama"),
    ("2026-06-23", "L", "England", "Ghana"),
    ("2026-06-23", "L", "Panama", "Croatia"),
    ("2026-06-27", "L", "Panama", "England"),
    ("2026-06-27", "L", "Croatia", "Ghana"),
]

# Cruces de Octavos de 32. "1"/"2" = 1ro/2do del grupo. "3:[...]" = el mejor
# tercero disponible entre esos grupos candidatos (ver limitación arriba).
BRACKET_R32: list[dict] = [
    {"id": 1, "slot_a": ("pos", "A", 2), "slot_b": ("pos", "B", 2)},
    {"id": 2, "slot_a": ("pos", "C", 1), "slot_b": ("pos", "F", 2)},
    {"id": 3, "slot_a": ("pos", "E", 1), "slot_b": ("tercero", ["A", "B", "C", "D", "F"])},
    {"id": 4, "slot_a": ("pos", "F", 1), "slot_b": ("pos", "C", 2)},
    {"id": 5, "slot_a": ("pos", "E", 2), "slot_b": ("pos", "I", 2)},
    {"id": 6, "slot_a": ("pos", "I", 1), "slot_b": ("tercero", ["C", "D", "F", "G", "H"])},
    {"id": 7, "slot_a": ("pos", "A", 1), "slot_b": ("tercero", ["C", "E", "F", "H", "I"])},
    {"id": 8, "slot_a": ("pos", "L", 1), "slot_b": ("tercero", ["E", "H", "I", "J", "K"])},
    {"id": 9, "slot_a": ("pos", "G", 1), "slot_b": ("tercero", ["A", "E", "H", "I", "J"])},
    {"id": 10, "slot_a": ("pos", "D", 1), "slot_b": ("tercero", ["B", "E", "F", "I", "J"])},
    {"id": 11, "slot_a": ("pos", "H", 1), "slot_b": ("pos", "J", 2)},
    {"id": 12, "slot_a": ("pos", "K", 2), "slot_b": ("pos", "L", 2)},
    {"id": 13, "slot_a": ("pos", "B", 1), "slot_b": ("tercero", ["E", "F", "G", "I", "J"])},
    {"id": 14, "slot_a": ("pos", "D", 2), "slot_b": ("pos", "G", 2)},
    {"id": 15, "slot_a": ("pos", "J", 1), "slot_b": ("pos", "H", 2)},
    {"id": 16, "slot_a": ("pos", "K", 1), "slot_b": ("tercero", ["D", "E", "I", "J", "L"])},
]

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


CALENDARIO_GRUPOS: list[tuple[str, str, str, str]] = [

    ("2026-06-11", "A", "Mexico", "South Africa"),
    ("2026-06-11", "A", "South Korea", "Czech Republic"),
    ("2026-06-18", "A", "Czech Republic", "South Africa"),
    ("2026-06-18", "A", "Mexico", "South Korea"),
    ("2026-06-24", "A", "Czech Republic", "Mexico"),
    ("2026-06-24", "A", "South Africa", "South Korea"),

    ("2026-06-12", "B", "Canada", "Bosnia and Herzegovina"),
    ("2026-06-13", "B", "Qatar", "Switzerland"),
    ("2026-06-18", "B", "Switzerland", "Bosnia and Herzegovina"),
    ("2026-06-18", "B", "Canada", "Qatar"),
    ("2026-06-24", "B", "Switzerland", "Canada"),
    ("2026-06-24", "B", "Bosnia and Herzegovina", "Qatar"),

    ("2026-06-13", "C", "Brazil", "Morocco"),
    ("2026-06-13", "C", "Haiti", "Scotland"),
    ("2026-06-19", "C", "Scotland", "Morocco"),
    ("2026-06-19", "C", "Brazil", "Haiti"),
    ("2026-06-24", "C", "Scotland", "Brazil"),
    ("2026-06-24", "C", "Morocco", "Haiti"),

    ("2026-06-12", "D", "United States", "Paraguay"),
    ("2026-06-13", "D", "Australia", "Turkey"),
    ("2026-06-19", "D", "United States", "Australia"),
    ("2026-06-19", "D", "Turkey", "Paraguay"),
    ("2026-06-25", "D", "Turkey", "United States"),
    ("2026-06-25", "D", "Paraguay", "Australia"),

    ("2026-06-14", "E", "Germany", "Curaçao"),
    ("2026-06-14", "E", "Ivory Coast", "Ecuador"),
    ("2026-06-20", "E", "Germany", "Ivory Coast"),
    ("2026-06-20", "E", "Ecuador", "Curaçao"),
    ("2026-06-25", "E", "Ecuador", "Germany"),
    ("2026-06-25", "E", "Curaçao", "Ivory Coast"),

    ("2026-06-14", "F", "Netherlands", "Japan"),
    ("2026-06-14", "F", "Sweden", "Tunisia"),
    ("2026-06-20", "F", "Netherlands", "Sweden"),
    ("2026-06-20", "F", "Tunisia", "Japan"),
    ("2026-06-25", "F", "Japan", "Sweden"),
    ("2026-06-25", "F", "Tunisia", "Netherlands"),

    ("2026-06-15", "G", "Belgium", "Egypt"),
    ("2026-06-15", "G", "Iran", "New Zealand"),
    ("2026-06-21", "G", "Belgium", "Iran"),
    ("2026-06-21", "G", "New Zealand", "Egypt"),
    ("2026-06-26", "G", "New Zealand", "Belgium"),
    ("2026-06-26", "G", "Egypt", "Iran"),

    ("2026-06-15", "H", "Spain", "Cape Verde"),
    ("2026-06-15", "H", "Saudi Arabia", "Uruguay"),
    ("2026-06-21", "H", "Spain", "Saudi Arabia"),
    ("2026-06-21", "H", "Uruguay", "Cape Verde"),
    ("2026-06-26", "H", "Cape Verde", "Saudi Arabia"),
    ("2026-06-26", "H", "Uruguay", "Spain"),

    ("2026-06-16", "I", "France", "Senegal"),
    ("2026-06-16", "I", "Iraq", "Norway"),
    ("2026-06-22", "I", "France", "Iraq"),
    ("2026-06-22", "I", "Norway", "Senegal"),
    ("2026-06-26", "I", "Norway", "France"),
    ("2026-06-26", "I", "Senegal", "Iraq"),

    ("2026-06-16", "J", "Argentina", "Algeria"),
    ("2026-06-16", "J", "Austria", "Jordan"),
    ("2026-06-22", "J", "Argentina", "Austria"),
    ("2026-06-22", "J", "Jordan", "Algeria"),
    ("2026-06-27", "J", "Algeria", "Austria"),
    ("2026-06-27", "J", "Jordan", "Argentina"),

    ("2026-06-17", "K", "Portugal", "DR Congo"),
    ("2026-06-17", "K", "Uzbekistan", "Colombia"),
    ("2026-06-23", "K", "Portugal", "Uzbekistan"),
    ("2026-06-23", "K", "Colombia", "DR Congo"),
    ("2026-06-27", "K", "Colombia", "Portugal"),
    ("2026-06-27", "K", "DR Congo", "Uzbekistan"),

    ("2026-06-17", "L", "England", "Croatia"),
    ("2026-06-17", "L", "Ghana", "Panama"),
    ("2026-06-23", "L", "England", "Ghana"),
    ("2026-06-23", "L", "Panama", "Croatia"),
    ("2026-06-27", "L", "Panama", "England"),
    ("2026-06-27", "L", "Croatia", "Ghana"),
]


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


# Árbol oficial de eliminación del Mundial 2026 (no el emparejamiento secuencial).
# Fuente: cuadro oficial FIFA (números de partido M73-M104), verificado contra
# https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
#
# OJO: el emparejamiento "secuencial" ingenuo (ganador id1 vs id2, id3 vs id4...)
# NO reproduce este árbol: solo 3 de los 8 cruces de octavos de final coinciden.
# Por eso la adyacencia se fija explícitamente aquí.
#
# Mapeo partido FIFA -> id de BRACKET_R32 (por sus slots):
#   M73=1  M74=3  M75=4  M76=2  M77=6  M78=5  M79=7  M80=8
#   M81=10 M82=9  M83=12 M84=11 M85=13 M86=15 M87=16 M88=14
#
# Octavos de final (M89-M96): cada cruce empareja a los ganadores de dos llaves
# de R32, identificadas por su id en BRACKET_R32.
BRACKET_R16: list[tuple[int, int]] = [
    (3, 6),    # M89: ganador M74 vs ganador M77
    (1, 4),    # M90: ganador M73 vs ganador M75
    (2, 5),    # M91: ganador M76 vs ganador M78
    (7, 8),    # M92: ganador M79 vs ganador M80
    (12, 11),  # M93: ganador M83 vs ganador M84
    (10, 9),   # M94: ganador M81 vs ganador M82
    (15, 14),  # M95: ganador M86 vs ganador M88
    (13, 16),  # M96: ganador M85 vs ganador M87
]

# De octavos de final en adelante: cada llave empareja a los ganadores de dos
# llaves de la ronda anterior, por su índice (0-based) dentro de esa ronda.
# Cuartos (M97-M100) desde R16; semis (M101-M102) desde cuartos; final (M104).
BRACKET_AVANCE: list[list[tuple[int, int]]] = [
    [(0, 1), (4, 5), (2, 3), (6, 7)],  # Cuartos: M97=R16[0]vR16[1], M98=R16[4]vR16[5], M99=R16[2]vR16[3], M100=R16[6]vR16[7]
    [(0, 1), (2, 3)],                  # Semis:   M101=QF[0]vQF[1], M102=QF[2]vQF[3]
    [(0, 1)],                          # Final:   M104=SF[0]vSF[1]
]

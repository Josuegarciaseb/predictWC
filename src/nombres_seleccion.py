"""Estandarización de nombres de selección entre fuentes.

El pipeline cruza datos de varias fuentes que escriben los nombres distinto:
  - martj42 / fixtures (la forma CANÓNICA del proyecto): "South Korea", "Ivory
    Coast", "DR Congo", "United States", "Iran", "Czech Republic"...
  - StatsBomb: "South Korea", "Congo DR", "Côte d'Ivoire", "Cape Verde Islands"...
  - FBref (soccerdata): "Korea Republic", "IR Iran", "Czechia", "China PR",
    "Côte d'Ivoire", "USA"...

`canonical(nombre)` mapea cualquier variante a la forma canónica. Las tablas se
siembran con los quirks conocidos; lo que NO se reconozca se devuelve igual y el
ingestor lo registra (con `reportar_desconocidos`) para refinar el mapeo en vez
de cruzar mal en silencio. NO se hardcodea a ciegas: se verifica con cada fuente.
"""
from __future__ import annotations

import unicodedata

# alias (cualquier fuente) -> forma canónica del proyecto
ALIAS: dict[str, str] = {
    # --- FBref (selecciones) ---
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "China PR": "China",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Republic of Ireland": "Ireland",
    "North Macedonia": "North Macedonia",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    # --- StatsBomb ---
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    # --- variantes comunes ---
    "United States of America": "United States",
    "South Korea Republic": "South Korea",
    "Czech Republic": "Czech Republic",
}


def _strip_acentos(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def canonical(nombre: str) -> str:
    if nombre is None:
        return nombre
    n = str(nombre).strip()
    if n in ALIAS:
        return ALIAS[n]
    # intento sin acentos (p.ej. "Cote d'Ivoire" -> "Côte d'Ivoire" no casa, pero
    # "Curacao" -> "Curaçao" sí conviene normalizar a una sola forma)
    sin = _strip_acentos(n)
    for k, v in ALIAS.items():
        if _strip_acentos(k) == sin:
            return v
    return n


def normalizar_columna(serie, registro_desconocidos: set | None = None,
                       conocidos: set | None = None):
    """Aplica canonical a una serie de pandas; si se pasa `conocidos` (el set de
    nombres canónicos válidos, p.ej. las 48 del Mundial), apunta los que no casen
    en `registro_desconocidos` para que el ingestor los reporte."""
    out = serie.map(canonical)
    if conocidos is not None and registro_desconocidos is not None:
        for v in out.unique():
            if v not in conocidos:
                registro_desconocidos.add(v)
    return out


# Las 48 selecciones del Mundial 2026 (forma canónica). Confirmado con los fixtures
# del propio proyecto (data/processed/partidos_a_predecir.csv): incluye Argelia, NO Gales.
SELECCIONES_WC2026 = {
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curaçao", "Czech Republic", "DR Congo", "Ecuador", "Egypt",
    "England", "France", "Germany", "Ghana", "Haiti", "Iran", "Iraq",
    "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
}

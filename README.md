# Predictor Mundial FIFA 2026

Sistema para estimar el marcador/resultado más probable de cada partido del
Mundial 2026, combinando Elo dinámico, modelos de goles (Poisson / Dixon-Coles),
modelos de machine learning (XGBoost / CatBoost), un ensemble por stacking y
simulación Monte Carlo del torneo completo.

Dataset fuente: [martj42/international_results](https://github.com/martj42/international_results)
(resultados de selecciones desde 1872, incluye penales y goleadores).

## Hallazgo clave de la exploración inicial

`results.csv` ya incluye los **40 partidos de la primera ronda de grupos del
Mundial 2026** (2026-06-20 a 2026-06-27) con `home_score`/`away_score` en
`NaN`. **No son datos sucios: son exactamente el target a predecir.** Por eso
el pipeline nunca los descarta — los separa en `por_predecir` y les calcula
el Elo "antes del partido" sin tocar resultados reales.

## Qué se corrigió del pseudocódigo original

| Problema en `pseudocodigo.py` | Corrección |
|---|---|
| `fila[fila['home_score'].notna() and fila['away_score'].notna()]` lanza `AttributeError` (un escalar no tiene `.notna()`) | Verificación explícita con `pd.notna(...)` antes de calcular `S_local` |
| `futuros = df[df['date'] > hoy]` trataba el target del proyecto como "basura a excluir" | Split explícito histórico/por-predecir; el Elo se calcula sobre la serie completa para que los partidos del Mundial arranquen con el Elo correcto |
| Elo sin ventaja de local, sin peso por goleada y con K fijo | `elo.py`: +100 al local (si no es sede neutral), multiplicador `G` por diferencia de goles, `K` variable según importancia del torneo |
| Nombres de selecciones que cambiaron (`former_names.csv`) detectados pero sin usar | `data_loader.estandarizar_nombres_equipos`: fusiona identidades 1:1 respetando fechas de vigencia. Casos sin sucesor único (Czechoslovakia, Yugoslavia, German DR) se dejan separados a propósito |
| Prob. de campeón del favorito inflada (~24%, casi el doble del mercado) por sobreconfianza del modelo | `poisson_dixon_coles.py`: `strength_shrinkage=0.85` encoge ataque/defensa hacia la media del campo. Actúa como proxy de la incertidumbre sobre la fuerza real de cada equipo, que pesa más al componerse sobre 6 rondas que en un solo partido. Baja Argentina de 24% a ~19% con coste mínimo de calibración (log-loss de partido +0.008; el stacking final no se mueve) |

## Estructura

```
wc2026_predictor/
├── data/
│   ├── raw/                          # CSV originales del repo
│   └── processed/
│       ├── historico_con_elo.csv     # output Fase 1 -> insumo de Fases 2-4
│       ├── partidos_a_predecir.csv   # los 40 partidos del Mundial 2026 con Elo
│       └── modelos_fase3/            # XGBoost/CatBoost entrenados (insumo Fase 4)
├── src/
│   ├── data_loader.py                # carga + limpieza + estandarización de nombres
│   ├── elo.py                        # Elo dinámico (corregido y mejorado)
│   ├── features.py                   # FASE 3 -- forma reciente + head-to-head (sin fuga)
│   ├── torneo.py                     # FASE 5 -- grupos oficiales + calendario + llave
│   ├── montecarlo.py                 # FASE 5 -- motor de simulación del torneo
│   └── models/
│       ├── poisson_dixon_coles.py    # FASE 2 -- ataque/defensa + Dixon-Coles
│       ├── ml_models.py              # FASE 3 -- XGBoost + CatBoost
│       └── stacking.py               # FASE 4 -- meta-modelo (regresión logística)
├── scripts/
│   ├── 01_pipeline_elo.py            # FASE 1
│   ├── 02_pipeline_poisson_dixon_coles.py  # FASE 2 -- con backtest temporal
│   ├── 03_pipeline_ml.py             # FASE 3 -- con backtest comparativo
│   ├── 04_pipeline_stacking.py       # FASE 4 -- walk-forward + ensemble final
│   └── 05_pipeline_montecarlo.py     # FASE 5 -- simulación del torneo completo
├── outputs/
│   ├── ranking_elo_actual.csv
│   ├── predicciones_fase1_mundial2026.csv
│   ├── predicciones_fase2_poisson_dixon_coles.csv  # marcador más probable
│   ├── predicciones_fase3_xgboost_catboost.csv     # probabilidades 1X2 ML
│   ├── predicciones_fase4_stacking_ensemble.csv    # tabla combinada (matchday 1)
│   └── predicciones_fase5_montecarlo.csv           # probabilidad de campeón, todo el torneo
├── scripts/   (mercados nuevos)
│   ├── A01_ingesta_statsbomb.py     # FASE A -- descarga eventos StatsBomb -> CSV
│   ├── A02_reporte_cobertura.py     # FASE A -- cobertura real por año/torneo/mercado
│   ├── A03_ingesta_jugadores.py     # FASE D -- jugador-partido: minutos + tiros a puerta
│   ├── B01_pipeline_corners.py      # FASE B -- córners: walk-forward + predicción WC2026
│   ├── B02_calibracion_corners.py   # FASE B -- diagnóstico + calibración de líneas Over
│   ├── B03_correlacion_corners.py   # FASE B -- cópula del 1x2 (corrige el empate inflado)
│   ├── C01_pipeline_tarjetas.py     # FASE C -- tarjetas: walk-forward + predicción WC2026
│   ├── D01_pipeline_tiros_jugador.py # FASE D -- tiros a puerta jugador: walk-forward + ranking
│   ├── predecir_corners.py          # CLI a demanda de córners (análogo a predecir_partido)
│   ├── predecir_tarjetas.py         # CLI a demanda de tarjetas
│   └── predecir_tiros_jugador.py    # CLI a demanda de tiros a puerta por jugador
├── src/   (mercados nuevos)
│   ├── statsbomb_loader.py          # ingesta+parseo de StatsBomb (córners/tiros/tarjetas/minutos)
│   ├── models/corners_dixon_coles.py # FASE B -- Dixon-Coles de córners (Poisson vs Bin.Neg.)
│   ├── models/tarjetas_model.py     # FASE C -- conteo de tarjetas + covariable de fase
│   └── models/tiros_jugador.py      # FASE D -- jerárquico Gamma-Poisson de tiros a puerta
└── requirements.txt
```

## Mercados adicionales (córners, tarjetas, tiros) — fuente StatsBomb

El dataset martj42 **solo trae el marcador**: no hay córners, tarjetas ni tiros.
Para esos mercados se abre una fuente nueva, **StatsBomb Open Data**, de la que se
derivan a nivel partido (y jugador) a partir de los eventos crudos. FBref vía
`soccerdata` se evaluó pero queda descartado como columna vertebral: responde 403
(Cloudflare) y no expone córners a nivel selección de forma fiable.

**Cobertura real (honesta).** Solo torneos de selecciones: WC 2018/2022, Euro
2020/2024, Copa América 2024, AFCON 2023 → **314 partidos modernos (2018+)**, 76
selecciones, mediana 7 partidos/selección. No hay profundidad histórica de córners
como sí la hay de goles (49k partidos). De las 48 selecciones del Mundial 2026, 40
tienen historial (3 por mapeo de nombre) y **8 caen al promedio del campo**
(Bosnia, Curaçao, Haití, Irak, Jordania, Nueva Zelanda, Noruega, Uzbekistán).
Reporte completo en `outputs/reporte_cobertura_mercados.md`.

**Modelo de córners (Fase B).** Clona el Dixon-Coles de goles cambiando el target a
córners (ataque/defensa de córner + ventaja de local), **quita la corrección τ** de
marcadores bajos (no aplica) y compara Poisson vs Binomial Negativa. Salidas:
Over/Under desde la cola de la distribución del total (convolución de marginales) y
1x2 vía **Skellam(λ,μ)**. Validación walk-forward (250 partidos, sin fuga):

| Métrica | Poisson | Bin.Neg. | Baseline |
|---|---|---|---|
| Total córners (log-loss) | 2.723 | **2.713** | 2.729 |
| O/U 9.5 (log-loss) | 0.678 | 0.678 | 0.676 |
| 1x2 córners (log-loss) | 0.876 | **0.873** | 0.916 |
| 1x2 córners (accuracy) | 0.588 | **0.592** | — |

> Lecturas honestas: (1) la sobre-dispersión condicional es **leve** (NB gana por
> poco; r≈15); (2) el **valor del modelo está en el 1x2** (bate al baseline), no en
> el O/U del total, que ~= a la tasa base (las fuerzas de equipo se cancelan en el
> total); (3) la ventaja de local no es identificable (casi todo es sede neutral) y
> no aplica al Mundial (fixtures neutrales).
>
> **Calibración de las líneas Over** (`B02_calibracion_corners.py`): el total se
> subestimaba ~5% out-of-sample por la deriva al alza de córners (tiempo añadido
> largo post-2022). Un multiplicador de localización **c=1.04** (elegido por
> walk-forward) centra la línea 9.5 (gap +0.036 → −0.006): la media predicha pasa de
> 8.68 a 9.02 (real 9.08). No crea edge en el O/U, pero hace **fiables** las
> probabilidades Over para decidir.
>
> **Empate de córners — cópula** (`B03_correlacion_corners.py`): la independencia
> sobreestima el empate (~12% vs ~8% real) porque ignora la dependencia negativa
> (el dominante saca más y concede menos; correlación residual ≈ −0.15). El 1x2 pasa
> a usar una **cópula gaussiana** (rho=−0.15): baja el empate a ~11% y mejora el 1x2
> (log-loss 0.876 → 0.873). Clave: la cópula se usa **solo para el 1x2** — aplicarla
> al total lo empeora (la correlación negativa estrecha la suma, pero el total real
> está sobre-dispersado), así que el O/U sigue en convolución independiente. Cada
> salida usa la dependencia que mejor le ajusta.

**Modelo de tarjetas (Fase C).** Mismo motor de conteo, con la "fuerza" de equipo
reinterpretada (propensión disciplinaria propia + cuánto induce tarjetas al rival)
y una covariable de **fase del torneo** (knockout): en los datos la eliminatoria
trae **x1.18** tarjetas. Validación walk-forward (250 partidos, línea O/U 3.5):

| Variante (calibrada c=1.10) | Total (log-loss) | O/U 3.5 (log-loss) | O/U 3.5 (acc) |
|---|---|---|---|
| Poisson + KO | 2.242 | 0.683 | 0.556 |
| **Bin.Neg. + KO** | **2.240** | **0.684** | 0.552 |
| Bin.Neg. sin KO | 2.243 | 0.688 | 0.560 |
| Baseline | 2.263 | 0.693 | — |

> Lecturas honestas: (1) las tarjetas están **más sobre-dispersas** que los córners
> (Var/Media=1.65), así que la NB sí ayuda y es el modelo desplegado; (2) la
> covariable knockout aporta señal real; (3) el **árbitro es el factor dominante y no
> es observable**, así que el techo es bajo. Aun así, tras **calibrar la línea Over**
> (la subestimación era ~11% por la deriva al alza de tarjetas en el fútbol moderno:
> Euro/Copa 2024 ≫ torneos antiguos), el O/U de tarjetas pasa a **batir al baseline
> modestamente** (0.684 vs 0.693; antes 0.698). Sigue siendo ruidoso, pero el sesgo
> de localización ocultaba el poco valor real que había.

**Modelo de tiros a puerta por jugador (Fase D).** Primero se derivan los minutos
jugados por jugador-partido (de `Starting XI` + `Substitution`), incluyendo a quien
jugó y no disparó (clave para una tasa no sesgada): **10 029 jugador-partidos, 2 579
jugadores**. El modelo es un **jerárquico empirical-Bayes Gamma-Poisson**: tasa de
tiros a puerta por 90' con efecto aleatorio por jugador (θ_p) y por rival (φ_o), ambos
encogidos hacia la media según el tamaño muestral, con la predictiva en Binomial
Negativa. Walk-forward (9 343 jugador-partidos, con minutos reales):

| Variante | Conteo (log-loss) | O/U 0.5 (log-loss) | O/U 0.5 (acc) |
|---|---|---|---|
| **Jugador + Rival** | **0.626** | **0.488** | 0.792 |
| Solo jugador | 0.627 | 0.488 | 0.793 |
| Baseline global | 0.677 | 0.524 | 0.785 |

> Lecturas honestas: (1) el **efecto jugador aporta señal fuerte** (k≈1: hay enormes
> diferencias reales entre jugadores) y bate al baseline; (2) el **efecto rival es
> despreciable** (k≈25, muy encogido): tu propia habilidad domina sobre a quién
> enfrentas; (3) **límite estructural**: la predicción es condicional a los minutos.
> Sin once probable del Mundial 2026, la incertidumbre de alineación domina, así que
> no hay tabla por fixture — se entrega un ranking de tiradores y un CLI por minutos.

```bash
python scripts/A01_ingesta_statsbomb.py     # descarga+cachea (~333 partidos)
python scripts/A02_reporte_cobertura.py      # reporte de cobertura
python scripts/A03_ingesta_jugadores.py      # jugador-partido: minutos + tiros a puerta
python scripts/B01_pipeline_corners.py       # córners: validación + predicciones WC2026
python scripts/B02_calibracion_corners.py    # córners: diagnóstico/calibración de líneas Over
python scripts/B03_correlacion_corners.py    # córners: cópula del 1x2 (empate)
python scripts/C01_pipeline_tarjetas.py      # tarjetas: validación + predicciones WC2026
python scripts/D01_pipeline_tiros_jugador.py # tiros a puerta jugador: validación + ranking
python scripts/predecir_corners.py "Spain" "Germany"            # córners a demanda
python scripts/predecir_tarjetas.py "Mexico" "Czech Republic"   # tarjetas a demanda
python scripts/predecir_tiros_jugador.py "Messi" "France"       # tiros a puerta a demanda
```

> **Estos modelos son ESTÁTICOS — no se re-entrenan a diario** (a diferencia de los de
> goles). Se entrenan con StatsBomb (eventos de torneos ya jugados), una fuente que no
> se actualiza cada día: martj42 trae el marcador diario, no córners/tarjetas/tiros, así
> que no hay nada nuevo de lo que aprender. Por eso el workflow `update-and-retrain.yml`
> NO los incluye (sería fingir un aprendizaje que no ocurre). Se regeneran a mano
> (`A01,A03,B01,C01,D01`) solo cuando StatsBomb publique un torneo nuevo. La fuente "en
> vivo" que lo habría permitido (Kaggle WC2026) resultó fabricada y se descartó.

**Integración en la web** (`web/app.js`): los CSV de córners y tarjetas se cargan en
la web y se muestran como mercados con el mismo marco de **edge/BET vs casa** que los
de goles (línea base "pegajosa" anclada a la tasa real de StatsBomb; el BET surge
donde el modelo se separa de la base). Reglas de honestidad aplicadas:
- Un mercado de córners/tarjetas solo aparece si **ambas** selecciones tienen
  historial en StatsBomb; si una no (8 selecciones del WC2026), no se muestra — no se
  rellena con el promedio del campo.
- Se marcan `unverifiable`: no hay feed de resultados reales de córners/tarjetas, así
  que se sugieren pero **no se autoliquidan** ni cuentan en el récord W-L (solo goles).
- Tiros por jugador NO se muestra por partido (depende del once, desconocido). Vive
  como sección aparte **"Amenaza de tiro a puerta"**: ranking histórico de tiros a
  puerta/90', con aviso de que es condicional a jugar, no un pick por fixture.

El efecto natural: en córners el **1x2 sale BET** (el modelo diverge real de la base)
y el **O/U sale PASA** (≈ base) — la mecánica revela dónde hay valor y dónde no.

## Cómo correr el proyecto

```bash
pip install -r requirements.txt 
o
py -m pip install -r requirements.txt

Igual aqui es con python o py
python scripts/01_pipeline_elo.py
python scripts/02_pipeline_poisson_dixon_coles.py
python scripts/03_pipeline_ml.py
python scripts/04_pipeline_stacking.py
python scripts/05_pipeline_montecarlo.py
python scripts/06_heatmap_marcador.py --todos
python scripts/07_export_matrices.py
python scripts/08_archivar_predicciones.py
```

`05_pipeline_montecarlo.py` simula el Mundial completo (no solo los 40
partidos del dataset) 5,000 veces: fase de grupos -> Octavos de 32 (formato
nuevo de 2026, no Octavos de 16) -> Octavos de Final -> Cuartos -> Semis ->
Final. La estructura oficial (12 grupos, calendario de 72 partidos, llave de
eliminación) se obtuvo por búsqueda web porque el dataset local solo trae 40
de esos 72 partidos -- los otros 32 ya tenían resultado real al momento de
armar el dataset y están en `historico_con_elo.csv`, no en
`partidos_a_predecir.csv`. Esos 32 partidos reales se usan tal cual en cada
simulación; los 40 restantes (y cualquier cruce de eliminación directa,
que puede ser entre cualquier par de las 48 selecciones según cómo avance
cada simulación) se generan muestreando de la matriz de marcador de
Dixon-Coles para ese par de equipos.

**Resultado** (top 10 favoritos a ganar el título, sobre 5,000 simulaciones):

| Selección | Llega a octavos32 | Cuartos | Semis | Final | **Campeón** |
|---|---|---|---|---|---|
| Argentina | 99.6% | 52.3% | 41.5% | 29.4% | **19.4%** |
| Brasil | 100% | 51.8% | 34.9% | 21.4% | **11.4%** |
| España | 92.4% | 41.0% | 30.1% | 18.2% | **11.1%** |
| Inglaterra | 99.4% | 48.7% | 29.1% | 16.8% | **9.2%** |
| Francia | 98.9% | 46.0% | 26.4% | 14.0% | **6.8%** |
| Colombia | 98.8% | 33.1% | 22.3% | 11.4% | **5.9%** |
| Portugal | 83.4% | 29.4% | 20.0% | 10.8% | **5.7%** |

> Las fuerzas de Dixon-Coles se calibran con `strength_shrinkage=0.85` (ver tabla de
> correcciones arriba): sin ese encogimiento el favorito daba ~24% de campeón, casi el
> doble del mercado. El reparto resultante queda alineado con las casas de apuestas.

Tabla completa de los 48 equipos en `outputs/predicciones_fase5_montecarlo.csv`.
Las probabilidades están internamente consistentes por construcción: suman
exactamente 32 equipos en "llega a octavos de 32", 2 en "llega a la final" y
1.0 en "campeón", en cada corrida.

**Limitaciones reconocidas de esta fase** (documentadas también en el código,
`src/torneo.py` y `src/montecarlo.py`):
- La calibración por `strength_shrinkage` es un ajuste deliberado: a nivel de
  partido individual el modelo está bien calibrado (el log-loss del backtest es
  mínimo sin shrinkage). El encogimiento existe para corregir la sobreconfianza
  **a nivel de torneo**, que surge al componer la ventaja del favorito sobre 6
  rondas con fuerzas tratadas como certezas. Es un proxy de incertidumbre, no una
  mejora de la predicción partido a partido.
- El Elo/ataque-defensa de cada selección se usa **estático** durante toda la
  simulación (el nivel de cierre de la fase de grupos real), sin actualizarlo
  partido a partido dentro de cada simulación -- es la simplificación
  estándar en simuladores públicos de este tipo.
- FIFA define una tabla oficial exacta para decidir qué grupo específico
  aporta cada "mejor tercero" a la llave de Octavos de 32 (depende de cuáles
  8 de los 12 grupos clasifican). Esa tabla completa (cientos de
  combinaciones) no se reprodujo aquí -- se usa una asignación determinista
  simplificada que respeta los grupos candidatos de cada cruce, pero no es
  la tabla oficial exacta.
- No está públicamente detallado qué ganador de Octavos de 32 enfrenta a cuál
  en Octavos de Final -- se asume el emparejamiento secuencial estándar
  (ganador M1 vs ganador M2, etc.).
- Los empates en eliminación directa (sin datos de tiempo extra/penales) se
  resuelven re-normalizando P(local)/P(visita) de la propia matriz de
  Dixon-Coles, sin modelar el detalle de penales.

---

## Resumen del proyecto (las 5 fases)

| Fase | Qué hace | Métrica de validación |
|---|---|---|
| 1. Elo dinámico | Rating base con ventaja de local y peso por goleada | Ranking de sanity-check (Argentina, España, Francia arriba) |
| 2. Poisson / Dixon-Coles | Marcador más probable por partido | 60.5% accuracy / 0.848 log-loss (backtest 18 meses, con `strength_shrinkage=0.85`) |
| 3. XGBoost / CatBoost | Probabilidades 1X2 vía gradient boosting | 60.6-60.7% accuracy (mismo backtest) |
| 4. Stacking ensemble | Combina los anteriores con un meta-modelo | **61.4% accuracy / 0.828 log-loss** -- mejor que cualquiera individual |
| 5. Monte Carlo | Simula el torneo completo 5,000 veces | Consistencia interna exacta (32/2/1.0) |

El proyecto está completo de punta a punta: desde un CSV crudo de Kaggle/GitHub
hasta la probabilidad de cada selección de salir campeona del Mundial 2026.

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
└── requirements.txt
```

## Cómo correr el proyecto

```bash
pip install -r requirements.txt 
o
py -m pip install -r requirements.txt

python scripts/01_pipeline_elo.py
python scripts/02_pipeline_poisson_dixon_coles.py
python scripts/03_pipeline_ml.py
python scripts/04_pipeline_stacking.py
python scripts/05_pipeline_montecarlo.py
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
| Argentina | 99.7% | 55.8% | 47.0% | 33.5% | **23.3%** |
| España | 94.0% | 43.2% | 34.2% | 20.7% | **13.1%** |
| Brasil | 100% | 55.1% | 37.6% | 22.9% | **12.3%** |
| Inglaterra | 99.7% | 51.3% | 31.7% | 18.3% | **9.6%** |
| Francia | 99.2% | 49.8% | 28.2% | 15.6% | **7.2%** |
| Colombia | 98.8% | 33.1% | 22.7% | 11.3% | **5.9%** |
| Portugal | 85.0% | 31.7% | 21.4% | 10.5% | **5.4%** |

Tabla completa de los 48 equipos en `outputs/predicciones_fase5_montecarlo.csv`.
Las probabilidades están internamente consistentes por construcción: suman
exactamente 32 equipos en "llega a octavos de 32", 2 en "llega a la final" y
1.0 en "campeón", en cada corrida.

**Limitaciones reconocidas de esta fase** (documentadas también en el código,
`src/torneo.py` y `src/montecarlo.py`):
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
| 2. Poisson / Dixon-Coles | Marcador más probable por partido | 60.5% accuracy / 0.840 log-loss (backtest 18 meses) |
| 3. XGBoost / CatBoost | Probabilidades 1X2 vía gradient boosting | 60.6-60.7% accuracy (mismo backtest) |
| 4. Stacking ensemble | Combina los anteriores con un meta-modelo | **61.4% accuracy / 0.828 log-loss** -- mejor que cualquiera individual |
| 5. Monte Carlo | Simula el torneo completo 5,000 veces | Consistencia interna exacta (32/2/1.0) |

El proyecto está completo de punta a punta: desde un CSV crudo de Kaggle/GitHub
hasta la probabilidad de cada selección de salir campeona del Mundial 2026.

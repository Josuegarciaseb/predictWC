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
│       └── historico_con_elo.csv     # output Fase 1 -> insumo de Fases 2-4
├── src/
│   ├── data_loader.py                # carga + limpieza + estandarización de nombres
│   ├── elo.py                        # Elo dinámico (corregido y mejorado)
│   └── models/                       # Fases 2-4 (ver roadmap)
├── scripts/
│   └── 01_pipeline_elo.py            # FASE 1 -- ejecutable hoy
├── outputs/
│   ├── ranking_elo_actual.csv
│   └── predicciones_fase1_mundial2026.csv
└── requirements.txt
```

## Cómo correr la Fase 1

```bash
pip install -r requirements.txt
python scripts/01_pipeline_elo.py
```

Esto imprime el ranking Elo actual y una tabla baseline (solo-Elo) de
probabilidad local/visitante para los 40 partidos del Mundial 2026, y guarda
`data/processed/historico_con_elo.csv`, que es el dataset que alimentará las
fases siguientes (ya traer las columnas `elo_local_antes` / `elo_visita_antes`
como feature por partido, calculadas correctamente sin leakage: el Elo de
cada fila es el que tenían los equipos *antes* de jugar ese partido).

---

## Roadmap (fases siguientes, no implementadas todavía)

### Fase 2 -- Poisson y Dixon-Coles
- Estimar `ataque_i`, `defensa_i` por selección (modelo Poisson bivariado al
  estilo Maher / Dixon-Coles 1997) usando goles anotados/recibidos en
  `historico_con_elo.csv`, con decaimiento exponencial por antigüedad (partidos
  de 2024 pesan más que partidos de 1990).
- Dixon-Coles agrega el ajuste `rho` para corregir la subestimación de
  empates 0-0/1-1 que tiene el Poisson independiente puro.
- Salida por partido: distribución completa de marcadores (matriz goles
  local × goles visitante), de la cual se deriva el marcador más probable,
  P(gana local/empate/gana visitante), y over/under.

### Fase 3 -- XGBoost y CatBoost (opcional)
- Feature engineering: diferencia de Elo, forma reciente (últimos N
  partidos, goles a favor/contra), historial head-to-head, si juega de local/
  neutral, importancia del torneo, descanso entre partidos, etc.
- Target: resultado (W/D/L) y/o goles por equipo. CatBoost maneja bien
  variables categóricas (selección, torneo, ciudad) sin one-hot manual.

### Fase 4 -- Stacking ensemble
- Meta-modelo (regresión logística o XGBoost liviano) que combina las
  probabilidades de Elo, Poisson/Dixon-Coles, XGBoost y CatBoost.
- Validación temporal (no k-fold aleatorio): entrenar con partidos hasta el
  año N, validar con el año N+1, deslizando la ventana — para no filtrar
  información del futuro.

### Fase 5 -- Simulación Monte Carlo del torneo
- Con las probabilidades de marcador de cada partido (salida de Fase 4),
  simular el Mundial completo (fase de grupos -> octavos -> ... -> final)
  10,000+ veces, respetando el formato de 48 equipos / 12 grupos.
- Salida: % de probabilidad de cada selección de ganar el grupo, llegar a
  octavos/cuartos/semis/final/campeón.

---

**Próximo paso sugerido:** Fase 2 (Poisson/Dixon-Coles), porque es la que
produce el "marcador más probable" que pediste como entregable central, y
porque XGBoost/CatBoost en la Fase 3 pueden usar las probabilidades de
Poisson como feature adicional.

"""
scripts/04_pipeline_stacking.py
==================================
Fase 4: Stacking ensemble.

1. Genera meta-features (Dixon-Coles + XGBoost + CatBoost, reentrenados
   walk-forward) para varios años de partidos pasados.
2. Entrena el meta-modelo (regresión logística) con esas meta-features.
3. Evalúa el ensemble en el MISMO holdout de 18 meses usado en las Fases 2-3,
   para comparar de forma justa.
4. Reentrena los modelos base con todo el histórico y genera la tabla final
   combinada para los 40 partidos del Mundial 2026: marcador más probable
   (Dixon-Coles) + probabilidades 1X2 ya calibradas por el ensemble.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from features import construir_features
from models.ml_models import etiquetar_resultado, ModelosML
from models.poisson_dixon_coles import DixonColesModel
from models.stacking import StackingEnsemble, generar_meta_features, generar_folds_walk_forward

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

COLS_COMUNES = ["date", "home_team", "away_team", "home_score", "away_score", "tournament",
                 "neutral", "decided_by_shootout", "elo_local_antes", "elo_visita_antes"]


def evaluar(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    acc = float((probs.argmax(axis=1) == y_true).mean())
    ll = float(log_loss(y_true, probs, labels=[0, 1, 2]))
    return acc, ll


def main():
    hist = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])
    pred = pd.read_csv(PROCESSED_DIR / "partidos_a_predecir.csv", parse_dates=["date"])
    todo = pd.concat([hist[COLS_COMUNES], pred[COLS_COMUNES]], ignore_index=True)
    todo = todo.sort_values("date").reset_index(drop=True)
    todo_feat = construir_features(todo)
    categorias_torneo = sorted(todo_feat["tournament"].astype(str).unique())

    historico_feat = todo_feat[todo_feat["home_score"].notna()].copy()
    predict_feat = todo_feat[todo_feat["home_score"].isna()].copy()

    fecha_max = historico_feat["date"].max()
    fecha_corte_holdout = fecha_max - pd.DateOffset(months=18)

    # ================================================================
    # 1) Walk-forward: generar meta-features de entrenamiento (sin fuga)
    # ================================================================
    folds = generar_folds_walk_forward(historico_feat, anio_inicio=2019, fecha_corte_holdout=fecha_corte_holdout)
    print(f"Folds walk-forward para entrenar el meta-modelo: {len(folds)}")

    meta_train_partes = []
    y_train_partes = []
    t0 = time.time()
    for i, (ini, fin) in enumerate(folds, 1):
        train_base = historico_feat[historico_feat["date"] < ini]
        target = historico_feat[(historico_feat["date"] >= ini) & (historico_feat["date"] < fin)]
        if len(train_base) < 500 or len(target) == 0:
            continue
        meta = generar_meta_features(train_base, target, hist, categorias_torneo, fecha_corte_dc=str(ini.date()))
        meta_train_partes.append(meta)
        y_train_partes.append(etiquetar_resultado(target).values)
        print(f"  Fold {i}/{len(folds)}  [{ini.date()} -> {fin.date()})  "
              f"train_base={len(train_base)}  target={len(target)}  ({time.time()-t0:.0f}s acumulado)")

    meta_train = pd.concat(meta_train_partes, ignore_index=True)
    y_train = np.concatenate(y_train_partes)
    print(f"\nMeta-training set: {len(meta_train)} partidos")

    # ================================================================
    # 2) Entrenar el meta-modelo
    # ================================================================
    ensemble = StackingEnsemble()
    ensemble.fit(meta_train, y_train)
    print("\nPesos del meta-modelo (coeficientes de la regresión logística):")
    print(ensemble.pesos().round(3).to_string())

    # ================================================================
    # 3) Evaluación en el holdout (misma ventana que Fases 2-3)
    # ================================================================
    train_pre_holdout = historico_feat[historico_feat["date"] <= fecha_corte_holdout]
    holdout = historico_feat[historico_feat["date"] > fecha_corte_holdout]
    y_holdout = etiquetar_resultado(holdout).values

    meta_holdout = generar_meta_features(
        train_pre_holdout, holdout, hist, categorias_torneo, fecha_corte_dc=str(fecha_corte_holdout.date())
    )
    probs_stack_holdout = ensemble.predict_proba(meta_holdout)
    acc_stack, ll_stack = evaluar(y_holdout, probs_stack_holdout)

    acc_dc, ll_dc = evaluar(y_holdout, meta_holdout[["prob_local_dc", "prob_empate_dc", "prob_visita_dc"]].values)
    acc_xgb, ll_xgb = evaluar(y_holdout, meta_holdout[["prob_local_xgb", "prob_empate_xgb", "prob_visita_xgb"]].values)
    acc_cb, ll_cb = evaluar(y_holdout, meta_holdout[["prob_local_cb", "prob_empate_cb", "prob_visita_cb"]].values)

    print(f"\n=== Comparación final en holdout ({len(holdout)} partidos, {fecha_corte_holdout.date()} -> {fecha_max.date()}) ===")
    print("  Modelo                 | Accuracy | Log-loss")
    print("  -----------------------|----------|----------")
    print(f"  dixon_coles            |  {acc_dc:.3f}   |  {ll_dc:.3f}")
    print(f"  xgboost                |  {acc_xgb:.3f}   |  {ll_xgb:.3f}")
    print(f"  catboost               |  {acc_cb:.3f}   |  {ll_cb:.3f}")
    print(f"  STACKING (ensemble)    |  {acc_stack:.3f}   |  {ll_stack:.3f}")

    # ================================================================
    # 4) Producción: reentrenar con TODO el histórico y predecir Mundial 2026
    # ================================================================
    print("\nReentrenando modelos base con todo el histórico para el Mundial 2026...")
    dc_full = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    dc_full.fit(hist)

    ml_full = ModelosML(categorias_torneo)
    ml_full.fit(historico_feat)

    probs_dc_final = np.array([
        [r["prob_local"], r["prob_empate"], r["prob_visita"]]
        for r in (dc_full.predecir_partido(f.home_team, f.away_team, bool(f.neutral))
                  for f in predict_feat.itertuples(index=False))
    ])
    probs_ml_final = ml_full.predict_proba(predict_feat)

    meta_final = pd.DataFrame({
        "elo_diff": predict_feat["elo_diff"].values,
        "prob_local_dc": probs_dc_final[:, 0], "prob_empate_dc": probs_dc_final[:, 1], "prob_visita_dc": probs_dc_final[:, 2],
        "prob_local_xgb": probs_ml_final["xgboost"][:, 0], "prob_empate_xgb": probs_ml_final["xgboost"][:, 1],
        "prob_visita_xgb": probs_ml_final["xgboost"][:, 2],
        "prob_local_cb": probs_ml_final["catboost"][:, 0], "prob_empate_cb": probs_ml_final["catboost"][:, 1],
        "prob_visita_cb": probs_ml_final["catboost"][:, 2],
    }, index=predict_feat.index)

    probs_finales = ensemble.predict_proba(meta_final)

    tabla = predict_feat[["date", "home_team", "away_team", "elo_local_antes", "elo_visita_antes"]].copy()
    tabla = tabla.rename(columns={
        "date": "fecha", "home_team": "local", "away_team": "visitante",
        "elo_local_antes": "elo_local", "elo_visita_antes": "elo_visitante",
    })
    tabla["fecha"] = tabla["fecha"].dt.date
    tabla["elo_local"] = tabla["elo_local"].round(1)
    tabla["elo_visitante"] = tabla["elo_visitante"].round(1)

    marcadores, prob_exacto, top3s = [], [], []
    for f in predict_feat.itertuples(index=False):
        p = dc_full.predecir_partido(f.home_team, f.away_team, bool(f.neutral))
        marcadores.append(p["marcador_mas_probable"])
        prob_exacto.append(round(p["prob_marcador_exacto"], 3))
        top3s.append(" | ".join(f"{g1}-{g2} ({pr*100:.1f}%)" for g1, g2, pr in p["top3_marcadores"]))

    tabla["marcador_mas_probable"] = marcadores
    tabla["prob_marcador_exacto"] = prob_exacto
    tabla["top3_marcadores"] = top3s
    tabla["prob_local_ensemble"] = probs_finales[:, 0].round(3)
    tabla["prob_empate_ensemble"] = probs_finales[:, 1].round(3)
    tabla["prob_visita_ensemble"] = probs_finales[:, 2].round(3)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUTS_DIR / "predicciones_fase4_stacking_ensemble.csv", index=False)

    print("\n=== Fase 4 -- TABLA FINAL: marcador más probable + probabilidades del ensemble ===")
    cols_mostrar = ["fecha", "local", "visitante", "marcador_mas_probable",
                     "prob_local_ensemble", "prob_empate_ensemble", "prob_visita_ensemble"]
    print(tabla[cols_mostrar].to_string(index=False))

    print(f"\nGuardado: {OUTPUTS_DIR / 'predicciones_fase4_stacking_ensemble.csv'}")


if __name__ == "__main__":
    main()

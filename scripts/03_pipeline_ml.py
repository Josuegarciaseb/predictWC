import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from features import construir_features
from models.ml_models import ModelosML, etiquetar_resultado
from models.poisson_dixon_coles import DixonColesModel

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

COLS_COMUNES = ["date", "home_team", "away_team", "home_score", "away_score", "tournament",
                 "neutral", "decided_by_shootout", "elo_local_antes", "elo_visita_antes"]


def construir_todo_con_features():
    hist = pd.read_csv(PROCESSED_DIR / "historico_con_elo.csv", parse_dates=["date"])
    pred = pd.read_csv(PROCESSED_DIR / "partidos_a_predecir.csv", parse_dates=["date"])
    todo = pd.concat([hist[COLS_COMUNES], pred[COLS_COMUNES]], ignore_index=True)
    todo = todo.sort_values("date").reset_index(drop=True)
    todo_feat = construir_features(todo)

    categorias_torneo = sorted(todo_feat["tournament"].astype(str).unique())
    return todo_feat, categorias_torneo, hist


def evaluar(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    acc = float((probs.argmax(axis=1) == y_true).mean())
    ll = float(log_loss(y_true, probs, labels=[0, 1, 2]))
    return acc, ll


def backtest(historico_feat: pd.DataFrame, historico_crudo: pd.DataFrame,
             categorias_torneo: list[str], meses_holdout: int = 18):
    fecha_max = historico_feat["date"].max()
    fecha_corte = fecha_max - pd.DateOffset(months=meses_holdout)

    train = historico_feat[historico_feat["date"] <= fecha_corte]
    test = historico_feat[historico_feat["date"] > fecha_corte]
    y_test = etiquetar_resultado(test).values

    print(f"\nBacktest (misma ventana que Fase 2): train={len(train)}  test={len(test)}  "
          f"({fecha_corte.date()} -> {fecha_max.date()})")


    modelos = ModelosML(categorias_torneo)
    modelos.fit(train)
    probs = modelos.predict_proba(test)

    resultados = {}
    for nombre, p in probs.items():
        resultados[nombre] = evaluar(y_test, p)


    dc = DixonColesModel(cutoff_years=11, half_life_years=2.5)
    dc.fit(historico_crudo, fecha_corte=str(fecha_corte.date()))

    probs_dc_list = []
    for f in test.itertuples(index=False):
        pred_dc = dc.predecir_partido(f.home_team, f.away_team, bool(f.neutral))
        probs_dc_list.append([pred_dc["prob_local"], pred_dc["prob_empate"], pred_dc["prob_visita"]])
    probs_dc = np.array(probs_dc_list)
    resultados["dixon_coles"] = evaluar(y_test, probs_dc)


    distrib_real = np.bincount(y_test, minlength=3) / len(y_test)
    resultados["baseline_ingenuo"] = evaluar(y_test, np.tile(distrib_real, (len(y_test), 1)))

    print("\n  Modelo                 | Accuracy | Log-loss")
    print("  -----------------------|----------|----------")
    orden = ["baseline_ingenuo", "dixon_coles", "xgboost", "catboost", "blend"]
    for nombre in orden:
        acc, ll = resultados[nombre]
        print(f"  {nombre:<22} |  {acc:.3f}   |  {ll:.3f}")

    print("\n  Top features XGBoost (importancia):")
    print(modelos.importancia_features(10).round(3).to_string())

    return resultados


def main():
    todo_feat, categorias_torneo, hist_crudo = construir_todo_con_features()
    historico_feat = todo_feat[todo_feat["home_score"].notna()].copy()
    predict_feat = todo_feat[todo_feat["home_score"].isna()].copy()

    backtest(historico_feat, hist_crudo, categorias_torneo, meses_holdout=18)


    print("\nEntrenando modelos finales con todo el histórico disponible...")
    modelos_finales = ModelosML(categorias_torneo)
    modelos_finales.fit(historico_feat)
    probs_finales = modelos_finales.predict_proba(predict_feat)

    tabla = predict_feat[["date", "home_team", "away_team", "elo_local_antes", "elo_visita_antes"]].copy()
    tabla = tabla.rename(columns={
        "date": "fecha", "home_team": "local", "away_team": "visitante",
        "elo_local_antes": "elo_local", "elo_visita_antes": "elo_visitante",
    })
    tabla["fecha"] = tabla["fecha"].dt.date
    tabla["elo_local"] = tabla["elo_local"].round(1)
    tabla["elo_visitante"] = tabla["elo_visitante"].round(1)

    for nombre, p in probs_finales.items():
        tabla[f"prob_local_{nombre}"] = p[:, 0].round(3)
        tabla[f"prob_empate_{nombre}"] = p[:, 1].round(3)
        tabla[f"prob_visita_{nombre}"] = p[:, 2].round(3)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUTPUTS_DIR / "predicciones_fase3_xgboost_catboost.csv", index=False)

    print("\n=== Fase 3 -- probabilidades 1X2 (blend XGBoost+CatBoost), Mundial 2026 ===")
    cols_mostrar = ["fecha", "local", "visitante", "prob_local_blend", "prob_empate_blend", "prob_visita_blend"]
    print(tabla[cols_mostrar].to_string(index=False))


    import joblib
    modelos_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "modelos_fase3"
    modelos_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelos_finales.xgb, modelos_dir / "xgboost.joblib")
    modelos_finales.cb.save_model(str(modelos_dir / "catboost.cbm"))
    joblib.dump(categorias_torneo, modelos_dir / "categorias_torneo.joblib")

    print(f"\nGuardado: {OUTPUTS_DIR / 'predicciones_fase3_xgboost_catboost.csv'}")
    print(f"Guardado: modelos entrenados en {modelos_dir}")


if __name__ == "__main__":
    main()

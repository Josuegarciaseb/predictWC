"""Modelo de conteo para TARJETAS totales (Over/Under).

Mismo motor de conteo que los córners (ataque/defensa por selección + GLM de
Poisson con time-decay y shrinkage, comparando Poisson vs Binomial Negativa),
pero adaptado a tarjetas, con dos diferencias:

  1. Las "fuerzas" de equipo se reinterpretan:
       - propensión = tendencia disciplinaria propia (tarjetas que recibe)
       - induce     = cuánto provoca que el rival sea amonestado
     Juntas capturan la "tendencia disciplinaria de cada selección".
  2. Se añade una covariable de FASE DEL TORNEO (knockout): los partidos de
     eliminatoria concentran más tensión y más tarjetas (en los datos, ~+18%).

ADVERTENCIA ESTRUCTURAL (documentada a propósito): en tarjetas el factor
dominante es el ÁRBITRO, que no se conoce de antemano y que este modelo no puede
incluir. Por eso el mercado de tarjetas es intrínsecamente más ruidoso que el de
córners: el techo de acierto es bajo, hay que ser honesto con eso. La rivalidad
("derbis") tampoco se modela: a nivel selección no hay una señal limpia y
disponible de qué cruces son rivalidades; usar un proxy inventado daría una
falsa sensación de información.

La Binomial Negativa suele ser preferible aquí (las tarjetas están más
sobre-dispersas que los córners). Sin fuga de datos: .fit(fecha_corte=...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson, nbinom
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor

LINEAS_OU_DEFECTO = (2.5, 3.5, 4.5, 5.5)


class TarjetasModel:
    def __init__(self, distribucion: str = "nb", min_year: int = 2018,
                 half_life_years: float = 3.0, alpha_ridge: float = 0.05,
                 max_cards: int = 20, strength_shrinkage: float = 0.6,
                 usar_knockout: bool = True, calibracion: float = 1.10):
        assert distribucion in ("poisson", "nb")
        self.distribucion = distribucion
        self.min_year = min_year
        self.half_life_years = half_life_years
        self.alpha_ridge = alpha_ridge
        self.max_cards = max_cards
        self.strength_shrinkage = strength_shrinkage
        self.usar_knockout = usar_knockout
        # Multiplicador de calibración de localización (como en córners): corrige
        # la subestimación sistemática out-of-sample (~11%), por la deriva al alza
        # de tarjetas en el fútbol moderno (Euro/Copa 2024 muy por encima de los
        # torneos antiguos). c=1.10 centra la media. 1.0 = sin corrección.
        self.calibracion = calibracion

        self.propension_: pd.Series | None = None
        self.induce_: pd.Series | None = None
        self.home_adv_: float = 0.0
        self.knockout_coef_: float = 0.0
        self.intercept_: float = 0.0
        self.r_: float = np.inf
        self.propension_global_: float = 0.0
        self.induce_global_: float = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, fecha_corte: str | None = None) -> "TarjetasModel":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= self.min_year]
        if fecha_corte is not None:
            df = df[df["date"] < pd.Timestamp(fecha_corte)]

        df["home_cards"] = df["home_yellow"] + df["home_red"]
        df["away_cards"] = df["away_yellow"] + df["away_red"]
        if "knockout" not in df.columns:
            df["knockout"] = False

        max_date = df["date"].max()
        dias = (max_date - df["date"]).dt.days
        decay = 0.5 ** (1 / (self.half_life_years * 365.25))
        df["peso"] = decay ** dias

        ko = df["knockout"].astype(float).values
        home_rows = pd.DataFrame({
            "team": df["home_team"].values, "opponent": df["away_team"].values,
            "cards": df["home_cards"].values,
            "is_home": np.where(df["neutral"].values, 0, 1),
            "knockout": ko, "peso": df["peso"].values,
        })
        away_rows = pd.DataFrame({
            "team": df["away_team"].values, "opponent": df["home_team"].values,
            "cards": df["away_cards"].values, "is_home": 0,
            "knockout": ko, "peso": df["peso"].values,
        })
        long_df = pd.concat([home_rows, away_rows], ignore_index=True)

        equipos = sorted(set(long_df["team"]) | set(long_df["opponent"]))
        idx = {e: i for i, e in enumerate(equipos)}
        n_eq = len(equipos)
        n = len(long_df)

        rows_p = long_df["team"].map(idx).values
        rows_i = long_df["opponent"].map(idx).values
        X_p = sparse.csr_matrix((np.ones(n), (np.arange(n), rows_p)), shape=(n, n_eq))
        X_i = sparse.csr_matrix((np.ones(n), (np.arange(n), rows_i)), shape=(n, n_eq))
        cols_extra = ["is_home", "knockout"] if self.usar_knockout else ["is_home"]
        X_extra = sparse.csr_matrix(long_df[cols_extra].values.astype(float))
        X = sparse.hstack([X_p, X_i, X_extra]).tocsr()
        y = long_df["cards"].values.astype(float)
        w = long_df["peso"].values

        modelo = PoissonRegressor(alpha=self.alpha_ridge, max_iter=1000, tol=1e-7)
        modelo.fit(X, y, sample_weight=w)

        coef = modelo.coef_
        self.propension_ = pd.Series(coef[:n_eq], index=equipos)
        self.induce_ = pd.Series(coef[n_eq:2 * n_eq], index=equipos)
        self.home_adv_ = float(coef[2 * n_eq])
        self.knockout_coef_ = float(coef[2 * n_eq + 1]) if self.usar_knockout else 0.0
        self.intercept_ = float(modelo.intercept_)
        self.propension_global_ = float(self.propension_.mean())
        self.induce_global_ = float(self.induce_.mean())

        if self.strength_shrinkage != 1.0:
            s = self.strength_shrinkage
            self.propension_ = self.propension_global_ + s * (self.propension_ - self.propension_global_)
            self.induce_ = self.induce_global_ + s * (self.induce_ - self.induce_global_)

        mean_long = self._mean_filas(long_df["team"].values, long_df["opponent"].values,
                                     long_df["is_home"].values, long_df["knockout"].values)
        self.r_ = self._mle_dispersion(y, mean_long, w)
        return self

    def _mle_dispersion(self, y, m, w) -> float:
        m = np.clip(m, 1e-6, None)

        def neg_ll(log_r):
            r = np.exp(log_r)
            ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                  + r * (np.log(r) - np.log(r + m)) + y * (np.log(m) - np.log(r + m)))
            return -np.sum(w * ll)

        res = minimize_scalar(neg_ll, bounds=(np.log(0.5), np.log(1000)), method="bounded")
        return float(np.exp(res.x))

    # ------------------------------------------------------------------ #
    def _fuerza(self, equipo: str) -> tuple[float, float]:
        p = self.propension_.get(equipo, self.propension_global_)
        i = self.induce_.get(equipo, self.induce_global_)
        return p, i

    def _mean_filas(self, teams, opponents, is_home, knockout) -> np.ndarray:
        teams = np.atleast_1d(teams)
        opponents = np.atleast_1d(opponents)
        is_home = np.atleast_1d(is_home).astype(float)
        knockout = np.atleast_1d(knockout).astype(float)
        prop = np.array([self._fuerza(t)[0] for t in teams])
        ind = np.array([self._fuerza(o)[1] for o in opponents])
        return np.exp(self.intercept_ + prop + ind
                      + self.home_adv_ * is_home + self.knockout_coef_ * knockout)

    def lambda_mu(self, home_team, away_team, neutral=False, knockout=False) -> tuple[float, float]:
        h = 0.0 if neutral else 1.0
        ko = 1.0 if knockout else 0.0
        lam = float(self._mean_filas([home_team], [away_team], [h], [ko])[0]) * self.calibracion
        mu = float(self._mean_filas([away_team], [home_team], [0.0], [ko])[0]) * self.calibracion
        return lam, mu

    def _pmf(self, mean: float, distribucion: str) -> np.ndarray:
        k = np.arange(0, self.max_cards + 1)
        if distribucion == "poisson":
            p = poisson.pmf(k, mean)
        else:
            r = self.r_
            p = nbinom.pmf(k, r, r / (r + mean))
        return p / p.sum()

    def predecir_partido(self, home_team, away_team, neutral=False, knockout=False,
                         distribucion=None, lineas=LINEAS_OU_DEFECTO,
                         factor_externo: float = 1.0) -> dict:
        """factor_externo: multiplicador opcional por partido (hook para una señal
        externa FIABLE, p.ej. designaciones arbitrales reales si se consiguen). 1.0 =
        sin ajuste. No se usa con datos de Kaggle (resultaron no fiables)."""
        dist = distribucion or self.distribucion
        lam, mu = self.lambda_mu(home_team, away_team, neutral, knockout)
        lam *= factor_externo
        mu *= factor_externo
        total_pmf = np.convolve(self._pmf(lam, dist), self._pmf(mu, dist))
        soporte = np.arange(len(total_pmf))
        prob_over = {ln: float(total_pmf[soporte > ln].sum()) for ln in lineas}
        return {
            "distribucion": dist,
            "tarjetas_local": lam,
            "tarjetas_visita": mu,
            "tarjetas_total": lam + mu,
            "total_pmf": total_pmf,
            "prob_over": prob_over,
        }

    def ranking_disciplina(self, top: int = 15) -> pd.DataFrame:
        df = pd.DataFrame({
            "propension": self.propension_ - self.propension_global_,
            "induce_rival": self.induce_ - self.induce_global_,
        })
        df["indice_disciplina"] = df["propension"] + df["induce_rival"]
        return df.sort_values("indice_disciplina", ascending=False).head(top)

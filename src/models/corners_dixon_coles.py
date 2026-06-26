"""Modelo Dixon-Coles adaptado a CÓRNERS (no a goles).

Clona la estructura del modelo de goles (ataque/defensa por selección + ventaja
de local, GLM de Poisson con time-decay y shrinkage hacia la media), con tres
cambios respecto a poisson_dixon_coles.py:

  1. El target son córners, no goles.
  2. Se QUITA la corrección τ de marcadores bajos (0-0, 1-0, 0-1, 1-1): es
     específica de la dependencia de goles bajos y no aplica a córners.
  3. Se compara Poisson vs Binomial Negativa (NB). Los córners suelen estar
     sobre-dispersos (Var > media), así que la NB modela mejor la cola — clave
     para Over/Under. La media se estima igual (GLM de Poisson, consistente
     bajo sobre-dispersión) y la NB solo añade un parámetro de dispersión 'r'
     estimado por máxima verosimilitud sobre el train.

Salidas por partido:
  - córners esperados local/visita/total
  - Over/Under: P(total > linea) desde la cola de la distribución del total
    (convolución de las marginales local y visita)
  - 1x2 de córners: Skellam(λ, μ) en el caso Poisson (diferencia local−visita);
    convolución del producto exterior en el caso NB.

No hay fuga de datos: el pipeline (B01) entrena con .fit(fecha_corte=...) usando
solo partidos anteriores y valida walk-forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson, nbinom, skellam, norm
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor

LINEAS_OU_DEFECTO = (7.5, 8.5, 9.5, 10.5, 11.5)

# Nodos/pesos Gauss-Legendre para la CDF normal bivariante (se calculan una vez).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(20)


def _bvn_cdf(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """CDF normal bivariante P(X<=h, Y<=k) con correlación rho, vectorizada.

    Usa la descomposición de Sheppard: Φ2(h,k;ρ) = Φ(h)Φ(k) + ∫_0^ρ φ2(h,k;t) dt,
    con cuadratura Gauss-Legendre en [0, ρ]. Precisión sobrada para una rejilla
    de córners. h, k son arrays de la misma forma; rho es escalar.
    """
    base = norm.cdf(h) * norm.cdf(k)
    if abs(rho) < 1e-9:
        return base
    t = 0.5 * rho * (_GL_X + 1.0)      # mapea [-1,1] -> [0, rho]
    wt = 0.5 * rho * _GL_W
    integ = np.zeros_like(base, dtype=float)
    for tm, wm in zip(t, wt):
        denom = 1.0 - tm * tm
        integ += wm * (1.0 / (2 * np.pi * np.sqrt(denom))) * \
            np.exp(-(h * h - 2 * tm * h * k + k * k) / (2 * denom))
    return base + integ


class CornersModel:
    def __init__(self, distribucion: str = "nb", min_year: int = 2018,
                 half_life_years: float = 3.0, alpha_ridge: float = 0.05,
                 max_corners: int = 25, strength_shrinkage: float = 0.65,
                 calibracion: float = 1.04, rho_corr: float = -0.15):
        assert distribucion in ("poisson", "nb")
        self.distribucion = distribucion
        # Correlación de la cópula gaussiana entre córners local y visita. La
        # independencia (rho=0) sobreestima el empate de córners (~12% vs ~6.7%
        # real) porque ignora que el equipo dominante saca MÁS y concede MENOS
        # (dependencia negativa). rho=-0.15 ≈ correlación residual observada.
        # Ver scripts/B03_correlacion_corners.py.
        self.rho_corr = rho_corr
        self.min_year = min_year
        self.half_life_years = half_life_years
        self.alpha_ridge = alpha_ridge
        self.max_corners = max_corners
        # Multiplicador de calibración de localización sobre los córners esperados
        # (lam, mu). Corrige el sesgo sistemático de la media medido out-of-sample
        # (el GLM se centra en el pasado y los córners derivan al alza con el tiempo
        # añadido largo post-2022). c=1.04 elegido por walk-forward (B02): centra la
        # línea 9.5. 1.0 = sin corrección. Ver scripts/B02_calibracion_corners.py.
        self.calibracion = calibracion
        # Encoge ataque/defensa de córner hacia la media del campo. Con ~314
        # partidos y selecciones de 3-26 apariciones, las fuerzas puntuales son
        # ruidosas: el shrinkage frena el sobreajuste a equipos con pocos datos.
        self.strength_shrinkage = strength_shrinkage

        self.attack_: pd.Series | None = None
        self.defense_: pd.Series | None = None
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0
        self.r_: float = np.inf  # dispersión NB (inf == Poisson)
        self.attack_global_: float = 0.0
        self.defense_global_: float = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, fecha_corte: str | None = None) -> "CornersModel":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= self.min_year]
        if fecha_corte is not None:
            df = df[df["date"] < pd.Timestamp(fecha_corte)]  # estricto: sin fuga

        max_date = df["date"].max()
        dias = (max_date - df["date"]).dt.days
        decay = 0.5 ** (1 / (self.half_life_years * 365.25))
        df["peso"] = decay ** dias

        home_rows = pd.DataFrame({
            "team": df["home_team"].values,
            "opponent": df["away_team"].values,
            "corners": df["home_corners"].values,
            "is_home": np.where(df["neutral"].values, 0, 1),
            "peso": df["peso"].values,
        })
        away_rows = pd.DataFrame({
            "team": df["away_team"].values,
            "opponent": df["home_team"].values,
            "corners": df["away_corners"].values,
            "is_home": 0,
            "peso": df["peso"].values,
        })
        long_df = pd.concat([home_rows, away_rows], ignore_index=True)

        equipos = sorted(set(long_df["team"]) | set(long_df["opponent"]))
        idx = {e: i for i, e in enumerate(equipos)}
        n_eq = len(equipos)
        n = len(long_df)

        rows_attack = long_df["team"].map(idx).values
        rows_defense = long_df["opponent"].map(idx).values
        X_attack = sparse.csr_matrix((np.ones(n), (np.arange(n), rows_attack)), shape=(n, n_eq))
        X_defense = sparse.csr_matrix((np.ones(n), (np.arange(n), rows_defense)), shape=(n, n_eq))
        X_home = sparse.csr_matrix(long_df[["is_home"]].values.astype(float))
        X = sparse.hstack([X_attack, X_defense, X_home]).tocsr()
        y = long_df["corners"].values.astype(float)
        w = long_df["peso"].values

        modelo = PoissonRegressor(alpha=self.alpha_ridge, max_iter=1000, tol=1e-7)
        modelo.fit(X, y, sample_weight=w)

        coef = modelo.coef_
        self.attack_ = pd.Series(coef[:n_eq], index=equipos)
        self.defense_ = pd.Series(coef[n_eq:2 * n_eq], index=equipos)
        self.home_adv_ = float(coef[2 * n_eq])
        self.intercept_ = float(modelo.intercept_)
        self.attack_global_ = float(self.attack_.mean())
        self.defense_global_ = float(self.defense_.mean())

        if self.strength_shrinkage != 1.0:
            s = self.strength_shrinkage
            self.attack_ = self.attack_global_ + s * (self.attack_ - self.attack_global_)
            self.defense_ = self.defense_global_ + s * (self.defense_ - self.defense_global_)

        # Dispersión NB por MLE de un solo 'r' compartido, sobre las medias ya
        # ajustadas (y encogidas). r->inf recupera Poisson.
        mean_long = self._mean_filas(long_df["team"].values, long_df["opponent"].values,
                                     long_df["is_home"].values)
        self.r_ = self._mle_dispersion(y, mean_long, w)
        return self

    def _mle_dispersion(self, y, m, w) -> float:
        m = np.clip(m, 1e-6, None)

        def neg_ll(log_r):
            r = np.exp(log_r)
            ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                  + r * (np.log(r) - np.log(r + m))
                  + y * (np.log(m) - np.log(r + m)))
            return -np.sum(w * ll)

        res = minimize_scalar(neg_ll, bounds=(np.log(0.5), np.log(1000)), method="bounded")
        return float(np.exp(res.x))

    # ------------------------------------------------------------------ #
    def _fuerza(self, equipo: str) -> tuple[float, float]:
        a = self.attack_.get(equipo, self.attack_global_)
        d = self.defense_.get(equipo, self.defense_global_)
        return a, d

    def _mean_filas(self, teams, opponents, is_home) -> np.ndarray:
        teams = np.atleast_1d(teams)
        opponents = np.atleast_1d(opponents)
        is_home = np.atleast_1d(is_home).astype(float)
        atk = np.array([self._fuerza(t)[0] for t in teams])
        dfn = np.array([self._fuerza(o)[1] for o in opponents])
        return np.exp(self.intercept_ + atk + dfn + self.home_adv_ * is_home)

    def lambda_mu(self, home_team: str, away_team: str, neutral: bool = False) -> tuple[float, float]:
        h = 0.0 if neutral else 1.0
        lam = float(self._mean_filas([home_team], [away_team], [h])[0]) * self.calibracion
        mu = float(self._mean_filas([away_team], [home_team], [0.0])[0]) * self.calibracion
        return lam, mu

    # ------------------------------------------------------------------ #
    def _pmf(self, mean: float, distribucion: str) -> np.ndarray:
        k = np.arange(0, self.max_corners + 1)
        if distribucion == "poisson":
            p = poisson.pmf(k, mean)
        else:
            r = self.r_
            prob = r / (r + mean)
            p = nbinom.pmf(k, r, prob)
        return p / p.sum()

    def _joint(self, ph: np.ndarray, pa: np.ndarray) -> np.ndarray:
        """Joint P(local=i, visita=j). Independencia si rho_corr=0; si no, cópula
        gaussiana con correlación rho_corr (preserva las marginales ph, pa)."""
        if abs(self.rho_corr) < 1e-9:
            return np.outer(ph, pa)
        # CDF marginales con un 0 antepuesto (frontera inferior).
        eps = 1e-12
        cdf_h = np.clip(np.concatenate([[0.0], np.cumsum(ph)]), eps, 1 - eps)
        cdf_a = np.clip(np.concatenate([[0.0], np.cumsum(pa)]), eps, 1 - eps)
        zh, za = norm.ppf(cdf_h), norm.ppf(cdf_a)
        H, A = np.meshgrid(zh, za, indexing="ij")           # (n+1, n+1)
        C = _bvn_cdf(H, A, self.rho_corr)                    # P(local<=i, visita<=j)
        # PMF por diferencias finitas de la CDF conjunta.
        M = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
        M = np.clip(M, 0.0, None)
        return M / M.sum()

    def predecir_partido(self, home_team: str, away_team: str, neutral: bool = False,
                         distribucion: str | None = None,
                         lineas=LINEAS_OU_DEFECTO) -> dict:
        dist = distribucion or self.distribucion
        lam, mu = self.lambda_mu(home_team, away_team, neutral)

        ph = self._pmf(lam, dist)
        pa = self._pmf(mu, dist)

        # Cada salida usa la dependencia que mejor le ajusta (validado en B03):
        #  - TOTAL / Over-Under: convolución de marginales (independencia). El total
        #    real está sobre-dispersado; la correlación negativa lo estrecharía y
        #    empeora el ajuste del total.
        #  - 1x2 (diferencia local-visita): cópula gaussiana con rho_corr<0. La
        #    dependencia negativa ensancha la diferencia y corrige el empate inflado.
        total_pmf = np.convolve(ph, pa)
        soporte = np.arange(len(total_pmf))
        prob_over = {ln: float(total_pmf[soporte > ln].sum()) for ln in lineas}

        M = self._joint(ph, pa)
        n = M.shape[0]
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        p_loc = float(M[ii > jj].sum())
        p_emp = float(M[ii == jj].sum())
        p_vis = float(M[ii < jj].sum())
        s = p_loc + p_emp + p_vis
        p_loc, p_emp, p_vis = p_loc / s, p_emp / s, p_vis / s

        # 1x2 vía Skellam(λ, μ) — referencia de INDEPENDENCIA (cierre exacto de la
        # diferencia de dos Poisson). Útil para comparar contra la cópula.
        sk_emp = float(skellam.pmf(0, lam, mu))
        sk_loc = float(skellam.sf(0, lam, mu))
        sk_vis = float(skellam.cdf(-1, lam, mu))
        sk_s = sk_loc + sk_emp + sk_vis

        return {
            "distribucion": dist,
            "corners_local": lam,
            "corners_visita": mu,
            "corners_total": lam + mu,
            "total_pmf": total_pmf,
            "prob_over": prob_over,
            "prob_mas_corners_local": p_loc,
            "prob_empate_corners": p_emp,
            "prob_mas_corners_visita": p_vis,
            "skellam_1x2": (sk_loc / sk_s, sk_emp / sk_s, sk_vis / sk_s),
        }

    def ranking_corners(self, top: int = 15) -> pd.DataFrame:
        df = pd.DataFrame({
            "ataque_corner": self.attack_ - self.attack_global_,
            "concede_corner": self.defense_ - self.defense_global_,
        })
        df["indice"] = df["ataque_corner"] - df["concede_corner"]
        return df.sort_values("indice", ascending=False).head(top)

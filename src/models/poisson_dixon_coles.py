from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import nbinom, poisson
from scipy.optimize import minimize_scalar
from sklearn.linear_model import PoissonRegressor


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    out = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1 - lam[m00] * mu[m00] * rho
    out[m01] = 1 + lam[m01] * rho
    out[m10] = 1 + mu[m10] * rho
    out[m11] = 1 - rho
    return out


class DixonColesModel:
    def __init__(self, cutoff_years: float = 11, half_life_years: float = 2.5,
                 alpha_ridge: float = 1e-4, max_goals: int = 10,
                 strength_shrinkage: float = 1.0, dispersion: float = 0.06):
        self.cutoff_years = cutoff_years
        self.half_life_years = half_life_years
        self.alpha_ridge = alpha_ridge
        self.max_goals = max_goals
        # Sobredispersión phi: los goles marginales pasan de Poisson(lam) a
        # Negative Binomial con media lam y varianza lam*(1+phi*lam), que es
        # el resultado de un shock multiplicativo Gamma sobre lam (forma del
        # partido, "partido loco"). La varianza extra crece con lam^2, así que
        # engorda la cola del favorito y casi no toca al débil. phi=0 recupera
        # el Poisson exacto. El GLM sigue estimando bien la media bajo
        # sobredispersión (quasi-Poisson), por lo que fit() no cambia.
        # phi=0.06 elegido por barrido walk-forward 2018-2025 (scripts/09):
        # ratio goleadas esp/obs 0.998 (vs 0.976 con Poisson) con +0.11% de
        # log-loss 1X2; con phi>=0.09 el visitante se sobre-ensancha.
        self.dispersion = dispersion
        # Encoge ataque/defensa hacia la media del campo (1.0 = sin cambio).
        # Se usó 0.85 para contener la sobreconfianza en prob_campeon del Monte
        # Carlo, pero el backtest (scripts/09) mostró que a nivel partido
        # aplasta la cola de goleadas (ratio esp/obs 0.87 vs 0.98) y ADEMÁS
        # empeora el log-loss 1X2. La sobreconfianza del torneo se controla
        # ahora con generar_variantes_bootstrap() en el Monte Carlo, que
        # propaga la incertidumbre de las fuerzas sin sesgar la media de
        # ningún partido. Se conserva el parámetro solo para experimentos.
        self.strength_shrinkage = strength_shrinkage

        self.equipos_: list[str] = []
        self.idx_equipo_: dict[str, int] = {}
        self.attack_: pd.Series | None = None
        self.defense_: pd.Series | None = None
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0
        self.rho_: float = 0.0
        self.attack_global_promedio_: float = 0.0


    def _matrices_entrenamiento(self, df_historico: pd.DataFrame, fecha_corte: str | None = None):
        """Prepara la ventana de entrenamiento y la matriz de diseño del GLM.
        Devuelve (train, X, y, w, equipos, idx_equipo)."""
        df = df_historico.copy()
        df["date"] = pd.to_datetime(df["date"])
        if fecha_corte is not None:
            df = df[df["date"] <= pd.Timestamp(fecha_corte)]

        fecha_inicio_ventana = df["date"].max() - pd.Timedelta(days=int(self.cutoff_years * 365.25))
        train = df[df["date"] >= fecha_inicio_ventana].copy()

        max_date = train["date"].max()
        dias = (max_date - train["date"]).dt.days
        decay = 0.5 ** (1 / (self.half_life_years * 365.25))
        train["peso"] = decay ** dias

        home_rows = pd.DataFrame({
            "team": train["home_team"].values,
            "opponent": train["away_team"].values,
            "goals": train["home_score"].values,
            "is_home": np.where(train["neutral"].values, 0, 1),
            "peso": train["peso"].values,
        })
        away_rows = pd.DataFrame({
            "team": train["away_team"].values,
            "opponent": train["home_team"].values,
            "goals": train["away_score"].values,
            "is_home": 0,
            "peso": train["peso"].values,
        })
        long_df = pd.concat([home_rows, away_rows], ignore_index=True)

        equipos = sorted(set(long_df["team"]) | set(long_df["opponent"]))
        idx_equipo = {e: i for i, e in enumerate(equipos)}
        n_eq = len(equipos)
        n = len(long_df)

        rows_attack = long_df["team"].map(idx_equipo).values
        rows_defense = long_df["opponent"].map(idx_equipo).values

        X_attack = sparse.csr_matrix((np.ones(n), (np.arange(n), rows_attack)), shape=(n, n_eq))
        X_defense = sparse.csr_matrix((np.ones(n), (np.arange(n), rows_defense)), shape=(n, n_eq))
        X_home = sparse.csr_matrix(long_df[["is_home"]].values.astype(float))
        X = sparse.hstack([X_attack, X_defense, X_home]).tocsr()
        y = long_df["goals"].values
        w = long_df["peso"].values
        return train, X, y, w, equipos, idx_equipo

    def _ajustar_glm(self, X, y, w, equipos: list[str]):
        """Ajusta el GLM Poisson y devuelve (attack, defense, home_adv, intercept)."""
        n_eq = len(equipos)
        modelo = PoissonRegressor(alpha=self.alpha_ridge, max_iter=500, tol=1e-6)
        modelo.fit(X, y, sample_weight=w)
        coef = modelo.coef_
        attack = pd.Series(coef[:n_eq], index=equipos)
        defense = pd.Series(coef[n_eq:2 * n_eq], index=equipos)
        return attack, defense, float(coef[2 * n_eq]), float(modelo.intercept_)

    def _aplicar_shrinkage(self):
        # Shrinkage hacia la media (preserva al equipo promedio: la media de las
        # series no cambia, por lo que intercept_ y home_adv_ siguen válidos).
        if self.strength_shrinkage != 1.0:
            s = self.strength_shrinkage
            self.attack_ = self.attack_global_promedio_ + s * (self.attack_ - self.attack_global_promedio_)
            self.defense_ = self._defense_global_promedio + s * (self.defense_ - self._defense_global_promedio)

    def fit(self, df_historico: pd.DataFrame, fecha_corte: str | None = None) -> "DixonColesModel":
        train, X, y, w, equipos, idx_equipo = self._matrices_entrenamiento(df_historico, fecha_corte)

        self.attack_, self.defense_, self.home_adv_, self.intercept_ = self._ajustar_glm(X, y, w, equipos)
        self.equipos_ = equipos
        self.idx_equipo_ = idx_equipo
        self.attack_global_promedio_ = float(self.attack_.mean())
        self._defense_global_promedio = float(self.defense_.mean())

        # El shrinkage (si se usa) va ANTES de estimar rho para que rho se
        # ajuste sobre los lambdas ya encogidos.
        self._aplicar_shrinkage()

        lam_train, mu_train = self._lambda_mu(
            train["home_team"].values, train["away_team"].values, train["neutral"].values
        )
        x_obs = train["home_score"].values.astype(int)
        y_obs = train["away_score"].values.astype(int)
        peso_train = train["peso"].values

        def neg_ll_rho(rho):
            t = _tau(x_obs, y_obs, lam_train, mu_train, rho)
            if np.any(t <= 0):
                return 1e10
            return -np.sum(peso_train * np.log(t))

        res = minimize_scalar(neg_ll_rho, bounds=(-0.2, 0.2), method="bounded")
        self.rho_ = float(res.x)
        return self

    def generar_variantes_bootstrap(self, df_historico: pd.DataFrame, fecha_corte: str | None = None,
                                    n_boot: int = 50, seed: int = 123) -> list["DixonColesModel"]:
        """Variantes del modelo por bootstrap bayesiano (pesos exponenciales).

        Cada variante es una muestra plausible de las fuerzas ataque/defensa
        dado el histórico: se re-ajusta el GLM multiplicando el peso de cada
        partido por un factor ~Exp(1) (así ningún partido desaparece y ningún
        equipo se queda sin datos, a diferencia del bootstrap por remuestreo).
        El Monte Carlo alterna variantes entre simulaciones para propagar la
        incertidumbre de parámetros: en las sims donde el favorito "salió
        débil" pierde temprano, lo que contiene su prob_campeon sin sesgar la
        media de ningún partido (reemplaza al viejo strength_shrinkage=0.85).

        rho no se re-estima (solo afecta celdas 0-0/0-1/1-0/1-1 y es estable);
        cada variante hereda el rho del modelo base, por lo que hay que llamar
        a fit() antes."""
        if self.attack_ is None:
            raise RuntimeError("Llama a fit() antes de generar variantes bootstrap.")

        train, X, y, w, equipos, _ = self._matrices_entrenamiento(df_historico, fecha_corte)
        n_partidos = len(train)
        rng = np.random.default_rng(seed)

        variantes = []
        for _ in range(n_boot):
            # el mismo factor para las dos filas (local/visita) de cada partido
            factor = rng.exponential(1.0, size=n_partidos)
            w_boot = w * np.tile(factor, 2)

            v = copy.copy(self)
            v.attack_, v.defense_, v.home_adv_, v.intercept_ = self._ajustar_glm(X, y, w_boot, equipos)
            v.attack_global_promedio_ = float(v.attack_.mean())
            v._defense_global_promedio = float(v.defense_.mean())
            v._aplicar_shrinkage()
            variantes.append(v)
        return variantes


    def _fuerza(self, equipo: str) -> tuple[float, float]:
        ataque = self.attack_.get(equipo, self.attack_global_promedio_)
        defensa = self.defense_.get(equipo, self._defense_global_promedio)
        return ataque, defensa

    def _lambda_mu(self, home_teams, away_teams, neutral):
        home_teams = np.atleast_1d(home_teams)
        away_teams = np.atleast_1d(away_teams)
        neutral = np.atleast_1d(neutral)

        ataque_h = np.array([self._fuerza(t)[0] for t in home_teams])
        defensa_h = np.array([self._fuerza(t)[1] for t in home_teams])
        ataque_a = np.array([self._fuerza(t)[0] for t in away_teams])
        defensa_a = np.array([self._fuerza(t)[1] for t in away_teams])

        lam = np.exp(self.intercept_ + ataque_h + defensa_a + self.home_adv_ * np.where(neutral, 0, 1))
        mu = np.exp(self.intercept_ + ataque_a + defensa_h)
        return lam, mu


    def matriz_marcador(self, home_team: str, away_team: str, neutral: bool = False) -> np.ndarray:
        lam, mu = self._lambda_mu([home_team], [away_team], [neutral])
        lam, mu = float(lam[0]), float(mu[0])

        g = np.arange(0, self.max_goals + 1)
        if self.dispersion > 0:
            n_nb = 1.0 / self.dispersion
            px = nbinom.pmf(g, n_nb, n_nb / (n_nb + lam))
            py = nbinom.pmf(g, n_nb, n_nb / (n_nb + mu))
        else:
            px = poisson.pmf(g, lam)
            py = poisson.pmf(g, mu)
        M = np.outer(px, py)

        for (xx, yy) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            t = _tau(np.array([xx]), np.array([yy]), np.array([lam]), np.array([mu]), self.rho_)[0]
            M[xx, yy] *= t

        M = M / M.sum()
        return M

    def predecir_partido(self, home_team: str, away_team: str, neutral: bool = False) -> dict:
        M = self.matriz_marcador(home_team, away_team, neutral)
        idx_max = np.unravel_index(np.argmax(M), M.shape)
        goles_local_idx, goles_visita_idx = idx_max

        xs, ys = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]), indexing="ij")
        p_local = M[xs > ys].sum()
        p_empate = M[xs == ys].sum()
        p_visita = M[xs < ys].sum()


        flat_idx = np.argsort(M.ravel())[::-1][:3]
        top3 = [(int(i // M.shape[1]), int(i % M.shape[1]), float(M.ravel()[i])) for i in flat_idx]

        lam, mu = self._lambda_mu([home_team], [away_team], [neutral])

        return {
            "marcador_mas_probable": f"{goles_local_idx}-{goles_visita_idx}",
            "prob_marcador_exacto": float(M[idx_max]),
            "prob_local": float(p_local),
            "prob_empate": float(p_empate),
            "prob_visita": float(p_visita),
            "goles_esperados_local": float(lam[0]),
            "goles_esperados_visita": float(mu[0]),
            "top3_marcadores": top3,
        }

    def ranking_ataque_defensa(self, top: int = 15) -> pd.DataFrame:
        df = pd.DataFrame({
            "ataque": self.attack_ - self.attack_global_promedio_,
            "defensa": self.defense_ - self._defense_global_promedio,
        })
        df["indice_ofensivo"] = df["ataque"] - df["defensa"]
        return df.sort_values("indice_ofensivo", ascending=False).head(top)

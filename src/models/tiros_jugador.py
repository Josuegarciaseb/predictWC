"""Modelo jerárquico de TIROS A PUERTA por jugador (Fase D).

Tasa Poisson de tiros a puerta por 90', escalada por minutos esperados y ajustada
por el rival, con efecto aleatorio por jugador para no sobreajustar a quienes
tienen pocos partidos. Se implementa como empirical-Bayes Gamma-Poisson (= modelo
jerárquico con efecto aleatorio): el multiplicador de cada jugador θ_p y el de cada
rival φ_o se encogen hacia 1 (la media del campo) según cuántos datos haya.

  λ0           = tasa global de tiros a puerta por 90'
  θ_p ~ Gamma  = factor del jugador (>1 más tirador), posterior Gamma(a_p, b_p)
  φ_o ~ Gamma  = factor del rival (>1 concede más tiros a puerta)
  media(jugador p, rival o, m min) = λ0 · θ_p · φ_o · (m/90)

La distribución predictiva del conteo es Binomial Negativa (Gamma-Poisson), que
para jugadores con pocos datos ensancha la incertidumbre automáticamente — justo
lo que se quiere para no dar falsa precisión.

LÍMITE HONESTO: la predicción es condicional a los minutos que juegue. El once
probable del Mundial 2026 no existe aún; por eso el CLI pide minutos esperados y
no hay tabla de fixtures por jugador (no hay alineaciones).

Sin fuga de datos: .fit(fecha_corte=...) usa solo partidos anteriores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

LINEAS_OU_DEFECTO = (0.5, 1.5, 2.5)


class TirosJugadorModel:
    def __init__(self, min_year: int = 2018, max_sot: int = 12):
        self.min_year = min_year
        self.max_sot = max_sot
        self.lambda0_: float = 0.357
        self.k_player_: float = 1.0
        self.k_opp_: float = 1.0
        # posteriores por jugador / rival: (a, b) de la Gamma
        self.player_ab_: dict[str, tuple[float, float]] = {}
        self.opp_ab_: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mle_k(conteos: np.ndarray, esperados: np.ndarray) -> float:
        """MLE de la forma k del prior Gamma vía marginal NB de los agregados."""
        conteos = conteos.astype(float)
        esperados = np.clip(esperados.astype(float), 1e-9, None)

        def neg_ll(log_k):
            k = np.exp(log_k)
            # Y ~ NB(r=k, p=k/(k+esperado))  (media = esperado, sobre-disp por k)
            p = k / (k + esperados)
            ll = (gammaln(conteos + k) - gammaln(k) - gammaln(conteos + 1)
                  + k * np.log(p) + conteos * np.log1p(-p))
            return -np.sum(ll)

        res = minimize_scalar(neg_ll, bounds=(np.log(0.1), np.log(100)), method="bounded")
        return float(np.exp(res.x))

    def fit(self, df: pd.DataFrame, fecha_corte: str | None = None) -> "TirosJugadorModel":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= self.min_year]
        if fecha_corte is not None:
            df = df[df["date"] < pd.Timestamp(fecha_corte)]
        df = df[df["minutes"] > 0]
        df["exp90"] = df["minutes"] / 90.0

        self.lambda0_ = float(df["sot"].sum() / df["exp90"].sum())

        # Efecto jugador.
        gp = df.groupby("player").agg(S=("sot", "sum"), E=("exp90", "sum"))
        gp["esp"] = self.lambda0_ * gp["E"]
        self.k_player_ = self._mle_k(gp["S"].values, gp["esp"].values)
        a = self.k_player_ + gp["S"].values
        b = self.k_player_ + gp["esp"].values
        self.player_ab_ = {p: (float(ai), float(bi)) for p, ai, bi in zip(gp.index, a, b)}

        # Efecto rival (tiros a puerta que concede).
        go = df.groupby("opponent").agg(S=("sot", "sum"), E=("exp90", "sum"))
        go["esp"] = self.lambda0_ * go["E"]
        self.k_opp_ = self._mle_k(go["S"].values, go["esp"].values)
        ao = self.k_opp_ + go["S"].values
        bo = self.k_opp_ + go["esp"].values
        self.opp_ab_ = {o: (float(ai), float(bi)) for o, ai, bi in zip(go.index, ao, bo)}
        return self

    # ------------------------------------------------------------------ #
    def theta_jugador(self, player: str) -> float:
        a, b = self.player_ab_.get(player, (self.k_player_, self.k_player_))
        return a / b  # media posterior (=1 si no hay datos)

    def phi_rival(self, opponent: str) -> float:
        a, b = self.opp_ab_.get(opponent, (self.k_opp_, self.k_opp_))
        return a / b

    def predecir(self, player: str, opponent: str, minutos: float = 90.0,
                 lineas=LINEAS_OU_DEFECTO) -> dict:
        e = minutos / 90.0
        phi = self.phi_rival(opponent)
        a_p, b_p = self.player_ab_.get(player, (self.k_player_, self.k_player_))
        # Predictiva NB: r = a_p (forma posterior del jugador), media = (a_p/b_p)*μ0.
        mu0 = self.lambda0_ * phi * e
        r = a_p
        p = b_p / (b_p + mu0)
        media = r * (1 - p) / p

        k = np.arange(0, self.max_sot + 1)
        pmf = nbinom.pmf(k, r, p)
        pmf = pmf / pmf.sum()
        soporte = np.arange(len(pmf))
        prob_over = {ln: float(pmf[soporte > ln].sum()) for ln in lineas}
        return {
            "player": player, "opponent": opponent, "minutos": minutos,
            "sot_esperados": float(media),
            "theta_jugador": a_p / b_p, "phi_rival": phi,
            "pmf": pmf, "prob_over": prob_over,
            "tiene_datos": player in self.player_ab_,
        }

    def ranking_tiradores(self, top: int = 20, min_exp90: float = 3.0) -> pd.DataFrame:
        filas = []
        for p, (a, b) in self.player_ab_.items():
            exp90 = (b - self.k_player_) / self.lambda0_  # E del jugador
            if exp90 < min_exp90:
                continue
            filas.append({"player": p, "sot_por_90": self.lambda0_ * a / b,
                          "theta": a / b, "exp90": exp90})
        return (pd.DataFrame(filas).sort_values("sot_por_90", ascending=False)
                .head(top).reset_index(drop=True))

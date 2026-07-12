# inverse power of tilted stable law
# primarily for FG's random variable generation

from typing import Optional
import numpy as np
from functools import lru_cache
from scipy.stats import uniform


TILT_GRID_SIZE = 200_000


# input: alpha in (0,1), beta > 0


def zolotarev_log_A(q, alpha):
    """Log Zolotarev/Kanter factor for a positive stable RV with Laplace exp(-s**alpha)."""
    v = np.pi * q
    c = (1.0 - alpha) / alpha
    return (
        np.log(np.sin(alpha * v))
        - np.log(np.sin(v)) / alpha
        + c * np.log(np.sin((1.0 - alpha) * v))
    )


def build_tilted_kanter_grid(alpha, beta, grid_size=TILT_GRID_SIZE):
    """Inverse-CDF grid for the polynomially tilted stable angle."""
    assert beta > 0
    eps = np.finfo(float).eps
    q_grid = np.linspace(eps, 1.0 - eps, grid_size)  # 0 to 1
    log_w = -beta * zolotarev_log_A(q_grid, alpha)
    log_w -= np.max(log_w)
    w = np.exp(log_w)  # the PDF

    q_cdf_grid = np.empty_like(q_grid)
    q_cdf_grid[0] = 0.0
    q_cdf_grid[1:] = np.cumsum(0.5 * (w[:-1] + w[1:]) * np.diff(q_grid))  # numerical integration on PDF
    q_cdf_grid /= q_cdf_grid[-1]  # normalize the total density, so we don't need to know I_beta
    q_cdf_grid[-1] = 1.0
    return q_grid, q_cdf_grid


class TiltedKanter:
    def __init__(self, alpha, beta, grid_size=TILT_GRID_SIZE, rng=None):
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.grid_size: int = int(grid_size)
        self.rng = np.random.default_rng() if rng is None else rng

        assert 0 < self.alpha < 1
        assert self.beta >= 0

        self.q_grid: Optional[np.ndarray] = None
        self.q_cdf_grid: Optional[np.ndarray] = None
        # don't use q_grid if beta is zero
        if self.beta > 0:
            self.q_grid, self.q_cdf_grid = build_tilted_kanter_grid(alpha, beta, grid_size=self.grid_size)

    def q_rvs(self, size):
        if self.beta == 0:
            return self.uniform_q_rvs(size)

        assert self.q_grid is not None
        assert self.q_cdf_grid is not None
        return np.interp(self.rng.random(size), self.q_cdf_grid, self.q_grid)

    def uniform_q_rvs(self, size):
        eps = np.finfo(float).eps
        return np.clip(self.rng.random(size), eps, 1.0 - eps)
    
    def gamma_rvs(self, size):
        c = (1.0 - self.alpha) / self.alpha
        return self.rng.gamma(shape=1.0 + c * self.beta, scale=1.0, size=size)
    
    def zolotarev_log_A(self, size):
        return zolotarev_log_A(self.q_rvs(size), self.alpha)



def _kanter_log_U_rvs(size, alpha, beta, rng, q_grid, q_cdf_grid):
    # Kanter representation, U = T_{alpha,beta}^{-alpha}
    # has density proportional to u**(beta/alpha) M_alpha(u)
    # and frac gamma's rvs X = sigma * U^(1/p).

    # this is legacy, kept for reference purpose
    c = (1.0 - alpha) / alpha
    q = np.interp(rng.random(size), q_cdf_grid, q_grid)
    e = rng.gamma(shape=1.0 + c * beta, scale=1.0, size=size)
    log_u = (1.0 - alpha) * np.log(e) - alpha * zolotarev_log_A(q, alpha)
    return log_u


# ---------------------------------------------------------------------
class InverseStable:
    def __init__(self, alpha, rng=None):
        # X = T_{alpha}^{-alpha} = M_alpha (M_Wright_One_Sided)
        self.alpha: float = float(alpha)
        self.rng = np.random.default_rng() if rng is None else rng

    def q_rvs(self, size):
        eps = np.finfo(float).eps
        return np.clip(self.rng.random(size), eps, 1.0 - eps)
    
    def gamma_rvs(self, size):
        return self.rng.gamma(shape=1.0, scale=1.0, size=size)  # Gamma(1,1) = Exp(1)

    def rvs(self, size):
        # Inverse stable law, using beta=0 Kanter variables: U0=E^{1-alpha} A_alpha(Q)^{-alpha}.
        # this should produce the same statistics as M_Wright_One_Sided.rvs()
        q = self.q_rvs(size)
        e = self.gamma_rvs(size)
        log_u = (1.0 - self.alpha) * np.log(e) - self.alpha * zolotarev_log_A(q, self.alpha)
        return np.exp(log_u)


# ---------------------------------------------------------------------
class TitledStable2(TiltedKanter):
    def __init__(self, alpha, beta, grid_size=TILT_GRID_SIZE, rng=None):
        # X = T_{alpha,beta}^{-alpha}
        super().__init__(alpha, beta, grid_size=grid_size, rng=rng)

    def log_U_rvs(self, size):
        # U = T_{alpha,beta}^{-alpha}, but we render with log_U, see fracdist.pdf
        # U has density proportional to u**(beta/alpha) M_alpha(u)
        e = self.gamma_rvs(size)
        log_u = (1.0 - self.alpha) * np.log(e) - self.alpha * self.zolotarev_log_A(size)
        return log_u

    def rvs(self, size):
        # this is U's rvs
        log_u = self.log_U_rvs(size)
        return np.exp(log_u)

    def fg_rvs(self, size, sigma, p):
        # X = sigma * U^(1/p), this is primarily for testing FG
        log_u = self.log_U_rvs(size)
        return sigma * np.exp(log_u / p)  # X


class TitledStable3(TiltedKanter):
    def __init__(self, alpha, beta, gamma, grid_size=TILT_GRID_SIZE, rng=None):
        # X = T_{alpha,beta}^{-gamma}; gamma is the positive exponent in the negative power.
        # The inverse stable/M-Wright special case is beta=0 and gamma=alpha.
        super().__init__(alpha, beta, grid_size=grid_size, rng=rng)
        self.gamma: float = float(gamma)
        # debug
        # print(f"TitledStable3: alpha {self.alpha:.3f} beta {self.beta:.3f} gamma {self.gamma:.3f}")


    def log_rvs(self, size):
        # X = T_{alpha,beta}^{-gamma}, but we render with log_X, see fracdist.pdf
        # without scale/sigma here
        c = (1.0 - self.alpha) / self.alpha
        e = self.gamma_rvs(size)
        log_x = c * self.gamma * np.log(e) - self.gamma * self.zolotarev_log_A(size)
        return log_x

    def rvs(self, size):
        # this is X's rvs
        return np.exp(self.log_rvs(size))


@lru_cache(maxsize=100)
def get_tilted_stable2(alpha, beta):
    return TitledStable2(alpha, beta)


@lru_cache(maxsize=100)
def get_tilted_stable3(alpha, beta, gamma):
    return TitledStable3(alpha, beta, gamma)

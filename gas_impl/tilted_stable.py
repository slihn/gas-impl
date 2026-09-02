# inverse power of tilted stable law
# primarily for FG's random variable generation

from typing import Optional
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.special import gamma
from scipy.stats import levy_stable, ks_2samp


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


@lru_cache(maxsize=100)
def get_tilted_kanter_grid(alpha, beta, grid_size=TILT_GRID_SIZE):
    """Cached inverse-CDF grid. The grid is the expensive part of a tilted Kanter sampler,
    so it is cached here, letting the sampler object itself stay cheap and carry a caller's rng.
    The arrays are shared by reference, hence read-only."""
    q_grid, q_cdf_grid = build_tilted_kanter_grid(alpha, beta, grid_size=grid_size)
    q_grid.flags.writeable = False
    q_cdf_grid.flags.writeable = False
    return q_grid, q_cdf_grid


def one_sided_stable(alpha: float):
    # this is S_alpha, or L_alpha
    assert 0 < alpha <= 1.0
    scale = np.power(np.cos(alpha * np.pi / 2.0), 1.0/alpha)  # type: ignore
    assert isinstance(scale, float), f"ERROR: scale={scale} is not float for alpha={alpha}"
    assert scale > 0
    return levy_stable(alpha, beta=1.0, loc=0, scale=scale)


def one_sided_stable_pdf(x, alpha: float):
    """Return the positive alpha-stable density with Laplace transform exp(-s**alpha)."""
    assert 0 < alpha <= 1.0
    return one_sided_stable(alpha).pdf(x)  # type: ignore


def inverse_stable_pdf(x, alpha: float):
    """Return the M-Wright density of X = S_alpha**(-alpha), for x > 0.

    If ``f_alpha`` is the positive stable density, the transformation
    ``S_alpha = x**(-1/alpha)`` gives

        M_alpha(x) = f_alpha(x**(-1/alpha))
                     * x**(-1/alpha - 1) / alpha.
    """
    assert 0 < alpha <= 1.0
    stable_x = np.power(x, -1.0 / alpha)
    jacobian = np.power(x, -1.0 / alpha - 1.0) / alpha
    return jacobian * one_sided_stable_pdf(stable_x, alpha)


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
            self.q_grid, self.q_cdf_grid = get_tilted_kanter_grid(self.alpha, self.beta, grid_size=self.grid_size)

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

    def pdf(self, x):
        """Return the density of X = T_alpha**(-alpha), for x > 0."""
        return inverse_stable_pdf(x, self.alpha)


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

    def pdf(self, x):
        """Return the density of U = T_{alpha,beta}**(-alpha), for x > 0."""
        c = gamma(1.0 + self.beta) / gamma(1.0 + self.beta / self.alpha)
        tilt = np.power(x, self.beta / self.alpha)
        return c * tilt * inverse_stable_pdf(x, self.alpha)


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

    def pdf(self, x):
        """Return the density of X = T_{alpha,beta}**(-gamma), for x > 0."""
        if self.gamma == 0:
            raise ValueError("gamma must be non-zero for a continuous density")
        c = gamma(1.0 + self.beta) / gamma(1.0 + self.beta / self.alpha)
        inverse_stable_x = np.power(x, self.alpha / self.gamma)
        jacobian = (
            abs(self.alpha / self.gamma)
            * np.power(x, self.alpha / self.gamma - 1.0)
        )
        tilt = np.power(inverse_stable_x, self.beta / self.alpha)
        return (
            c
            * tilt
            * jacobian
            * inverse_stable_pdf(inverse_stable_x, self.alpha)
        )

    def negative_moment(self, q):
        """Return E[T_{alpha,beta}^{-q}] for the underlying tilted stable law."""
        return (
            gamma(1.0 + self.beta)
            / gamma(1.0 + self.beta / self.alpha)
            * gamma(1.0 + (self.beta + q) / self.alpha)
            / gamma(1.0 + self.beta + q)
        )

    def moment(self, order):
        """Return E[X**order] for X = T_{alpha,beta}^{-gamma}."""
        return self.negative_moment(order * self.gamma)

    def mean(self):
        """Return the expected value of X = T_{alpha,beta}^{-gamma}."""
        return self.moment(1)

    def variance(self):
        """Return the variance of X = T_{alpha,beta}^{-gamma}."""
        return self.moment(2) - self.mean() ** 2

    def std(self):
        """Return the standard deviation of X = T_{alpha,beta}^{-gamma}."""
        return np.sqrt(self.variance())


class TitledStable(TitledStable3):
    # X = T_{alpha,beta}, the two-parameter base of the tilted stable law
    # created for validation, e.g. the PDF formula mentioned in the Introduction
    def __init__(self, alpha, beta, grid_size=TILT_GRID_SIZE, rng=None):
        super().__init__(alpha, beta, gamma=-1.0, grid_size=grid_size, rng=rng)

    def pdf(self, x):
        c = gamma(1 + self.beta) / gamma(1 + self.beta/self.alpha)
        return c * np.power(x, -self.beta) * one_sided_stable_pdf(x, self.alpha)


# NOT lru_cached: these objects carry an rng, so caching one would pin a single random
# stream for the life of the process and silently ignore a caller's seed. The costly part,
# the inverse-CDF grid, is cached in get_tilted_kanter_grid instead.
def get_tilted_stable2(alpha, beta, rng=None):
    return TitledStable2(alpha, beta, rng=rng)


def get_tilted_stable3(alpha, beta, gamma, rng=None):
    return TitledStable3(alpha, beta, gamma, rng=rng)


# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
class PitmanYorRestaurant(TitledStable3):
    def __init__(self, alpha: float, beta: float, gamma: float, num_customers: int, rng=None):
        # T_{alpha,beta}^{-gamma} == T_{discount,strength}^{-power}. Hence,
        #    alpha is discount
        #    beta is strength
        #    gamma is power
        super().__init__(alpha, beta, gamma=gamma, grid_size=TILT_GRID_SIZE, rng=rng)

        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if int(num_customers) != num_customers or num_customers < 1:
            raise ValueError("num_customers must be a positive integer")
        self.num_customers: int = int(num_customers)

    def new_table_probability(self, table_counts, customers):
        """Return P(K_{n+1}=K_n+1 | K_n) for the Pitman--Yor process.

        ``customers`` is the current restaurant size n, before the next
        customer is seated. Inputs follow NumPy broadcasting rules.
        """
        return (self.beta + self.alpha * table_counts) / (self.beta + customers)

    def alpha_diversity_estimate(self, table_counts, customers=None):
        """Return the finite-n estimate K_n / n**alpha at the terminal customer count.
        This approaches T_{alpha,beta}^{-alpha}.

        ``customers`` is default to the terminal number of customers.
        But if you want to observe transient effect, you can set it to the current restaurant size n.
        """
        if customers is None:
            customers = float(self.num_customers)
        assert customers is not None
        return table_counts / customers ** self.alpha

    def tilted_stable_estimate(self, table_counts, customers=None):
        """Return the finite-n estimate of T_{alpha,beta}^{-gamma}."""
        diversity = self.alpha_diversity_estimate(table_counts, customers=customers)
        return diversity ** (self.gamma / self.alpha)

    def pitman_yor_path(self, num_checkpoints: Optional[int]):
        """Simulate one restaurant and return checkpoint estimates as a DataFrame."""
        return self.pitman_yor_paths(size=1, num_checkpoints=num_checkpoints).drop(
            columns="path"
        )

    def pitman_yor_paths(self, size, num_checkpoints: Optional[int]):
        """Simulate multiple restaurants together and return their checkpoint estimates.

        Vectorizing the table counts across ``size`` paths avoids running a
        separate Python customer loop for every path.
        Checkpoints are geometrically spaced from customer 100 (or the terminal
        customer count, when smaller) through ``self.num_customers``. Passing
        ``None`` records every customer on every path and can require substantial
        memory.
        This method advances ``self.rng``.
        """
        if int(size) != size or size < 1:
            raise ValueError("size must be a positive integer")
        size = int(size)

        if num_checkpoints is None:
            checkpoints = np.arange(1, self.num_customers + 1, dtype=np.int64)
        elif int(num_checkpoints) != num_checkpoints or num_checkpoints < 1:
            raise ValueError("num_checkpoints must be a positive integer or None")
        elif num_checkpoints == 1:
            checkpoints = np.array([self.num_customers])
        else:
            num_checkpoints = int(num_checkpoints)
            first_checkpoint = min(100, self.num_customers)
            checkpoints = np.unique(
                np.geomspace(first_checkpoint, self.num_customers, num_checkpoints).astype(int)
            )

        checkpoint_count = checkpoints.size
        checkpoint_table_counts = np.empty((size, checkpoint_count), dtype=np.int64)
        checkpoint_estimates = np.empty((size, checkpoint_count), dtype=float)
        checkpoint_index = 0
        table_counts = np.ones(size, dtype=np.int64)
        if checkpoints[0] == 1:
            checkpoint_table_counts[:, 0] = table_counts
            checkpoint_estimates[:, 0] = self.tilted_stable_estimate(
                table_counts, customers=1
            )
            checkpoint_index = 1

        for customer in range(1, self.num_customers):
            # the following three lines are the core logic
            p_new = self.new_table_probability(table_counts, customer)
            table_counts += self.rng.random(size) < p_new
            customers = customer + 1
            # save details only at checkpoints, to save space
            if checkpoint_index < checkpoint_count and customers == checkpoints[checkpoint_index]:
                checkpoint_table_counts[:, checkpoint_index] = table_counts
                checkpoint_estimates[:, checkpoint_index] = self.tilted_stable_estimate(
                    table_counts, customers=customers
                )
                checkpoint_index += 1

        return pd.DataFrame(
            {
                "customers": np.tile(checkpoints, size),
                "tables": checkpoint_table_counts.reshape(-1),
                "tilted-stable estimate": checkpoint_estimates.reshape(-1),
                "path": np.repeat(np.arange(1, size + 1), checkpoint_count),
            },
            copy=False,
        )

    def pitman_yor_rvs(self, size):
        """Approximate ``T_{alpha,beta}^{-gamma}`` using Pitman--Yor table counts."""
        if int(size) != size or size < 1:
            raise ValueError("size must be a positive integer")

        # table_counts and samples are array(size)
        # both are returned to allow maximum usage flexiblity
        table_counts = np.ones(int(size), dtype=np.int64)

        # `customer` is the current restaurant size.  The update seats customer+1.
        for customer in range(1, self.num_customers):
            p_new = self.new_table_probability(table_counts, customer)
            table_counts += self.rng.random(table_counts.size) < p_new

        samples = self.tilted_stable_estimate(table_counts)
        return samples, table_counts

    def stick_breaking_rvs(self, size, num_breaks, return_residuals: bool = False):
        """Approximate ``T_{alpha,beta}^{-gamma}`` from Pitman--Yor stick residuals.

        After ``m`` GEM(alpha, beta) breaks, let ``R_m`` be the unallocated
        stick mass. The alpha-diversity identity

        ``m**((1-alpha)/alpha) * R_m -> alpha * T**(-1)``

        gives ``T**(-gamma)`` by raising the scaled residual to ``gamma``.
        Computation stays in log space so large ``num_breaks`` values do not
        underflow. Set ``return_residuals`` to also return ``R_m`` for
        convergence diagnostics.
        """
        if int(size) != size or size < 1:
            raise ValueError("size must be a positive integer")
        if int(num_breaks) != num_breaks or num_breaks < 1:
            raise ValueError("num_breaks must be a positive integer")
        size = int(size)
        num_breaks = int(num_breaks)

        log_residuals = np.zeros(size)
        smallest_positive = np.nextafter(0.0, 1.0)
        for break_number in range(1, num_breaks + 1):
            # If V_j ~ Beta(1-alpha, beta+j*alpha), then 1-V_j has
            # Beta(beta+j*alpha, 1-alpha) law. Drawing the remainder directly
            # avoids loss of precision from subtracting a break near one.
            residual_fraction = self.rng.beta(
                self.beta + break_number * self.alpha,
                1.0 - self.alpha,
                size=size,
            )
            log_residuals += np.log(np.maximum(residual_fraction, smallest_positive))

        residual_power = (1.0 - self.alpha) / self.alpha
        log_scaled_residuals = (
            residual_power * np.log(num_breaks)
            + log_residuals
            - np.log(self.alpha)
        )
        samples = np.exp(self.gamma * log_scaled_residuals)

        if return_residuals:
            return samples, np.exp(log_residuals)
        return samples

    def ks_kanter_crp(self, size, return_samples: bool = False):
        """Compare the Pitman--Yor CRP and Kanter samplers with a two-sample KS test.

        ``size`` is the number of paths/samples generated by each method.
        Return both sample arrays when they are needed for further diagnostics.
        """
        py_samples, _ = self.pitman_yor_rvs(size=size)
        kanter_samples = self.rvs(size)  # this is TitledStable3's rvs
        ks = ks_2samp(py_samples, kanter_samples)
        if return_samples:
            return ks, py_samples, kanter_samples
        return ks

    def ks_kanter_stick(self, size, num_breaks, return_samples: bool = False):
        """Compare the stick-breaking and Kanter samplers with a two-sample KS test.

        ``size`` is the number of samples generated by each method, and
        ``num_breaks`` is the finite GEM truncation used by stick breaking.
        Return both sample arrays when they are needed for further diagnostics.
        """
        stick_samples = self.stick_breaking_rvs(
            size=size,
            num_breaks=num_breaks,
        )
        kanter_samples = self.rvs(size=size)
        ks = ks_2samp(stick_samples, kanter_samples)
        if return_samples:
            return ks, stick_samples, kanter_samples
        return ks

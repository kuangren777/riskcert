"""RiskCert statistical core.

Per-cell confidence sequences for a Bernoulli mean by betting (hedged capital process,
Waudby-Smith & Ramdas, JRSS-B 2024), evaluated on a grid of candidate means. A candidate m stays
in the confidence set while the hedged capital  theta*K+(m) + (1-theta)*K-(m) < 1/alpha
(the sum; THEORY.md Remark 1.3: a max with adaptively chosen side would cost 2*alpha).
Any predictable bet sizes lambda_t and any fixed hedge theta keep the set anytime-valid, which is
what lets a prior (e.g. the previous model generation's table) steer theta without breaking
coverage.

`interval` returns the hull of every grid cell [g_i, g_{i+1}] that cannot be excluded, using the
monotonicity of K+ (nonincreasing in m) and K- (nondecreasing), so the returned interval contains
every candidate mean in the continuous set, not only grid points (THEORY.md Remark 1.4).

Two bet schedules: `BettingCS` (WSR-style predictable plug-in, THEORY.md Conj 2.4) and `RCMixCS`
(fixed-weight mixture over constant bets c0*2^-g, the schedule Theorem 2 is proved for).

Decisions over K cells use a Bonferroni split alpha/K, so all K sets hold simultaneously with
probability >= 1-alpha at every time and under any predictable allocation.
"""
import math

import numpy as np

GRID = np.linspace(0.0, 1.0, 1001)


class BettingCS:
    def __init__(self, alpha: float, theta=0.5, c: float = 0.5, grid=GRID):
        self.alpha, self.c, self.grid = alpha, c, grid
        self.theta = np.broadcast_to(np.asarray(theta, float), grid.shape).copy()
        self.logk_plus = np.zeros_like(grid)
        self.logk_minus = np.zeros_like(grid)
        self.n, self.s, self.ss = 0, 0.0, 0.0  # count, sum, sum of squares of residuals

    def _lam(self) -> float:
        # predictable plug-in bet size (WSR eq. 26 style), uses only past data
        t = self.n + 1
        var = (0.25 + self.ss) / t
        return math.sqrt(2 * math.log(2 / self.alpha) / (var * t * math.log(t + 1)))

    def update(self, x: float):
        lam = self._lam()
        m = self.grid
        lp = np.minimum(lam, self.c / np.maximum(m, 1e-12))
        lm = np.minimum(lam, self.c / np.maximum(1 - m, 1e-12))
        with np.errstate(divide="ignore"):
            self.logk_plus += np.log1p(lp * (x - m))
            self.logk_minus += np.log1p(-lm * (x - m))
        mu_prev = (0.5 + self.s) / (self.n + 1)
        self.n += 1
        self.s += x
        self.ss += (x - mu_prev) ** 2

    def _logk(self):
        """(log K+, log K-) on the grid."""
        return self.logk_plus, self.logk_minus

    def inside(self) -> np.ndarray:
        """Grid points m with hedged capital K(m) < 1/alpha."""
        lkp, lkm = self._logk()
        thr = math.log(1 / self.alpha)
        with np.errstate(divide="ignore"):
            a = np.log(np.maximum(self.theta, 1e-300)) + lkp
            b = np.log(np.maximum(1 - self.theta, 1e-300)) + lkm
        return np.logaddexp(a, b) < thr

    def cells_kept(self) -> np.ndarray:
        """Grid cell i = [g_i, g_{i+1}] is kept unless a lower bound of K on the whole cell,
        min(theta)*K+(g_{i+1}) + min(1-theta)*K-(g_i), reaches 1/alpha (theta is taken as the
        left-continuous step function of its grid values)."""
        lkp, lkm = self._logk()
        thr = math.log(1 / self.alpha)
        th_lo = np.minimum(self.theta[:-1], self.theta[1:])
        om_lo = np.minimum(1 - self.theta[:-1], 1 - self.theta[1:])
        with np.errstate(divide="ignore"):
            a = np.log(np.maximum(th_lo, 1e-300)) + lkp[1:]
            b = np.log(np.maximum(om_lo, 1e-300)) + lkm[:-1]
        return np.logaddexp(a, b) < thr

    def interval(self):
        kept = self.cells_kept()
        if not kept.any():  # empty only on the failure event G^c: collapse to the running mean
            mu = self.s / max(self.n, 1)
            return mu, mu
        idx = np.flatnonzero(kept)
        return float(self.grid[idx[0]]), float(self.grid[idx[-1] + 1])


class RCMixCS(BettingCS):
    """RC-mix (THEORY.md §2): K±(m) = sum_g w_g prod_i (1 ± lam_g (X_i - m)), lam_g = c0 2^-g,
    w_g = 1/((g+1)(g+2)), truncated at g_max (truncation only lowers capital, validity kept).
    With c0 = 1/2 no cap c0/m binds, every factor is >= 1/2. Theorem 2 needs g_max >= log2(4/Delta)."""

    def __init__(self, alpha: float, theta=0.5, c: float = 0.5, grid=GRID, g_max: int = 12):
        super().__init__(alpha, theta=theta, c=c, grid=grid)
        g = np.arange(g_max + 1)
        self.lams = c * 2.0 ** (-g)
        self.logw = -np.log((g + 1.0) * (g + 2.0))
        self.lp_g = np.zeros((g_max + 1, grid.size))
        self.lm_g = np.zeros((g_max + 1, grid.size))

    def update(self, x: float):
        d = (x - self.grid)[None, :]
        lam = self.lams[:, None]
        self.lp_g += np.log1p(lam * d)
        self.lm_g += np.log1p(-lam * d)
        self.n += 1
        self.s += x

    def _logk(self):
        w = self.logw[:, None]
        return (np.logaddexp.reduce(w + self.lp_g, axis=0), np.logaddexp.reduce(w + self.lm_g, axis=0))


def threshold_decision(lo: float, hi: float, tau: float):
    """'above' / 'below' when the interval excludes tau, else None."""
    if lo > tau:
        return "above"
    if hi < tau:
        return "below"
    return None


def prior_theta(prior_p: float, strength: float = 0.8, grid=GRID):
    """Hedge weight per candidate m: bet more on 'p > m' where the prior says p > m."""
    return np.where(grid < prior_p, strength, 1 - strength)


# ---------------------------------------------------------------- sequential procedures on pools

def riskcert_threshold(pools, tau, alpha, n_max, rng, thetas=None, cs_cls=BettingCS):
    """Sample each cell until its CS (alpha/K) excludes tau or n_max is hit.
    pools: list of 0/1 arrays (real outcomes), sampled with replacement. Returns (n_used, decisions)."""
    k = len(pools)
    out_n, out_d = [], []
    for j, pool in enumerate(pools):
        cs = cs_cls(alpha / k, theta=0.5 if thetas is None else thetas[j])
        d = None
        while cs.n < n_max:
            cs.update(float(pool[rng.integers(len(pool))]))
            d = threshold_decision(*cs.interval(), tau)
            if d:
                break
        out_n.append(cs.n)
        out_d.append(d)
    return out_n, out_d


def wald_sprt(pool, p0, p1, alpha, beta, n_max, rng):
    """Per-cell Wald SPRT H0: p<=p0 vs H1: p>=p1 (AgentAssay-style). Returns (n, 'above'/'below'/None)."""
    la, lb = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    w1, w0 = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    llr, n = 0.0, 0
    while n < n_max:
        x = pool[rng.integers(len(pool))]
        n += 1
        llr += w1 if x else w0
        if llr >= la:
            return n, "above"
        if llr <= lb:
            return n, "below"
    return n, None


def fixed_n(p0, p1, alpha, beta, n_cap=2000):
    """Smallest n with critical count c: P(K>=c|p0)<=alpha and P(K<c|p1)<=beta (exact binomial)."""
    from scipy.stats import binom
    for n in range(1, n_cap):
        c = int(binom.isf(alpha, n, p0)) + 1  # smallest c with P(K>=c|p0) <= alpha
        if binom.sf(c - 1, n, p0) <= alpha and binom.cdf(c - 1, n, p1) <= beta:
            return n, c
    raise ValueError("no n found")


# ---------------------------------------------------------------- R1 paired contrasts, R2 zone / three-way rules

class ContrastCS:
    """R1: CS for delta = p_a - p_b from paired outcomes on a shared unit (THEORY_v2 Thm 1-P).
    Runs a [0,1] betting CS on (X_a - X_b + 1)/2 and maps the interval back to [-1, 1]."""

    def __init__(self, alpha: float, cls=RCMixCS, **kw):
        self.cs = cls(alpha, **kw)

    @property
    def n(self):
        return self.cs.n

    def update(self, xa: float, xb: float):
        self.cs.update((xa - xb + 1) / 2)

    def interval(self):
        lo, hi = self.cs.interval()
        return 2 * lo - 1, 2 * hi - 1


def sign_decision(lo: float, hi: float, eps: float = 0.0):
    """'>' / '<' when the contrast interval excludes 0; 'tie' when it lies inside (-eps, eps) (R2 futility)."""
    if lo > 0:
        return ">"
    if hi < 0:
        return "<"
    if eps > 0 and -eps < lo and hi < eps:
        return "tie"
    return None


def zone_decision(lo: float, hi: float, p0: float, p1: float):
    """SPRT contract (THEORY_v2 Prop 4-Z): 'below' (p < p1) when sup C < p1, 'above' (p > p0) when inf C > p0.
    Wrong only if it says 'below' while p >= p1 or 'above' while p <= p0."""
    if hi < p1:
        return "below"
    if lo > p0:
        return "above"
    return None


def three_way(lo: float, hi: float, tau: float, eps: float):
    """R2: 'above' / 'below' tau, or 'near' when the interval lies inside (tau - eps, tau + eps)."""
    d = threshold_decision(lo, hi, tau)
    if d:
        return d
    if tau - eps < lo and hi < tau + eps:
        return "near"
    return None


def paired_sprt_step(llr: float, xa: int, xb: int, eta: float):
    """Wald SPRT on discordant pairs (McNemar form): q = P(X_a=1 | X_a != X_b), H0: q <= 1/2 - eta vs
    H1: q >= 1/2 + eta. Concordant pairs leave the LLR unchanged."""
    if xa == xb:
        return llr
    q0, q1 = 0.5 - eta, 0.5 + eta
    return llr + (math.log(q1 / q0) if xa > xb else math.log((1 - q1) / (1 - q0)))


class SignCS:
    """R1' (THEORY_v2 Lemma D): CS for q = P(X_a = 1 | X_a != X_b) from discordant pairs only.
    sign(delta) = sign(q - 1/2) because delta = P(X_a != X_b) * (2q - 1). Concordant pairs multiply the capital by 1,
    so the process stays a nonnegative martingale on the global clock; the sign decision needs no estimate of
    P(X_a != X_b). Default bet = plug-in BettingCS (bets near m = 1/2 may reach 1; RC-mix caps at 1/2)."""

    def __init__(self, alpha: float, cls=BettingCS, **kw):
        self.cs = cls(alpha, **kw)
        self.n = 0  # pairs seen, discordant or not (cost accounting)

    def update(self, xa: float, xb: float):
        self.n += 1
        if xa != xb:
            self.cs.update(float(xa > xb))

    def interval(self):
        """Interval on delta's sign scale: (2q_lo - 1, 2q_hi - 1); its sign equals sign(delta)."""
        lo, hi = self.cs.interval()
        return 2 * lo - 1, 2 * hi - 1

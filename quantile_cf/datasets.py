"""Simulated data with a known counterfactual truth."""

import numpy as np
import torch


def normal_cdf(e):
    """Standard normal CDF: the true quantile level of noise ``e ~ N(0, 1)`` when Y increases in E."""
    return torch.special.ndtr(torch.as_tensor(e, dtype=torch.float64)).numpy()


class SyntheticSCM:
    """Five structural causal models with treatment X ~ U[0, 1], covariate Z ~ U[0, 1] and noise E ~ N(0, 1).

    ====  =========================  =======================================
    case  name                       outcome
    ====  =========================  =======================================
    1     linear                     Y = X + Z + E
    2     nonlinear, additive        Y = sin(2 pi X + Z) + E
    3     nonlinear, multiplicative  Y = exp(X - Z + 0.5) E
    4     post-nonlinear             Y = exp(sin(pi X + Z) + E)
    5     heteroscedastic            Y = exp(-5 X + Z) + exp(X + Z - 0.5) E
    ====  =========================  =======================================

    Y increases in E in every case, so a sample's true quantile level is ``normal_cdf(e)`` and
    its counterfactual outcome under X = x' is ``outcome(x', z, e)``.
    """

    names = {1: "linear", 2: "nonlinear additive", 3: "nonlinear multiplicative",
             4: "post-nonlinear", 5: "heteroscedastic"}

    def __init__(self, case=2):
        if case not in self.names:
            raise ValueError("case must be 1, 2, 3, 4 or 5")
        self.case = case
        self.name = self.names[case]

    def outcome(self, x, z, e):
        """Y for treatment ``x``, covariate ``z`` and noise ``e`` (numpy arrays, broadcast)."""
        x, z, e = (np.asarray(a, dtype=np.float64) for a in (x, z, e))
        if self.case == 1:
            return x + z + e
        if self.case == 2:
            return np.sin(2 * np.pi * x + z) + e
        if self.case == 3:
            return np.exp(x - z + 0.5) * e
        if self.case == 4:
            return np.exp(np.sin(np.pi * x + z) + e)
        return np.exp(-5 * x + z) + np.exp(x + z - 0.5) * e

    def sample(self, n, seed=None):
        """``n`` observational samples: a dict of ``(n, 1)`` arrays ``x``, ``z``, ``e`` and ``y``."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(size=(n, 1))
        z = rng.uniform(size=(n, 1))
        e = rng.normal(size=(n, 1))
        return {"x": x, "z": z, "e": e, "y": self.outcome(x, z, e)}

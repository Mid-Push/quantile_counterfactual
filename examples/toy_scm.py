"""Counterfactual outcomes in a simulated structural causal model with a known answer.

    python examples/toy_scm.py --case 3

Trains on 100,000 observational samples of ``SyntheticSCM(case)`` with the default settings, then
checks against the truth: the quantile levels and counterfactual outcomes of 2,000 new individuals,
and the outcome under every treatment of one individual with X = 0.5, Z = 0.5 and noise E = 0.5
(true quantile level Phi(0.5) = 0.691). A GPU makes this much faster.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from quantile_cf import MLPOutcomeNet, MLPQuantileNet, QuantileCF
from quantile_cf.datasets import SyntheticSCM, normal_cdf


def main():
    parser = argparse.ArgumentParser(description="Counterfactual prediction on a simulated SCM.")
    parser.add_argument("--case", type=int, default=3, choices=range(1, 6), help="which of the five SCMs")
    parser.add_argument("--n", type=int, default=100_000, help="number of observational samples")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="toy_scm.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    scm = SyntheticSCM(args.case)
    data = scm.sample(args.n, seed=args.seed)
    print(f"SCM {args.case} ({scm.name}), {args.n} observational samples")

    model = QuantileCF(MLPQuantileNet(x_dim=1, y_dim=1, z_dim=1), MLPOutcomeNet(x_dim=1, y_dim=1, z_dim=1),
                       device=args.device)
    model.fit(data["x"], data["y"], data["z"], steps=args.steps, log_every=2000)

    # 2,000 new individuals, each under a new random treatment
    test = scm.sample(2000, seed=args.seed + 1)
    tau = normal_cdf(test["e"])
    tau_hat = model.quantile(test["x"], test["y"], test["z"]).numpy()
    x_cf = np.random.default_rng(args.seed + 2).uniform(size=test["x"].shape)
    y_cf = scm.outcome(x_cf, test["z"], test["e"])
    y_cf_hat = model.counterfactual(test["x"], test["y"], x_cf, test["z"]).numpy()
    error = np.abs(tau_hat - tau)
    print(f"quantile level: mean absolute error {error.mean():.4f}, 90% below {np.percentile(error, 90):.4f}")
    print(f"counterfactual outcome: RMSE {np.sqrt(np.mean((y_cf_hat - y_cf) ** 2)):.4f} "
          f"(standard deviation of the truth {y_cf.std():.4f})")

    # one individual under every treatment
    x0 = z0 = e0 = 0.5
    y0 = float(scm.outcome(x0, z0, e0))
    tau0 = float(model.quantile([[x0]], [[y0]], [[z0]]))
    grid = np.linspace(0, 1, 200).reshape(-1, 1)
    ones = np.ones_like(grid)
    curve = model.counterfactual(x0 * ones, y0 * ones, grid, z0 * ones).numpy()
    truth = scm.outcome(grid, z0, e0)
    print(f"individual (x, z, e) = (0.5, 0.5, 0.5): learned quantile level {tau0:.4f}, "
          f"true {float(normal_cdf(e0)):.4f}; outcome under x' in [0, 1]: RMSE {np.sqrt(np.mean((curve - truth) ** 2)):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(tau, tau_hat, s=3, alpha=0.4)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="true quantile level", ylabel="learned quantile level", title="2,000 new individuals")
    axes[1].plot(grid, truth, lw=2.5, alpha=0.6, label="truth")
    axes[1].plot(grid, curve, "k--", label=f"predicted (learned quantile level {tau0:.3f})")
    axes[1].scatter([x0], [y0], color="C1", edgecolor="k", zorder=3, label="observed")
    axes[1].set(xlabel="treatment x'", ylabel="outcome", title="one individual under every treatment")
    axes[1].legend()
    fig.suptitle(f"SCM {args.case}: {scm.name}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()

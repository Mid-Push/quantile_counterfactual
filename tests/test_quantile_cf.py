import numpy as np
import torch
from torch.utils.data import TensorDataset

from quantile_cf import ConvOutcomeNet, ConvQuantileNet, MLPOutcomeNet, MLPQuantileNet, QuantileCF, pinball_loss
from quantile_cf.datasets import SyntheticSCM, normal_cdf

SMALL = dict(batch_size=32, num_quantiles=4, query_batch_size=8, inner_steps=5, device="cpu")


def test_pinball_loss():
    residual = torch.tensor([[2.0], [-2.0]])
    tau = torch.tensor([[0.9], [0.9]])
    assert torch.allclose(pinball_loss(residual, tau), torch.tensor([[1.8], [0.2]]))


def test_shapes_with_covariates():
    data = SyntheticSCM(1).sample(300, seed=0)
    model = QuantileCF(MLPQuantileNet(1, 1, z_dim=1, hidden_dim=16), MLPOutcomeNet(1, 1, z_dim=1, hidden_dim=16), **SMALL)
    model.fit(data["x"], data["y"], data["z"], steps=12, log_every=0)
    tau = model.quantile(data["x"], data["y"], data["z"])
    assert tau.shape == (300, 1) and bool(((tau > 0) & (tau < 1)).all())
    assert model.counterfactual(data["x"], data["y"], 1 - data["x"], data["z"]).shape == (300, 1)


def test_no_covariates_and_two_arms():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=200).astype(float)
    y = 2 * x + rng.normal(size=200)
    model = QuantileCF(MLPQuantileNet(1, 1, hidden_dim=16, num_arms=2), MLPOutcomeNet(1, 1, hidden_dim=16, num_arms=2), **SMALL)
    model.fit(x, y, steps=12, log_every=0)
    assert model.counterfactual(x, y, 1 - x).shape == (200, 1)


def test_dataset_input_and_query():
    data = SyntheticSCM(2).sample(200, seed=1)
    dataset = TensorDataset(*(torch.tensor(data[k]).float() for k in ("x", "y", "z")))
    model = QuantileCF(MLPQuantileNet(1, 1, z_dim=1, hidden_dim=16), MLPOutcomeNet(1, 1, z_dim=1, hidden_dim=16), **SMALL)
    model.fit(dataset=dataset, query=(data["x"][:10], data["y"][:10], data["z"][:10]), steps=12, log_every=0)
    assert model.quantile(data["x"][:10], data["y"][:10], data["z"][:10]).shape == (10, 1)


def test_images():
    z, y = torch.rand(40, 1, 16, 16), torch.rand(40, 1, 16, 16)
    x = torch.rand(40, 1)
    model = QuantileCF(ConvQuantileNet(16, z_channels=1, y_channels=1, hidden_dim=8),
                       ConvOutcomeNet(16, z_channels=1, y_channels=1, hidden_dim=8), **SMALL)
    model.fit(x, y, z, steps=12, log_every=0)
    assert model.counterfactual(x, y, -x, z).shape == (40, 1, 16, 16)


def test_save_and_load(tmp_path):
    data = SyntheticSCM(1).sample(100, seed=2)

    def build():
        torch.manual_seed(0)
        return QuantileCF(MLPQuantileNet(1, 1, z_dim=1, hidden_dim=16), MLPOutcomeNet(1, 1, z_dim=1, hidden_dim=16), **SMALL)

    model = build()
    model.fit(data["x"], data["y"], data["z"], steps=12, log_every=0)
    model.save(tmp_path / "model.pt")
    other = build().load(tmp_path / "model.pt")
    assert torch.equal(model.quantile(data["x"], data["y"], data["z"]), other.quantile(data["x"], data["y"], data["z"]))


def test_recovers_quantile_level_of_one_individual():
    torch.manual_seed(0)
    scm = SyntheticSCM(1)
    data = scm.sample(20000, seed=0)
    y0 = float(scm.outcome(0.5, 0.5, 0.5))
    individual = ([[0.5]], [[y0]], [[0.5]])
    model = QuantileCF(MLPQuantileNet(1, 1, z_dim=1, hidden_dim=32), MLPOutcomeNet(1, 1, z_dim=1, hidden_dim=32),
                       batch_size=512, num_quantiles=1, query_batch_size=1, lr_decay_every=1000, device="cpu")
    model.fit(data["x"], data["y"], data["z"], steps=3000, query=individual, log_every=0)
    assert abs(float(model.quantile(*individual)) - float(normal_cdf(0.5))) < 0.02

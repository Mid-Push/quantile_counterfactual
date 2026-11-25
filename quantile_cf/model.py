"""Counterfactual prediction by bi-level quantile regression.

Two networks are trained together:

* the outcome network ``g(x, z, tau)`` is a conditional quantile function of ``y``. It is fitted
  with the pinball loss at the quantile levels produced by the quantile network (lower level);
* the quantile network ``h(x, z, y)`` gives the quantile level of a factual outcome. It is
  trained so that ``g(x, z, h(x, z, y))`` reconstructs ``y`` (upper level); its gradient also
  accounts for how ``g`` responds, approximated with SAMA (Choe et al., 2023).

If ``y`` is strictly monotonic in the (transformed) noise, a sample keeps its quantile level
under an intervention, so the counterfactual outcome had ``X`` been ``x'`` is
``g(x', z, h(x, z, y))``.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sama import precondition


def pinball_loss(residual, tau):
    """Elementwise pinball (quantile) loss of ``residual = y - prediction`` at level ``tau``."""
    return torch.where(residual >= 0, tau * residual, (tau - 1) * residual)


def _as_tensor(a):
    """Float tensor with one row per sample (a 1-d input becomes a column)."""
    if a is None:
        return None
    t = torch.as_tensor(a).float()
    return t.reshape(-1, 1) if t.dim() == 1 else t


def _as_triple(data):
    """(x, y, z) from a tuple/list of two or three arrays or a dict with keys x, y and z."""
    if isinstance(data, dict):
        data = (data["x"], data["y"], data.get("z"))
    data = tuple(data) + (None,) * (3 - len(data))
    return tuple(_as_tensor(t) for t in data)


def _repeat(t, k):
    """k stacked copies of a batch: row ``i * len(t) + j`` is sample ``j``."""
    return None if t is None else t.repeat(k, *([1] * (t.dim() - 1)))


def _grad(loss, params, retain_graph=False):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


class _Batches:
    """Endless shuffled mini-batches of (x, y, z) from tensors or from a Dataset.

    Each pass over the data uses a new order; the last batch of a pass may be smaller.
    """

    def __init__(self, source, batch_size, device, num_workers=0):
        self.device = device
        self.batch_size = batch_size
        if isinstance(source, tuple):
            self.tensors = source
            self.size = len(source[0])
            self.order, self.pos = None, self.size
            self.loader = None
        else:
            self.loader = DataLoader(source, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.loader is None:
            if self.pos >= self.size:
                self.order, self.pos = torch.randperm(self.size).to(self.device), 0
            idx = self.order[self.pos:self.pos + self.batch_size]
            self.pos += len(idx)
            return tuple(None if t is None else t[idx] for t in self.tensors)
        batch = next(self.iterator, None) if self.iterator is not None else None
        if batch is None:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return tuple(None if t is None else t.to(self.device) for t in _as_triple(batch))


class QuantileCF:
    """Counterfactual prediction by bi-level quantile regression.

    Args:
        quantile_net: module ``h(x, z, y) -> tau`` with values in (0, 1).
        outcome_net: module ``g(x, z, tau) -> y_hat`` with the shape of ``y``.
        lr: Adam learning rate of both networks.
        weight_decay: Adam weight decay of both networks.
        inner_steps: outcome-network updates per quantile-network update.
        batch_size: samples in each outcome-network update.
        num_quantiles: factual samples whose quantile levels are used in each outcome-network
            update; the outcome network is fitted on the whole batch at each of these levels.
        query_batch_size: samples in each quantile-network update.
        hypergradient: ``"sama"`` (as in the paper) or ``"none"`` (only the direct gradient of
            the quantile network, ignoring how the outcome network responds).
        sama_alpha: length of the finite-difference step in the outcome network's parameters.
        lr_decay_every: multiply both learning rates by ``lr_decay_rate`` every so many steps
            (``None`` or 0 to keep them fixed).
        lr_decay_rate: see ``lr_decay_every``.
        upper_loss: ``"mse"``, ``"l1"`` or a function ``(y_hat, y) -> loss`` for the upper level.
        device: torch device; defaults to CUDA when available.

    The networks are called as ``quantile_net(x, z, y)`` and ``outcome_net(x, z, tau)``, with
    ``z=None`` when there are no covariates.
    """

    def __init__(self, quantile_net, outcome_net, lr=1e-3, weight_decay=0.0, inner_steps=30,
                 batch_size=256, num_quantiles=64, query_batch_size=64, hypergradient="sama",
                 sama_alpha=1.0, lr_decay_every=3000, lr_decay_rate=0.5, upper_loss="mse", device=None):
        if hypergradient not in ("sama", "none"):
            raise ValueError('hypergradient must be "sama" or "none"')
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.quantile_net = quantile_net.to(self.device)
        self.outcome_net = outcome_net.to(self.device)
        self.quantile_optimizer = torch.optim.Adam(self.quantile_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.outcome_optimizer = torch.optim.Adam(self.outcome_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.inner_steps = inner_steps
        self.batch_size = batch_size
        self.num_quantiles = num_quantiles
        self.query_batch_size = query_batch_size
        self.hypergradient = hypergradient
        self.sama_alpha = sama_alpha
        self.lr_decay_every = lr_decay_every
        self.lr_decay_rate = lr_decay_rate
        if upper_loss == "mse":
            upper_loss = F.mse_loss
        elif upper_loss == "l1":
            upper_loss = F.l1_loss
        self.upper_loss = upper_loss
        self.step = 0
        self.history = {"step": [], "lower_loss": [], "upper_loss": []}

    def __repr__(self):
        return (f"QuantileCF(quantile_net={type(self.quantile_net).__name__}, "
                f"outcome_net={type(self.outcome_net).__name__}, step={self.step}, device={self.device})")

    # ------------------------------------------------------------------ training

    def fit(self, x=None, y=None, z=None, steps=10000, query=None, dataset=None, query_dataset=None,
            num_workers=0, log_every=1000):
        """Train both networks for ``steps`` outcome-network updates.

        Args:
            x, y, z: factual treatments, outcomes and (optional) covariates, one row per sample
                (numpy arrays or tensors; 1-d arrays are read as one column).
            steps: number of outcome-network updates; the quantile network is updated after
                every ``inner_steps`` of them.
            query: optional ``(x, y)``, ``(x, y, z)`` or dict of factual samples to use, instead of
                the training data, for the quantile levels at the lower level and for the upper level.
            dataset, query_dataset: ``torch.utils.data.Dataset`` objects returning ``(x, y)``,
                ``(x, y, z)`` or dicts with keys ``x``, ``y``, ``z``; use them instead of arrays for
                data made on the fly, such as images.
            num_workers: data-loading workers for datasets.
            log_every: print the losses every so many steps (0 to stay silent).
        """
        if dataset is not None:
            train = dataset
        elif x is not None and y is not None:
            train = tuple(None if t is None else t.to(self.device) for t in _as_triple((x, y, z)))
        else:
            raise ValueError("give x and y (and z if there are covariates), or a dataset")
        if query is not None:
            query = tuple(None if t is None else t.to(self.device) for t in _as_triple(query))
        elif query_dataset is not None:
            query = query_dataset
        else:
            query = train

        lower_batches = _Batches(train, self.batch_size, self.device, num_workers)
        query_batches = _Batches(query, self.num_quantiles, self.device, num_workers)
        upper_batches = _Batches(query, self.query_batch_size, self.device, num_workers)

        self.quantile_net.train()
        self.outcome_net.train()
        lower = [0.0, 0]
        upper = [0.0, 0]
        for _ in range(steps):
            self.step += 1
            batch, levels = next(lower_batches), next(query_batches)
            lower[0] += self._lower_step(batch, levels)
            lower[1] += 1
            if self.step % self.inner_steps == 0:
                upper[0] += self._upper_step(next(upper_batches), batch, query_batches)
                upper[1] += 1
            if self.lr_decay_every and self.step % self.lr_decay_every == 0:
                for optimizer in (self.quantile_optimizer, self.outcome_optimizer):
                    for group in optimizer.param_groups:
                        group["lr"] = max(1e-8, group["lr"] * self.lr_decay_rate)
            if log_every and self.step % log_every == 0:
                lower_loss = lower[0] / lower[1]
                upper_loss = upper[0] / upper[1] if upper[1] else float("nan")
                self.history["step"].append(self.step)
                self.history["lower_loss"].append(lower_loss)
                self.history["upper_loss"].append(upper_loss)
                print(f"step {self.step:6d}  lower (pinball) {lower_loss:.4f}  "
                      f"upper (reconstruction) {upper_loss:.4f}", flush=True)
                lower, upper = [0.0, 0], [0.0, 0]
        return self

    def _lower_loss(self, batch, levels, track_tau):
        """Pinball loss of the outcome network on the batch at every quantile level from ``levels``."""
        x, y, z = batch
        xq, yq, zq = levels
        with torch.set_grad_enabled(track_tau):
            tau = self.quantile_net(xq, zq, yq)
        k, n = len(tau), len(y)
        tau = tau.repeat_interleave(n, dim=0)
        pred = self.outcome_net(_repeat(x, k), _repeat(z, k), tau)
        residual = (_repeat(y, k) - pred).reshape(k * n, -1)
        return pinball_loss(residual, tau.reshape(k * n, -1)).mean()

    def _lower_step(self, batch, levels):
        loss = self._lower_loss(batch, levels, track_tau=False)
        self.outcome_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.outcome_optimizer.step()
        self.outcome_optimizer.zero_grad(set_to_none=True)
        return loss.item()

    def _upper_step(self, batch, lower_batch, query_batches):
        h_params = [p for p in self.quantile_net.parameters() if p.requires_grad]
        g_params = [p for p in self.outcome_net.parameters() if p.requires_grad]
        x, y, z = batch
        loss = self.upper_loss(self.outcome_net(x, z, self.quantile_net(x, z, y)), y)
        grads = _grad(loss, h_params, retain_graph=self.hypergradient == "sama")

        if self.hypergradient == "sama":
            # Implicit part: -(d^2 L_lower / dh dg) v, with v the preconditioned gradient of the
            # upper loss in g, by central differences of the lower loss's gradient in h. Each of
            # the two lower-loss evaluations draws its own quantile levels.
            v = precondition(_grad(loss, g_params), g_params, self.outcome_optimizer)
            eps = self.sama_alpha / torch.cat([t.reshape(-1) for t in v]).norm().add(1e-15).item()
            with torch.no_grad():
                for p, t in zip(g_params, v):
                    p.add_(t, alpha=eps)
            loss_plus = self._lower_loss(lower_batch, next(query_batches), track_tau=True)
            grads = [g - gp / (2 * eps) for g, gp in zip(grads, _grad(loss_plus, h_params))]
            with torch.no_grad():
                for p, t in zip(g_params, v):
                    p.sub_(t, alpha=2 * eps)
            loss_minus = self._lower_loss(lower_batch, next(query_batches), track_tau=True)
            grads = [g + gm for g, gm in zip(grads, _grad(loss_minus / (2 * eps), h_params))]
            with torch.no_grad():
                for p, t in zip(g_params, v):
                    p.add_(t, alpha=eps)

        for p, g in zip(h_params, grads):
            p.grad = g
        self.quantile_optimizer.step()
        self.quantile_optimizer.zero_grad(set_to_none=True)
        return loss.item()

    # ------------------------------------------------------------------ prediction

    @torch.no_grad()
    def _evaluate(self, net, fn, tensors, batch_size):
        was_training = net.training
        net.eval()
        try:
            outputs = []
            for start in range(0, len(tensors[0]), batch_size):
                chunk = [None if t is None else t[start:start + batch_size].to(self.device) for t in tensors]
                outputs.append(fn(*chunk).cpu())
            return torch.cat(outputs)
        finally:
            net.train(was_training)

    def quantile(self, x, y, z=None, batch_size=4096):
        """Quantile level ``tau = h(x, z, y)`` of each factual sample (a CPU tensor)."""
        return self._evaluate(self.quantile_net, lambda x, y, z: self.quantile_net(x, z, y),
                              _as_triple((x, y, z)), batch_size)

    def predict(self, x, tau, z=None, batch_size=4096):
        """Conditional ``tau``-quantile of ``y`` given ``x`` and ``z``: ``g(x, z, tau)`` (a CPU tensor)."""
        return self._evaluate(self.outcome_net, lambda x, tau, z: self.outcome_net(x, z, tau),
                              _as_triple((x, tau, z)), batch_size)

    def counterfactual(self, x, y, x_cf, z=None, batch_size=4096):
        """Outcome each sample would have had under ``X = x_cf``: ``g(x_cf, z, h(x, z, y))``."""
        return self.predict(x_cf, self.quantile(x, y, z, batch_size), z, batch_size)

    # ------------------------------------------------------------------ saving

    def state_dict(self):
        return {
            "quantile_net": self.quantile_net.state_dict(),
            "outcome_net": self.outcome_net.state_dict(),
            "quantile_optimizer": self.quantile_optimizer.state_dict(),
            "outcome_optimizer": self.outcome_optimizer.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state):
        self.quantile_net.load_state_dict(state["quantile_net"])
        self.outcome_net.load_state_dict(state["outcome_net"])
        self.quantile_optimizer.load_state_dict(state["quantile_optimizer"])
        self.outcome_optimizer.load_state_dict(state["outcome_optimizer"])
        self.step = state["step"]
        return self

    def save(self, path):
        """Save both networks and their optimizers."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load what ``save`` wrote into this model (built with the same networks)."""
        return self.load_state_dict(torch.load(path, map_location=self.device))

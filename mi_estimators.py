"""Neural mutual-information estimators for I(X; Y).

Two estimators with a common interface, so they can be swapped at the call site:

    est = CLUB(x_dim, y_dim, hidden, layers, act, norm)
    est = MINE(x_dim, y_dim, hidden, layers, act, norm)

    mi  = est.mi_est(x, y)         # differentiable estimate w.r.t. x and y
    aux = est.learning_loss(x, y)  # fits the estimator's own net, on detached inputs

Alternate the two: `learning_loss` is stepped by a separate optimizer that owns
only this module's parameters; `mi_est` is added to the outer objective.

CLUB is an upper bound and MINE is a lower bound on I(X; Y). Both `mi_est`
methods return larger values for more dependence, but only the upper bound is
valid to minimize as a penalty -- minimizing a lower bound says nothing about the
true MI. Inputs may have any leading shape; they are flattened to (N, dim).
"""

import torch
from torch import nn


def _mlp(in_dim, out_dim, hidden, layers, act, norm, tail=()):
    act = getattr(nn, act) if isinstance(act, str) else act
    norm = getattr(nn, norm) if isinstance(norm, str) else norm
    mods, inp = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(inp, hidden, bias=False), norm(hidden, eps=1e-3), act()]
        inp = hidden
    return nn.Sequential(*mods, nn.Linear(inp, out_dim), *tail)


def _flat(t):
    return t.reshape(-1, t.shape[-1])


class CLUB(nn.Module):
    """Contrastive Log-ratio Upper Bound (Cheng et al., 2020).

    Fits q(y|x) = N(mu(x), sigma(x)^2) and returns

        I(X; Y) <= E_i[log q(y_i|x_i)] - E_ij[log q(y_j|x_i)]

    valid as an upper bound only once q approximates p(y|x).
    """

    def __init__(self, x_dim, y_dim, hidden=256, layers=2, act="SiLU",
                 norm="LayerNorm"):
        super(CLUB, self).__init__()
        self._mu = _mlp(x_dim, y_dim, hidden, layers, act, norm)
        # Tanh-bounded: an unbounded logvar can shrink without limit and the
        # negative-pair term blows up.
        self._logvar = _mlp(x_dim, y_dim, hidden, layers, act, norm, [nn.Tanh()])

    def _stats(self, x):
        return self._mu(x), self._logvar(x)

    def learning_loss(self, x, y):
        """Negative log-likelihood of q(y|x), on detached inputs."""
        x, y = _flat(x).detach(), _flat(y).detach()
        mu, logvar = self._stats(x)
        return ((mu - y) ** 2 / logvar.exp() + logvar).sum(-1).mean() / 2

    def mi_est(self, x, y):
        """The upper bound. Gradients flow to x and y."""
        x, y = _flat(x), _flat(y)
        mu, logvar = self._stats(x)
        var = logvar.exp()
        pos = -((mu - y) ** 2 / var).sum(-1) / 2
        # (n, 1, d) vs (1, n, d) -> (n, n), averaged over j.
        neg = -(
            (mu.unsqueeze(1) - y.unsqueeze(0)) ** 2 / var.unsqueeze(1)
        ).sum(-1).mean(1) / 2
        # The logvar term is identical in both and cancels, so it is dropped.
        return (pos - neg).mean()


class MINE(nn.Module):
    """Mutual Information Neural Estimation (Belghazi et al., 2018).

    A statistics net T(x, y) gives the Donsker-Varadhan lower bound

        I(X; Y) >= E_p(x,y)[T] - log E_p(x)p(y)[e^T]

    with negatives formed by pairing every x with every y in the batch.

    The DV gradient is biased because log E[e^T] is a minibatch estimate. The
    standard correction divides that term's gradient by an EMA of E[e^T]; it is on
    by default and changes only the gradient, never the returned value.
    """

    def __init__(self, x_dim, y_dim, hidden=256, layers=2, act="SiLU",
                 norm="LayerNorm", ema_decay=0.99, unbiased_grad=True, clip=10.0):
        super(MINE, self).__init__()
        self._T = _mlp(x_dim + y_dim, 1, hidden, layers, act, norm)
        self._ema_decay = ema_decay
        self._unbiased_grad = unbiased_grad
        # Bound T before exponentiating; one large score otherwise dominates
        # log E[e^T] and the gradient explodes.
        self._clip = clip
        self.register_buffer("_et_ema", torch.ones(()))
        self.register_buffer("_ema_inited", torch.zeros((), dtype=torch.bool))

    def _scores(self, x, y):
        """T on matched pairs, shape (n,), and on all pairs, shape (n, n)."""
        n = x.shape[0]
        pos = self._T(torch.cat([x, y], -1)).squeeze(-1)
        x_rep = x.unsqueeze(1).expand(n, n, x.shape[-1])
        y_rep = y.unsqueeze(0).expand(n, n, y.shape[-1])
        neg = self._T(torch.cat([x_rep, y_rep], -1)).squeeze(-1)
        return pos.clamp(-self._clip, self._clip), neg.clamp(-self._clip, self._clip)

    @torch.no_grad()
    def _update_ema(self, value):
        if not bool(self._ema_inited):
            self._et_ema.copy_(value)
            self._ema_inited.fill_(True)
        else:
            self._et_ema.mul_(self._ema_decay).add_(value, alpha=1 - self._ema_decay)

    def _dv(self, pos, neg, update_ema):
        et = neg.exp().mean()
        log_et = et.clamp(min=1e-8).log()
        if not self._unbiased_grad:
            return pos.mean() - log_et
        if update_ema:
            self._update_ema(et.detach())
        # Value is log_et; gradient is d(et)/ema, the unbiased estimator of
        # d log E[e^T].
        surrogate = et / self._et_ema.clamp(min=1e-8)
        surrogate = surrogate - surrogate.detach() + log_et.detach()
        return pos.mean() - surrogate

    def learning_loss(self, x, y):
        """Negated bound on detached inputs; maximizing DV w.r.t. T fits it."""
        x, y = _flat(x).detach(), _flat(y).detach()
        pos, neg = self._scores(x, y)
        return -self._dv(pos, neg, update_ema=True)

    def mi_est(self, x, y):
        """The lower bound. Gradients flow to x and y."""
        x, y = _flat(x), _flat(y)
        pos, neg = self._scores(x, y)
        return self._dv(pos, neg, update_ema=False)

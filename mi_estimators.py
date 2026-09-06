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

import time

import torch
from torch import nn

to_np = lambda x: x.detach().cpu().numpy()


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


class _Term:
    """One MI term: an estimator, its optimizer, and how it enters the objective.

    `sign` is +1 for a bound that is minimized (an upper bound -- CLUB) and -1 for
    one that is maximized (a lower bound -- MINE). `clamp` holds only for upper
    bounds: I >= 0, so a negative estimate means the estimator has not fit and the
    number is an artifact, not a measurement; clamping kills the gradient in that
    regime. A lower bound must not be clamped -- a negative value there means the
    critic is behind, which is exactly the regime the term exists to escape.
    """

    def __init__(self, name, estimator, opt, scale, inner_steps, sign, clamp):
        self.name = name
        self.estimator = estimator
        self.opt = opt
        self.scale = scale
        self.inner_steps = inner_steps
        self.sign = sign
        self.clamp = clamp
        self.clear()

    def clear(self):
        self.pair = None
        self.bound = None
        self.loss = None
        self.fit = None

    def compute(self, pair):
        """The scaled, signed contribution to the world-model objective."""
        self.pair = pair
        if pair is None:
            return 0.0
        self.bound = self.estimator.mi_est(*pair)
        bound = self.bound.clamp(min=0.0) if self.clamp else self.bound
        self.loss = self.sign * self.scale * bound
        return self.loss

    def step(self):
        """Fit the estimator on detached inputs, by its own optimizer.

        RequiresGrad on the owning world model has cleared requires_grad across
        everything reached through it, this module included, so re-enable it here.
        The inputs carry no grad, so this backward touches only the estimator.
        """
        if self.pair is None:
            return
        self.estimator.requires_grad_(True)
        x, y = (t.detach() for t in self.pair)
        with torch.enable_grad():
            for _ in range(max(1, self.inner_steps)):
                loss = self.estimator.learning_loss(x, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.fit = loss.detach()

    def metrics(self):
        """mi/<term>/{bound,loss,fit} plus whatever diagnostics the estimator has."""
        if self.bound is None:
            return {}
        m = {f"mi/{self.name}/bound": to_np(self.bound)}
        if self.loss is not None and torch.is_tensor(self.loss):
            m[f"mi/{self.name}/loss"] = to_np(self.loss)
        if self.fit is not None:
            m[f"mi/{self.name}/fit"] = to_np(self.fit)
        x, y = self.pair
        # Only the gaussian CLUB has a variance parameter to saturate; the
        # categorical one has none, which is the point of it.
        if hasattr(self.estimator, "logvar_stats"):
            for k, v in self.estimator.logvar_stats(x).items():
                m[f"mi/{self.name}/{k}"] = to_np(v)
        if hasattr(self.estimator, "mi_stats"):
            for k, v in self.estimator.mi_stats(x, y).items():
                m[f"mi/{self.name}/{k}"] = to_np(v)
        return m


class InfoLosses(nn.Module):
    """Every mutual-information term the world model carries.

    Three of them, all gated behind `use_infolosses` and each disabled
    individually by a zero scale:

        endo_exo_mi  I(z; xi)            minimized -- the endogenous stream forgets
                                         what the exogenous one already explains
        rew_endo_mi  I((e, s, s'); G)    maximized -- endogenous stays return-informative
        rew_exo_mi   I((e, xi, xi'); G)  minimized -- exogenous does not encode return

    G is the discounted return-to-go computed from the rewards actually observed
    in the buffer, two-hot encoded. Not a critic estimate: a bootstrapped value is
    a function of the endogenous stream these terms measure, which would make the
    bounds partly self-referential.

    No estimator here is a world-model parameter. Each is fit by its own optimizer
    on detached inputs, or the world model would learn to make the estimator bad
    instead of making the streams behave.

    **Two-phase, and the order is load-bearing.** `compute` builds the bounds and
    returns the term added to the world-model objective; `fit` steps the
    estimators afterwards. They cannot be merged: the bounds hold the estimator
    weights in the world model's graph, so stepping the estimators first is an
    in-place modification that breaks that backward pass.

        info_loss = info.compute(post, feat, xi_post, xi_feat, data, timing)
        ...                                   # prediction losses, may subsample feat
        metrics = model_opt(... + info_loss, params)
        info.fit()                            # only now
        metrics.update(info.metrics())

    `fit` and `metrics` without a preceding `compute` are no-ops.
    """

    def __init__(self, config, use_exo):
        super(InfoLosses, self).__init__()
        # Imported here, not at module scope: networks imports this module for
        # MINE/CLUB, so a top-level import would be circular.
        import networks

        self._config = config
        self._use_exo = use_exo
        self._terms = []

        z_stoch = (config.dyn_stoch * config.dyn_discrete
                   if config.dyn_discrete else config.dyn_stoch)
        xi_stoch = (config.xi_stoch * config.xi_discrete
                    if config.xi_discrete else config.xi_stoch)
        feat_size = config.dyn_deter + z_stoch
        xi_feat_size = (config.xi_deter + xi_stoch
                        if config.xi_head_input == "feat" else config.xi_deter)

        # --- endo_exo_mi: I(z; xi) ---
        if use_exo and self._opt_for("endo_exo_mi", "scale") > 0:
            mode = self._opt_for("endo_exo_mi", "input")
            q = self._opt_for("endo_exo_mi", "q")
            # 'logit' has the same flattened width as 'stoch' -- the logits over
            # the same categorical, before sampling.
            z_size = {"feat": feat_size, "deter": config.dyn_deter,
                      "stoch": z_stoch, "logit": z_stoch}[mode]
            xi_size = {"feat": config.xi_deter + xi_stoch, "deter": config.xi_deter,
                       "stoch": xi_stoch, "logit": xi_stoch}[mode]
            if mode == "logit" and not (config.dyn_discrete and config.xi_discrete):
                raise ValueError(
                    "endo_exo_mi_input='logit' needs dyn_discrete and xi_discrete; "
                    "continuous RSSMs carry 'mean'/'std', not 'logit'."
                )
            if q == "categorical":
                if not config.xi_discrete:
                    raise ValueError(
                        "endo_exo_mi_q='categorical' needs xi_discrete; a continuous "
                        "xi has no categorical to classify."
                    )
                # q reads the endogenous stream only -- see _endo_exo_inputs for why
                # xi_deter is not an input -- and predicts the exogenous categorical.
                est = networks.CategoricalCLUB(
                    z_size, config.xi_stoch, config.xi_discrete,
                    self._opt_for("endo_exo_mi", "hidden"),
                    self._opt_for("endo_exo_mi", "layers"),
                    config.act, config.norm,
                )
            else:
                est = networks.CLUB(
                    z_size, xi_size,
                    self._opt_for("endo_exo_mi", "hidden"),
                    self._opt_for("endo_exo_mi", "layers"),
                    config.act, config.norm,
                    self._opt_for("endo_exo_mi", "normalize"),
                    self._opt_for("endo_exo_mi", "norm_momentum"),
                )
            self._endo_exo = self._register("endo_exo_mi", est, sign=1, clamp=True)
        else:
            self._endo_exo = None

        # --- Action terms. Both condition on a fixed one-hot of the task id: a
        # learned embedding would take gradients from these same objectives and
        # could satisfy them by reshaping the task code rather than the streams.
        task_code = config.task_train_size
        if self._opt_for("act_endo_mi", "scale") > 0:
            est = MINE(
                task_code + 2 * feat_size, config.num_actions,
                self._opt_for("act_endo_mi", "hidden"),
                self._opt_for("act_endo_mi", "layers"),
                config.act, config.norm,
                ema_decay=self._opt_for("act_endo_mi", "ema_decay"),
                clip=self._opt_for("act_endo_mi", "clip"),
            )
            self._act_endo = self._register("act_endo_mi", est, sign=-1, clamp=False)
        else:
            self._act_endo = None

        if use_exo and self._opt_for("act_exo_mi", "scale") > 0:
            est = CLUB(
                task_code + 2 * xi_feat_size, config.num_actions,
                self._opt_for("act_exo_mi", "hidden"),
                self._opt_for("act_exo_mi", "layers"),
                config.act, config.norm,
            )
            self._act_exo = self._register("act_exo_mi", est, sign=1, clamp=True)
        else:
            self._act_exo = None

        # --- Reward terms. Same pairing and conditioning as the action ones, with
        # the two-hot return in place of the action, so the target width is the
        # DiscDist bucket count rather than num_actions.
        self._rew_bins = config.rew_mi_bins
        if self._opt_for("rew_endo_mi", "scale") > 0:
            est = MINE(
                task_code + 2 * feat_size, self._rew_bins,
                self._opt_for("rew_endo_mi", "hidden"),
                self._opt_for("rew_endo_mi", "layers"),
                config.act, config.norm,
                ema_decay=self._opt_for("rew_endo_mi", "ema_decay"),
                clip=self._opt_for("rew_endo_mi", "clip"),
            )
            self._rew_endo = self._register("rew_endo_mi", est, sign=-1, clamp=False)
        else:
            self._rew_endo = None

        if use_exo and self._opt_for("rew_exo_mi", "scale") > 0:
            # Categorical, not Gaussian: the two-hot return is a distribution over
            # buckets, and a diagonal Gaussian on it fails the same two ways run #12
            # found for one-hot xi -- variance pinned at the floor, and no ability to
            # discriminate, since distinct two-hot vectors are near-equidistant in
            # L2. One group of `bins` classes, so chance is ln(bins).
            est = networks.CategoricalCLUB(
                task_code + 2 * xi_feat_size, 1, self._rew_bins,
                self._opt_for("rew_exo_mi", "hidden"),
                self._opt_for("rew_exo_mi", "layers"),
                config.act, config.norm,
            )
            self._rew_exo = self._register("rew_exo_mi", est, sign=1, clamp=True)
        else:
            self._rew_exo = None

    def _opt_for(self, term, key):
        """`<term>_<key>` if set, else the shared `mi_<key>`.

        Lets every term override any net/optimizer default without repeating the
        shared ones in the config. `scale` is additionally gated on
        use_infolosses, so the master switch reaches every term through the one
        lookup each of them already makes.
        """
        if key == "scale" and not self._config.use_infolosses:
            return 0.0
        specific = f"{term}_{key}"
        if hasattr(self._config, specific):
            return getattr(self._config, specific)
        return getattr(self._config, f"mi_{key}")

    def _register(self, name, estimator, sign, clamp):
        """Build a _Term, move the net to the device, and give it an optimizer."""
        estimator = estimator.to(self._config.device)
        # Registered as a submodule so .to()/.train() reach it, but deliberately
        # excluded from the world model's optimizer by the caller.
        self.add_module(f"_est_{name}", estimator)
        term = _Term(
            name.replace("_mi", ""),
            estimator,
            torch.optim.Adam(estimator.parameters(),
                             lr=self._opt_for(name, "lr")),
            self._opt_for(name, "scale"),
            self._opt_for(name, "inner_steps"),
            sign,
            clamp,
        )
        self._terms.append(term)
        return term

    @property
    def active(self):
        """False when every term is off, so callers can skip the phases entirely."""
        return bool(self._terms)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _endo_exo_inputs(self, post, feat, xi_post, xi_feat):
        """The slice of each stream the I(z; xi) bound is taken over.

        Built from the state dicts rather than by slicing `feat`, so it does not
        depend on how get_feat happens to order deter and stoch.
        """
        mode = self._opt_for("endo_exo_mi", "input")
        if self._opt_for("endo_exo_mi", "q") == "categorical":
            # q reads the endogenous stream alone and predicts the exogenous
            # categorical, so the bound is the unconditional I(z; xi_stoch).
            # xi_deter is deliberately *not* an input: conditioning on it let q
            # predict xi_stoch from the exogenous side, which explained the shared
            # information away and drove the bound to ~0 (pos -6.0 vs neg -6.2)
            # while the fit sat well below chance. The endogenous stream is the one
            # that has to forget xi, so it is the only thing q may look at.
            z = {"feat": feat, "deter": post["deter"]}.get(mode)
            if z is None:
                z = post["logit" if mode == "logit" else "stoch"]
                z = z.reshape(*z.shape[:2], -1)
            x = xi_post["stoch"]
            return z, x.reshape(*x.shape[:2], -1)
        if mode == "feat":
            return feat, xi_feat
        if mode == "deter":
            return post["deter"], xi_post["deter"]
        # 'stoch' is one straight-through sample, so the estimate carries the
        # sampling noise; 'logit' is the distribution it was drawn from, which is
        # continuous and unsampled -- a better match for the Gaussian q, and it
        # does not jitter step to step. Both flatten the discrete groups.
        key = "logit" if mode == "logit" else "stoch"
        z, x = post[key], xi_post[key]
        return z.reshape(*z.shape[:2], -1), x.reshape(*x.shape[:2], -1)

    def _pair_inputs(self, stream_feat, data, target):
        """Build ((task_onehot, s, s'), target) for an action or reward term.

        Returns (x, y) already flattened to (N, dim), or None when no step in the
        batch carries a usable label. `stream_feat` is the endogenous or exogenous
        feature and `target` the action or two-hot return; the pairing and masking
        are identical for all four terms.

        s' is the next observed step, so the last timestep has no successor and is
        dropped. Steps whose task_id falls outside the train split are dropped too:
        the one-hot has one column per train task, and an unlabelled step (-1)
        would set the wrong column rather than error.
        """
        n = self._config.task_train_size
        task = data["task_id"].long()
        # A step is usable if it is labelled *and* its successor exists.
        valid = ((task >= 0) & (task < n))[:, :-1]
        if not torch.any(valid):
            return None
        # Clamp before the one-hot: invalid rows are masked out immediately after,
        # but the index itself must be in range or the scatter faults.
        e = torch.nn.functional.one_hot(
            task[:, :-1].clamp(0, n - 1), n
        ).to(stream_feat.dtype)
        x = torch.cat([e, stream_feat[:, :-1], stream_feat[:, 1:]], -1)
        y = target[:, :-1]
        # Masking after the concat keeps the pairing intact: flattening first and
        # then selecting would let a dropped row's successor pair with the wrong
        # step. The result is a flat (N, dim) with only usable rows.
        return x[valid], y[valid]

    def _return_target(self, data):
        """Two-hot discounted return-to-go, from the rewards in the buffer.

        `tools.lambda_return` with lambda_=1 and a zero value function is the
        discounted Monte Carlo return, so no critic enters: a bootstrapped value
        is a function of the endogenous stream these terms measure, which would
        make the bounds partly self-referential.
        """
        import tools

        reward = data["reward"]
        if reward.dim() == 2:
            reward = reward.unsqueeze(-1)
        # lambda_return wants time-major; it unbinds along batch, so stacking the
        # result gives (batch, time, 1) back directly -- no second permute.
        reward_t = reward.permute(1, 0, 2)
        zeros = torch.zeros_like(reward_t)
        ret = tools.lambda_return(
            reward_t, zeros, self._config.discount, None, 1.0, axis=0
        )
        ret = torch.stack(list(ret), dim=0) if isinstance(ret, tuple) else ret
        return tools.DiscDist(
            logits=torch.zeros(*ret.shape[:2], self._rew_bins, device=ret.device),
            device=ret.device,
        ).two_hot(ret).to(reward.dtype)

    # ------------------------------------------------------------------
    # The three phases
    # ------------------------------------------------------------------

    def compute(self, post, feat, xi_post, xi_feat, data, timing=None):
        """The MI term added to the world-model objective.

        Call BEFORE the prediction losses, which may subsample `feat`, and before
        the optimizer step. Returns 0.0 when every term is off.
        """
        for term in self._terms:
            term.clear()
        if not self.active:
            return 0.0

        total = 0.0
        if self._endo_exo is not None:
            t = time.time()
            total = total + self._endo_exo.compute(
                self._endo_exo_inputs(post, feat, xi_post, xi_feat)
            )
            self._add_timing(timing, "endo_exo_mi", time.time() - t)
        if self._act_endo is not None or self._act_exo is not None:
            t = time.time()
            action = data["action"]
            if self._act_endo is not None:
                total = total + self._act_endo.compute(
                    self._pair_inputs(feat, data, action))
            if self._act_exo is not None:
                total = total + self._act_exo.compute(
                    self._pair_inputs(xi_feat, data, action))
            self._add_timing(timing, "act_mi", time.time() - t)
        if self._rew_endo is not None or self._rew_exo is not None:
            t = time.time()
            # Computed once and shared: both terms take the same target.
            ret = self._return_target(data)
            if self._rew_endo is not None:
                total = total + self._rew_endo.compute(
                    self._pair_inputs(feat, data, ret))
            if self._rew_exo is not None:
                total = total + self._rew_exo.compute(
                    self._pair_inputs(xi_feat, data, ret))
            self._add_timing(timing, "rew_mi", time.time() - t)
        return total

    def fit(self):
        """Step every estimator. Call AFTER the world-model optimizer step.

        The bounds from `compute` hold these weights in the world model's graph, so
        stepping them any earlier is an in-place modification that breaks that
        backward pass.
        """
        for term in self._terms:
            term.step()

    def metrics(self):
        """Everything logged for these terms, already numpy."""
        metrics = {}
        for term in self._terms:
            metrics.update(term.metrics())
        # The gap each pair jointly widens. Read it only alongside that pair's two
        # fit metrics -- it differences a lower bound against an upper bound
        # estimated by a different net, so it is a direction of travel, not a
        # calibrated quantity in nats.
        for name, endo, exo in (("act", self._act_endo, self._act_exo),
                                ("rew", self._rew_endo, self._rew_exo)):
            if (endo is not None and exo is not None
                    and endo.bound is not None and exo.bound is not None):
                metrics[f"mi/{name}_diff"] = to_np(endo.bound - exo.bound)
        return metrics

    @staticmethod
    def _add_timing(timing, name, value):
        """Same key convention as WorldModel._add_wm_timing: a position counter,
        then the name."""
        if timing is None:
            return
        timing[f"wm_{len(timing)}_{name}_time"] = value

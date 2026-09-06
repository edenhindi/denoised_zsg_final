import copy
import time

import torch
from torch import nn
import numpy as np

import mi_estimators
import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        # Reward-free encoder for xi, mirroring online-zsg's q(y_t | y_prev, o_t):
        # only the endogenous stream may integrate reward feedback.
        self.xi_encoder = None
        if config.use_exo:
            self.xi_encoder = networks.MultiEncoder(
                shapes, **{**config.encoder, "input_reward": False}
            )
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            config.rnn_detach_every
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            stoch_size = config.dyn_stoch * config.dyn_discrete
        else:
            stoch_size = config.dyn_stoch
        feat_size = stoch_size + config.dyn_deter
        self._stoch_size = stoch_size

        # Exogenous stream: an action-free RSSM whose decoded output is added to the
        # endogenous decoder's. It feeds the observation reconstruction only -- the
        # reward and cont heads stay endogenous, because the policy imagines with the
        # endogenous state alone and must never be asked to predict returns that
        # depend on a latent it cannot roll forward.
        self._use_exo = config.use_exo
        self.xi_dynamics = None
        xi_feat_size = 0
        if self._use_exo:
            self.xi_dynamics = networks.XiRSSM(
                config.xi_stoch,
                config.xi_deter,
                config.xi_hidden,
                config.xi_input_layers,
                config.xi_output_layers,
                config.xi_discrete,
                config.act,
                config.norm,
                config.dyn_mean_act,
                config.dyn_std_act,
                config.dyn_min_std,
                config.dyn_cell,
                config.unimix_ratio,
                config.initial,
                self.xi_encoder.outdim if self.xi_encoder is not None else self.embed_size,
                config.device,
                config.task_train_size if config.xi_ctx_cond else 0,
                config.xi_ctx_dim,
            )
            xi_feat_size = (
                self.xi_dynamics.feat_size
                if config.xi_head_input == "feat"
                else config.xi_deter
            )

        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder, xi_feat_size=xi_feat_size
        )
        if config.reconstruction_window > 0:
            self.heads["multi_decoder"] = networks.MultiDecoder(
                feat_size + self.embed_size,
                shapes,
                **config.decoder,
                xi_feat_size=xi_feat_size,
            )
        reward_mlp_shape = (255,) if config.reward_head == "symlog_disc" else []
        # With reward_head_action the head is r = f(feat, a) rather than f(feat), so it
        # cannot fit reward without representing which action was taken.
        # reward_head_input picks the slice it reads: 'stoch' hides the recurrent state,
        # where a per-episode-constant distractor would otherwise sit.
        self._reward_head_action = config.reward_head_action
        self._reward_head_input = config.reward_head_input
        reward_feat = {"feat": feat_size, "stoch": stoch_size,
                       "deter": config.dyn_deter}[self._reward_head_input]
        reward_in = reward_feat + (config.num_actions if self._reward_head_action else 0)
        self.heads["reward"] = networks.MLP(
            reward_in,  # pytorch version
            reward_mlp_shape,
            config.reward_layers,
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head,
            outscale=0.0,
            device=config.device,
        )
        if config.reconstruction_window > 0:
            self.heads["multi_reward"] = networks.MLP(
                feat_size + self.embed_size,  # pytorch version
                reward_mlp_shape,
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )

        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        if config.reconstruction_window > 0:
            self.heads["multi_cont"] = networks.MLP(
                feat_size + self.embed_size,  # pytorch version
                [],
                config.cont_layers,
                config.units,
                config.act,
                config.norm,
                dist="binary",
                device=config.device,
            )
        self._config.grad_heads = list(self._config.grad_heads)
        new_grad_heads = []
        if config.reconstruction_window > 0:
            for name in config.grad_heads:
                new_grad_heads.append("multi_" + name)
        self._config.grad_heads.extend(new_grad_heads)
        self._config.grad_heads = tuple(self._config.grad_heads)

        for name in self._config.grad_heads:
            assert name in self.heads, name

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale)

        # Diagnostic probes. Built after _model_opt so they are not among its
        # parameters, and trained on detached features by their own optimizer, so
        # they read the state without shaping it.
        self.probes = nn.ModuleDict()
        if config.probe_state:
            stoch_size = (config.dyn_stoch * config.dyn_discrete
                          if config.dyn_discrete else config.dyn_stoch)
            targets = {"taskid": config.task_train_size}
            # arm is bandits-specific: optimal_arm = task_id % num_arms.
            if getattr(config, "num_arms", 0):
                targets["arm"] = config.num_arms
            for part, size in (("deter", config.dyn_deter), ("stoch", stoch_size)):
                for target, classes in targets.items():
                    self.probes[f"{part}_{target}"] = nn.Linear(size, classes)
            self.probes.to(config.device)
            self._probe_opt = torch.optim.Adam(self.probes.parameters(), lr=3e-4)

        # CLUB. Built after _model_opt for the same reason as the probes: the
        # approximator is fit by its own optimizer and must not be a world-model
        # parameter, or the model would learn to make q bad instead of making the
        # streams independent.
        self._club = None
        if self._use_exo and config.club_scale > 0:
            z_stoch = (config.dyn_stoch * config.dyn_discrete
                       if config.dyn_discrete else config.dyn_stoch)
            xi_stoch = (config.xi_stoch * config.xi_discrete
                        if config.xi_discrete else config.xi_stoch)
            # 'logit' has the same flattened width as 'stoch' -- it is the logits
            # over the same categorical, before sampling.
            club_z_size = {
                "feat": config.dyn_deter + z_stoch,
                "deter": config.dyn_deter,
                "stoch": z_stoch,
                "logit": z_stoch,
            }[config.club_input]
            club_xi_size = {
                "feat": config.xi_deter + xi_stoch,
                "deter": config.xi_deter,
                "stoch": xi_stoch,
                "logit": xi_stoch,
            }[config.club_input]
            if config.club_input == "logit" and not (
                config.dyn_discrete and config.xi_discrete
            ):
                raise ValueError(
                    "club_input='logit' needs dyn_discrete and xi_discrete; "
                    "continuous RSSMs carry 'mean'/'std', not 'logit'."
                )
            if config.club_q == "categorical":
                if not config.xi_discrete:
                    raise ValueError(
                        "club_q='categorical' needs xi_discrete; a continuous xi "
                        "has no categorical to classify."
                    )
                # q reads the endogenous stream only -- see _club_inputs for why
                # xi_deter is not an input -- and predicts the exogenous
                # categorical, bounding I(z; xi_stoch).
                self._club = networks.CategoricalCLUB(
                    club_z_size,
                    config.xi_stoch,
                    config.xi_discrete,
                    config.club_hidden,
                    config.club_layers,
                    config.act,
                    config.norm,
                ).to(config.device)
            else:
                self._club = networks.CLUB(
                    club_z_size,
                    club_xi_size,
                    config.club_hidden,
                    config.club_layers,
                    config.act,
                    config.norm,
                    config.club_normalize_xi,
                    config.club_norm_momentum,
                ).to(config.device)
            self._club_opt = torch.optim.Adam(
                self._club.parameters(), lr=config.club_lr
            )

        # Action-MI terms: MINE on I((e, s, s'); a) and CLUB on I((e, xi, xi'); a),
        # where e is a fixed one-hot of the task id. Nothing here is a world-model
        # parameter: the one-hot is constant, and the estimator nets are fit by
        # their own optimizers for the same reason as the CLUB above.
        #
        # A one-hot rather than a learned nn.Embedding because a learned table
        # takes gradients from these same objectives, so it can make them easier to
        # satisfy by reshaping the task code instead of by changing the streams --
        # the estimate then moves for a reason that has nothing to do with the
        # world model. The per-task weights in each estimator's first layer still
        # give it whatever task-specific capacity it needs.
        self._act_mine = None
        self._act_club = None
        want_mine = config.act_mine_scale > 0
        want_act_club = config.act_club_scale > 0 and self._use_exo
        if want_mine or want_act_club:
            task_code_size = config.task_train_size
            feat_size = config.dyn_deter + (
                config.dyn_stoch * config.dyn_discrete
                if config.dyn_discrete
                else config.dyn_stoch
            )
            xi_stoch = (
                config.xi_stoch * config.xi_discrete
                if config.xi_discrete
                else config.xi_stoch
            )
            xi_feat_size = (
                config.xi_deter + xi_stoch
                if config.xi_head_input == "feat"
                else config.xi_deter
            )
            if want_mine:
                self._act_mine = mi_estimators.MINE(
                    task_code_size + 2 * feat_size,
                    config.num_actions,
                    config.act_mi_hidden,
                    config.act_mi_layers,
                    config.act,
                    config.norm,
                    ema_decay=config.act_mine_ema_decay,
                    clip=config.act_mine_clip,
                ).to(config.device)
                self._act_mine_opt = torch.optim.Adam(
                    self._act_mine.parameters(), lr=config.act_mi_lr
                )
            if want_act_club:
                self._act_club = mi_estimators.CLUB(
                    task_code_size + 2 * xi_feat_size,
                    config.num_actions,
                    config.act_mi_hidden,
                    config.act_mi_layers,
                    config.act,
                    config.norm,
                ).to(config.device)
                self._act_club_opt = torch.optim.Adam(
                    self._act_club.parameters(), lr=config.act_mi_lr
                )

    def _probe_metrics(self, post, data):
        """Linear-probe accuracy from detached state parts to task id and arm.

        Early steps are the informative ones: before exploration the arm is only
        knowable from the distractor, so high early accuracy is the shortcut.
        """
        deter = post["deter"].detach()
        stoch = post["stoch"].detach()
        stoch = stoch.reshape(*stoch.shape[:2], -1)
        task = data["task_id"].long()
        valid = (task >= 0) & (task < self._config.task_train_size)
        if not torch.any(valid):
            return {}
        early = max(1, self._config.probe_early_steps)
        labels = {"taskid": task}
        if getattr(self._config, "num_arms", 0):
            labels["arm"] = task % self._config.num_arms

        metrics, loss = {}, 0.0
        # RequiresGrad(self) clears requires_grad on the whole world model on exit,
        # probes included, so re-enable them here. Inputs stay detached either way.
        self.probes.requires_grad_(True)
        with torch.enable_grad():
            for part, feat in (("deter", deter), ("stoch", stoch)):
                for target, label in labels.items():
                    logits = self.probes[f"{part}_{target}"](feat)
                    loss = loss + torch.nn.functional.cross_entropy(
                        logits[valid], label[valid]
                    )
                    with torch.no_grad():
                        correct = (logits.argmax(-1) == label) & valid
                        for span, sl in (("early", slice(0, early)),
                                         ("late", slice(early, None))):
                            v = valid[:, sl]
                            if torch.any(v):
                                metrics[f"probe/{part}_{target}_{span}_acc"] = to_np(
                                    correct[:, sl].sum() / v.sum())
        self._probe_opt.zero_grad()
        loss.backward()
        self._probe_opt.step()
        metrics["probe/chance_taskid"] = 1.0 / self._config.task_train_size
        if getattr(self._config, "num_arms", 0):
            metrics["probe/chance_arm"] = 1.0 / self._config.num_arms
        return metrics

    def xi_embed(self, data, embed):
        """Embedding for the exogenous stream: its own reward-free one, or shared."""
        return self.xi_encoder(data) if self.xi_encoder is not None else embed

    def get_xi_feat(self, xi_state):
        """The exogenous feature the decoder branch reads."""
        if xi_state is None:
            return None
        if self._config.xi_head_input == "feat":
            return self.xi_dynamics.get_feat(xi_state)
        return xi_state["deter"]

    def _club_inputs(self, post, feat, xi_post, xi_feat):
        """The slice of each stream the CLUB bound is taken over.

        Built from the state dicts rather than by slicing `feat`, so it does not
        depend on how get_feat happens to order deter and stoch.
        """
        mode = self._config.club_input
        if self._config.club_q == "categorical":
            # q reads the endogenous stream alone and predicts the exogenous
            # categorical, so the bound is the unconditional I(z; xi_stoch).
            # xi_deter is deliberately *not* an input: conditioning on it let q
            # predict xi_stoch from the exogenous side, which explained the shared
            # information away and drove the bound to ~0 (pos -6.0 vs neg -6.2)
            # while club_nll sat well below chance. The endogenous stream is the
            # one that has to forget xi, so it is the only thing q may look at.
            z = {
                "feat": feat,
                "deter": post["deter"],
            }.get(mode)
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

    def _club_learn(self, feat, xi_feat):
        """Fit the CLUB approximator q(xi|z). Returns its NLL for logging.

        Inputs are detached inside `learning_loss`, and RequiresGrad(self) has
        cleared requires_grad across the world model -- the CLUB net is not a
        world-model parameter, but re-enable it here for the same reason the probes
        do, since it is reached through this module.
        """
        self._club.requires_grad_(True)
        feat, xi_feat = feat.detach(), xi_feat.detach()
        with torch.enable_grad():
            for _ in range(max(1, self._config.club_inner_steps)):
                nll = self._club.learning_loss(feat, xi_feat)
                # feat/xi_feat carry no grad here, so this backward touches only q.
                self._club_opt.zero_grad()
                nll.backward()
                self._club_opt.step()
        return nll.detach()

    def _act_mi_inputs(self, stream_feat, data):
        """Build ((task_onehot, s, s'), a) for the action-MI terms.

        Returns (x, y) already flattened to (N, dim), or None when no step in the
        batch carries a usable label. `stream_feat` is the endogenous or exogenous
        feature; the pairing and masking are identical for both.

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
        y = data["action"][:, :-1]
        # Masking after the concat keeps the pairing intact: flattening first and
        # then selecting would let a dropped row's successor pair with the wrong
        # step. The result is a flat (N, dim) with only usable rows.
        return x[valid], y[valid]

    def _act_mi_learn(self, estimator, opt, x, y):
        """Fit one action-MI estimator on detached inputs. Returns its loss.

        Same contract as `_club_learn`: the estimator is not a world-model
        parameter, but it is reached through this module, so RequiresGrad(self) has
        cleared its requires_grad on the way in.
        """
        estimator.requires_grad_(True)
        x, y = x.detach(), y.detach()
        with torch.enable_grad():
            for _ in range(max(1, self._config.act_mi_inner_steps)):
                loss = estimator.learning_loss(x, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return loss.detach()

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        time_metrics = {}
        data_preprocess_time = time.time()
        data = self.preprocess(data)
        self._add_wm_timing(time_metrics, 'data_preprocess', time.time() - data_preprocess_time)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embedding_time = time.time()
                embed = self.encoder(data)
                self._add_wm_timing(time_metrics, 'embedding', time.time() - embedding_time)
                rssm_time = time.time()
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                self._add_wm_timing(time_metrics, 'rssm', time.time() - rssm_time)
                kl_free = tools.schedule(self._config.kl_free, self._step)
                dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                rep_scale = tools.schedule(self._config.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                # Exogenous stream: no action input, and no reward when xi_no_reward.
                xi_post, xi_prior, xi_kl_loss = None, None, 0.0
                if self._use_exo:
                    xi_rssm_time = time.time()
                    xi_post, xi_prior = self.xi_dynamics.observe(
                        self.xi_embed(data, embed), data["is_first"],
                        task_id=data.get("task_id"),
                    )
                    self._add_wm_timing(time_metrics, 'xi_rssm', time.time() - xi_rssm_time)
                    xi_kl_loss, xi_kl_value, xi_dyn_loss, xi_rep_loss = (
                        self.xi_dynamics.kl_loss(
                            xi_post,
                            xi_prior,
                            self._config.xi_kl_free,
                            self._config.xi_dyn_scale,
                            self._config.xi_rep_scale,
                        )
                    )
                    xi_kl_loss = self._config.xi_kl_scale * xi_kl_loss

                get_features_time = time.time()
                feat = self.dynamics.get_feat(post)
                xi_feat = self.get_xi_feat(xi_post)
                self._add_wm_timing(time_metrics, 'get_features', time.time() - get_features_time)

                # CLUB, before the prediction losses because they may subsample feat.
                club_loss = 0.0
                if self._club is not None:
                    club_time = time.time()
                    club_z, club_xi = self._club_inputs(
                        post, feat, xi_post, xi_feat
                    )
                    club_mi = self._club.mi_est(club_z, club_xi)
                    # I(z; xi) >= 0 always, so a negative bound means q has not fit
                    # p(xi|z) and the estimate is an artifact, not a measurement.
                    # Clamping kills the gradient in that regime rather than letting
                    # the world model descend on it. club_mi is still logged raw --
                    # a clamped-to-zero club_loss with a negative club_mi is the
                    # signal that q is behind, so read both.
                    club_loss = self._config.club_scale * club_mi.clamp(min=0.0)
                    self._add_wm_timing(time_metrics, 'club', time.time() - club_time)

                # Action-MI terms, on the same features and before the prediction
                # losses for the same reason as the CLUB above.
                act_mi_loss = 0.0
                mine_pair, act_club_pair = None, None
                mine_mi, act_club_mi = None, None
                if self._act_mine is not None or self._act_club is not None:
                    act_mi_time = time.time()
                    if self._act_mine is not None:
                        mine_pair = self._act_mi_inputs(feat, data)
                        if mine_pair is not None:
                            mine_mi = self._act_mine.mi_est(*mine_pair)
                            # Maximized, so it enters negated. No clamp: MINE is a
                            # lower bound, and a negative value means the critic has
                            # not fit rather than that the quantity is negative --
                            # clamping there would silently kill the gradient in
                            # exactly the regime the term is meant to escape.
                            act_mi_loss = (
                                act_mi_loss - self._config.act_mine_scale * mine_mi
                            )
                    if self._act_club is not None:
                        act_club_pair = self._act_mi_inputs(xi_feat, data)
                        if act_club_pair is not None:
                            act_club_mi = self._act_club.mi_est(*act_club_pair)
                            # Minimized, and clamped for the same reason as the
                            # I(z; xi) bound: I >= 0, so a negative estimate is an
                            # unfit q rather than a measurement.
                            act_mi_loss = (
                                act_mi_loss
                                + self._config.act_club_scale
                                * act_club_mi.clamp(min=0.0)
                            )
                    self._add_wm_timing(
                        time_metrics, 'act_mi', time.time() - act_mi_time
                    )

                losses, mses = self._compute_prediction_losses(
                    feat, embed, data, time_metrics, xi_feat
                )

            optimizer_time = time.time()
            metrics = self._model_opt(
                sum(losses.values()) + kl_loss + xi_kl_loss + club_loss + act_mi_loss,
                self.parameters(),
            )
            # Fit the action-MI estimators after the world-model step, for the same
            # reason as the CLUB below: the bounds above hold their weights in the
            # graph, and stepping them first is an in-place modification that breaks
            # that backward pass.
            mine_loss, act_club_nll = None, None
            if mine_pair is not None:
                mine_loss = self._act_mi_learn(
                    self._act_mine, self._act_mine_opt, *mine_pair
                )
            if act_club_pair is not None:
                act_club_nll = self._act_mi_learn(
                    self._act_club, self._act_club_opt, *act_club_pair
                )
            if self._club is not None:
                # Fit q(xi|z) on detached features, by its own optimizer, *after*
                # the world-model step: the bound above holds the CLUB weights in
                # its graph, and stepping them first is an in-place modification
                # that breaks that backward pass.
                club_nll = self._club_learn(club_z, club_xi)
            self._add_wm_timing(time_metrics, 'optimizer', time.time() - optimizer_time)
            self._add_wm_timing(time_metrics, 'total', time.time() - data_preprocess_time, use_counter=False)
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        if self._use_exo:
            metrics["xi_kl_loss"] = to_np(xi_kl_loss)
            metrics["xi_dyn_loss"] = to_np(xi_dyn_loss)
            metrics["xi_rep_loss"] = to_np(xi_rep_loss)
            metrics["xi_kl"] = to_np(torch.mean(xi_kl_value))
            if self._club is not None:
                # club_mi is the bound on I(z; xi) in nats; club_nll says whether q
                # has fit well enough for that bound to mean anything.
                metrics["club_mi"] = to_np(club_mi)
                metrics["club_loss"] = to_np(club_loss)
                metrics["club_nll"] = to_np(club_nll)
                # Whether q's Gaussian is straining against its targets. See
                # CLUB.logvar_stats. The categorical q has no variance parameter,
                # so there is nothing to saturate and no analogue to report.
                if hasattr(self._club, "logvar_stats"):
                    for k, v in self._club.logvar_stats(club_z).items():
                        metrics[f"club_{k}"] = to_np(v)
                # Is the bound's gradient noise-dominated, and is it a small
                # residual of two large terms? See CLUB.mi_stats.
                for k, v in self._club.mi_stats(club_z, club_xi).items():
                    metrics[f"club_{k}"] = to_np(v)
        # Action-MI. mi is the bound in nats; the companion loss says whether the
        # estimator has fit well enough for that bound to mean anything. Both are
        # logged raw -- for the CLUB term, a clamped-to-zero contribution with a
        # negative act_club_mi is the signal that q is behind.
        if mine_mi is not None:
            metrics["action_info/endo_mi"] = to_np(mine_mi)
            metrics["action_info/endo_fit"] = to_np(mine_loss)
        if act_club_mi is not None:
            metrics["action_info/exo_mi"] = to_np(act_club_mi)
            metrics["action_info/exo_fit"] = to_np(act_club_nll)
        if mine_mi is not None and act_club_mi is not None:
            # The gap the two terms jointly widen: action information in the
            # endogenous stream minus that in the exogenous one. Read it only
            # alongside the two fit metrics -- it differences a lower bound
            # against an upper bound estimated by a different net, so it is a
            # direction of travel, not a calibrated quantity in nats.
            metrics["action_info/diff"] = to_np(mine_mi - act_club_mi)
        if torch.is_tensor(act_mi_loss):
            metrics["action_info/loss"] = to_np(act_mi_loss)
        for name, mse in mses.items():
            metrics[f"{name}_mse"] = to_np(mse)
        metrics.update(time_metrics)
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            prior_ent = self.dynamics.get_dist(prior).entropy()
            post_ent = self.dynamics.get_dist(post).entropy()
            metrics["prior_ent"] = to_np(torch.mean(prior_ent))
            metrics["post_ent"] = to_np(torch.mean(post_ent))
            if self._use_exo:
                metrics["xi_prior_ent"] = to_np(
                    torch.mean(self.xi_dynamics.get_dist(xi_prior).entropy())
                )
                metrics["xi_post_ent"] = to_np(
                    torch.mean(self.xi_dynamics.get_dist(xi_post).entropy())
                )
            context = dict(
                embed=embed,
                feat=feat,
                kl=kl_value,
                postent=post_ent,
            )
            # xi rides in context, deliberately not in post: post becomes `start` for
            # ImagBehavior, and imagination must stay endogenous-only.
            if self._use_exo:
                context["xi_post"] = {k: v.detach() for k, v in xi_post.items()}
        post = {k: v.detach() for k, v in post.items()}
        if self._config.probe_state and "task_id" in data:
            metrics.update(self._probe_metrics(post, data))
        return post, context, metrics

    @staticmethod
    def _add_wm_timing(time_metrics, name, t, use_counter=True):
        if use_counter:
            name = f'{len(time_metrics)}_{name}'
        name = f'wm_{name}_time'
        time_metrics[name] = t

    def get_uncertainty_measure(self, features):
        reward_dist = self.heads["reward"](features)
        reward_logits = reward_dist.logits
        logits_pairs_mean = (reward_logits[:, :, 1:] + reward_logits[:, :, :-1]) / 2
        max_pair = logits_pairs_mean.max(dim=-1)[0]
        return (logits_pairs_mean.sum(dim=-1) - max_pair).unsqueeze(-1)

    def _compute_prediction_losses(self, feat, embed, data, time_metrics, xi_feat=None):
        preds = {}
        sequence_indices = None

        sample_features_time = time.time()
        if self._config.hidden_states_subsample != -1:
            # randomly subsample a minibatch of hidden states
            batch_size, batch_length = feat.shape[:2]

            sequence_indices = np.array([np.random.choice(
                np.arange(batch_length),
                self._config.hidden_states_subsample,
                replace=True,
            ) for _ in range(batch_size)]).reshape(batch_size, self._config.hidden_states_subsample)
            batch_indices = np.arange(batch_size)[:, np.newaxis]
            feat = feat[batch_indices, sequence_indices]
            # Same indices, or xi desynchronizes from the endogenous state it is
            # being added to.
            if xi_feat is not None:
                xi_feat = xi_feat[batch_indices, sequence_indices]

        # Same indices as feat, for the same reason.
        reward_action = None
        if self._reward_head_action:
            reward_action = data["action"]
            if sequence_indices is not None:
                idx = np.arange(reward_action.shape[0])[:, np.newaxis]
                reward_action = reward_action[idx, sequence_indices]

        self._add_wm_timing(time_metrics, 'sample_features', time.time() - sample_features_time)

        repeat_data_time = time.time()
        windowed_embed = None
        if self._config.reconstruction_window > 0:
            windowed_embed, _ = tools.window_data_repeat(embed, self._config.reconstruction_window, sequence_indices)
        self._add_wm_timing(time_metrics, 'repeat_data', time.time() - repeat_data_time)

        feature_to_prediction_time = time.time()
        for name, head in self.heads.items():
            grad_head = name in self._config.grad_heads
            curr_feat = feat if grad_head else feat.detach()
            # Only the decoder heads get the exogenous branch; reward and cont are
            # endogenous by design.
            is_decoder = name in ("decoder", "multi_decoder")
            curr_xi = None
            if xi_feat is not None and is_decoder:
                curr_xi = xi_feat if grad_head else xi_feat.detach()
            if "multi" in name:
                feat_repeat = curr_feat.repeat(1, self._config.reconstruction_window, 1)
                feat_and_embed_input = torch.cat([feat_repeat, windowed_embed], dim=-1)
                if curr_xi is not None:
                    # Repeat xi the same way, so each repeated feature keeps its own
                    # exogenous state.
                    pred = head(
                        feat_and_embed_input,
                        curr_xi.repeat(1, self._config.reconstruction_window, 1),
                    )
                else:
                    pred = head(feat_and_embed_input)
            elif curr_xi is not None:
                pred = head(curr_feat, curr_xi)
            elif name == "reward":
                rf = self.reward_feat(curr_feat)
                if reward_action is not None:
                    rf = torch.cat([rf, reward_action], -1)
                pred = head(rf)
            else:
                pred = head(curr_feat)

            if type(pred) is dict:
                preds.update({f"{name},{k}": v for k, v in pred.items()})
            else:
                preds[name] = pred
        self._add_wm_timing(time_metrics, 'feature_to_prediction', time.time() - feature_to_prediction_time)

        compute_loss_time = time.time()
        losses, mses = {}, {}
        for name, pred in preds.items():
            if ',' in name:
                head_name, key_name = name.split(",")
            else:
                head_name, key_name = None, name
            if head_name is not None and head_name.startswith("multi_"):
                curr_data, masking = tools.window_data_repeat(data[key_name], self._config.reconstruction_window,
                                                              sequence_indices, shift_left=True)
            elif head_name is None and key_name.startswith("multi_"):
                curr_data, masking = tools.window_data_repeat(data[key_name[6:]], self._config.reconstruction_window,
                                                              sequence_indices, shift_left=True)
            else:
                curr_data = data[key_name]
                if sequence_indices is not None:
                    batch_indices = np.arange(curr_data.shape[0])[:, np.newaxis]
                    curr_data = curr_data[batch_indices, sequence_indices]
                masking = None
            # negative log likelihood loss
            like = pred.log_prob(curr_data)
            mse = (pred.mode().detach() - curr_data).square().sum(-1).sqrt()
            if masking is not None:
                # if there is masking we apply it:
                # note, we don't have to count the mean with exactly the same number of elements as the masking,
                # because the window size if fixed so the number of zeroed elements is fixed, therefore it is
                # like multiplying by a constant which could be adjusted by the scale.
                like = like * masking.resize_as(like, )
                mse = mse * masking.resize_as(mse, )
            loss = -torch.mean(like) * self._scales.get(name, 1.0)
            losses[name] = loss
            mses[name] = torch.mean(mse)

        self._add_wm_timing(time_metrics, 'compute_loss', time.time() - compute_loss_time)
        return losses, mses

    def reward_feat(self, feat):
        """The slice of get_feat's cat([stoch, deter]) the reward head reads."""
        if self._reward_head_input == "stoch":
            return feat[..., :self._stoch_size]
        if self._reward_head_input == "deter":
            return feat[..., self._stoch_size:]
        return feat

    def preprocess(self, obs):
        obs = obs.copy()
        if "image" in obs.keys():
            obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        if "reward" in obs:
            obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        if "is_terminal" in obs:
            # this label is necessary to train cont_head
            obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        else:
            raise ValueError('"is_terminal" was not found in observation.')
        if "time_step" in obs:
            obs["time_step"] = torch.Tensor(obs["time_step"]).unsqueeze(-1)

        task_id = obs.pop("task_id", None)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        if task_id is not None:
            obs["task_id"] = torch.as_tensor(
                np.asarray(task_id), dtype=torch.long, device=self._config.device
            )
        return obs

    def _rollout_for_logging(self, data, batch=6, context=5):
        """Posterior over the first `context` steps, then open-loop imagination.

        Returns (data, recon, openl) where recon and openl are each the
        (combined, endo_only, exo_only) triple from the decoder. The endogenous half
        is rolled with actions; the exogenous half is action-free.
        """
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:batch, :context],
            data["action"][:batch, :context],
            data["is_first"][:batch, :context],
        )
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:batch, context:], init)

        xi_post_feat, xi_prior_feat = None, None
        if self._use_exo:
            xi_states, _ = self.xi_dynamics.observe(
                self.xi_embed(data, embed)[:batch, :context],
                data["is_first"][:batch, :context],
            )
            xi_init = {k: v[:, -1] for k, v in xi_states.items()}
            horizon = data["action"][:batch, context:].shape[1]
            xi_post_feat = self.get_xi_feat(xi_states)
            xi_prior_feat = self.get_xi_feat(self.xi_dynamics.imagine(horizon, xi_init))

        decode = self.heads["decoder"].forward_parts
        recon = decode(self.dynamics.get_feat(states), xi_post_feat)
        openl = decode(self.dynamics.get_feat(prior), xi_prior_feat)
        return data, recon, openl

    def video_pred(self, data):
        data, recon, openl = self._rollout_for_logging(data)
        truth = data["image"][:6] + 0.5

        def stitch(recon_dists, openl_dists):
            """Observed segment followed by the open-loop one."""
            observed = recon_dists["image"].mode()[:6][:, :5]
            return torch.cat([observed, openl_dists["image"].mode()], 1) + 0.5

        model = stitch(recon[0], openl[0])
        rows = [truth, model]
        if self._use_exo:
            # The same sequence decoded from each stream alone.
            rows.append(stitch(recon[1], openl[1]))
            rows.append(stitch(recon[2], openl[2]))
        rows.append((model - truth + 1.0) / 2.0)
        return torch.cat(rows, 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, logger, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._logger = logger
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor

        self._reward = reward

        if config.dyn_discrete:
            stoch_size = config.dyn_stoch * config.dyn_discrete
        else:
            stoch_size = config.dyn_stoch
        feat_size = stoch_size + config.dyn_deter
        # get_feat returns cat([stoch, deter], -1); actor_input picks the slice the
        # actor reads. The value always gets the full feature.
        self._actor_input = config.actor_input
        self._stoch_size = stoch_size
        actor_feat_size = {
            "feat": feat_size,
            "stoch": stoch_size,
            "deter": config.dyn_deter,
        }[self._actor_input]
        self.actor = networks.ActionHead(
            actor_feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        value_mlp_shape = (255,) if config.value_head == "symlog_disc" else []
        self.value = networks.MLP(
            feat_size,
            value_mlp_shape,
            config.value_layers,
            config.units,
            config.act,
            config.norm,
            config.value_head,
            outscale=0.0,
            device=config.device,
        )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def actor_feat(self, feat):
        """The slice of get_feat's cat([stoch, deter]) that the actor reads."""
        if self._actor_input == "stoch":
            return feat[..., :self._stoch_size]
        if self._actor_input == "deter":
            return feat[..., self._stoch_size:]
        return feat

    def _train(
            self,
            start,
            objective=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with (tools.RequiresGrad(self.actor)):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)

                # this target is not scaled
                target, weights, base, actor_ent, state_ent = self._compute_target(
                    imag_feat, imag_state, imag_action, reward,
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat[:-1].detach()

        if self._config.log_imagined_horizon_effective_length:
            for q in [0.9, 0.2, 0.1, 0.01]:
                metrics[f'imagination_horizon_above_{q}'] = to_np(
                    torch.sum(torch.greater(weights, q)) / weights.shape[1])
        with tools.RequiresGrad(self.value):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                value = self.value(value_input)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                # slow is flag to indicate whether slow_target is used for lambda-return
                if self._config.slow_value_target:
                    slow_target = self._slow_value(value_input)
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        # Endogenous only, by construction: `dynamics` is the endogenous RSSM and
        # `start` is its posterior. The exogenous state is never imagined and the
        # policy never sees it, so the actor's feature stays the endogenous feature.
        dynamics = self._world_model.dynamics
        assert not any(k.startswith("xi_") for k in start), (
            "exogenous state leaked into the imagination start state", list(start)
        )
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(self.actor_feat(inp)).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    def _compute_target(
            self, imag_feat, imag_state, imag_action, reward
    ):
        if "cont" in self._world_model.heads:
            discount = self._config.discount * self._world_model.heads["cont"](imag_feat).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        actor_ent = None
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            actor_ent = self.actor(self.actor_feat(imag_feat)).entropy()
            reward += self._config.actor_entropy() * actor_ent
        state_ent = None
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
            reward += self._config.actor_state_entropy() * state_ent
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        target = torch.stack(target, dim=1)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1], actor_ent, state_ent

    def _compute_actor_loss(
            self,
            imag_feat,
            imag_state,
            imag_action,
            target,
            actor_ent,
            state_ent,
            weights,
            base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(self.actor_feat(inp))
        actor_ent = policy.entropy()
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        # Q-val for actor is not transformed using symlog
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])
        else:
            adv = target - base

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                    policy.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act="SiLU",
        norm="LayerNorm",
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="gru",
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
        detach_every=None
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device
        self._detach_every = detach_every

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools.weight_init)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
            self._cell.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
            self._cell.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, eps=1e-03))
            img_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.weight_init)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        should_detach = lambda i: (i+1)%self._detach_every == 0 if self._detach_every != -1 else False
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first, i: self.obs_step(
                prev_state[0], prev_act, embed, is_first, should_detach=should_detach(i)
            ),
            (action, embed, is_first, range(len(is_first))),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        # Used for video prediction
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                    tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, should_detach=False):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()

        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first  # zero out prev_action if is_first is True
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r  # zero out prev_state if is_first is True
        if should_detach:
            prev_state = {k: v.detach() for k,v in prev_state.items()}
        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self._obs_out_layers(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    # this is used for making future image
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape)
            # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_stoch, prev_action, embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._inp_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.

        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class XiRSSM(nn.Module):
    """Action-free RSSM for the exogenous latent xi.

    Mirrors RSSM's dict-state convention ({stoch, deter, logit|mean,std}) but the
    transition is xi_t -> xi_{t+1} with no action input, so the exogenous stream
    evolves independently of what the agent does. Used by the world model only:
    the policy and the imagination rollout never see xi.
    """

    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        discrete=False,
        act="SiLU",
        norm="LayerNorm",
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        cell="gru",
        unimix_ratio=0.01,
        initial="learned",
        embed=None,
        device=None,
        num_contexts=0,
        ctx_dim=16,
        ctx_cond=True,
        ctx_cls=False,
    ):
        super(XiRSSM, self).__init__()
        self._ctx_dim = (int(ctx_dim)
                         if int(num_contexts) > 0 and ctx_cond else 0)
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._discrete = discrete
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device

        # No action term here: this is the only structural difference from RSSM.
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete
        else:
            inp_dim = self._stoch
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools.weight_init)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
            self._cell.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
            self._cell.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter + self._ctx_dim
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, eps=1e-03))
            img_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.weight_init)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.weight_init)

        # Two independent ways to tie xi to the task, either or both:
        #   ctx_cond -- task-conditional prior p(xi_t | xi_{t-1}, task_id); the last
        #               embedding row is the unlabelled id.
        #   ctx_cls  -- classifier xi_deter -> task_id, supervising the *posterior*.
        # Conditioning alone leaves the posterior free to ignore the observation:
        # a prior that knows the task can predict xi outright, so the KL goes to
        # zero and xi_post drifts to uniform.
        self._num_contexts = int(num_contexts)
        if self._num_contexts > 0 and ctx_cond:
            self._ctx_emb = nn.Embedding(self._num_contexts + 1, ctx_dim)
            self._ctx_emb.apply(tools.weight_init)
        if self._num_contexts > 0 and ctx_cls:
            self._ctx_cls = nn.Linear(self._deter, self._num_contexts)
            self._ctx_cls.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def context_logits(self, state):
        return self._ctx_cls(state["deter"])

    @property
    def feat_size(self):
        if self._discrete:
            return self._stoch * self._discrete + self._deter
        return self._stoch + self._deter

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, is_first, state=None, task_id=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(embed.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        if task_id is None:
            task_id = torch.full(embed.shape[:2], -1, dtype=torch.long,
                                 device=embed.device)
        embed, is_first, task_id = swap(embed), swap(is_first), swap(task_id[..., None])
        post, prior = tools.static_scan(
            lambda prev_state, embed, is_first, task_id: self.obs_step(
                prev_state[0], embed, is_first, task_id=task_id.squeeze(-1)
            ),
            (embed, is_first, task_id),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, horizon, state, task_id=None):
        """Action-free prior rollout. Used for logging only, never for behavior."""
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        priors = []
        prior = state
        for _ in range(horizon):
            prior = self.img_step(prior, task_id=task_id)
            priors.append(prior)
        prior = {k: torch.stack([p[k] for p in priors], 0) for k in priors[0].keys()}
        return {k: swap(v) for k, v in prior.items()}

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, embed, is_first, sample=True, task_id=None):
        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r
        prior = self.img_step(prev_state, sample, task_id=task_id)
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def _ctx_feat(self, task_id, ref):
        """Embedding for task_id, mapping the -1 "unlabelled" id to the last row."""
        if self._ctx_dim == 0:
            return None
        if task_id is None:
            idx = torch.full(ref.shape[:-1], self._num_contexts,
                             dtype=torch.long, device=ref.device)
        else:
            idx = task_id.long()
            idx = torch.where((idx >= 0) & (idx < self._num_contexts),
                              idx, torch.full_like(idx, self._num_contexts))
        return self._ctx_emb(idx)

    def img_step(self, prev_state, sample=True, task_id=None):
        """xi_t -> xi_{t+1}, with no dependence on the action."""
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        x = self._inp_layers(prev_stoch)
        deter = prev_state["deter"]
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.
        ctx = self._ctx_feat(task_id, deter)
        x = self._img_out_layers(x if ctx is None else torch.cat([x, ctx], -1))
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter, task_id=None):
        ctx = self._ctx_feat(task_id, deter)
        x = self._img_out_layers(deter if ctx is None else torch.cat([deter, ctx], -1))
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if name == "ims":
            x = self._ims_stat_layer(x)
        elif name == "obs":
            x = self._obs_stat_layer(x)
        else:
            raise NotImplementedError
        if self._discrete:
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        mean, std = torch.split(x, [self._stoch] * 2, -1)
        mean = {
            "none": lambda: mean,
            "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
        }[self._mean_act]()
        std = {
            "softplus": lambda: torch.softplus(std),
            "abs": lambda: torch.abs(std + 1),
            "sigmoid": lambda: torch.sigmoid(std),
            "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
        }[self._std_act]()
        std = std + self._min_std
        return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class CategoricalCLUB(nn.Module):
    """CLUB with a categorical q(xi_stoch | z, context), for discrete xi.

    Same bound as `CLUB` -- only the likelihood family changes. CLUB needs *some*
    conditional density it can evaluate at any (z, xi) pair; nothing about it
    requires a Gaussian.

    The Gaussian version is misspecified when xi is a flattened one-hot draw, and
    it fails in two linked ways that the diagnostics show directly:

    - It drives the variance to the floor (`logvar_frac_floor` ~0.96), because a
      near-deterministic target makes q want sigma -> 0. The 1/var factor then
      inflates both terms of the bound to |pos|, |neg| ~ 48.
    - Worse, it cannot discriminate. Under a Gaussian, log q is -||mu - xi||^2 /
      2 sigma^2, and the L2 distance between any two *distinct* one-hot patterns is
      roughly constant -- so a wrong xi scores about as well as the right one,
      pos ~ neg, and the bound carries no signal.

    A categorical head fixes both: no variance parameter to saturate, and a wrong
    one-hot gets a genuinely low log-probability.

    q reads the endogenous stream alone, so the bound is the unconditional
    I(z; xi_stoch). Feeding xi_deter in as a conditioning context was tried and is
    a trap: it lets q predict xi_stoch from the exogenous side, which explains the
    shared information away and drives the bound to ~0 (pos -6.0 vs neg -6.2) even
    though q is predicting well below chance. The endogenous stream is the one that
    has to forget xi, so it is the only thing q may look at.

    This leaves xi_deter unpenalized as a *target*: if the distractor rides in the
    exogenous recurrent state, this bound will not see it. That is deliberate --
    xi is supposed to encode the distractor -- but it means the deter probe stays
    the check on where the shortcut actually lives.
    """

    def __init__(self, z_dim, groups, classes, hidden=256, layers=2,
                 act="SiLU", norm="LayerNorm"):
        super(CategoricalCLUB, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._groups = groups
        self._classes = classes

        mods, inp = [], z_dim
        for _ in range(layers):
            mods += [nn.Linear(inp, hidden, bias=False), norm(hidden, eps=1e-03), act()]
            inp = hidden
        mods += [nn.Linear(inp, groups * classes)]
        self._logits = nn.Sequential(*mods)
        self._logits.apply(tools.weight_init)

    @staticmethod
    def _flat(x):
        return x.reshape(-1, x.shape[-1])

    def _log_prob(self, z, xi):
        """log q(xi | z) for flat z (n, d) and flat one-hot xi (m, groups*classes).

        Returns (n, m): every z scored against every xi, which is what the negative
        term needs. The positives are the diagonal.
        """
        logits = self._logits(z).reshape(-1, self._groups, self._classes)
        logp = torch.log_softmax(logits, dim=-1)
        onehot = xi.reshape(-1, self._groups, self._classes)
        # (n, 1, g, c) * (1, m, g, c) summed over classes then groups: picks each
        # target's log-probability and adds the (independent) groups.
        return torch.einsum("ngc,mgc->nm", logp, onehot)

    def learning_loss(self, z, xi):
        """Cross-entropy of q(xi_stoch | z). Detached: fits q only."""
        z, xi = self._flat(z).detach(), self._flat(xi).detach()
        # Only the diagonal (each z with its own xi) trains q.
        return -self._log_prob(z, xi).diagonal().mean()

    def mi_est(self, z, xi):
        """The CLUB upper bound. Gradients flow to z and xi."""
        z, xi = self._flat(z), self._flat(xi)
        logp = self._log_prob(z, xi)
        return (logp.diagonal() - logp.mean(1)).mean()

    @torch.no_grad()
    def mi_stats(self, z, xi):
        """Decomposition of the bound. See CLUB.mi_stats -- same fields, minus the
        logvar diagnostics, which have no analogue here (that is the point)."""
        z, xi = self._flat(z), self._flat(xi)
        logp = self._log_prob(z, xi)
        pos, neg = logp.diagonal(), logp.mean(1)
        diff = pos - neg
        std = diff.std()
        return {
            "pos_mean": pos.mean(),
            "neg_mean": neg.mean(),
            "diff_std": std,
            "diff_snr": diff.mean().abs() / std.clamp(min=1e-8),
            "pos_abs_mean": pos.abs().mean(),
        }


class CLUB(nn.Module):
    """Contrastive Log-ratio Upper Bound on I(z; xi)  (Cheng et al., 2020).

    A variational net q(xi | z) = N(mu(z), sigma(z)^2) gives

        I(z; xi) <= E_i[log q(xi_i|z_i)] - E_{i,j}[log q(xi_j|z_i)]

    which is a valid upper bound only when q has actually fit p(xi|z). So the
    module is used in two alternating passes:

    - `learning_loss(z, xi)`: -log q(xi|z) on *detached* inputs, stepped by its own
      optimizer. Fits the approximator; never touches world-model gradients.
    - `mi_est(z, xi)`: the bound itself, added to the world-model loss so that
      minimizing it pushes the endogenous and exogenous streams apart.

    Inputs are flattened over batch and time, so the negative pairs (i, j) mix
    across both -- the "wrong" xi for a given z is usually a different task.
    """

    def __init__(self, z_dim, xi_dim, hidden=256, layers=2, act="SiLU", norm="LayerNorm",
                 normalize_xi=True, norm_momentum=0.99):
        super(CLUB, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)

        def trunk(out_dim, tail):
            mods, inp = [], z_dim
            for _ in range(layers):
                mods += [nn.Linear(inp, hidden, bias=False), norm(hidden, eps=1e-03), act()]
                inp = hidden
            mods += [nn.Linear(inp, out_dim)] + tail
            net = nn.Sequential(*mods)
            net.apply(tools.weight_init)
            return net

        self._mu = trunk(xi_dim, [])
        # Tanh-bounded logvar: an unbounded one lets q shrink its variance without
        # limit, and the bound blows up on the negative pairs.
        self._logvar = trunk(xi_dim, [nn.Tanh()])

        # Running standardizer for xi. The logvar head is bounded to [-1, 1], so q
        # can only express variances in [e^-1, e^1]. When xi's own per-dimension
        # scale sits far below that, q pins every dimension at the floor and the
        # squared errors are inflated by 1/var -- which is what makes |pos| and
        # |neg| both large while their difference (the actual bound) stays small.
        # Standardizing xi puts its scale inside the range q can represent, so the
        # floor stops binding.
        #
        # Buffers, not parameters: these are updated from data under no_grad, never
        # by the optimizer. Registered so they survive checkpoint save/load -- the
        # normalization must be identical when a run resumes.
        self._normalize_xi = normalize_xi
        self._norm_momentum = norm_momentum
        self.register_buffer("_xi_mean", torch.zeros(xi_dim))
        self.register_buffer("_xi_var", torch.ones(xi_dim))
        self.register_buffer("_norm_inited", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _update_norm(self, xi):
        """Refresh the running xi statistics. Call once per step, on flat xi."""
        mean, var = xi.mean(0), xi.var(0, unbiased=False)
        if not bool(self._norm_inited):
            # Seed from the first batch rather than crawling up from (0, 1), which
            # would leave the scale wrong for the first few hundred steps.
            self._xi_mean.copy_(mean)
            self._xi_var.copy_(var)
            self._norm_inited.fill_(True)
        else:
            m = self._norm_momentum
            self._xi_mean.mul_(m).add_(mean, alpha=1 - m)
            self._xi_var.mul_(m).add_(var, alpha=1 - m)

    def _normalize(self, xi):
        """Apply the running standardization to xi.

        The statistics are buffers updated under no_grad, so this is a fixed affine
        map at each step: gradients reach xi itself but never flow through the
        mean/std. Normalizing with in-graph batch statistics would both reroute
        world-model gradients through them and give q a target that rescales every
        step.
        """
        if not self._normalize_xi:
            return xi
        return (xi - self._xi_mean) * torch.rsqrt(self._xi_var + 1e-8)

    def _stats(self, z):
        return self._mu(z), self._logvar(z)

    @staticmethod
    def _flat(x):
        return x.reshape(-1, x.shape[-1])

    @torch.no_grad()
    def logvar_stats(self, z):
        """Diagnostics on q's predicted log-variance.

        The logvar head is tanh-bounded to [-1, 1]. A mean pinned near -1, or a
        large `frac_floor`, means q is driving its variance to the floor -- what
        a Gaussian does when asked to model near-deterministic targets such as
        one-hot samples. That is the signature of a misspecified q: the fit can
        still look converged while the bound it supports is loose.
        """
        lv = self._logvar(self._flat(z).detach())
        return {
            "logvar_mean": lv.mean(),
            "logvar_min": lv.min(),
            "logvar_frac_floor": (lv < -0.95).float().mean(),
        }

    @torch.no_grad()
    def mi_stats(self, z, xi):
        """Decomposition of the bound, for telling variance from cancellation.

        `mi_est` returns only the mean of (pos - neg). Two different pathologies
        produce a noisy gradient from it, and they need different fixes:

        - High variance: the per-sample spread of (pos - neg) is large relative
          to its mean. The distractor is constant within an episode, so the
          negative average has an effective sample size of roughly the number of
          episodes in the batch, not the number of flattened rows.
        - Catastrophic cancellation: `pos` and `neg` are both large and nearly
          equal while their difference is small, so the relative error in the
          difference is far larger than in either term -- and it worsens as the
          bound approaches zero, i.e. as training succeeds.

        `diff_std` is the spread across samples within one batch, not across
        training steps; `diff_snr` is |mean| / std, so small means noise-dominated.
        """
        z, xi = self._flat(z), self._flat(xi)
        xi = self._normalize(xi)
        mu, logvar = self._stats(z)
        var = logvar.exp()
        pos = -((mu - xi) ** 2 / var).sum(-1) / 2
        neg = -(
            (mu.unsqueeze(1) - xi.unsqueeze(0)) ** 2 / var.unsqueeze(1)
        ).sum(-1).mean(1) / 2
        diff = pos - neg
        std = diff.std()
        return {
            "pos_mean": pos.mean(),
            "neg_mean": neg.mean(),
            "diff_std": std,
            # Guarded so an exactly-constant diff logs 0 rather than inf/nan.
            "diff_snr": diff.mean().abs() / std.clamp(min=1e-8),
            # Scale of the terms being differenced. If |pos| >> |diff|, the bound
            # is a small residual of large numbers.
            "pos_abs_mean": pos.abs().mean(),
        }

    def learning_loss(self, z, xi):
        """Negative log-likelihood of q(xi|z). Detached: fits q only."""
        z, xi = self._flat(z).detach(), self._flat(xi).detach()
        # The single per-step update of the running statistics. This runs after
        # mi_est in the training loop, so the bound and the fit both use the same
        # (previous-step) statistics -- q is never asked to chase a target that was
        # rescaled after the bound that shaped it was taken.
        if self._normalize_xi:
            self._update_norm(xi)
        xi = self._normalize(xi)
        mu, logvar = self._stats(z)
        return ((mu - xi) ** 2 / logvar.exp() + logvar).sum(-1).mean() / 2

    def mi_est(self, z, xi):
        """The CLUB upper bound on I(z; xi). Gradients flow to z and xi."""
        z, xi = self._flat(z), self._flat(xi)
        # Fixed affine map (buffers), so gradients still reach xi. MI is invariant
        # under an invertible transform of either argument, so the bound is on the
        # same quantity as before -- only its conditioning changes.
        xi = self._normalize(xi)
        mu, logvar = self._stats(z)
        # Positive pairs: each z with its own xi.
        pos = -((mu - xi) ** 2 / logvar.exp()).sum(-1) / 2
        # Negative pairs: every z against every xi in the batch, averaged over j.
        # (n, 1, d) vs (1, n, d) -> (n, n); the mean over dim 1 is E_j.
        neg = -(
            (mu.unsqueeze(1) - xi.unsqueeze(0)) ** 2 / logvar.exp().unsqueeze(1)
        ).sum(-1).mean(1) / 2
        # The log-variance term is identical in pos and neg and cancels, so it is
        # dropped from both above.
        return (pos - neg).mean()


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        input_reward
    ):
        super(MultiEncoder, self).__init__()
        if input_reward is True:
            excluded = ("is_first", "is_last", "is_terminal", "task_id")
        else:
            excluded = ("is_first", "is_last", "is_terminal", "reward", "task_id")

        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
            input_reward,
            xi_feat_size=0,
            cnn_upsample="transpose",
    ):
        super(MultiDecoder, self).__init__()
        if input_reward is True:
            excluded = ("is_first", "is_last", "is_terminal", "task_id")
        else:
            excluded = ("is_first", "is_last", "is_terminal", "reward", "task_id")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        # When on, a second decoder branch reads the exogenous feature and its output
        # is added to the endogenous one. Only genuine observation keys get that
        # branch: reward and time_step are auxiliary signals the policy depends on,
        # so they stay purely endogenous. For an image env that leaves the CNN keys
        # plus nothing else; for bandits it leaves "state" alone.
        self._use_xi = xi_feat_size > 0
        self._xi_excluded = ("reward", "time_step")
        self.xi_mlp_shapes = {
            k: v for k, v in self.mlp_shapes.items() if k not in self._xi_excluded
        }
        if self._use_xi:
            print("Decoder exogenous CNN shapes:", self.cnn_shapes)
            print("Decoder exogenous MLP shapes:", self.xi_mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            # Both branches emit raw logits; forward_parts applies cnn_sigmoid once,
            # to their sum, as r2dreamer does -- sigmoid(endo + exo), not
            # sigmoid(endo) + sigmoid(exo), which would span [0, 2].
            self._cnn_sigmoid = cnn_sigmoid
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                cnn_sigmoid=False,
                upsample=cnn_upsample,
            )
            if self._use_xi:
                # Same output shape as the endogenous branch, so the two means add.
                self._cnn_xi = ConvDecoder(
                    xi_feat_size,
                    shape,
                    cnn_depth,
                    act,
                    norm,
                    kernel_size,
                    minres,
                    cnn_sigmoid=False,
                    upsample=cnn_upsample,
                )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
            )
            # Sized to xi_mlp_shapes, so this branch cannot emit reward/time_step.
            if self._use_xi and self.xi_mlp_shapes:
                self._mlp_xi = MLP(
                    xi_feat_size,
                    self.xi_mlp_shapes,
                    mlp_layers,
                    mlp_units,
                    act,
                    norm,
                    vector_dist,
                )
        self._image_dist = image_dist

    def forward(self, features, xi_features=None, split=False):
        """Unchanged contract: a dict holding every key this decoder emits.

        With `split`, the two branches are returned as separate predictions --
        exogenous keys prefixed 'exo_' -- instead of one distribution over their
        sum, so each can be trained against its own target.
        """
        combined, endo_only, exo_only = self.forward_parts(features, xi_features)
        if not split or exo_only is None:
            return combined
        out = {k: endo_only.get(k, v) for k, v in combined.items()}
        out.update({f"exo_{k}": v for k, v in exo_only.items()})
        return out

    def forward_parts(self, features, xi_features=None):
        """Decode into (combined, endo_only, exo_only), each a dict keyed as before.

        `combined` holds every key, exactly as `forward` always has, and is what the
        loss consumes. Observation keys are the sum of the two branches, taken on the
        pre-distribution mean so that one distribution -- and so one likelihood term
        -- is built from the sum. Keys with no exogenous branch (reward, time_step)
        pass through as the endogenous prediction alone.

        `endo_only` and `exo_only` decode the branches separately for logging and
        carry no loss; `exo_only` holds only the keys that have an exogenous branch.
        Both are None when xi is off.
        """
        use_xi = self._use_xi and xi_features is not None
        combined, endo_only, exo_only = {}, {}, {}

        if self.cnn_shapes:
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]

            def _split_image(x):
                # With cnn_sigmoid, x is a logit: squash into [0, 1], then shift to
                # the [-0.5, 0.5] space preprocess puts images in. The branches are
                # summed *before* this, so combined is sigmoid(endo + exo) and each
                # branch alone renders as sigmoid(branch), like r2dreamer's video.
                if self._cnn_sigmoid:
                    x = torch.sigmoid(x) - 0.5
                return {
                    key: self._make_image_dist(out)
                    for key, out in zip(
                        self.cnn_shapes.keys(), torch.split(x, split_sizes, -1)
                    )
                }

            mean = self._cnn(features)
            if use_xi:
                xi_mean = self._cnn_xi(xi_features)
                combined.update(_split_image(mean + xi_mean))
                endo_only.update(_split_image(mean))
                exo_only.update(_split_image(xi_mean))
            else:
                combined.update(_split_image(mean))

        if self.mlp_shapes:
            stats = self._mlp.forward_stats(features)

            def _make_vector(source):
                return {
                    name: self._mlp.dist(
                        self._mlp._dist, mean, std, self.mlp_shapes[name]
                    )
                    for name, (mean, std) in source.items()
                }

            if use_xi and self.xi_mlp_shapes:
                xi_stats = self._mlp_xi.forward_stats(xi_features)
                # Every key survives; only those with an exogenous branch are summed.
                total = {
                    name: (mean + xi_stats[name][0], std)
                    if name in xi_stats
                    else (mean, std)
                    for name, (mean, std) in stats.items()
                }
                combined.update(_make_vector(total))
                endo_only.update(_make_vector(stats))
                exo_only.update(_make_vector(xi_stats))
            else:
                combined.update(_make_vector(stats))

        if not use_xi:
            return combined, None, None
        return combined, endo_only, exo_only

    def branch_means(self, features, xi_features):
        """Pre-distribution (endo_mean, exo_mean) per key, for the decorrelation
        penalty. The decoder only ever scores their sum, so the two are free to
        drift into large opposite-signed values that cancel -- which they do."""
        out = {}
        if not (self._use_xi and xi_features is not None):
            return out
        if self.mlp_shapes and self.xi_mlp_shapes:
            stats = self._mlp.forward_stats(features)
            xi_stats = self._mlp_xi.forward_stats(xi_features)
            for name in xi_stats:
                if name in stats:
                    out[name] = (stats[name][0], xi_stats[name][0])
        if self.cnn_shapes:
            out["_cnn"] = (self._cnn(features), self._cnn_xi(xi_features))
        return out

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm="LayerNorm",
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        h, w, input_ch = input_shape
        layers = []
        for i in range(int(np.log2(h) - np.log2(minres))):
            if i == 0:
                in_dim = input_ch
            else:
                in_dim = 2 ** (i - 1) * depth
            out_dim = 2**i * depth
            layers.append(
                Conv2dSame(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            layers.append(ChLayerNorm(out_dim))
            layers.append(act())
            h, w = h // 2, w // 2

        self.outdim = out_dim * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=nn.LayerNorm,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
        upsample="transpose",
    ):
        super(ConvDecoder, self).__init__()
        if upsample not in ("transpose", "nearest"):
            raise ValueError(f"upsample must be 'transpose' or 'nearest', got {upsample!r}")
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres**2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools.weight_init)
        in_dim = self._embed_size // (minres**2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres**2) // (2 ** (i + 1))
            bias = False
            initializer = tools.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            if upsample == "nearest":
                # r2dreamer's decoder: nearest upsample, then a stride-1 conv. A
                # stride-2 transposed conv overlaps its kernel unevenly and paints a
                # checkerboard from initialisation; with an additive exo branch the
                # two branches' checkerboards co-adapt to cancel in the sum, so each
                # branch alone renders as noise.
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                layers.append(
                    nn.Conv2d(in_dim, out_dim, kernel_size, 1, padding="same", bias=bias)
                )
            else:
                pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
                pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
                layers.append(
                    nn.ConvTranspose2d(
                        in_dim,
                        out_dim,
                        kernel_size,
                        2,
                        padding=(pad_h, pad_w),
                        output_padding=(outpad_h, outpad_w),
                        bias=bias,
                    )
                )
            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm="LayerNorm",
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward_stats(self, features):
        """Return raw (mean, std) before the distribution is built.

        Split out of forward so that an additive decoder can sum the endogenous and
        exogenous means and build a single distribution from the sum.
        """
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            stats = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                stats[name] = (mean, std)
            return stats
        mean = self.mean_layer(out)
        if self._std == "learned":
            std = self.std_layer(out)
        else:
            std = self._std
        return (mean, std)

    def forward(self, features, dtype=None):
        stats = self.forward_stats(features)
        if self._shape is None:
            return stats
        if isinstance(self._shape, dict):
            return {
                name: self.dist(self._dist, mean, std, self._shape[name])
                for name, (mean, std) in stats.items()
            }
        mean, std = stats
        return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return tools.SymlogDist(mean)
        raise NotImplementedError(dist)


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            x = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

import argparse
import datetime
import functools
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

os.environ["GYM_NOTICES_ENABLED"] = "false"  # silence gym's unmaintained-version banner
import gym

os.environ.setdefault("MUJOCO_GL", "glfw" if os.environ.get("DISPLAY") else "egl")

import numpy as np
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import task_split
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        # if mamba_context, we need to clip the data
        self._should_increase_wm_horizon = False
        if config.mamba_context:
            horizon_diff = config.batch_length - config.initial_wm_horizon
            if horizon_diff < 0:
                raise ValueError(
                    "batch_length should be larger than initial_wm_horizon in mamba_context mode"
                )
            if horizon_diff > 0:
                self._horizon_increase_rate = horizon_diff / config.reach_wm_horizon_limit
                self._should_increase_wm_horizon = True
        self._wm_horizon = config.initial_wm_horizon

        self._metrics = {}
        self._update_count = 0
        # Schedules.
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._logger.get_agent_frames()
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._logger.get_agent_frames())
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._logger.get_agent_frames()
        )
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._logger.get_agent_frames(), config)
        if config.compile and os.name != "nt":  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
        reward_prediction = lambda f, s, a: self._wm.heads["reward"](f).mode()
        self._task_behavior = models.ImagBehavior(
            config, logger, self._wm, config.behavior_stop_grad, reward_prediction)
        if config.compile and os.name != "nt":  # compilation is not supported on windows
            self._task_behavior = torch.compile(self._task_behavior)
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            # random=lambda: expl.Random(config, act_space),
            # epsilon_greedy=lambda: expl.EpsilonGreedy(config, act_space, self._task_behavior),
            # plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        if self._should_reset(self._logger.get_agent_frames()):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]

        with torch.no_grad():
            policy_output, state = self._policy(obs, state, training)

        if training:
            self._logger.step += self._config.action_repeat * len(reset)

        return policy_output, state

    def update_models(self, steps):
        batch_times, update_times = [], []
        for _ in range(steps):
            batch_time = time.time()
            batch = next(self._dataset)
            batch_times.append(time.time() - batch_time)

            update_time = time.time()
            self._train(batch)
            update_times.append(time.time() - update_time)

            self._update_count += 1
            self._metrics["update_count"] = self._update_count
        if self._should_log(self._logger.get_agent_frames()):
            for name, values in self._metrics.items():
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []
            if self._config.video_pred_log:
                openl = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl", to_np(openl))
            self._logger.scalar("train_batch_time", float(np.mean(batch_times)))
            self._logger.scalar("train_update_time", float(np.mean(update_times)))
            self._logger.write(fps=True)

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = next(iter(obs.values())).shape[0]
            latent = self._wm.dynamics.initial(batch_size)
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        # Must match the slice the actor was trained on.
        actor_feat = self._task_behavior.actor_feat(feat)
        if not training:
            actor = self._task_behavior.actor(actor_feat)
            action = actor.mode()
        elif self._should_expl(self._logger.get_agent_frames()):
            actor = self._expl_behavior.actor(self._expl_behavior.actor_feat(feat))
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(actor_feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)

    def _train(self, data):
        if self._config.mamba_context:
            # only use the first wm_horizon data points
            data = {k: data[k][:, :self._wm_horizon] for k in data}
            # handle horizon increase
            if self._should_increase_wm_horizon:
                if self._config.batch_length > self._wm_horizon:
                    counter = self._logger.get_agent_frames()
                    self._wm_horizon = int(
                        np.ceil(counter * self._horizon_increase_rate + self._config.initial_wm_horizon))
                    self._wm_horizon = min(self._wm_horizon, self._config.batch_length)
        metrics = {
            'batch_rewards_mean': data['reward'].mean(),
            'batch_rewards_min': data['reward'].min(),
            'batch_rewards_max': data['reward'].max(),
            'batch_rewards_positive': (data['reward'] > 0).mean(),
        }
        if self._config.mamba_context:
            metrics['wm_horizon'] = self._wm_horizon
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        # start['deter'] (16, 64, 512)

        batch_indices = None
        if self._config.behavior_batch_length != -1:
            # randomly subsample a minibatch to train the behavior
            batch_size, batch_length = start["deter"].shape[:2]

            subsample_batch_for_policy_time = time.time()
            idx = np.array([np.random.choice(
                np.arange(batch_length),
                self._config.behavior_batch_length,
                replace=True,
            ) for _ in range(batch_size)]).reshape(batch_size, self._config.behavior_batch_length)
            batch_indices = np.arange(batch_size)[:, np.newaxis]
            start = {k: v[batch_indices, idx] for k, v in start.items()}
            metrics["subsample_batch_for_policy_time"] = time.time() - subsample_batch_for_policy_time

        task_policy_train_time = time.time()
        metrics.update(self._task_behavior._train(start)[-1])
        metrics["task_policy_train_time"] = time.time() - task_policy_train_time

        if self._config.expl_behavior != "greedy" and self._config.expl_behavior != "epsilon_greedy":
            if batch_indices is not None:
                context = {k: v[batch_indices, idx] for k, v in context.items()}
                data = {k: v[batch_indices, idx] for k, v in data.items()}
            expl_policy_train_time = time.time()
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics['expl_policy_train_time'] = time.time() - expl_policy_train_time
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length, sample_first=config.sample_first)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, pool=None):
    suite, task = config.task.split("_", 1)
    if suite == "bandits":
        import envs.bandits as bandits

        generator = bandits.DistractorDataset(
            num_tasks=config.task_pool_size,
            num_arms=config.num_arms,
            distractor_dim=config.distractor_dim,
            seed=config.generator_seed,
            reward_mode=config.reward_mode,
            static_distractor=config.static_distractor,
        )
        # One list per split, never the val+test union: surplus envs reset with
        # no task and fall through to sample_task(), which draws from allowed_ids.
        allowed_ids = None
        if pool is not None:
            allowed_ids = {
                "train": pool.train_tasks,
                "eval": pool.val_tasks,
                "test": pool.test_tasks,
            }[mode]()
        env = bandits.BanditEnv(generator, num_steps=config.max_episode_length,
                                allowed_ids=allowed_ids,
                                use_distractor=config.use_distractor)
        env = wrappers.OneHotAction(env)
    elif suite == "dmc":
        import envs.dmc as dmc
        if config.meta_learning:
            task = task + "meta"
        environment_kwargs = {"theta": config.goal_dist_theta, "radius": config.goal_dist_radius,
                              "dense_reward": config.dense_reward,
                              "num_goals": config.num_goals} if "goal" in task else {}
        env = dmc.DeepMindControl(task, config.action_repeat, config.size,
                                  detach_image_from_obs=config.detach_image_from_obs,
                                  environment_kwargs=environment_kwargs,
                                  small_state_space=config.small_state_space)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari
        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task, mode if "train" in mode else "test", config.action_repeat
        )
        env = wrappers.OneHotAction(env)
    elif suite == "rooms":
        import envs.rooms as rooms

        env = rooms.RoomNaviNew(num_cells=config.num_cells,
                                num_rooms=config.num_rooms)
        env = wrappers.OneHotAction(env)
    elif suite == "halfcircle":
        import envs.point_robot as point_robot
        if not config.meta_learning:
            raise ValueError("Halfcircle only supports meta-learning")
        env = point_robot.SparsePointEnv(goal_radius=config.goal_radius)
    elif suite == "halfcirclewind":
        import envs.point_robot_wind as point_robot_wind
        if not config.meta_learning:
            raise ValueError("HalfcircleWind only supports meta-learning")
        env = point_robot_wind.SparsePointWindEnv(goal_radius=config.goal_radius, wind_force=config.wind_force)
    elif suite == "escaperoom":
        import envs.point_robot_barrier as point_robot_barrier
        if not config.meta_learning:
            raise ValueError("EscapeRoom only supports meta-learning")
        env = point_robot_barrier.PointEnvBarrier()
    elif suite == "humanoiddir":
        import envs.mujoco.humanoid_dir as humanoid_dir
        if not config.meta_learning:
            raise ValueError("HumanoidDir only supports meta-learning")
        env = humanoid_dir.HumanoidDirEnv()
        env = wrappers.NormalizeActions(env)
    elif suite == "crafter":
        import envs.crafter as crafter
        env = crafter.Crafter(
            task, outdir="./stats"
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memory-maze":
        env = gym.make(task)
        if hasattr(env.action_space, 'n'):
            env = wrappers.OneHotAction(env)
        return env
    elif suite == "panda-reach":
        from envs.custom_reach import CustomReachEnv
        env = CustomReachEnv()
    else:
        raise NotImplementedError(suite)

    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)
    if config.meta_learning:
        max_time_steps = config.max_episode_length * config.num_meta_episodes
        env = wrappers.TimeAugmentedState(env, max_time_steps)
        env = wrappers.MetaLearningEnv(env, config.num_meta_episodes, config.max_episode_length)
    else:
        env = wrappers.TimeLimit(env, config.time_limit)
    return env


def main(config):
    curr_log_file = pathlib.Path(
        datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "_" + hashlib.sha256(str(config).encode()).hexdigest())
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir = logdir / curr_log_file
    print("Logging to ", logdir)
    if config.all_layers != 0:
        config.reward_layers = config.all_layers
        config.cont_layers = config.all_layers
        config.value_layers = config.all_layers
        config.actor_layers = config.all_layers

    config.decoder["input_reward"] = config.meta_learning
    config.encoder["input_reward"] = config.meta_learning

    if config.mamba_context:
        config.batch_length = config.num_meta_episodes * config.max_episode_length

    config.sample_first = config.mamba_context

    if not config.meta_learning and config.num_meta_episodes > 1:
        raise ValueError("Cannot use more than one meta episode without meta learning")

    # Previously dead config: without this, repeats of one config were unreproducible.
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(logdir, 'config.txt'), 'wb') as f:
        pickle.dump(config, f)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(config.action_repeat * step, config)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    # The split has to exist before the envs, since each env is constructed with
    # the ids it is allowed to use.
    pool = None
    if config.task_split:
        pool = task_split.TaskPool(
            exhaustive_fn=lambda: list(range(config.task_pool_size)),
            train_size=config.task_train_size,
            val_size=config.task_val_size,
            test_size=config.task_test_size,
            seed=config.task_split_seed,
            contiguous=config.task_split_contiguous,
        )
        described = task_split.describe_pool(pool)
        print(f"Task split: {described}")
        with open(logdir / "task_split.json", "w") as f:
            json.dump({**described,
                       "train": [int(t) for t in pool.train_tasks()],
                       "val": [int(t) for t in pool.val_tasks()],
                       "test": [int(t) for t in pool.test_tasks()]}, f, indent=2)
    make = lambda mode: make_env(config, mode, pool=pool)
    helper_env = make("eval")
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    # Separate from eval_envs, which are restricted to val.
    test_envs = [make("test") for _ in range(config.envs)] if pool is not None else eval_envs
    state2img = helper_env.state2image if hasattr(helper_env, "state2image") else None
    if pool is not None:
        sample_train_task = pool.sample_train
        sample_val_task = pool.sample_val
        sample_test_task = pool.sample_test
    else:
        sample_task = lambda: helper_env.sample_task() if config.meta_learning else None
        sample_train_task = sample_val_task = sample_test_task = sample_task
    if config.envs > 1:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
        test_envs = ([Parallel(env, "process") for env in test_envs]
                     if pool is not None else eval_envs)
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
        test_envs = [Damy(env) for env in test_envs] if pool is not None else eval_envs
    acts = helper_env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} episodes).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        prefill_tasks = [sample_train_task() for _ in range(config.prefill)]
        _, steps_taken, _ = tools.simulate(
            random_agent,
            train_envs,
            prefill_tasks,
            train_eps,
            config.traindir,
            logger,
            is_eval=True,
            limit=config.dataset_size,
            state2image=state2img,
            num_meta_episodes=config.num_meta_episodes,
        )
        logger.step += steps_taken * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        helper_env.observation_space,
        helper_env.action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest_model.pt").exists():
        agent.load_state_dict(torch.load(logdir / "latest_model.pt"))
        agent._should_pretrain._once = False

    eval_scheduler = tools.Every(config.eval_every_collection_episodes)
    collected_episodes = 0
    # Fixed across evaluations, so best_return comparisons over training are
    # measuring the model rather than the task draw. Sampled without replacement
    # so a small val pool is covered evenly.
    if pool is not None:
        eval_tasks = pool.sample_val_batch(config.eval_episode_num)
        # Held-out return is only interpretable next to seen-task return under the
        # same policy, so evaluate a matched set of train tasks too.
        train_eval_tasks = pool.sample_train_batch(config.eval_episode_num)
    else:
        eval_tasks = [sample_val_task() for _ in range(config.eval_episode_num)]
        train_eval_tasks = None
    training_times = []
    best_return = -np.inf
    eval_policy = functools.partial(agent, training=False)
    while logger.get_agent_frames() < config.steps:
        logger.write()
        if eval_scheduler(collected_episodes):
            dreamer_eval_time = time.time()
            print(f"Start evaluation (episodes {collected_episodes}).")
            _, _, eval_return = tools.simulate(
                eval_policy,
                eval_envs,
                eval_tasks,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                state2image=state2img,
                num_meta_episodes=config.num_meta_episodes,
            )
            if train_eval_tasks is not None:
                # Same policy, same episode count, seen tasks: the difference is
                # attributable to task familiarity rather than to policy mode.
                _, _, train_eval_return = tools.simulate(
                    eval_policy,
                    eval_envs if pool is None else train_envs,
                    train_eval_tasks,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    state2image=state2img,
                    num_meta_episodes=config.num_meta_episodes,
                    # Without its own prefix this pass overwrites the val pass's
                    # eval_* keys at the same env_step.
                    metric_prefix="train_eval",
                )
                logger.scalar("generalization_gap", train_eval_return - eval_return)
            if eval_return > best_return:
                best_return = eval_return
                torch.save(agent.state_dict(), logdir / "best_model.pt")
                logger.scalar("best_return", best_return)
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
            logger.scalar("dreamer_eval_time", time.time() - dreamer_eval_time)
            logger.scalar("dreamer_training_time", np.sum(training_times))
            training_times.clear()
        dreamer_training_time = time.time()
        print("Start training.")
        train_tasks = [sample_train_task() for _ in range(config.envs)]
        _, steps_taken, _ = tools.simulate(
            agent,
            train_envs,
            train_tasks,
            train_eps,
            config.traindir,
            logger,
            is_eval=False,
            limit=config.dataset_size,
            num_meta_episodes=config.num_meta_episodes
        )
        collected_episodes += config.envs
        logger.step += steps_taken * config.action_repeat
        agent.update_models(int(config.train_ratio * steps_taken / config.num_meta_episodes))

        torch.save(agent.state_dict(), logdir / "latest_model.pt")
        training_times.append(time.time() - dreamer_training_time)

    agent.load_state_dict(torch.load(logdir / "best_model.pt"), strict=False)
    _, _, test_return = tools.simulate(
        eval_policy,
        test_envs,
        [sample_test_task() for _ in range(config.test_episode_num)],
        eval_eps,
        config.evaldir,
        logger,
        is_eval=True,
        state2image=state2img,
        num_meta_episodes=config.num_meta_episodes,
        metric_prefix="test",
    )
    print(f"Test return: {test_return:.1f}.")

    for env in train_envs + eval_envs + (test_envs if pool is not None else []):
        try:
            env.close()
        except Exception:
            pass
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    yaml = YAML(typ='rt')
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )


    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value


    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))

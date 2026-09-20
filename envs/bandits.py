"""Multi-armed bandits with a per-task exogenous distractor.

The agent should learn to explore for the best arm and then exploit it. The
observation carries a distractor that is fixed per task, so an agent can instead
learn the shortcut "distractor -> optimal arm". That shortcut is spurious: on
held-out tasks the distractor is unfamiliar and the agent that relied on it falls
back to roughly random-arm reward.

Task generation lives in `DistractorSampler`, outside the environment: the driver
samples a dataset of task configs once (see task_sampler.py) and hands one to the
env, which simply runs it. `BanditEnv` therefore holds no task table and cannot
invent a task, which is what keeps a train env from ever touching a held-out task.
"""
import gym
import numpy as np
from gym import spaces


class DistractorSampler:
    """Yields one bandit task config per call.

    Owns no ids and no task table -- TaskSampler assigns ids and holds the dataset.
    `index` is passed in only so the optimal arm can be spread deterministically
    across tasks; everything else is drawn from this sampler's own RNG.
    """

    def __init__(self, num_arms, distractor_dim, seed=0, reward_mode="uniform",
                 static_distractor=False):
        assert reward_mode in ("uniform", "binary"), f"Unknown reward_mode: {reward_mode}"
        self.num_arms = num_arms
        self.distractor_dim = distractor_dim
        self.reward_mode = reward_mode
        self.static_distractor = static_distractor
        self._rng = np.random.default_rng(seed)

    def __call__(self, index):
        # optimal_arm cycles with the index rather than being drawn: every arm is
        # then optimal for an equal number of tasks in *every* split, which is what
        # gives the task-id probe its exact 1/num_arms ceiling and what stops a
        # split from being lopsided (a 4-task val set once drew arms [2, 2, 3, 4],
        # so a policy that always picked arm 2 scored optimal on half of val).
        optimal_arm = int(index) % self.num_arms
        return {
            "optimal_arm": optimal_arm,
            "rewards": self._rewards(optimal_arm),
            # Both endpoints are fixed per task, so the distractor trajectory is
            # fully determined by the task.
            "distractor_start": self._rng.uniform(
                0.0, 1.0, self.distractor_dim).astype(np.float32),
            "distractor_end": self._rng.uniform(
                0.0, 1.0, self.distractor_dim).astype(np.float32),
            "static_distractor": self.static_distractor,
        }

    def _rewards(self, optimal_arm):
        """1.0 on the optimal arm. Under 'binary' every other arm pays exactly 0,
        so a confidently wrong policy scores 0.00 rather than something random-ish
        -- that is what makes wrong-arm lock-in visible as an exact zero."""
        rewards = np.empty(self.num_arms, dtype=np.float32)
        rewards[optimal_arm] = 1.0
        subopt_idx = [i for i in range(self.num_arms) if i != optimal_arm]
        if self.reward_mode == "uniform":
            rewards[subopt_idx] = self._rng.uniform(
                0.0, 1.0, self.num_arms - 1).astype(np.float32)
        else:
            rewards[subopt_idx] = 0.0
        return rewards


def make_sampler(config):
    return DistractorSampler(
        config.num_arms,
        config.distractor_dim,
        seed=config.generator_seed,
        reward_mode=config.reward_mode,
        static_distractor=config.static_distractor,
    )


class BanditEnv(gym.Env):
    def __init__(self, num_arms, distractor_dim, num_steps, use_distractor=True):
        super().__init__()
        self.num_steps = num_steps
        self.num_arms = num_arms
        self.distractor_dim = distractor_dim
        # Turn off for the control condition: zeros the distractor everywhere, so the
        # shortcut "distractor -> optimal arm" does not exist and only exploration
        # works.
        self.use_distractor = use_distractor

        obs_dim = distractor_dim + num_arms + 1  # distractor + prev one-hot action + prev reward
        self.observation_space = gym.spaces.Dict({
            "state": spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=bool),
            "is_first": spaces.Box(low=0, high=1, shape=(), dtype=bool),
        })
        self.action_space = spaces.Discrete(num_arms)

        self._task = None
        self._rewards = None
        self._prev_action_idx = -1
        self._prev_reward = 0.0
        self._t = 0

    def set_task(self, task):
        """Takes the exact config the driver chose -- the only source of tasks."""
        self._task = task
        self._rewards = np.asarray(task["rewards"], dtype=np.float32)

    def get_task(self):
        # The id, not the config: tools.py writes int(get_task()) into every
        # transition, and the xi context classifier reads that label.
        return int(self._task["id"])

    def _distractor(self, frac):
        """Constant `start` under static_distractor, else start -> end by `frac`."""
        start = np.asarray(self._task["distractor_start"], dtype=np.float32)
        if self._task.get("static_distractor", False):
            return start
        end = np.asarray(self._task["distractor_end"], dtype=np.float32)
        return (start + (end - start) * frac).astype(np.float32)

    def _obs(self, is_first=False):
        distractor = self._distractor(self._t / self.num_steps)
        if not self.use_distractor:
            distractor = np.zeros_like(distractor)
        one_hot = np.zeros(self.num_arms, dtype=np.float32)
        if self._prev_action_idx >= 0:
            one_hot[self._prev_action_idx] = 1.0
        state = np.concatenate([distractor, one_hot, [self._prev_reward]], dtype=np.float32)
        # Episodes end on the step budget, never on failure, so is_terminal stays
        # False throughout: models.py turns it into the `cont` signal, and marking a
        # time-limit truncation terminal would tell the world model that value stops
        # there.
        return {"state": state, "is_terminal": False, "is_first": is_first}

    def reset_model(self):
        """Restart the trial without changing the task."""
        self._t = 0
        self._prev_action_idx = -1
        self._prev_reward = 0.0

    def reset(self, task=None):
        if task is not None:
            self.set_task(task)
        assert self._task is not None, (
            "BanditEnv.reset needs a task config on first use -- the env does not "
            "generate tasks")
        self.reset_model()
        return self._obs(is_first=True)

    def step(self, action):
        action_idx = int(action)
        reward = float(self._rewards[action_idx])
        self._t += 1
        self._prev_action_idx = action_idx
        self._prev_reward = reward
        done = self._t == self.num_steps
        return self._obs(), reward, done, {"task_id": self.get_task()}

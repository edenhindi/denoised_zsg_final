"""Multi-armed bandits with a per-task exogenous distractor.

The agent should learn to explore for the best arm and then exploit it. The
observation carries a distractor that is fixed per task, so an agent can instead
learn the shortcut "distractor -> optimal arm". That shortcut is spurious: on
held-out tasks the distractor is unfamiliar and the agent that relied on it falls
back to roughly random-arm reward.

Ported from /home/eden.hindi/online-zsg/envs/bandits.py with the task API this
repo's MetaLearningEnv requires, dict observations, and a per-instance pool of
permitted task ids.
"""
import gym
import numpy as np
from gym import spaces


class DistractorDataset:
    """Generates the distractor vectors and reward tables for every task id.

    Not a train/test dataset -- the train/val/test partition lives in task_split.py.
    The same generator (same `seed`) must be shared by train and eval envs, or a
    given task id would mean a different task in each.
    """

    def __init__(self, num_tasks, num_arms, distractor_dim, seed=0, reward_mode="uniform",
                 static_distractor=False):
        assert reward_mode in ("uniform", "binary"), f"Unknown reward_mode: {reward_mode}"
        self.num_tasks = num_tasks
        self.num_arms = num_arms
        self.distractor_dim = distractor_dim
        self.reward_mode = reward_mode
        self.static_distractor = static_distractor

        rng = np.random.default_rng(seed)
        # Per-task linear-drift distractor: drifts from `distractor_start` to
        # `distractor_end` across the episode. Both endpoints are fixed per task,
        # so the trajectory is fully determined by task_id (context-specific).
        self.distractor_start = rng.uniform(0.0, 1.0, (num_tasks, distractor_dim)).astype(np.float32)
        self.distractor_end = rng.uniform(0.0, 1.0, (num_tasks, distractor_dim)).astype(np.float32)
        self.suboptimal_rewards = rng.uniform(0.0, 1.0, (num_tasks, num_arms - 1)).astype(np.float32)

    def distractor(self, task_id, frac):
        """Distractor for `task_id` at episode fraction `frac` in [0, 1].

        With `static_distractor`, returns the constant per-task `start` vector
        (frac ignored); otherwise linearly drifts from `start` to `end`.
        """
        start = self.distractor_start[task_id]
        if self.static_distractor:
            return start.astype(np.float32)
        end = self.distractor_end[task_id]
        return (start + (end - start) * frac).astype(np.float32)

    def optimal_arm(self, task_id):
        return task_id % self.num_arms

    def get_rewards(self, task_id):
        rewards = np.empty(self.num_arms, dtype=np.float32)
        opt = self.optimal_arm(task_id)
        rewards[opt] = 1.0
        subopt_idx = [i for i in range(self.num_arms) if i != opt]
        if self.reward_mode == "uniform":
            rewards[subopt_idx] = self.suboptimal_rewards[task_id]
        else:
            rewards[subopt_idx] = 0.0
        return rewards


class BanditEnv(gym.Env):
    def __init__(self, generator: DistractorDataset, num_steps: int, allowed_ids=None,
                 seed=0, use_distractor=True):
        super().__init__()
        self.generator = generator
        self.num_steps = num_steps
        # Turn off for the control condition: zeros the distractor everywhere, so the
        # shortcut "distractor -> optimal arm" does not exist and only exploration
        # works.
        self.use_distractor = use_distractor
        self.num_arms = generator.num_arms
        self.distractor_dim = generator.distractor_dim

        # The task ids this instance may use. Train and eval envs get disjoint
        # sets, which is what keeps the split intact: sampling below draws only
        # from here, and set_task refuses anything outside it.
        if allowed_ids is None:
            allowed_ids = range(generator.num_tasks)
        self._allowed_ids = [int(i) for i in allowed_ids]
        assert self._allowed_ids, "allowed_ids must not be empty"
        self._allowed_set = set(self._allowed_ids)
        self._rng = np.random.default_rng(seed)

        obs_dim = self.distractor_dim + self.num_arms + 1  # distractor + prev one-hot action + prev reward
        self.observation_space = gym.spaces.Dict({
            "state": spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=bool),
            "is_first": spaces.Box(low=0, high=1, shape=(), dtype=bool),
        })
        self.action_space = spaces.Discrete(self.num_arms)

        self._current_task_id = self._allowed_ids[0]
        self._rewards = generator.get_rewards(self._current_task_id)
        self._prev_action_idx = -1
        self._prev_reward = 0.0
        self._t = 0

    def enumerate_tasks(self):
        """The full task space, for TaskPool. Independent of allowed_ids."""
        return list(range(self.generator.num_tasks))

    def sample_task(self):
        """Draw from this instance's permitted ids only.

        MetaLearningEnv falls back to this whenever reset() is called without a
        task (tools.py:113), so restricting it here is what makes that path safe.
        """
        return int(self._rng.choice(self._allowed_ids))

    def set_task(self, task):
        task_id = int(task)
        if task_id not in self._allowed_set:
            raise ValueError(
                f"task {task_id} is outside this env's allowed ids "
                f"({sorted(self._allowed_set)}) -- the driver assigned a task from "
                f"the wrong split")
        self._current_task_id = task_id
        self._rewards = self.generator.get_rewards(task_id)

    def get_task(self):
        return self._current_task_id

    # Kept for compatibility with the original env's caller.
    def set_task_id(self, task_id, rank=0):
        self.set_task(task_id)

    def _obs(self, is_first=False):
        frac = self._t / self.num_steps
        distractor = self.generator.distractor(self._current_task_id, frac)
        if not self.use_distractor:
            distractor = np.zeros_like(distractor)
        one_hot = np.zeros(self.num_arms, dtype=np.float32)
        if self._prev_action_idx >= 0:
            one_hot[self._prev_action_idx] = 1.0
        state = np.concatenate([distractor, one_hot, [self._prev_reward]], dtype=np.float32)
        # Episodes end on the step budget, never on failure, so is_terminal stays
        # False throughout: models.py:312 turns it into the `cont` signal, and
        # marking a time-limit truncation terminal would tell the world model that
        # value stops there.
        return {"state": state, "is_terminal": False, "is_first": is_first}

    def reset_model(self):
        """Restart the trial without changing the task."""
        self._t = 0
        self._prev_action_idx = -1
        self._prev_reward = 0.0

    def reset(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self.reset_model()
        return self._obs(is_first=True)

    def step(self, action):
        action_idx = int(action)
        reward = float(self._rewards[action_idx])
        self._t += 1
        self._prev_action_idx = action_idx
        self._prev_reward = reward
        done = self._t == self.num_steps
        return self._obs(), reward, done, {"task_id": self._current_task_id}

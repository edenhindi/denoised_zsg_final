"""MiniGrid navigation with a per-task exogenous colour distractor.

The image counterpart of `envs/bandits.py`. The agent should learn to explore the
room, find the goal, and then return to it on later meta-episodes. Each task
instead carries a fixed RGB tint over the whole frame, so the agent can learn the
shortcut "tint -> goal position". That shortcut is spurious: on held-out tasks the
tint is unfamiliar and the policy falls back to searching -- or, worse, walks to
the wrong memorised corner.

Same three-part separation as bandits (see CLAUDE.md):

  ConfoundSampler  what a task *is* -- one config dict per call, no ids
  TaskSampler      how many, ids, and the split (task_sampler.py)
  MiniGridConfound what to do with one config -- never generates one

The layout family (empty room / four rooms) is a `Task` spec, which owns the grid
size, the inner walls, the candidate goal positions and the agent start. The env
is layout-agnostic: it asks the spec to build the grid and place things.
"""
import gym
import numpy as np
from gym import spaces

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv as _MiniGridBase


class HiddenGoal(Goal):
    """Rewards normally but renders as empty floor.

    With `hide_goal: True` the goal cannot be seen even when the agent is looking
    straight at it, so the only way to locate it is to walk over it -- that is what
    makes the first meta-episode a genuine search and the later ones genuine
    recall. With a visible goal, a single frame gives the answer away and there is
    nothing for the distractor to short-cut.
    """

    def render(self, img):
        pass


# ---------------------------------------------------------------------------
# Layout specs
# ---------------------------------------------------------------------------


class Layout:
    """One MiniGrid layout family.

    `grid_size` sets the overall NxN grid; None falls back to the family default.
    """

    default_grid_size = 8

    def __init__(self, grid_size=None):
        size = self.default_grid_size if grid_size is None else int(grid_size)
        self._validate_grid_size(size)
        self.grid_size = size

    def _validate_grid_size(self, size):
        if size < 4:
            raise ValueError(f"grid_size must be >= 4; got {size}.")

    @property
    def goal_positions(self):
        """Candidate goal cells as (row, col).

        The sampler cycles through these with the task index, exactly as bandits
        cycles `optimal_arm`, so every goal appears in every split and each split
        holds `split_size / len(goal_positions)` tasks per goal. That is also what
        gives the task-id probe its ceiling -- see the module docstring of
        `envs/bandits.py` and the "Load-bearing design points" section of
        CLAUDE.md.
        """
        raise NotImplementedError

    def build_walls(self, grid, width, height, rng):
        """Add inner walls to `grid`. The outer wall is already drawn."""
        raise NotImplementedError

    def start_pos(self):
        """(top, size) for `place_agent`, or None for a random start."""
        return None


class EmptyRoom(Layout):
    """Empty room. One goal in each of the four inner corners.

    Four rather than two: with two, guessing a corner is right half the time, and
    because the reward is step-discounted the "always walk to one corner" policy is
    a strong local optimum -- it scores ~0.44 against ~0.78 for actually searching,
    a gap small enough that the agent settles for committing to one goal. Four
    corners drop a guess to 1/4 and make that shortcut clearly worse, without
    making the task unsolvable: every corner is still reachable by exploration.
    """

    default_grid_size = 8

    @property
    def goal_positions(self):
        s = self.grid_size
        return [(1, 1), (1, s - 2), (s - 2, 1), (s - 2, s - 2)]

    def build_walls(self, grid, width, height, rng):
        pass  # only the outer wall, drawn by the caller

    def start_pos(self):
        # Only consulted under fixed_start. Every corner now holds a goal, so the
        # old bottom-left start would sit on one; start in the middle instead,
        # which is also equidistant from all four rather than favouring one.
        c = self.grid_size // 2
        return ((c, c), (1, 1))


class EmptyRoomTwoGoals(EmptyRoom):
    """Empty room, two goals: upper-left and bottom-right; start upper-right.

    Both goals are adjacent to the start and equidistant from it, so neither is
    favoured. On 9x9 with a 50-step budget: memorizing scores 0.86, the best search
    0.75 (a 21-step tour), and a memorizer on an unfamiliar tint ~0.43. The shortcut
    still pays on train (+0.12) and still shows on held-out tasks, but a whole search
    fits inside a 25-step imagination.

    Not upper-left + bottom-left: from the upper-right start the search passes one
    goal on the way to the other, so memorizing gains ~0.01 and there is no shortcut
    for a method to remove. With two goals a guess is right half the time -- the
    reason `EmptyRoom` has four -- so read held-out return with the task probe
    (ceiling 1 / tasks-per-goal), not alone.
    """

    @property
    def goal_positions(self):
        s = self.grid_size
        return [(1, 1), (s - 2, s - 2)]  # (row, col): upper-left, bottom-right

    def start_pos(self):
        # place_agent takes (x, y), unlike goal_positions' (row, col).
        return ((self.grid_size - 2, 1), (1, 1))  # upper-right inner corner


class FourRooms(Layout):
    """Four rooms split by cross walls, one door per wall segment.

    Goals sit in the centres of the two diagonal rooms. `grid_size` must be odd so
    the cross walls land on the centre row/column.
    """

    default_grid_size = 9

    def _validate_grid_size(self, size):
        super()._validate_grid_size(size)
        if size % 2 == 0:
            raise ValueError(
                f"FourRooms grid_size must be odd (cross walls need a centre "
                f"row/col); got {size}.")

    @property
    def goal_positions(self):
        s = self.grid_size
        q = s // 4  # roughly the centre of a room
        # centres of the bottom-left and top-right rooms, as (row, col)
        return [(s - 1 - q, q), (q, s - 1 - q)]

    def build_walls(self, grid, width, height, rng):
        room_w, room_h = width // 2, height // 2
        for j in range(2):
            for i in range(2):
                xL, yT = i * room_w, j * room_h
                xR, yB = xL + room_w, yT + room_h
                if i + 1 < 2:  # vertical wall, one door gap
                    grid.vert_wall(xR, yT, room_h)
                    grid.set(xR, int(rng.integers(yT + 1, yB)), None)
                if j + 1 < 2:  # horizontal wall, one door gap
                    grid.horz_wall(xL, yB, room_w)
                    grid.set(int(rng.integers(xL + 1, xR)), yB, None)

    def start_pos(self):
        return ((self.grid_size - 2, 1), (1, 1))  # bottom-left inner corner


LAYOUTS = {
    "empty": EmptyRoom,
    "empty2": EmptyRoomTwoGoals,
    "fourrooms": FourRooms,
}


def make_layout(name, grid_size=None):
    if name not in LAYOUTS:
        raise NotImplementedError(
            f"Unknown minigrid layout {name!r}. Available: {sorted(LAYOUTS)}")
    return LAYOUTS[name](grid_size=grid_size)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class ConfoundSampler:
    """Yields one minigrid task config per call.

    Owns no ids and no task table -- TaskSampler assigns ids and holds the dataset.
    `index` is passed in only so the goal can be spread deterministically across
    tasks; the tint and the door layout are drawn from this sampler's own RNG.
    """

    def __init__(self, layout, seed=0):
        self.layout = layout
        self._goal_positions = list(layout.goal_positions)
        self._rng = np.random.default_rng(seed)

    def __call__(self, index):
        # Cycled, not drawn: see Layout.goal_positions.
        goal = self._goal_positions[int(index) % len(self._goal_positions)]
        return {
            # The answer: recoverable by walking the room on any task.
            "goal_pos": (int(goal[0]), int(goal[1])),
            # The shortcut: an arbitrary RGB vector, perfectly predictive of the
            # goal on train tasks and unfamiliar on held-out ones. The mapping
            # tint -> goal can only be memorised, never generalised.
            "tint": self._rng.uniform(0.0, 1.0, 3).astype(np.float32),
            # Fixed per task, so the walls are part of the task rather than a
            # per-episode draw -- an env that re-rolled its doors each reset would
            # make a fixed task fail to pin down an episode.
            "door_seed": int(self._rng.integers(0, 2 ** 31 - 1)),
        }


def make_sampler(config):
    return ConfoundSampler(
        make_layout(config.minigrid_layout, config.minigrid_size),
        seed=config.generator_seed,
    )


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class _MiniGridCore(_MiniGridBase):
    """Layout-agnostic MiniGrid that renders whatever task it is handed.

    Holds no task table and never samples: goal, doors and start all come from the
    config dict set by the outer env.
    """

    def __init__(self, layout, num_steps, agent_view_size=7, hide_goal=True,
                 fixed_start=True, tile_size=8, partial_obs=False,
                 random_start_dir=False, start_dir_seed=0):
        self.layout = layout
        self._goal_pos = layout.goal_positions[0]
        self._door_seed = 0
        self._hide_goal = hide_goal
        self._fixed_start = fixed_start
        # Per-episode facing, as r2dreamer's place_agent draws it: a fixed action
        # script then cannot solve the task, and the opening states vary. Drawn from
        # this env's own seeded RNG, not MiniGrid's, so runs are reproducible -- but
        # it is deliberately outside the task dict, like the per-step randomness of
        # SparsePointWindEnv: a fixed task no longer pins down the first frame.
        self._random_start_dir = random_start_dir
        self._dir_rng = np.random.default_rng(start_dir_seed)
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "get to the goal square"),
            grid_size=layout.grid_size,
            see_through_walls=False,
            max_steps=num_steps,
            render_mode="rgb_array",
            agent_view_size=agent_view_size,
            tile_size=tile_size,
            # True renders only the agent's egocentric view (true partial
            # observability) instead of the full map.
            agent_pov=partial_obs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.layout.build_walls(
            self.grid, width, height, np.random.default_rng(self._door_seed))

        row, col = self._goal_pos
        self.put_obj(HiddenGoal() if self._hide_goal else Goal(), col, row)

        start = self.layout.start_pos() if self._fixed_start else None
        if start is not None:
            top, size = start
            # place_agent draws the facing from MiniGrid's own RNG, which is not
            # part of the task config: that would randomise outside the task dict,
            # so a fixed task would no longer pin down an episode and the first
            # frame of a hidden-goal search would vary run to run. rand_dir=False
            # leaves agent_dir at -1 ("invalid direction"), so set it explicitly.
            self.place_agent(top=top, size=size, rand_dir=False)
            # Facing right unless random_start_dir; see __init__.
            self.agent_dir = int(self._dir_rng.integers(4)) if self._random_start_dir else 0
        else:
            self.place_agent()
        self.mission = "get to the goal square"


class MiniGridConfound(gym.Env):
    """Image-observation meta-RL env with a per-task tint distractor."""

    def __init__(self, layout, num_steps, size=(64, 64), use_distractor=True,
                 distractor_strength=0.3, agent_view_size=7, hide_goal=True,
                 fixed_start=True, partial_obs=False, random_start_dir=False,
                 start_dir_seed=0):
        super().__init__()
        self.num_steps = num_steps
        self._size = tuple(size)
        # Turn off for the control condition: zeros the tint everywhere, train
        # included, so the shortcut "tint -> goal" does not exist and only
        # exploration works. This is the ceiling a distractor-present run is
        # measured against.
        self.use_distractor = use_distractor
        self._distractor_strength = float(distractor_strength)
        self._partial_obs = partial_obs

        self._env = _MiniGridCore(
            layout=layout,
            num_steps=num_steps,
            agent_view_size=agent_view_size,
            hide_goal=hide_goal,
            fixed_start=fixed_start,
            tile_size=max(1, self._size[0] // layout.grid_size),
            partial_obs=partial_obs,
            random_start_dir=random_start_dir,
            start_dir_seed=start_dir_seed,
        )
        self._num_actions = 3  # left, right, forward -- pickup/drop/toggle are no-ops here

        self.observation_space = gym.spaces.Dict({
            "image": spaces.Box(0, 255, self._size + (3,), np.uint8),
            # Previous action and reward live *inside* the observation, as in
            # bandits: without them the agent cannot tell a fresh meta-episode's
            # first step from a repeat visit, so it cannot infer the task within
            # the trial. Keep `meta_learning: True` -- turning it off also disables
            # reward-input to the encoder/decoder.
            "state": spaces.Box(-np.inf, np.inf, (self._num_actions + 1,), np.float32),
            "is_terminal": spaces.Box(0, 1, (), bool),
            "is_first": spaces.Box(0, 1, (), bool),
        })
        self.action_space = spaces.Discrete(self._num_actions)

        self._task = None
        self._prev_action_idx = -1
        self._prev_reward = 0.0
        self._t = 0

    def set_task(self, task):
        """Takes the exact config the driver chose -- the only source of tasks."""
        self._task = task
        self._env._goal_pos = tuple(int(v) for v in task["goal_pos"])
        self._env._door_seed = int(task["door_seed"])

    def get_task(self):
        # The id, not the config: tools.py writes int(get_task()) into every
        # transition, and the xi context classifier reads that label.
        return int(self._task["id"])

    def _tint(self, image):
        """Blend this task's colour over the frame.

        Under partial observability the unseen region is left untinted, so the fog
        stays black instead of becoming a coloured block that leaks the tint even
        where the agent can see nothing.
        """
        if not self.use_distractor or self._distractor_strength <= 0:
            return image
        colour = np.asarray(self._task["tint"], dtype=np.float32)
        overlay = (colour * 255).astype(np.uint8)[None, None, :]
        tinted = ((1 - self._distractor_strength) * image.astype(np.float32)
                  + self._distractor_strength * overlay).clip(0, 255).astype(np.uint8)
        if self._partial_obs:
            seen = image.max(axis=2) >= 20
            return np.where(seen[..., None], tinted, image)
        return tinted

    def _obs(self, is_first=False, is_terminal=False):
        image = self._env.render()
        if image.shape[:2] != self._size:
            image = _resize(image, self._size)
        one_hot = np.zeros(self._num_actions, dtype=np.float32)
        if self._prev_action_idx >= 0:
            one_hot[self._prev_action_idx] = 1.0
        state = np.concatenate([one_hot, [self._prev_reward]], dtype=np.float32)
        # is_terminal is True only for reaching the goal, never for the step budget
        # running out. models.py turns it into `cont` (1 - is_terminal), which is the
        # imagination discount: marking a time-limit truncation terminal would tell
        # the world model value stops there, while leaving goal contact non-terminal
        # tells it value *continues past the goal* -- and since the episode really
        # does end there, imagined rollouts then collect the goal reward several
        # times over.
        return {
            "image": self._tint(image),
            "state": state,
            "is_terminal": is_terminal,
            "is_first": is_first,
        }

    def reset_model(self):
        """Restart the trial without changing the task."""
        self._t = 0
        self._prev_action_idx = -1
        self._prev_reward = 0.0
        self._env.reset()

    def reset(self, task=None):
        if task is not None:
            self.set_task(task)
        assert self._task is not None, (
            "MiniGridConfound.reset needs a task config on first use -- the env "
            "does not generate tasks")
        self.reset_model()
        return self._obs(is_first=True)

    def step(self, action):
        action_idx = int(np.asarray(action).flat[0])
        _, reward, terminated, _, _ = self._env.step(action_idx)
        self._t += 1
        self._prev_action_idx = action_idx
        self._prev_reward = float(reward)
        # Two different endings: `terminated` is goal contact, which is a real
        # terminal state (the episode is over and no further value is available),
        # while the step budget running out is a truncation and must not be.
        terminated = bool(terminated)
        done = terminated or self._t >= self.num_steps
        return (self._obs(is_terminal=terminated), np.float32(reward), done,
                {"task_id": self.get_task()})

    def close(self):
        self._env.close()


def _resize(image, size):
    """Nearest-neighbour resize of an (H, W, 3) uint8 image."""
    h, w = image.shape[:2]
    th, tw = size
    ys = (np.arange(th) * h // th).clip(0, h - 1)
    xs = (np.arange(tw) * w // tw).clip(0, w - 1)
    return image[ys][:, xs]

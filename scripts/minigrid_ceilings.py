"""Reference returns for the minigrid_confound env, computed by planning.

A measured return means nothing without knowing what the task pays. This prints
the two numbers that bracket every run:

  ORACLE  knows the goal and walks straight to it. The ceiling.
  BLIND   does not know the goal and sweeps the candidate cells until it finds
          one. What a policy that ignores task identity scores.

Their difference is the *whole* value of knowing the task -- i.e. the most any
shortcut (the tint) or any memory mechanism can be worth. If a run plateaus at
BLIND, it has learned to search efficiently and is using no task information at
all, whatever its absolute number looks like. That is the reading that separates
"the confound is not being learned" from "the confound is barely worth learning".

Usage:
    python scripts/minigrid_ceilings.py
    python scripts/minigrid_ceilings.py --layout fourrooms --grid-size 9
    python scripts/minigrid_ceilings.py --num-meta-episodes 4 --fixed-start

MiniGrid pays `1 - 0.9 * steps / max_steps` on reaching the goal and 0 otherwise,
so every number here is a *discounted* return: the cost of searching is the point.
"""
import argparse
import pathlib
import sys
from collections import deque

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import envs.minigrid_confound as minigrid_confound
from task_sampler import TaskSampler

# MiniGrid actions: 0 = turn left, 1 = turn right, 2 = forward.
_TURN_LEFT, _TURN_RIGHT, _FORWARD = 0, 1, 2
# agent_dir indexes into these (dx, dy) steps.
_DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def shortest_steps(core, target_rc):
    """Fewest actions to stand on `target_rc` from the current (pos, dir).

    BFS over (position, direction) rather than position alone: turning costs an
    action, so Manhattan distance understates the true cost and would inflate
    every ceiling. Walls are read from the live grid, so this is correct for any
    layout. Returns None if unreachable.
    """
    start = (tuple(core.agent_pos), core.agent_dir)
    # The env stores goals as (row, col); grid coordinates are (x=col, y=row).
    target = (target_rc[1], target_rc[0])
    if start[0] == target:
        return 0

    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (pos, direction), dist = queue.popleft()
        for action in (_TURN_LEFT, _TURN_RIGHT, _FORWARD):
            if action == _TURN_LEFT:
                nxt = (pos, (direction - 1) % 4)
            elif action == _TURN_RIGHT:
                nxt = (pos, (direction + 1) % 4)
            else:
                dx, dy = _DIR_TO_VEC[direction]
                ahead = (pos[0] + dx, pos[1] + dy)
                cell = core.grid.get(*ahead)
                # Walking into a wall is legal but does not move the agent.
                moved = ahead if (cell is None or cell.can_overlap()) else pos
                nxt = (moved, direction)
            if nxt[0] == target:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def _reward(steps, max_steps):
    """MiniGrid's step-discounted goal reward; 0 if the budget ran out."""
    if steps is None or steps > max_steps:
        return 0.0
    return 1.0 - 0.9 * steps / max_steps


def oracle_return(env, task, max_steps, num_meta_episodes):
    """Knows the goal from the start of every meta-episode."""
    total = 0.0
    for episode in range(num_meta_episodes):
        env.reset(task) if episode == 0 else env.reset_model()
        total += _reward(shortest_steps(env._env, task["goal_pos"]), max_steps)
    return total


def blind_return(env, task, goals, max_steps, num_meta_episodes, rng):
    """Sweeps the candidate goals until it lands on the right one.

    The sweep order is shuffled per meta-episode: a fixed order would let the
    first cell be systematically cheap and quietly flatter the blind policy.
    Within a meta-episode the agent walks cell to cell, accumulating steps, and
    is paid only when it reaches the real goal -- so this is the return of a
    policy that navigates perfectly but knows nothing about the task.

    Crucially it re-searches every meta-episode: not knowing the task means not
    remembering it either. The gap against `oracle_return` at
    num_meta_episodes > 1 is therefore the value of carrying information across
    episodes, which is what the meta-RL setup is meant to reward.
    """
    total = 0.0
    for episode in range(num_meta_episodes):
        env.reset(task) if episode == 0 else env.reset_model()
        used = 0
        for goal in rng.permutation(len(goals)):
            cell = goals[goal]
            steps = shortest_steps(env._env, cell)
            if steps is None:
                continue
            used += steps
            if tuple(cell) == tuple(task["goal_pos"]):
                total += _reward(used, max_steps)
                break
            # Walking to a wrong cell costs its steps and pays nothing; the agent
            # continues from there, so the next leg is measured from this cell.
            _walk(env, cell)
    return total


def _walk(env, target_rc):
    """Move the agent onto `target_rc` so the next leg starts from there."""
    core = env._env
    target = (target_rc[1], target_rc[0])
    for _ in range(4 * core.grid.width * core.grid.height):
        if tuple(core.agent_pos) == target:
            return
        dx = target[0] - core.agent_pos[0]
        dy = target[1] - core.agent_pos[1]
        want = 0 if dx > 0 else 2 if dx < 0 else (1 if dy > 0 else 3)
        env.step(_FORWARD if core.agent_dir == want else _TURN_RIGHT)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", default="empty",
                        choices=sorted(minigrid_confound.LAYOUTS))
    parser.add_argument("--grid-size", type=int, default=None,
                        help="None uses the layout's own default.")
    parser.add_argument("--max-episode-length", type=int, default=50)
    parser.add_argument("--num-meta-episodes", type=int, default=1)
    parser.add_argument("--fixed-start", action="store_true",
                        help="Match fixed_start: True (default here is random).")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Episodes per task. The start cell is random, so the "
                             "blind sweep needs repeats to average over it.")
    parser.add_argument("--task-train-size", type=int, default=20)
    parser.add_argument("--task-val-size", type=int, default=20)
    parser.add_argument("--task-test-size", type=int, default=20)
    parser.add_argument("--generator-seed", type=int, default=0)
    args = parser.parse_args()

    layout = minigrid_confound.make_layout(args.layout, args.grid_size)
    sampler = minigrid_confound.ConfoundSampler(layout, seed=args.generator_seed)
    tasks = TaskSampler(sampler, args.task_train_size, args.task_val_size,
                        args.task_test_size, seed=args.generator_seed)
    env = minigrid_confound.MiniGridConfound(
        layout=layout,
        num_steps=args.max_episode_length,
        fixed_start=args.fixed_start,
    )
    goals = layout.goal_positions
    rng = np.random.default_rng(args.generator_seed)

    print(f"layout={args.layout} grid={layout.grid_size} goals={len(goals)} "
          f"max_episode_length={args.max_episode_length} "
          f"num_meta_episodes={args.num_meta_episodes} "
          f"fixed_start={args.fixed_start}")
    print(f"reward = 1 - 0.9*steps/{args.max_episode_length} on success, else 0\n")

    for split_name, split in (("train", tasks.train_tasks()),
                              ("val", tasks.val_tasks()),
                              ("test", tasks.test_tasks())):
        oracle, blind = [], []
        for task in split:
            for _ in range(args.repeats):
                oracle.append(oracle_return(
                    env, task, args.max_episode_length, args.num_meta_episodes))
                blind.append(blind_return(
                    env, task, goals, args.max_episode_length,
                    args.num_meta_episodes, rng))
        # Between-task standard error: with a fixed task set the uncertainty is
        # bounded by the number of distinct tasks, not the episode count.
        per_task = np.array(blind).reshape(len(split), args.repeats).mean(1)
        stderr = per_task.std(ddof=1) / np.sqrt(len(split))
        print(f"{split_name:5s}  ORACLE {np.mean(oracle):6.3f}   "
              f"BLIND {np.mean(blind):6.3f} +/- {stderr:.3f}   "
              f"value of knowing the task {np.mean(oracle) - np.mean(blind):+.3f}")

    print("\nA run plateauing at BLIND is navigating well but using no task\n"
          "information. The ORACLE-BLIND gap is the most any shortcut or memory\n"
          "mechanism can be worth -- if it is small, the confound is not failing\n"
          "to be learned, it is not worth learning.")


if __name__ == "__main__":
    main()

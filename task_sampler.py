"""Train/val/test partitioning of a task space, for zero-shot generalization.

Environments do not generate tasks. Each env module exposes a sampler -- a callable
returning one task config per call -- and this module samples a fixed dataset from it,
labels each task with an id, and partitions it. The driver then decides which task goes
to which rollout and hands the config to the env.

Keeping generation and partitioning here (rather than inside an env) means the split is
seeded, loggable, and identical across the parallel worker processes that each hold
their own env -- the env receives a task, so there is no generator to keep in sync.
"""
import random


class TaskSampler:
    """Samples a task dataset once, stamps ids, and partitions it train/val/test.

    Env-agnostic: `sampler(index)` is any callable returning one task config, so a
    task can hold anything -- this class only counts, labels and slices. `index` is
    passed so a sampler can spread structure deterministically across the dataset
    (bandits uses it for `optimal_arm = index % num_arms`); samplers that do not
    care may ignore it.

    Tasks are generated exactly once, here. The sample_* methods draw from those
    fixed lists and never call `sampler` again, so a task's contents are stable for
    the whole run and a given id always means the same task.

    Ids are assigned here rather than by the env samplers, so every experiment
    labels its tasks the same way and the envs stay free of task bookkeeping.

    The partition is by sample order, never shuffled. The xi context classifier and
    prior embedding (models.py, networks.py `_ctx_feat`) are sized `task_train_size`
    and fold ids outside [0, task_train_size) into an "unlabelled" row, so the train
    split must be the *first* `train_size` samples, carrying ids 0..train_size-1.
    Shuffling here would mask out most of the classifier's supervision with no error
    -- only a drift in xi_ctx_loss would show it.

    Split sizes are absolute task counts, not fractions: these splits are small
    enough that rounding decides whether a split covers the task space at all.
    """

    def __init__(self, sampler, train_size, val_size, test_size, seed=0):
        n_train, n_val, n_test = int(train_size), int(val_size), int(test_size)
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                f"non-positive split: train={n_train} val={n_val} test={n_test}")
        tasks = [dict(sampler(i), id=i) for i in range(n_train + n_val + n_test)]
        self._train = tasks[:n_train]
        self._val = tasks[n_train:n_train + n_val]
        self._test = tasks[n_train + n_val:]
        self._rng = random.Random(seed)

    def train_tasks(self):
        return list(self._train)

    def val_tasks(self):
        return list(self._val)

    def test_tasks(self):
        return list(self._test)

    def sample_train_task(self):
        return self._rng.choice(self._train)

    def sample_val_task(self):
        return self._rng.choice(self._val)

    def sample_test_task(self):
        return self._rng.choice(self._test)


def task_id(task):
    """The id a task was stamped with, or -1 when it carries none.

    tools.py writes this into every transition; the xi classifier masks out -1.
    """
    if isinstance(task, dict) and "id" in task:
        return int(task["id"])
    return -1


def describe_split(sampler):
    """Sizes plus a hash of the test set, so two runs claiming the same split can be
    checked to actually have it."""
    import hashlib
    import json

    test_ids = sorted(task_id(t) for t in sampler.test_tasks())
    digest = hashlib.sha256(json.dumps(test_ids).encode()).hexdigest()
    return {
        "num_train": len(sampler.train_tasks()),
        "num_val": len(sampler.val_tasks()),
        "num_test": len(sampler.test_tasks()),
        "test_hash": digest[:16],
    }

"""Train/val/test partitioning of a task space, for zero-shot generalization.

The environment sets tasks; this module decides which tasks belong to which split.
Keeping the partition here (rather than inside an env) means it is seeded, loggable,
and consistent across the parallel worker processes that each hold their own env.
"""
import hashlib
import random

import numpy as np


def canonicalize_task(task, decimals=6):
    """Map a task to a hashable key, so tasks can be deduplicated and compared.

    Tasks are heterogeneous across environments: a bare array (rooms), an int
    (bandits), or a dict holding a list of arrays (dmc_meta). Order is preserved
    throughout, since a sequence of goals is not the same task as its permutation.
    """
    if task is None:
        return None
    if isinstance(task, dict):
        return tuple(sorted((k, canonicalize_task(v, decimals)) for k, v in task.items()))
    if isinstance(task, (list, tuple)):
        return tuple(canonicalize_task(v, decimals) for v in task)
    if isinstance(task, np.ndarray):
        if task.ndim == 0:
            return canonicalize_task(task.item(), decimals)
        return tuple(canonicalize_task(v, decimals) for v in task)
    if isinstance(task, np.generic):
        return canonicalize_task(task.item(), decimals)
    if isinstance(task, bool):
        return task
    if isinstance(task, float):
        return round(task, decimals)
    if isinstance(task, (int, str)):
        return task
    raise TypeError(f"Cannot canonicalize task component of type {type(task)}")


class TaskPool:
    """A task space partitioned into disjoint train/val/test sets.

    Either `exhaustive_fn` (finite space, enumerated) or `sample_fn` (drawn `size`
    times and deduplicated) supplies the pool. The pool is sorted by canonical key
    before shuffling, so the partition depends only on `seed` and not on the order
    the tasks happened to be generated in -- the envs sample from unseeded global
    RNG, so generation order is not reproducible.

    Split sizes are absolute task counts. `test_size` defaults to whatever is
    left over. Counts rather than fractions because the splits here are small
    enough that rounding decides whether a split covers the task space at all.
    """

    def __init__(self, train_size, val_size, test_size=None, sample_fn=None,
                 size=None, seed=0, exhaustive_fn=None, contiguous=False):
        if exhaustive_fn is not None:
            pool = list(exhaustive_fn())
        elif sample_fn is not None:
            if size is None:
                raise ValueError("size is required when sampling the pool")
            pool = [sample_fn() for _ in range(size)]
        else:
            raise ValueError("provide either exhaustive_fn or sample_fn")

        by_key = {}
        for task in pool:
            by_key.setdefault(canonicalize_task(task), task)
        ordered = [by_key[key] for key in sorted(by_key, key=repr)]

        # Contiguous keeps train ids dense for the exogenous context classifier, but
        # arm coverage then needs each split size to be a multiple of num_arms.
        if contiguous:
            # repr-order is lexicographic ("10" < "2"), which is not contiguous.
            if not all(isinstance(t, (int, np.integer)) for t in ordered):
                raise ValueError(
                    "contiguous splits require integer tasks; "
                    f"got {type(ordered[0]).__name__}")
            ordered = sorted(int(t) for t in ordered)
        else:
            random.Random(seed).shuffle(ordered)
        n = len(ordered)
        n_train, n_val = int(train_size), int(val_size)
        n_test = n - n_train - n_val if test_size is None else int(test_size)
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                f"non-positive split: train={n_train} val={n_val} test={n_test}")
        if n_train + n_val + n_test > n:
            raise ValueError(
                f"split sizes {n_train}+{n_val}+{n_test} exceed the "
                f"{n} distinct tasks available")
        self._train = ordered[:n_train]
        self._val = ordered[n_train:n_train + n_val]
        self._test = ordered[n_train + n_val:n_train + n_val + n_test]

        self.train_keys = {canonicalize_task(t) for t in self._train}
        self.val_keys = {canonicalize_task(t) for t in self._val}
        self.test_keys = {canonicalize_task(t) for t in self._test}

        assert not (self.train_keys & self.val_keys), "train/val task overlap"
        assert not (self.train_keys & self.test_keys), "train/test task overlap"
        assert not (self.val_keys & self.test_keys), "val/test task overlap"
        assert self._train and self._val and self._test, (
            f"empty split: train={len(self._train)} val={len(self._val)} "
            f"test={len(self._test)} from a pool of {n}")

        self._rng = random.Random(seed + 1)

    def train_tasks(self):
        return list(self._train)

    def val_tasks(self):
        return list(self._val)

    def test_tasks(self):
        return list(self._test)

    def eval_tasks(self):
        """Every held-out task. Eval envs serve both val and test, so they are
        permitted the union -- which of the two is used is decided per rollout."""
        return list(self._val) + list(self._test)

    def sample_train(self):
        return self._rng.choice(self._train)

    def sample_val(self):
        return self._rng.choice(self._val)

    def sample_test(self):
        return self._rng.choice(self._test)

    def sample_train_batch(self, n):
        return self._sample_batch(self._train, n)

    def sample_val_batch(self, n):
        return self._sample_batch(self._val, n)

    def sample_test_batch(self, n):
        return self._sample_batch(self._test, n)

    def _sample_batch(self, tasks, n):
        if n <= len(tasks):
            return self._rng.sample(tasks, n)
        return [self._rng.choice(tasks) for _ in range(n)]


def describe_pool(pool):
    """Sizes plus a hash of the test set, so two runs claiming the same split can
    be checked to actually have it."""
    digest = hashlib.sha256(repr(sorted(pool.test_keys, key=repr)).encode()).hexdigest()
    return {
        "num_train": len(pool.train_keys),
        "num_val": len(pool.val_keys),
        "num_test": len(pool.test_keys),
        "test_hash": digest[:16],
    }

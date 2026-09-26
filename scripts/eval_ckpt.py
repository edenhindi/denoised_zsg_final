"""Evaluate saved checkpoints on val and test, to separate the argmax from the gap.

    python scripts/eval_ckpt.py <logdir> [--episodes 200]

best_model.pt is chosen by argmax over ~100 val evals, so its val number is biased
upward. latest_model.pt is not selected on anything, so comparing the four cells
says how much of a val/test gap is selection rather than generalization.
"""
import argparse
import functools
import os
import pathlib
import pickle
import sys

os.environ.setdefault("WANDB_MODE", "disabled")  # measurement only, no run

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import dreamer
import task_split
import tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=pathlib.Path)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    cfg = pickle.load(open(args.logdir / "config.txt", "rb"))
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    pool = task_split.TaskPool(
        exhaustive_fn=lambda: list(range(cfg.task_pool_size)),
        train_size=cfg.task_train_size,
        val_size=cfg.task_val_size,
        test_size=cfg.task_test_size,
        seed=cfg.task_split_seed,
        contiguous=cfg.task_split_contiguous,
    )

    make = lambda mode, i: dreamer.make_env(cfg, mode, pool=pool, index=i)
    envs = {"val": [make("eval", 100 + i) for i in range(cfg.envs)],
            "test": [make("test", 200 + i) for i in range(cfg.envs)]}
    envs = {k: [dreamer.Damy(e) for e in v] for k, v in envs.items()}

    acts = envs["val"][0].action_space
    cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    logger = tools.Logger(0, cfg)
    eval_eps = tools.load_episodes(cfg.evaldir, limit=1)
    dataset = dreamer.make_dataset(tools.load_episodes(cfg.traindir, limit=1), cfg)
    agent = dreamer.Dreamer(envs["val"][0].observation_space, acts, cfg, logger,
                            dataset).to(cfg.device)
    agent.requires_grad_(requires_grad=False)
    policy = functools.partial(agent, training=False)

    # Same fixed val set the run selected on, so the val column is comparable.
    tasks = {"val": pool.sample_val_batch(args.episodes),
             "test": pool.sample_test_batch(args.episodes)}

    print(f"{'checkpoint':<16}{'val':>10}{'test':>10}{'gap':>10}")
    for name in ["best_model.pt", "latest_model.pt"]:
        path = args.logdir / name
        if not path.exists():
            continue
        agent.load_state_dict(torch.load(path, map_location=cfg.device), strict=True)
        scores = {}
        for split in ["val", "test"]:
            _, _, scores[split] = tools.simulate(
                policy, envs[split], tasks[split], eval_eps, cfg.evaldir, logger,
                is_eval=True, num_meta_episodes=cfg.num_meta_episodes,
                metric_prefix=f"ckpt_{split}")
        print(f"{name:<16}{scores['val']:>10.2f}{scores['test']:>10.2f}"
              f"{scores['val'] - scores['test']:>10.2f}")


if __name__ == "__main__":
    main()

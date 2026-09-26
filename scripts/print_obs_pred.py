"""Print real vs endogenous vs exogenous observation, per dimension.

The logged state_mse is an L2 norm over every dimension at once, so it cannot say
*which* dimensions are wrong. On bandits the layout is known -- distractor, then
the previous-action one-hot, then reward -- so a per-dimension view says whether
the error sits in the part the exogenous stream is meant to explain or elsewhere.

    python scripts/print_obs_pred.py <logdir>
"""
import argparse
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import models  # noqa: E402
import tools  # noqa: E402


def load_config(logdir):
    """The run's own config, as dreamer.py pickled it into config.txt."""
    import pickle

    with open(pathlib.Path(logdir) / "config.txt", "rb") as f:
        cfg = pickle.load(f)
    if not isinstance(cfg, argparse.Namespace):
        cfg = argparse.Namespace(**cfg)
    cfg.device = "cpu"
    cfg.compile = False
    cfg.precision = 32
    cfg.encoder["input_reward"] = cfg.meta_learning
    cfg.decoder["input_reward"] = cfg.meta_learning
    for name in ("actor_entropy", "actor_state_entropy", "imag_gradient_mix"):
        val = getattr(cfg, name, 0.0)
        setattr(cfg, name, (lambda v=val: tools.schedule(v, 0)))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--key", default="state")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=5,
                    help="posterior context length; step must be < this")
    ap.add_argument("--step", type=int, default=-1,
                    help="which step to print the raw vectors for (-1 = last)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--per-step", action="store_true",
                    help="also print the per-step L2, to see how error evolves")
    args = ap.parse_args()

    cfg = load_config(args.logdir)

    # The full wrapper chain, so the observation matches what the run saw
    # (RewardObs and TimeAugmentedState add dimensions past the raw env's).
    import dreamer as dreamer_mod

    env = dreamer_mod.make_env(cfg, "train")
    acts = env.action_space
    cfg.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    wm = models.WorldModel(env.observation_space, env.action_space, 0, cfg)
    ckpt = torch.load(pathlib.Path(args.logdir) / "latest_model.pt",
                      map_location="cpu")
    state = {k[len("_wm."):]: v for k, v in ckpt.items() if k.startswith("_wm.")}
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = wm.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    wm.eval()

    eps = sorted((pathlib.Path(args.logdir) / "train_eps").glob("*.npz"))
    if not eps:
        print("no episodes in train_eps")
        return
    batch = {}
    for path in eps[: args.episodes]:
        with np.load(path) as f:
            for k in f.files:
                batch.setdefault(k, []).append(f[k])
    n = min(len(v) for v in batch.values())
    data = {k: torch.as_tensor(np.stack(v[:n])).float() for k, v in batch.items()}
    if "task_id" in data:
        data["task_id"] = data["task_id"].long()

    with torch.no_grad():
        proc, recon, _ = wm._rollout_for_logging(
            data, batch=min(args.episodes, n), context=args.steps
        )

    key = args.key
    truth = proc[key][: args.episodes, : args.steps]
    combined = recon[0][key].mode()[: args.episodes, : args.steps]
    endo = recon[1][key].mode()[: args.episodes, : args.steps] if recon[1] else None
    exo = recon[2][key].mode()[: args.episodes, : args.steps] \
        if recon[2] and key in recon[2] else None

    d = truth.shape[-1]
    nd = getattr(cfg, "distractor_dim", 0)
    na = cfg.num_actions

    def label(i):
        if i < nd:
            return f"distract{i}"
        if i < nd + na:
            return f"prevact{i - nd}"
        return "reward"

    print(f"\nlogdir : {args.logdir}")
    print(f"key    : {key}  ({d} dims: {nd} distractor, {na} prev-action, "
          f"{d - nd - na} reward)")
    print(f"sample : {truth.shape[0]} episodes x {truth.shape[1]} steps\n")

    hdr = f"{'dim':<12}{'|truth|':>9}{'combined':>10}{'endo':>10}{'exo':>10}"
    print(hdr)
    print("-" * len(hdr))
    for i in range(d):
        t = truth[..., i].abs().mean().item()
        c = (combined[..., i] - truth[..., i]).abs().mean().item()
        e = (endo[..., i] - truth[..., i]).abs().mean().item() if endo is not None else float("nan")
        x = (exo[..., i] - truth[..., i]).abs().mean().item() if exo is not None else float("nan")
        print(f"{label(i):<12}{t:>9.3f}{c:>10.3f}{e:>10.3f}{x:>10.3f}")

    print("-" * len(hdr))
    tot = (combined - truth).square().sum(-1).sqrt().mean().item()
    print(f"{'L2 (the logged state_mse)':<41}{tot:>10.3f}")
    print("\ncolumns are mean |prediction - truth| per dimension;")
    print("|truth| is the mean magnitude, for scale.\n")

    if args.per_step:
        # Step 0 is special: is_first resets the state, so xi has no history and
        # the prior sits at its learned initial value.
        print(f"{'step':<12}{'L2':>9}{'endo L2':>10}{'exo L2':>10}")
        print("-" * 41)
        for t in range(truth.shape[1]):
            c = (combined[:, t] - truth[:, t]).square().sum(-1).sqrt().mean().item()
            e = (endo[:, t] - truth[:, t]).square().sum(-1).sqrt().mean().item() \
                if endo is not None else float("nan")
            x = (exo[:, t] - truth[:, t]).square().sum(-1).sqrt().mean().item() \
                if exo is not None else float("nan")
            print(f"{t:<12}{c:>9.3f}{e:>10.3f}{x:>10.3f}")
        print()

    step = args.step if args.step >= 0 else truth.shape[1] - 1
    ep = args.episode
    print(f"episode {ep}, step {step}:")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("  truth   ", truth[ep, step].numpy())
    print("  combined", combined[ep, step].numpy())
    if endo is not None:
        print("  endo    ", endo[ep, step].numpy())
    if exo is not None:
        print("  exo     ", exo[ep, step].numpy())
        if endo is not None:
            print("  endo+exo", (endo[ep, step] + exo[ep, step]).numpy())


if __name__ == "__main__":
    main()

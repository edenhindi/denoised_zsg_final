"""Ablation plots: validation return of our config vs. one design choice changed.

Pulls the last N sweeps from wandb, takes --baseline as "ours" and groups the other
runs by their config diff against it. One PNG per ablated knob, mean +- 95% CI
(t-interval over seeds).

    .venv/bin/python scripts/compare_sweeps.py
    .venv/bin/python scripts/compare_sweeps.py --baseline 7az9h5gl --refresh
"""

import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy import stats

METRIC = "eval_return"
X_KEY = "env_step"
ABLATIONS = ["branch_decorr_scale", "xi_ctx_cond", "reward_head_task", "xi_ctx_scale"]
IGNORED_PARAMS = {"seed", "wandb_group"}
N_SEEDS = 5  # ours has 10 seeds; keep the first 5 so every curve has the same n

# knob -> (title, legend label for a value)
LABELS = {
    "branch_decorr_scale": ("Branch decorrelation weight ablation",
                            lambda v: rf"$\lambda_{{\mathrm{{decorr}}}} = {v}$"),
    "xi_ctx_cond": ("With or without instance-dependent exo prior",
                    lambda v: "with" if v else "without"),
    "reward_head_task": ("With or without task-conditioned reward head",
                         lambda v: "with" if v else "without"),
    "xi_ctx_scale": ("Exo task-classifier weight ablation",
                     lambda v: rf"$\lambda_{{\mathrm{{ctx}}}} = {v}$"),
}

OURS_COLOR = "#1f1f1f"
COLORS = ["#d1495b", "#2e86ab", "#edae49", "#66a182"]


def norm(v):
    """wandb returns '1.0', 1, 1.0 for the same value -- make them comparable."""
    if isinstance(v, bool) or v is None:
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def fetch(run):
    df = run.history(keys=[METRIC], x_axis=X_KEY, samples=5000, pandas=True)
    if not len(df):
        return None
    return df.groupby(X_KEY)[METRIC].mean().sort_index()


def load(args):
    cache = args.out / "history.pkl"
    if cache.exists() and not args.refresh:
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if (data["baseline"], data["n_sweeps"]) == (args.baseline, args.n_sweeps) \
                and "seed" in data["records"][0]:
            return data

    api = wandb.Api(timeout=120)
    entity = args.entity or api.default_entity
    sweeps = sorted(api.project(args.project, entity=entity).sweeps(),
                    key=lambda s: s.created_at, reverse=True)[: args.n_sweeps]
    if args.baseline not in [s.id for s in sweeps]:
        raise SystemExit(f"{args.baseline} is not among the last {args.n_sweeps} sweeps")

    params = set()
    for s in sweeps:
        params |= set((s.config.get("parameters") or {}).keys())
    params = sorted(params - IGNORED_PARAMS)

    runs = [(s.id, r) for s in sweeps for r in s.runs if r.state == "finished"]
    print(f"downloading {len(runs)} runs from {len(sweeps)} sweeps ...")
    with ThreadPoolExecutor(8) as ex:
        hists = list(ex.map(lambda sr: fetch(sr[1]), runs))

    records = [{"sweep": sid, "seed": norm(r.config.get("seed")), "params": {p: norm(r.config.get(p)) for p in params},
                "hist": h} for (sid, r), h in zip(runs, hists) if h is not None]
    data = {"baseline": args.baseline, "n_sweeps": args.n_sweeps, "records": records}
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def curve(runs):
    """Mean and 95% t-interval half-width over seeds on a common step grid."""
    lo = max(r["hist"].index.min() for r in runs)
    hi = min(r["hist"].index.max() for r in runs)
    grid = np.unique(np.concatenate([r["hist"].index.values for r in runs]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    Y = np.stack([np.interp(grid, r["hist"].index, r["hist"].values) for r in runs])
    n = len(runs)
    m = Y.mean(0)
    h = stats.t.ppf(0.975, n - 1) * Y.std(0, ddof=1) / np.sqrt(n) if n > 1 else 0 * m
    return grid, m, h


def plot(ours, variants, knob, ours_value, path):
    title, fmt = LABELS[knob]
    groups = sorted([(ours_value, ours)] + list(variants.items()), key=lambda kv: kv[0])
    others = iter(COLORS)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for value, runs in groups:
        is_ours = value == ours_value
        color = OURS_COLOR if is_ours else next(others)
        x, m, h = curve(runs)
        ax.fill_between(x, m - h, m + h, color=color, alpha=0.15, lw=0)
        ax.plot(x, m, color=color, lw=1.8,
                label=fmt(value) + (" (ours)" if is_ours else ""), zorder=3 if is_ours else 2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Validation return")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k" if x else "0"))
    ax.set_xlim(left=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", default="7az9h5gl")
    p.add_argument("--n_sweeps", type=int, default=5)
    p.add_argument("--project", default="dreamer-v3")
    p.add_argument("--entity", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--refresh", action="store_true", help="re-download from wandb")
    args = p.parse_args()
    args.out = args.out or Path("plots") / f"ablations_{args.baseline}"
    args.out.mkdir(parents=True, exist_ok=True)

    data = load(args)
    ours = sorted((r for r in data["records"] if r["sweep"] == args.baseline),
                  key=lambda r: r["seed"])[:N_SEEDS]
    ref = ours[0]["params"]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444444",
                         "xtick.color": "#444444", "ytick.color": "#444444"})
    for knob in ABLATIONS:
        variants = {}
        for r in data["records"]:
            if r["sweep"] == args.baseline:
                continue
            diff = {k for k, v in r["params"].items() if v != ref.get(k)}
            if diff == {knob}:
                variants.setdefault(r["params"][knob], []).append(r)
        if not variants:
            print(f"{knob}: no ablation runs found, skipping")
            continue
        path = args.out / f"ablation_{knob}.png"
        plot(ours, variants, knob, ref[knob], path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

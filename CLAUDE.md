# CLAUDE.md

Mamba: a DreamerV3-style model-based agent with a Mamba/SSM world model, used for
meta-RL and zero-shot generalization experiments.

## Running

Every run needs `--task <suite>_<rest>`; `make_env` splits on the first `_` to pick the
suite. **No named config block sets `task`** — only `defaults` does — so it must always
be passed explicitly.

```bash
python dreamer.py --configs bandits --task bandits_distractor
python dreamer.py --configs bandits bandits_drift --task bandits_distractor
python dreamer.py --configs rooms --task rooms_navigation
```

Headless machines: `dreamer.py` picks `egl` when `DISPLAY` is unset, `glfw` otherwise.
Override with `MUJOCO_GL=osmesa`. With multiple GPUs, keep rendering and training on the
same card: `CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0`.

## Config system

`configs.yaml` has one `defaults` block; every other block is a **sparse patch** applied
left to right (`dreamer.py` `recursive_update`). Nested dicts merge one level deep, so
`encoder: { mlp_keys: '.*' }` keeps the other encoder fields.

The arg parser is built from the *merged* dict, so any key in `defaults` automatically
becomes a `--flag`. Two consequences:

- Env-specific keys (`num_arms`, `goal_radius`, `wind_force`) live only in their own
  block. They are not CLI-overridable unless that block is passed — and a `--flag` that
  is in no loaded block is a hard parse error, which is how a sweep dies at startup.
- Booleans must be exactly `True`/`False` — parsing is `["False","True"].index(x)`.
  Use bare ints, not `1e3`, or the value is typed as float.

### Traps

- **`reach_wm_horizon_limit: -1`** in `defaults` is an unset sentinel. Any config using
  the world-model horizon curriculum must override it with a positive value. Leaving it
  at `-1` makes the growth rate negative, so `_wm_horizon` goes negative mid-run, the
  batch slice `data[k][:, :negative]` comes out empty, and training dies on
  `zero-size array to reduction operation minimum`. Every working block sets it.
- **`batch_length` is derived**, not configured: under `mamba_context` it becomes
  `num_meta_episodes * max_episode_length`, overriding any YAML value (including
  `debug`'s). It must exceed `initial_wm_horizon` or `Dreamer.__init__` raises.
- **`seed` does not seed the env dynamics or the task set.** It seeds torch/numpy/python
  global RNG; the task dataset is seeded separately (`task_split_seed`, `generator_seed`).
  On bandits about **one run in three** never learns explore-exploit at all (see "Reading
  results"). Run 3+ repeats before believing anything.

## Task sampling and the zero-shot split

Turned on with `task_split: True`. The keys live in `defaults`: `task_split_seed` and the
three sizes `task_train_size`, `task_val_size`, `task_test_size` — currently 20/20/20.
**Sizes are absolute counts**, and the dataset is exactly their sum.

The design has three parts, and the separation is the point:

| part | what it knows |
|---|---|
| **sampler** (per env module) | what a task *is* — returns one config dict per call |
| **`TaskSampler`** (`task_sampler.py`) | how many, ids, and the split. Never inspects a task |
| **env** | what to *do* with one config. Never generates one |

`TaskSampler` calls `sampler(i)` for `i` in `range(train+val+test)`, stamps each result
with `id=i`, and slices in order. The driver then hands a config dict to `reset()` per
episode.

**Envs cannot invent a task.** There is no `sample_task()` on `BanditEnv` and
`MetaLearningEnv` no longer requires one, so a train env physically cannot run a held-out
task. `simulate`'s surplus-env path (more envs than tasks left) takes an explicit
`fallback_task=` from the driver rather than asking the env.

**Train ids must be `0..task_train_size-1`.** This is a *range check*, not set membership:
`models.py` sizes the xi context embedding at `task_train_size` and masks to
`(task >= 0) & (task < task_train_size)`; `networks.py` `_ctx_feat` folds anything outside
into an "unlabelled" row. A val id of 20 is not "a task outside train" — it is simply out
of range. Sampling in order is what guarantees the alignment; **shuffling the split would
silently mask out most of the classifier's supervision**, showing up only as `xi_ctx_loss`
drifting toward chance.

Task routing: prefill and training draw with `sample_train_task()`; the periodic eval and
the final test **iterate their whole split**, each task `eval_repeats` / `test_repeats`
times. Held-out coverage is therefore balanced by construction and eval-to-eval movement
contains zero task-sampling noise. `best_model.pt` is selected on val, so the reported
`test_return` is not biased by checkpoint selection.

The split is written to `<logdir>/task_split.json` with the full task configs and a hash
of the test set, so two runs claiming the same split can be checked.

## Creating a new environment

Two objects and three lines of wiring. `envs/bandits.py` is the reference.

**1. A sampler** — decides what a task is. One config dict per call:

```python
class MySampler:
    def __init__(self, ..., seed=0):
        self._rng = np.random.default_rng(seed)

    def __call__(self, index):
        # Cycle the answer with the index; do not draw it. See below.
        goal = index % self.num_goals
        return {"goal": goal, "confounder": self._rng.uniform(0., 1., dim)}


def make_sampler(config):
    return MySampler(..., seed=config.generator_seed)
```

**2. An env** — decides what to do with that dict:

```python
class MyEnv(gym.Env):
    def set_task(self, task):
        self._task = task           # read whatever keys the sampler emits

    def get_task(self):
        return int(self._task["id"])   # the id, never the dict

    def reset(self, task=None):
        if task is not None:
            self.set_task(task)
        self.reset_model()             # restart the trial, keep the task
        return self._obs(is_first=True)

    def reset_model(self):
        ...                            # trial state only
```

**3. Wiring** — a branch in `dreamer.py`'s `make_sampler` and one in `make_env`, plus a
`configs.yaml` block.

### Requirements that fail silently

- **`get_task()` must return the int id.** `tools.py` does `int(env.get_task())` and falls
  back to `-1` on anything it cannot cast. Return the config dict and every transition is
  labelled `-1`, at which point `xi_ctx_cond`, `xi_ctx_scale` and the task-id probe go
  inert — with no error, just worse numbers.
- **`reset(task=None)` keeps the current task.** It must not sample one. That is what
  makes the split airtight.
- **Cycle the answer with `index`, don't draw it.** `goal = index % num_goals` gives
  exactly `split_size / num_goals` tasks per goal in *every* split. A random draw gives
  binomial counts, and a lopsided split is undetectable — see the arm-coverage trap below.
- **Never split by the answer.** Every goal/arm must appear in train *and* test, so a test
  task is always solvable by exploration and only a memorizing agent fails. Splitting by
  the answer makes test unsolvable even for a perfect explorer and proves nothing.
- **`is_terminal` stays False on time-limit truncation.** `models.py` turns it into the
  `cont` signal; marking a step-budget end as terminal tells the world model value stops
  there.
- Put previous action and reward **inside** the observation vector if the agent must infer
  the task within an episode — and keep `meta_learning: True`, since turning it off also
  disables reward-input to the encoder/decoder.

### Designing a confounded env

The pattern bandits demonstrates: the task config carries both a **shortcut** (visible in
the observation, perfectly predictive on train tasks, unfamiliar on held-out ones) and a
**solution** (recoverable by behavior on any task). A fresh random vector per task works
because the mapping confounder → answer is arbitrary, so it can only be memorized, never
generalized.

The effect is not guaranteed by the design: memorizing has to be *easier* than exploring.
That depends on the confounder's width and on how costly exploration is. If the gap comes
out small, check train return first, then widen the confounder.

## The bandits experiment

`envs/bandits.py` demonstrates a specific generalization failure. The agent should learn
explore-then-exploit, but each task carries a fixed exogenous distractor in the
observation, so it can instead learn "distractor → optimal arm". That shortcut is
spurious: on held-out tasks the distractor is unfamiliar and the policy collapses toward
random-arm reward.

Two conditions differing in one key: `bandits` (static distractor, the pathological case)
and `bandits bandits_drift` (drifts within the episode, harder to bind).

`DistractorSampler` draws each task's content from one RNG seeded by `generator_seed`:

- `distractor_start` / `distractor_end` — with `static_distractor: True` only `start` is
  used, constant for the whole episode; otherwise the distractor drifts linearly from one
  to the other across the episode fraction.
- `rewards` — under `reward_mode: 'binary'` every non-optimal arm pays exactly 0, so a
  confidently wrong policy scores 0.00, not random-ish. That is what makes wrong-arm
  lock-in visible as an exact zero.

The observation is `state = [distractor, prev_action_onehot, prev_reward]`, width
`distractor_dim + num_arms + 1`. Previous action and reward live **inside** `state`, not
in separate keys — so anything that masks or subtracts observation keys touches them too.

### Load-bearing design points

- `optimal_arm = index % num_arms`, so **arms are shared across the split** — every arm is
  optimal for some train task and some test task.
- The same modulo gives a **probe ceiling**: with 20 train tasks and 5 arms there are 4
  tasks per arm, so an agent that knows only the arm can score at most **1/4 = 0.25** on a
  task-id probe. Anything above that is non-behavioral information, i.e. the distractor.
  Sharper than any MI bound — no estimator, no fit diagnostic.
- **Every split must cover the arms.** This bit once: a 4-task val set drew optimal arms
  `[2, 2, 3, 4]`, so arm 2 was half of val and arms 0 and 1 were absent — a policy that
  always picked arm 2 scored optimal on half of val, undetectably. Cycling by index now
  makes coverage exact, but check per-arm counts after changing any size.
- `num_meta_episodes: 1` is correct here, but not because adaptation is removed — one
  bandit episode *is* the explore/exploit trial.
- `--use_distractor False` is the control: it zeros the distractor everywhere, train
  included, so the shortcut cannot exist. Use it as the ceiling any distractor-present run
  is measured against.

### Making the exogenous stream carry the distractor

Closed as of 2026-09-13: ~17.75 out of an 18 ceiling *with* the distractor, against 17.55
for the `--use_distractor False` control. The fix was two knobs **together**:

```
--xi_ctx_scale 1.0 --xi_ctx_cond True
```

- `xi_ctx_cond` conditions the xi **prior** on `task_id`.
- `xi_ctx_scale` weights a classifier from `xi_deter` to `task_id`, supervising the
  **posterior**.

Neither alone works, and the reason is specific to this env. With `static_distractor:
True`, xi is a deterministic function of `task_id`, so `H(xi | task) = 0` — a
task-conditional prior can predict xi outright, the posterior matches it at zero KL cost,
and xi collapses to uniform (`xi_kl ≈ 0.00`, `xi_post_ent` near its `4·ln(16) = 11.09`
ceiling) while carrying nothing. The classifier forces the posterior to *infer* the task
from the observation instead. Healthy values: `xi_post_ent` ~1.2 **below** `xi_prior_ent`
~1.8, and `xi_ctx_loss` well under chance `ln(task_train_size)`.

**Directions already tried that did not work** (see `progress.md` runs #7-#15 for detail):
CLUB on I(z; xi) at `feat`/`logit`/`deter` — each moved the shortcut rather than removing
it, and #10 is the clean counterexample (`club_mi` fell to 0.026 while the deter arm probe
*rose* to 0.82/0.99). Also: `endo_obs_diff` residual encoding, the split decoder,
narrowing `dyn_stoch`, and `actor_input: stoch` — that last one fails because
`dyn_temp_post: True` computes `stoch` from `deter`, so restricting the actor does not
isolate it.

### Reading results

`generalization_gap = train_eval_return - eval_return`, both measured under the same
greedy eval policy on the same number of episodes, so the difference is task familiarity
rather than policy mode. The three return *series* are separated by `metric_prefix`
(`eval`, `train_eval`, `test`) inside `simulate`. Note that `generalization_gap`,
`train_task_return`, and `best_return` are logged from `dreamer.py` as **bare scalars**
with no prefix — don't go looking for `eval_generalization_gap`.

**Check train return before interpreting any gap.** A gap near zero means nothing if the
agent never learned explore-exploit — that looks identical to perfect generalization. For
the default 60-task/5-arm/20-step setup the reference points are roughly: random ≈ 4,
always-wrong-arm ≈ 0, oracle = 20.

**One run is not a result.** Twelve August 2026 runs at 500k frames, same config family,
gave train returns of 19, 17, 20, **5**, 19, 19, **3**, 20, **0**, 19, 19, **0** — a third
never learned. Two of the failures had `generalization_gap` near zero (0.0 train / 12.40
eval), which is exactly the trap above. Run 3+ repeats per condition, and share a
`wandb_group` so they aggregate.

`scripts/print_obs_pred.py <logdir>` prints per-dimension truth vs endogenous vs exogenous
reconstruction for one key. Use it when `state_mse` looks wrong: the logged number is an
L2 norm over all 14 dims at once and cannot say *which* are failing. It is what showed the
two decoder branches cancelling (endo −1.82, exo +2.09, truth 0.24) — a sum that
reconstructs correctly while neither branch means anything.

**Per-episode return spread is mostly exploration cost, not measurement noise.** With 20
binary-reward steps, an episode that identifies the arm on step 2 scores ~18 while one
that takes until step 6 scores ~14 — same policy, same task. That spread is real agent
behavior and does not average away with more episodes; only the standard error of the
*mean* shrinks. If returns look bimodal rather than a wide single mode, that is a
different failure — the agent locking onto a wrong arm and never recovering — so plot the
histogram before concluding anything.

The test set is 20 tasks, so variance is between-task and does not shrink with
`test_repeats`. Report per-task means and a between-task standard error; a pooled std over
200 episodes would understate uncertainty — the confidence in a generalization claim is
bounded by the 20 distinct tasks, not by the episode count.

## Gotchas elsewhere in the repo

- **Eval episodes are no longer written to disk** — `simulate` saves only when
  `is_eval=False`, since eval episodes are consumed straight from `cache`. Analysis
  scripts that glob `eval_eps/*.npz` will find nothing.
- **`tools.py`'s `meta_episode_reset` fallback is inverted.** The `info.get(...,
  1 - float(d))` default is a copy-paste of the `discount` line above it and yields the
  opposite of a reset flag. It is dormant because `MetaLearningEnv` always supplies the
  real key. Do not "fix" it mid-experiment — it would change recorded episode data and
  confound any measured gap.
- **`simulate` returns `None` for score when `is_eval=False`**, so a training call cannot
  supply a return value for metrics.
- **`len(tasks)` is the number of episodes, not envs.** Surplus envs run throwaway
  episodes that are discarded, not stored — `add_to_cache` sits inside the task-carrying
  guard.
- Several envs randomize outside the task dict (`SparsePointWindEnv` per-step wind,
  `ReacherGoalMeta` joint init, `PointEnvBarrier` start position), so a fixed task does
  not pin down an episode there.
- Most envs other than `bandits` still carry their own `sample_task()`. It is unused by
  the driver and not required by `MetaLearningEnv` — do not wire it back in.

## Environment

`requirements.txt` leaves `numpy` unpinned, which resolves to 2.x — **keep numpy < 2.0**,
gym 0.23 breaks on it. The file says `gym==0.23.0` while what has been used is 0.22.0;
that drift already caused one bug (two wrappers in `envs/wrappers.py` missing `@property`
on `action_space`, which older gym tolerated).

Torch must match the CUDA driver. On the 2080 Ti boxes (driver 535, CUDA 12.2) a default
`pip install torch` pulls a cu130 wheel that dies with "NVIDIA driver is too old" at the
first CUDA op; install from the cu121 index instead.

The gym "unmaintained" banner is a static print at import from `gym_notices`, not a
diagnostic. `GYM_NOTICES_ENABLED` does not suppress it on 0.22.0 — that env var landed in
0.24+.

`labmaze` (a `dm_control` dependency) has no wheels past cp312 and falls back to a bazel
source build. Install into a Python ≤ 3.12 environment.

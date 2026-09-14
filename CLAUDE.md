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
  block, matching existing convention. They are not CLI-overridable unless that block is
  passed.
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
- **`seed: 0` is dead config** — never read anywhere. Only the task split is seeded
  (`task_split_seed`); network init, exploration, and env dynamics are not. This is not
  academic: on bandits about **one run in three** never learns explore-exploit at all
  (see "Reading results"). Run 3+ repeats before believing anything.

## Zero-shot task split

Turned on with `task_split: True`. The generic keys live in `defaults`:
`task_split_seed`, `task_pool_size`, and the three split sizes `task_train_size`,
`task_val_size`, `task_test_size` — currently 60 tasks split 20/20/20.

**Sizes are absolute counts, not fractions.** `TaskPool` requires `train_size` and
`val_size`; `test_size` defaults to the remainder. It raises if any split is
non-positive or if the three exceed the pool. Fractions were removed on purpose: these
splits are small enough that rounding decided whether a split covered the task space at
all (see the arm-coverage trap below).

**Sampling lives outside the env.** `task_split.py` computes a seeded, disjoint
train/val/test partition; the driver decides which task goes to which rollout. Each env
*also* receives its own `allowed_ids` at construction, selected by `make_env`'s `mode`
("train" vs "eval") — a parameter previously ignored by every suite except `dmlab`.

The split is enforced twice, on purpose:

- `sample_task()` draws only from `allowed_ids`, so an env cannot invent an off-split
  task. This is what makes `tools.py`'s no-task reset path safe: when the task queue
  empties before envs finish, `simulate` calls `reset()` with no argument, which falls
  through to the env's own sampler.
- `set_task()` raises on an id outside `allowed_ids`, catching a driver wiring bug.

A train env therefore cannot run a test task by either route. The split is written to
`<logdir>/task_split.json` with a hash of the test set, so two runs claiming the same
split can be checked.

Task routing: prefill and training draw train tasks; the periodic eval draws **val**
(sampled once without replacement, then fixed, so `best_return` tracks the model rather
than the task draw); the final test draws **test**. `best_model.pt` is selected on val,
so the reported `test_return` is not biased by checkpoint selection.

**`eval_episode_num` is a task count, not episodes-per-task.** It is the argument to
`sample_val_batch`, so at the current 20 it draws *the entire* 20-task val set — eval-to-eval
movement therefore contains zero task-sampling noise. There is no config knob for "N
episodes per val task"; getting repeats would need a code change (or a deliberately
repeated task list).

**Checkpoint selection is a max over noisy evals.** `eval_return > best_return` runs at
every eval, so with `eval_every_collection_episodes: 200` and `envs: 10` it fires on the
order of a hundred times per run. A small `eval_episode_num` makes that a maximization
over sampling noise — it saves whichever checkpoint got the luckiest draw, and that
checkpoint is what the final test then measures precisely. Keep `eval_episode_num` large;
this is the main reason it is not 4 anymore.

`task_split.py` is env-agnostic — `canonicalize_task` handles bare arrays (`rooms`),
ints (`bandits`), and nested dicts (`dmc_meta`) — but only `bandits` is wired up today.

## The bandits experiment

`envs/bandits.py` demonstrates a specific generalization failure. The agent should learn
explore-then-exploit, but each task carries a fixed exogenous distractor in the
observation, so it can instead learn "distractor → optimal arm". That shortcut is
spurious: on held-out tasks the distractor is unfamiliar and the policy collapses toward
random-arm reward.

Two conditions differing in one key: `bandits` (static distractor, the pathological case)
and `bandits bandits_drift` (drifts within the episode, harder to bind).

### What `envs/bandits.py` builds

`DistractorDataset` generates everything from one seeded RNG, so a task id means the same
task in every env. **Train and eval envs must share the generator** (`generator_seed`),
or the split is meaningless. It holds three arrays over `num_tasks`:

- `distractor_start` / `distractor_end` — with `static_distractor: True` only `start` is
  used, constant for the whole episode; otherwise the distractor drifts linearly from one
  to the other across the episode fraction.
- `suboptimal_rewards` — only under `reward_mode: 'uniform'`. Under `'binary'` every
  non-optimal arm pays exactly 0, so a confidently wrong policy scores 0.00, not
  random-ish. That is what makes wrong-arm lock-in visible as an exact zero.

The observation is `state = [distractor, prev_action_onehot, prev_reward]`, width
`distractor_dim + num_arms + 1`. Note the previous action and reward live **inside**
`state`, not in separate keys — so anything that masks or subtracts observation keys
touches them too.

`BanditEnv` takes `allowed_ids`, and enforces the split twice: `sample_task()` draws only
from it, and `set_task()` raises on anything outside it. The first makes `tools.py`'s
no-task reset path safe; the second catches driver wiring bugs.

### Load-bearing design points

- `optimal_arm = task_id % num_arms`, so **arms are shared across the split** — every arm
  is optimal for some train task and some test task. A test task's answer is always
  reachable by exploration, so only a memorizing agent fails. **Do not split by arm**:
  that would make test tasks unsolvable even for a perfect explorer and prove nothing.
- The same `task_id % num_arms` gives a **probe ceiling**: with 20 train tasks and 5 arms
  there are 4 tasks per arm, so an agent that knows only the arm can score at most
  **1/4 = 0.25** on a task-id probe. Anything above that is non-behavioral information,
  i.e. the distractor. This is a sharper test than any MI bound — no estimator, no fit
  diagnostic, and late accuracy alone is diagnostic.
- **Every split must be large enough to cover the arms.** A split of fewer than `num_arms`
  tasks *cannot* contain all arms, and one of a few multiples of it will cover them
  lopsidedly. This bit once already: a 4-task val set drew ids `[19, 8, 17, 2]` → optimal
  arms `[2, 2, 3, 4]`, so arm 2 was half of val and arms 0 and 1 were absent. A policy
  that always picked arm 2 scored optimal on half of val, and never selecting arms 0 or 1
  was undetectable. Keep each split at several × `num_arms` and check the per-arm counts
  after changing any size or the seed.
- `num_meta_episodes: 1` is correct here, but not because adaptation is removed — one
  bandit episode *is* the explore/exploit trial. `meta_learning: True` must stay on:
  `MetaLearningEnv` is the only wrapper whose `reset()` accepts a task, and turning it off
  also disables reward-input to the encoder/decoder, which the agent needs to infer the
  arm.
- `is_terminal` stays **False even on the final step** — episodes end on the step budget,
  not on failure. `models.py` turns `is_terminal` into the `cont` signal, so marking a
  time-limit truncation terminal would tell the world model that value stops there.
- `--use_distractor False` is the control: it zeros the distractor everywhere, train
  included, so the shortcut cannot exist and exploration is the only route. Use it as the
  ceiling any distractor-present run is measured against.

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

Commit `d1e9f56` swapped the classifier out for the conditioning; it was restored
2026-09-13 as an independent knob, so either or both can run.

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
eval), which is exactly the trap above. `seed: 0` is dead config, so nothing but the task
split is seeded and these are genuinely independent draws. Run 3+ repeats per condition,
and share a `wandb_group` so they aggregate.

`scripts/print_obs_pred.py <logdir>` prints per-dimension truth vs endogenous vs exogenous
reconstruction for one key. Use it when `state_mse` looks wrong: the logged number is an
L2 norm over all 14 dims at once and cannot say *which* are failing. It is what showed the
two decoder branches cancelling (endo −1.82, exo +2.09, truth 0.24) — a sum that
reconstructs correctly while neither branch means anything.

**Per-episode return spread is mostly exploration cost, not measurement noise.** With 20
binary-reward steps, an episode that identifies the arm on step 2 scores ~18 while one
that takes until step 6 scores ~14 — same policy, same task. That spread is real agent
behavior and does not average away with more episodes; only the standard error of the
*mean* shrinks. Don't diagnose it as a sampling problem. If returns look bimodal rather
than a wide single mode, that is a different failure — the agent locking onto a wrong arm
and never recovering — so plot the histogram before concluding anything.

The effect is not guaranteed by the design: memorizing the distractors has to be *easier*
than learning to explore, which depends on `distractor_dim` and `task_pool_size`. If the
gap is small, check train return first, then raise `distractor_dim` — and check whether
the exogenous knobs above are on, since a small gap is now the *intended* outcome rather
than a sign the setup is broken.

The test set is 20 tasks, so variance is between-task and does not shrink with
`test_episode_num`. Report per-task means and a between-task standard error; a pooled std
over 200 episodes would understate uncertainty — the confidence in a generalization claim
is bounded by the 20 distinct tasks, not by the episode count.

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
  episodes that are discarded, not stored — `add_to_cache` sits inside the
  task-carrying guard.
- Several envs randomize outside the task dict (`SparsePointWindEnv` per-step wind,
  `ReacherGoalMeta` joint init, `PointEnvBarrier` start position), so a fixed task does
  not pin down an episode there.

## Environment

`requirements.txt` is currently unpinned in the working tree; the installed `gym` is
0.22.0 while the file says 0.23.0. That drift already caused one bug — two wrappers in
`envs/wrappers.py` were missing `@property` on `action_space`, which older gym tolerated.
Keep numpy < 2.0; gym 0.23 breaks on it.

The gym "unmaintained" banner is a static print at import from `gym_notices`, not a
diagnostic. `GYM_NOTICES_ENABLED` does not suppress it on 0.22.0 — that env var landed in
0.24+.

`labmaze` (a `dm_control` dependency) has no wheels past cp312 and falls back to a bazel
source build. Install into a Python ≤ 3.12 environment.

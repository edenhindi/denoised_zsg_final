# CLUB on mamba/bandits — session handoff

Companion to `progress.md` (run log). This covers what was *built*, what was *ruled
out*, and where things stand. Read `CLAUDE.md` in the repo first — it has the config
traps and the bandits experiment design.

Branch `CLUB`. Modified but uncommitted: `configs.yaml`, `models.py`, `networks.py`,
`progress.md`.

---

## 1. What was implemented this session

**CLUB estimator** — `networks.py:632` (`class CLUB`).

Upper-bounds `I(z; xi)` as `E_i[log q(xi_i|z_i)] - E_{i,j}[log q(xi_j|z_i)]`, with
`q(xi|z)` a diagonal Gaussian `N(mu(z), sigma^2(z))` from two MLPs. Inputs are flattened
over batch x time so negatives mix across both.

- `mi_est(z, xi)` — the bound. Gradients reach z and xi, not q's weights. The `+logvar`
  normalizer is identical in the positive and negative terms and cancels, so it is
  dropped from both.
- `learning_loss(z, xi)` — NLL on **detached** inputs, fits q. Keeps the `+logvar` term
  (it is a real likelihood here).
- `logvar_stats(z)` — diagnostics; see §3.

The logvar head is **tanh-bounded** to [-1, 1]. Unbounded, q shrinks its variance without
limit and the negative-pair term blows up. This bound is a choice, not something
fundamental — widening it (e.g. `2*tanh`) is an option if saturation turns out to matter,
but it would make runs incomparable.

**Wiring** — `models.py`.

- Built *after* `_model_opt` (`models.py:211`), like the existing probes, so the CLUB net
  is **not** a world-model parameter. Otherwise the model minimizes the bound by making q
  bad instead of by making the streams independent.
- Bound added to the WM loss, clamped: `club_scale * club_mi.clamp(min=0.0)`
  (`models.py:425`). `I >= 0` always, so a negative bound means q has not fit and the
  estimate is an artifact — clamping kills the gradient there. Verified: exactly zero
  gradient below 0, full gradient above. `club_mi` is still logged raw, so a clamped
  `club_loss` with negative `club_mi` is the signal that q is behind.
- q is fit **after** `_model_opt` (`models.py:437`). Stepping q first mutates weights the
  bound's graph still needs → `one of the variables needed for gradient computation has
  been modified by an inplace operation`. This ordering is load-bearing.
- `_club_inputs` (`models.py:319`) slices each stream, built from the state dicts rather
  than by slicing `feat`, so it does not depend on `get_feat`'s deter/stoch ordering.

**Config** (`configs.yaml`, in `defaults`, so all are CLI flags):
`club_scale` (0 = off), `club_hidden` 256, `club_layers` 2, `club_lr` 3e-4,
`club_inner_steps` 1, `club_input` 'feat'.

`club_input` ∈ {feat, deter, stoch, logit}, applied to **both** streams, independent of
`xi_head_input` (which controls what the *decoder* branch reads). Widths with the current
config: feat 192, deter 128, stoch 64, logit 64. 'logit' raises if latents aren't
discrete.

Also set this session: `precision: 32` in `defaults` — note that is **global**, so it
also hits the dmc/panda image suites where fp32 costs real speed. Moving it into the
`bandits` block would be tidier.

---

## 2. Current standing

**Best result is still run #7: `club_input: 'feat'`, `club_scale: 0.1`, fp32, otherwise
stock. Test 16.98.** Four attempts to improve on it all did worse:

| run | change | test | note |
|---|---|---|---|
| #7 | feat, fp32 | **16.98** | best |
| #8 | `club_inner_steps: 4` | 13.1 | jumps survived → not q lag |
| #9 | `club_input: logit` | (stopped) | stoch clean, shortcut moved to deter |
| #10 | `club_input: deter` | 13.3 | deter arm probe *rose* to 0.82/0.99 |
| #11 | `actor_entropy 1e-3` + `discount 0.99` | 12.29 | both targets moved, returns worse |

User's goal: **18 test with smooth return curves.**

---

## 3. Ruled out (do not re-litigate)

- **fp16 overflow** — was corrupting runs #1-#6. `model_grad_norm` 300-10000 with
  inf/nan while every WM loss term was small and falling. fp32 gives ~2-4. This explained
  the hard return collapses and the mi spikes *in those runs*.
- **q lag as the cause of the remaining spikes** — `club_inner_steps: 4` (#8) did not
  remove them.
- **wm_horizon curriculum** — `reach_wm_horizon_limit: 2e4` vs `steps: 5e5`, saturates in
  the first 4%, cannot explain spikes across the run.
- **CLUB causing the entropy collapse** — the `club_scale: 0.0` control collapses too,
  and harder/faster. Entropy is a property of the setup.
- **`club_nll ≈ -60` being a problem per se** — it is a log *density* over 192 dims
  (~-0.3/dim) and it was stable in a -50..-61 band for 13k steps. Stability is what
  matters, not sign or magnitude. It also **cannot** detect Gaussian-on-categorical
  misspecification — a badly-specified q fits its own wrong family fine. Use
  `club_logvar_frac_floor` for that instead.

---

## 4. Open threads, roughly by promise

**(a) The mi spikes ↔ return drops correlation, under fp32.** Still unexplained. User
reports probe accuracy **drops** during spikes, so it is *not* the shortcut re-forming —
it looks like the representation is transiently losing task information, and mi, ctx
loss, nll and returns all move together. Candidates not yet tested: replay-buffer task
composition shifting; `hidden_states_subsample: 64` resampling from only 20 timesteps
with `replace=True`. A discriminator: run the same config twice and see whether spikes
land at the *same steps* (schedule-driven) or different ones (data/optimization noise).

**(b) The shortcut relocates to whatever is not penalized.** #9 (logit) drove `stoch` to
near-chance and the shortcut moved to `deter`; #10 (deter) made `deter` memorize *harder*
(arm probe 0.82 early / 0.99 late, the highest anywhere) while `club_mi` fell to 0.026.
That pair is a clean demonstration that a low bound with a saturated q (`frac_floor`
1.000) is worthless. Suggests the target may not be a slice of the latent at all — CLUB
constrains `I(z; xi)` but nothing stops `z` itself from encoding the distractor, which
enters through the encoder directly.

**(c) The eval zeros.** Eval is greedy (`dreamer.py:148`, `actor.mode()`), so a wrong-arm
commitment is unrecoverable within 20 steps. Per-task eval returns are bimodal — a dozen
tasks at 17-20 alongside tasks at exactly 0. Arithmetic on run #3's final eval: 3 tasks
at 0 and one at 2; recovering just those to ~15 moves the mean 14.55 → ~17.4. **This is
the largest single available gain and it is not a representation problem.** But #11's
entropy bump did not deliver it, so the mechanism is not simply "raise the entropy
bonus." Worth checking whether the argmax *moves within an episode* on the failing tasks
(log the action sequence) — if it is `[3,3,3,...]` the state is not being used, which is
a meta-learning failure, not an exploration one.

**(d) Not done, arguably worth it.** Cross-episode negative masking in `mi_est`: with
`batch_size 16` x `batch_length 20`, the ~320 flattened samples contain only ~16
*independent* xi draws (the distractor is constant within an episode), so negatives are
contaminated by same-episode pairs, biasing the estimate down. Also: `club_input`
currently has no `deter ++ logit` option, which is the combination run #7's `feat`
approximates but with sampling noise.

**(e) `discount: 0.997` on a 20-step task** still in the bandits path. Values grew to
176-217 while imagined reward saturates at 0.66. #11 changed it to 0.99 (value 217 →
48.7) bundled with the entropy change, so its individual effect is unmeasured. Note it
lives in `defaults` and is shared with dmc/panda, which have 50-300 step episodes.

---

## 5. Reference numbers

- task id chance 1/20 = 0.05; `xi_ctx_loss` at chance = ln(20) = **2.996**
- optimal arm chance 1/5 = **0.20**
- actor entropy max ln(5) = **1.609** (0.08 ≈ 5% of max — genuine collapse)
- stoch/xi entropy max 4·ln(16) = **11.09** (so `post_ent` 2-4 is normal, *not* collapse)
- returns: random ≈12, always-wrong-arm ≈10, oracle 20
- `club_mi` is **not comparable across `club_input` modes** — different quantity,
  different dimensionality.

---

## 6. Verification status

Verified by me: the estimator's math (reads ~36 nats when xi is a deterministic function
of z, ~0 when independent); the clamp's gradient behavior; all four `club_input` modes
construct with correct widths and run end-to-end on `bandits debug`.

**Not** verified: that CLUB improves the generalization gap. The user ran a
`club_scale: 0.0` control and reported it improved, but that comparison was made on
fp16-corrupted runs and has not been redone under fp32. **The headline claim of the
project is currently unsupported by a clean control.** That rerun is probably the single
most valuable thing to do next.

Also unverified: everything in §4 is hypothesis, not measurement.

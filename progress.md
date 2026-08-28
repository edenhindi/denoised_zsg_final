# Experiment Progress log
I'm now making the bandits environment work with the mamba world model code. 
This repo was originaly used for meta rl properly.   
---
The bandits environment embody the problem of overfitting by exogenous observation.
Suppose for each instance we roll a random optimal arm, and a distractor vector can uniquely identify the training instance. The distractor vector will be concatenated to the observation (prev reward, prev action).
Our wanted policy is the one that tries each action in turn and finds the optimal arm automatically and roll it until the end.
In truth what happens is that the trained policy uses the distractor vector to uniquely identify the training instance, then applies the optimal action right from the start. But in test it will completely fail.
---

## Experiments:

###  1. bandits_smaller_latents_xi_encoder_higher_lr

bandits smaller deterministic latents: endogenous deterministic: 512 -> 128, endogenous stochastic: 16 -> 4
with Exogenous Dynamics with instance classifier head on the exogenous determ + stoch state
Higher lr -> each lr * 10

Results:
* Train Reward: 19.2
* Val Reward: 10.4

probes suggest, stochastic state and determ state are too knowing of the task_id at the start. 

### 2. bandits_smaller_latents_xi_encoder_vclub

club_scale 1

Added CLUB implementation.
Completely collapsed the latents, the exogenous state became almost a constant (low entropy)

Results:
* Train Reward: 5.25
* Val Reward: 4.85
  
constant too high, the information was -3 through the entire run. and the instance pred loss was too high (3.1) malarkey!
  
### 3. 
club_scale 0.1
Much better results, even the best one.

mi starting at 0.3 goes down to 0.1 gracefully
the instance pred loss goes down too 0

The training curves are almost the same for train and eval. 
Both achieve at 2k steps very high reward (14.7, 13.9)
Then they drop and oscillate for the next 10k steps between 7 and 15 until the end where they achieve around (18, 14.55)

Results: 
* Train Reward: 18.2
* Vale Reward: 14.55
* Test Reward: 16.41 (Best one yet!)

The value function values are insane, like 170 etc. the EMA's are also huge 178, 169

Also noticed some jumps in the mi. 

### 4. 
club_scale 0.05
club not going down fast enough stays around 0.4
actor_entropy stays high 0.3
the value function values are still insane

eval return and train return are very close
it was too choppy and peaked at 8 and went down so I stopped it.

Results: 
* Train Reward: 5.15
* Vale Reward: 5.3

My hypothesis is that it didn't work because of value function too high. something is unstable. so I turned down the discount factor

### 5.bandits_smaller_latents_xi_encoder_vclub0.1discount_0.95
club_scale 0.1
discount_factor 0.997 -> 0.95

Very unstable, throughout the entire run it stayed below 10, very choppy. then at the end it achieved 11.
But the value function went down so that's cool!
Still not goog results.


Results: 
* Train Reward: 11.5
* Vale Reward: 11.75
* Test Reward: 11.65

### 6. bandits_smaller_latents_xi_encoder_vclub0.1trial2
club_scale 0.1
discount_factor 0.997
 
like the previous run, but it was way choppier.

Results: 
* Train Reward: 19.9
* Vale Reward: 12.61
* Test Reward: 15.755


### 7. bandits_smaller_latents_xi_encoder_vclub0.1fp32 
like the previous run, but it was way less choppy. the mi values are still going down and there are spikes but it's quite ok. 
fron now on i'll try fp32

what's important is that the model gradients are not exploding. it stays near 2

Choppy train and val curves, that correspond to the mi values not smoothly decreasing
between 16.45 to 11

Results: 
* Train Reward: 11.3
* Vale Reward: 11.5
* Test Reward: 16.98 The new Best!!

### 8. bandits_smaller_latents_xi_encoder_vclub0.1inner_steps4
doing 4 inner steps in the club.
Still doing jumps so it's not the problem

The hypothesis was that the jumps come from q lagging behind a moving target (the bound
is only valid once q has fit). 4x the fitting steps and the jumps survive, so it isn't
estimator lag. mi still falls fine (0.65 -> 0.06) and grad norm is healthy (188 -> 3.9).
But the run itself is worse than #7 - eval peaks at 13 and ends at 6.35.

Results:
* Train Reward: 8.5
* Val Reward: 6.35
* Test Reward: 13.1

### 9. club_input logit
club_scale 0.1, fp32, club_input 'logit'

Added a `club_input` knob (feat | deter | stoch | logit) so CLUB's target is
independent of what the decoder reads (xi_head_input). 'logit' is the categorical's
parameters instead of a draw from it - continuous and free of per-step sampling noise,
so the diagonal-Gaussian q is better specified. Also added logvar diagnostics
(club_logvar_mean/min/frac_floor) to see if q is straining.

mi sits at ~0 the whole run (-0.05 -> -0.015) and nll *rises* (9.9 -> 34.8), so the
bound is not obviously measuring anything. frac_floor climbs 0.014 -> 0.27 (peaked 0.76),
i.e. a quarter of dims pinned at the variance floor - the discreteness leaks through into
the logits as the categorical sharpens.

Probes are the interesting part: stoch is clean (taskid 0.10 vs chance 0.05, arm 0.34)
but deter is *not* - arm early 0.79, late 0.96, taskid early 0.23. So CLUB on the logits
drove the stochastic latent to near-chance and the shortcut simply relocated to the
recurrent state.

Train and val track early then diverge hard, train overfits (19.95 vs 11.3).
Stopped at ~90k env steps.

Results (at stop):
* Train Reward: 19.95
* Val Reward: 11.3

### 10. club_input deter
club_scale 0.1, fp32, club_input 'deter'

Follow-up to #9: put the pressure where the probe says the shortcut is. Didn't work.

deter arm probe goes *up*, not down - early 0.23 -> 0.82, late 0.99. So CLUB on deter
did not remove the shortcut; the recurrent state memorizes anyway. club_mi still falls
(0.28 -> 0.026) while the probe says the information is very much still there, which is
a direct demonstration that a low bound is not evidence of independence.

frac_floor hits 1.000 - q fully saturated, so the bound is slack. post_ent also drops to
1.56 (vs ~4 in #8), the endogenous latent is much more collapsed than usual.
value_mean 217 (discount still 0.997).

Big train/val gap: 19.7 train vs 12.3 val.

Results:
* Train Reward: 19.7
* Val Reward: 12.3
* Test Reward: 13.3

### 11. actor_entropy 1e-3 + discount 0.99
club_scale 0.1, fp32, club_input 'feat', actor_entropy 3e-4 -> 1e-3, discount 0.997 -> 0.99

Not a CLUB change. Two separate problems found while debugging:
- eval is greedy (`actor.mode()`), and training entropy collapses to ~0.08 vs a max of
  ln(5)=1.61. Per-task eval returns are bimodal - a dozen tasks at 17-20 and a few at
  exactly 0, which on a 20-step binary bandit means a wrong arm held for all 20 steps.
- discount 0.997 on a 20-step task -> value_mean grew to ~176 with imagined reward
  saturating at 0.66.

1e-3 is online-zsg's value for the same task family (their bandits config uses
gamma 0.99, dreamer_entropy_coef 1e-3).

Both targets moved: value_mean 217 -> 48.7, entropy floor 0.06 -> ends at 0.21. But the
returns got *worse*, not better - eval never exceeds 12.2. Whatever is capping the score
now, it isn't only the entropy bonus.

Results:
* Train Reward: 10.25
* Val Reward: 11.15
* Test Reward: 12.29

---

## Settled findings (not runs)

**fp16 was corrupting the runs (#1-#6).** `model_grad_norm` was 300-10000 and hit
inf/nan, while every world-model loss term was small and *falling* (decoder 0.001, kl
0.5, cont 2e-10). A shrinking loss with exploding gradients is numerical, not
optimization. Under fp32 the same config gives grad norm ~2-4. The return collapses in
#3 and #6, and the mi spikes that co-occurred with them, were downstream of this.

**Chance levels, for reading the probes and losses.**
- task id: 1/20 = 0.05; xi_ctx_loss at chance = ln(20) = 2.996
- optimal arm: 1/5 = 0.20
- actor entropy max = ln(5) = 1.609
- stoch/xi entropy max = 4*ln(16) = 11.09, so post_ent of 2-4 is normal, not collapse
- returns: random ~12, always-wrong-arm ~10, oracle 20

So run #2's "instance pred loss 3.1" was xi collapsing to a constant (at chance), not
noise - the club_scale of 1.0 outweighed xi_ctx_scale and the cheapest way to minimize
I(z;xi) is to make xi constant.

**A low club_mi does not mean the streams are independent.** #10 is the clean
counterexample: mi falls to 0.026 while the deter arm probe reads 0.82/0.99. Always read
club_nll and club_logvar_frac_floor alongside it - if q is saturated the bound is slack.

**Test vs val are not the same estimator.** test runs on `best_model.pt`, selected by
max val return over ~60 evals, with test_episode_num 200 over 20 test tasks; val is a
single pass over the 20-task val set (eval_episode_num 20 = the whole set). That is why
#7 can report test 16.98 above train 11.3 - the selection is a max over noisy evals.

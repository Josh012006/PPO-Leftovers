# PPO Exploitation Gap

**Research question.** Given a fixed amount of experience `D`, what prevents
PPO from reaching the best policy that can be extracted from `D`, and can we
modify PPO to measurably reduce that gap?

This is a study of PPO's *optimization/exploitation* behavior in isolation
from exploration: the environment, the data-collection policy, and the
dataset itself are all held fixed, so that any performance difference
between algorithm variants is attributable to how well each one extracts
information already present in `D` — not to differences in what was
collected.

## Motivating paper

This project is directly motivated by:

> Glen Berseth, ["Is Exploration or Optimization the Problem for Deep
> Reinforcement Learning?"](https://arxiv.org/pdf/2508.01329) (2025)

That paper introduces the **experience optimal policy** `π̂*` — the best
policy recoverable from an agent's own collected experience — and defines
**practical sub-optimality** as the gap between it and the learned policy,
`V(π̂*(s0)) − V(π_θ(s0))`. It reports that across many
environments and both PPO and DQN, the experience-optimal value is typically 
2–3× the learned policy's own value — in the paper's own words, these algorithms 
"only exploit half of the good experience they generate." — i.e. that much of 
deep RL's difficulty on hard tasks is an
*exploitation/optimization* problem, not an *exploration* problem.

This project adapts that core idea into a controlled, single-algorithm
study: instead of comparing PPO against DQN across many environments with
different data-generating processes, we fix one environment, one dataset
`D`, and study PPO against modified versions of itself, so that the
resulting exploitation-gap numbers are cleanly attributable to a single
algorithmic change at a time. See "Relationship to the motivating paper"
below for the precise correspondence and the places this project
deliberately diverges.

## The core decomposition

For a learned policy `π_alg` trained on a fixed dataset `D`, define the
**experience-optimal policy** `π_D*` as the best policy extractable from the
information in `D` (not the globally optimal policy `π*` for the
environment). Then:

```
J(π*) − J(π_alg) = [J(π*) − J(π_D*)]  +  [J(π_D*) − J(π_alg)]
                     exploration/data       exploitation/optimization
                          gap                        gap
```

This project only ever intervenes on the second term. The first term is a
property of how `D` was collected and is held fixed throughout.

**Terminology, kept strictly distinct throughout the codebase:**

| Term | Means | Does NOT mean |
|---|---|---|
| Exploitation *frequency* | how often the agent acts greedily w.r.t. its current policy (e.g. deterministic/mode action at eval time) | how *good* that greedy behavior actually is |
| Exploitation *quality* | how much of the value latent in `D` the learned policy actually recovers | how deterministic the policy is |
| Optimization | the mechanical process of updating θ to improve the PPO objective | value estimation |

A policy can be 100% "exploitative" in the frequency sense (always takes
its best-known action) while still being very bad at exploitation in the
quality sense (its best-known action is wrong, because the critic/advantage
estimate or the policy update failed to extract what `D` actually supports).
This project is about the latter.

## Environment

A custom **stochastic maze** (`src/ppo_exploitation/envs/stochastic_maze.py`),
designed specifically so that `π_D*` can be computed *exactly* while PPO
still faces a genuine function-approximation challenge:

- **Ground-truth state** is a single integer (`row * width + col`) — finite,
  fully enumerable. This is what the reference solver operates on, and what
  every rollout (of every policy) actually executes in.
- **PPO's observation** is a separate, richer 8-dim continuous encoding
  (normalized position, local wall sensors, hazard flag, BFS-based
  normalized distance to goal). PPO never sees the integer state directly —
  it has to generalize across positions rather than memorize a lookup table.
  Exactness of `π_D*` is a property of *how it's
  solved*, never a constraint on what PPO is allowed to see.
- **Stochasticity** is a simple, closed-form "slip" model: with probability
  `slip_prob`, the executed action is resampled uniformly at random from all
  four actions instead of the intended one. Because this is analytic, the
  *exact* transition kernel `P(s'|s,a)` is computable without sampling —
  this is what makes the true-dynamics definition of `π_D*` possible (see
  below).
- **Multiple paths of different length/risk**: a randomized-DFS spanning
  tree guarantees start↔goal connectivity, then a small fraction of extra
  walls are knocked down to create loops/alternate routes, and hazard cells
  are scattered off the guaranteed-safe path — manufacturing genuine local
  optima (a short risky route vs. a longer safe one) for PPO to potentially
  fail to fully exploit.
- **Sparse reward, long horizon** (default 30×30, max 200 steps, only
  terminal reward + a small per-step penalty) — deliberately non-trivial,
  so the exploitation gap doesn't collapse to ~0 for lack of a real credit
  assignment problem.

## Approximating π_D*: two definitions, never conflated

`π_D*` is generally impossible to know exactly for a real environment — so
this project restricts itself to an environment small and structured enough
that it *can* be known exactly, and then computes it two different, clearly
labeled ways (`src/ppo_exploitation/reference/experience_optimal.py`):

**(i) Empirical / MLE `π_D*`** — solve the Bellman equations of the
maximum-likelihood MDP estimated purely from `D`'s own transition counts,
via exact tabular value iteration. This is the honest ceiling on what *any*
learner operating only on `D` could figure out: finite-sample noise in `D`
is a real, inescapable part of this ceiling, not something to be corrected
away. This is the **primary definition** used in all exploitation-gap
arithmetic.

**(ii) True-dynamics, support-restricted `π_D*`** — solve using the
environment's actual, exact transition kernel (available to us because we
built the environment), but restricted to only the `(state, action)` pairs
`D` actually visited. This uses privileged knowledge no real learner has, so
it is reported as a **diagnostic**, not the headline ceiling: the gap
between (i) and (ii) tells you how much of an apparent "PPO exploitation
gap" is really just sampling noise in the definition of the ceiling itself,
versus a genuine PPO limitation.

**Unvisited `(s,a)` convention.** Any `(state, action)` pair never observed
in `D` is routed, **during the DP solve only**, to a dedicated absorbing
state with a large fixed negative reward (`unseen_penalty`, default `-50`,
configured in `configs/reference.yaml`). This must be set more negative
than any achievable real return so that greedy value iteration always
prefers a *visited* action at a state whenever one exists. `D` supplies
value iteration with a transition and reward **model** — how the
environment behaves — never a default **behavior** to fall back on: the
Bellman equations are solved from that model alone, with no notion of
"what the data-collecting policy `π_β` would have done." So there is no
`π_β`-shaped default to borrow either, only the arbitrary (but fixed, and
reproducible) tie-break value iteration itself produces when every action
at a state is equally unvisited-and-penalized.

**This penalty is deliberately *not* reapplied at live-rollout evaluation
time.** Reapplying the same logic would mean that if a tabular
`π_D*` policy landed on a state with zero visited actions in `D`, the
evaluator forcibly ended the episode there and charged `unseen_penalty` a
second time. That would be a mistake, for a reason worth stating
plainly: a neural policy has no notion of an "unseen state" at all. Every
observation gets a forward pass and an action, informed by whatever the
network generalizes from nearby states — there is no special branch, no
early termination, in training or in production. `π_D*`, being a lookup
table, structurally cannot generalize like that, and that's *intentional*
— crediting it with neural-style generalization would stop it from
measuring "the best extractable from `D` alone" and start measuring "the
best extractable from `D` plus borrowed function-approximation capacity."
But forcibly ending the episode doesn't just withhold that credit; it
manufactures a penalty the real environment would never actually impose,
silently deflating `J(π_D*)` by an amount that has nothing to do with what
`D` supports. (Querying `π_β` directly for a fallback action was
considered and rejected for the same underlying reason: it would smuggle
`π_β`'s own neural generalization back into a ceiling that's supposed to
depend on `D` alone.)

**What actually happens:** `ReferenceSolution.act(state)` always
returns a real action — value iteration's own default at that state, an
uninformed tie-break when the state has zero coverage, the genuinely
optimal action when it doesn't — and the live evaluator just lets the real
environment continue normally, exactly as it would for any other state.
What's tracked instead is *how often this happens*: `evaluate_policy`
accepts an optional `covered_states` set and reports
`uncovered_state_step_rate` / `uncovered_state_episode_rate` — the
fraction of steps/episodes where the policy acted on a state `D` never
informed it about. `scripts/05_evaluate_all.py` reports these alongside
every other column for both `π_D*` variants (`NaN` for neural policies,
which have no such notion). Read `J(π_D*)` together with this diagnostic:
a low uncovered-state rate means the reported ceiling is trustworthy more
or less as-is; a high one is a flag that `D`'s coverage, not `π_D*`'s
solve, is the limiting factor for that particular run — which is itself
useful information, since it points at the exploration/data term of the
decomposition rather than the exploitation term this project studies.

**Evaluation is always live-rollout, for every policy.** The DP solve gives
`π_D*` a closed-form `V(s0)` too, but that's a *theoretical* value under the
(possibly imperfect) solved model. The number that actually goes into the
exploitation-gap arithmetic is always the empirical mean return from live
rollouts in the real environment, using identical seeds/episode counts
across every policy — the prior, both `π_D*` variants, standard PPO, and
modified PPO. `scripts/03_compute_pi_d_star.py` prints both numbers so you
can see how close they are, but only `scripts/05_evaluate_all.py`'s output
is the headline result.

## The four policies we compare

Every result in this project is a comparison across exactly four policies,
all scored the same way (live rollouts, same seeds, same episode count —
see "Evaluation is always live-rollout" above):

1. **`π_D*` (empirical / MLE)** — the primary ceiling, solved exactly via
   value iteration on the MLE model of `D` (definition (i) above).
2. **`π_D*` (true-restricted)** — the diagnostic ceiling, solved exactly
   using the environment's real dynamics restricted to `D`'s support
   (definition (ii) above).
3. **PPO standard** — one rigorous PPO trust-region window on `D` (see
   below), with ordinary, unmodified hyperparameters
   (`configs/ppo_fixed_d_standard.yaml`).
4. **PPO modified** — the same trainer, same code path, with exactly one
   hyperparameter changed to test a specific hypothesis
   (`configs/ppo_fixed_d_modified.yaml`, or any `configs/ppo_fixed_d_h*.yaml`
   variant added later).

`scripts/05_evaluate_all.py` produces one table with all four (plus the
prior, `π_β`, as an extra reference point). The headline exploitation gap
is always `J(π_D*_empirical) − J(PPO standard)`; `π_D*` (true-restricted)
bounds how much of that gap is sampling noise in the ceiling itself rather
than a real PPO limitation, and PPO modified tests whether a specific
mechanism closes any of the remainder.

## What "fixed-D training" means for PPO, operationally

PPO's normal rhythm is: collect a batch with `π_old` → run `epochs` epochs
of clipped updates against it → throw the batch away → collect a *new*
batch with the now-updated policy → repeat. PPO never reuses a batch
across more than one such window — once a batch has been used to update
`θ`, continuing to use it for a second window would mean updating against
a `π_old` that no longer reflects any policy that actually generated data,
which is exactly why real PPO always collects fresh data before
continuing.

Once `D` is frozen, there is no fresh data to collect. This project models
exactly the ONE window PPO's own rules actually license, as rigorously as
possible, rather than inventing a multi-window scheme PPO itself has no
equivalent of:

```
π_old  <- π_β                           # fixed for the entire run, never
                                          # refreshed -- π_β is the literal
                                          # checkpoint that generated D
θ      <- π_β's weights                  # theta starts where pi_old is
advantages, returns <- computed ONCE, from π_β's own critic
for epoch in range(epochs):
    shuffle D into minibatches
    clipped PPO update of θ against π_β
```

`src/ppo_exploitation/ppo/fixed_d_trainer.py` implements exactly this.
`epochs` (`PPOHyperparams.epochs`) is the swept hyperparameter, standing in
for "how much PPO can extract from `D` within the one window its own
update rule allows it." No call to `env.step()` ever happens here, and `D`
is read-only.

**Why not reuse `D` for multiple windows (refresh `π_old`, keep going)?**
Tha would mean periodicallysnapshotting `π_old ← θ` and continuing training on the same `D` for many
outer iterations. That scheme has a real correctness problem, not just a
stylistic one. The actions in `D` were generated by `π_β`, and PPO's ratio
`r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is only a genuine importance-
sampling ratio relative to the policy that actually produced `a_t` when
`π_old = π_β`. As soon as `π_old` is refreshed away from `π_β`, the ratio's
denominator stops being the real behavior policy's probability and becomes
a moving, undocumented reference — the clip still bounds movement
*relative to that reference*, but says nothing about the real, growing
mismatch with `π_β`. Restricting to a single window removes this problem
by construction (`π_old = π_β` is exactly true throughout, by definition)
rather than by measuring a drift and hoping it stays small. See "Further
analyses" at the bottom for what the multi-window scheme would still be
useful for, as a separate, clearly-labeled question.

**For now, standard PPO and modified PPO are the same code path.** They differ only
in which YAML config is loaded (`configs/ppo_fixed_d_standard.yaml` vs.
`configs/ppo_fixed_d_modified.yaml`) — this guarantees any measured
difference in the resulting exploitation gap is attributable to the
hyperparameter/mechanism that changed, never to an accidental code
divergence.

**How much training is "enough"? The clip mechanism tells you, empirically
— it doesn't have to be decided in advance.** Sweeping `epochs` (H4 below)
is the right way to ask "did we really extract the maximum PPO can give
us," but a fixed epoch count chosen up front can't answer that by itself.
Two signals, both already logged in `results/*_history.csv`, can:
`clip_frac` (the fraction of the batch where the clip is currently active)
and live evaluation on held-out seeds never used anywhere in training or
dataset collection. As `clip_frac` approaches saturation, most of the
batch contributes zero gradient and further epochs stop moving `θ` in any
way PPO's own trust region actually licenses — and if held-out performance
has already plateaued by that same point, that plateau is the maximum this
mechanism can give you for this `D`, not an arbitrary cutoff chosen in
advance. `FixedDPPOTrainer.train(verbose=True)` prints a note when
`clip_frac` crosses 0.9 for exactly this reason.

## Hypotheses (H1–H7)

The exploitation gap could come from the value/advantage-estimation side or
the policy-optimization side. Every field below is a YAML-reachable knob in
`configs/ppo_fixed_d_*.yaml` — no code changes needed to test any of these:

| # | Hypothesis | Config field(s) |
|---|---|---|
| H1 | Critic doesn't accurately represent the information in `D` | network capacity, `value_coef` |
| H2 | Advantage estimates misrank actions | `gae_lambda`, GAE implementation |
| H3 | The clip range prevents sufficient policy improvement | `clip_eps` |
| H4 | The single trust-region window is too short to extract what the clipped objective would otherwise allow | `epochs` |
| H5 | Entropy regularization prevents deterministic exploitation | `entropy_coef` |
| H6 | Policy network lacks capacity | `hidden_sizes` |
| H7 | Optimization hyperparameters (lr, minibatch size, grad clipping) limit convergence | `lr`, `minibatch_size`, `max_grad_norm` |

`configs/ppo_fixed_d_modified.yaml` ships as a worked example testing H4
(`epochs: 10 → 30`) with every other field held identical to the standard
config — copy it as a template for testing any other single hypothesis
(change exactly one field per copy, per the project's own "modify ONE
component at a time" rule). When sweeping `epochs` specifically, read it
alongside `clip_frac` in the saved history, per the note above — a higher
`epochs` value that only pushes `clip_frac` further into saturation
without changing held-out performance isn't evidence of a larger true
ceiling, it's evidence the window had already stopped mattering.

Note on H5 specifically, since it's easy to get wrong: **entropy_coef is
never zeroed by default.** Zeroing it *during training* is itself an
intervention on the optimization trajectory (a candidate for H5), not a
neutral preprocessing step — it changes what θ you end up with. It is not
the PPO analogue of DQN's `epsilon=0`. The actual analogue of `epsilon=0`
is evaluating with the deterministic/mode action rather than sampling
(`make_neural_act_fn(net, deterministic=True)` in `eval/evaluate.py`,
the default for all evaluation scripts) — that changes only how a *fixed* θ
is read out, never how it was trained.

## Two kinds of tests, kept deliberately separate

`tests/` contains two things that answer different questions and should
never be conflated:

**`tests/test_env.py`, `tests/test_reference.py`, `tests/test_fixed_d_trainer.py`
(Tier-1 unit / pipeline-bug tests).** "Is this specific piece of code
correct?" These check individual components of the actual research
library against hand-derived or structurally-guaranteed expectations —
e.g. `test_reference.py` hand-derives the closed-form fixed point of a
tiny stochastic self-loop MDP and checks the value-iteration solver
matches it to `1e-4`; `test_fixed_d_trainer.py` checks that `π_old`'s
stored log-probabilities exactly match `π_β`'s own, and that `θ` starts at
exactly `π_β`'s weights. Fast (seconds), narrow, no claim about the real
30×30 maze study beyond "the code that will run it isn't buggy."

**`tests/tier0/` (Tier-0 cross-environment pipeline validation).** "Does the
*full* pipeline — collect `D` → compute `π_D*` → train fixed-D PPO →
evaluate everything under one protocol — actually run start to finish and
produce well-formed output on an environment we did **not** build
ourselves?" This runs the exact same library code
(`src/ppo_exploitation/...`, completely unmodified) against gymnasium's
`FrozenLake-v1` (4×4, slippery) instead of the custom maze, via a small
adapter (`tests/tier0/frozen_lake_env.py`) that exposes the same minimal
interface (`get_state()`, `n_states`, `is_terminal_state`,
`true_transition_probs`/`true_reward` sourced directly from gymnasium's own
`env.unwrapped.P` rather than hand-derived). It exercises **both** `π_D*`
definitions (empirical and true-restricted), trains a short fixed-D PPO
run, and checks the final gap report is well-formed (finite returns,
success rates in `[0, 1]`). It does not attempt to produce a
scientifically meaningful result — for that, see the real Tier-1 study.
Takes about a minute.

If Tier-0 ever needed pipeline code special-cased for FrozenLake to pass,
that would itself be a signal the pipeline had quietly become overfit to
the maze; so far it hasn't needed any.

```bash
pytest tests/ -v           # everything: Tier-1 unit tests + Tier-0
pytest tests/tier0/ -v     # just the cross-environment validation
```

## Results analysis

### Baseline run (target_success_rate = 0.35)

This is the first complete pipeline run and establishes the frozen base
for the rest of this study: `π_β`, `D`, and both `π_D*` solutions below do
not change as later sections test individual PPO modifications (H1–H7) —
only the fixed-D PPO config varies from here on. `configs/ppo_fixed_d_modified.yaml`'s
current value (`epochs: 30`) is an arbitrary placeholder inherited from
earlier pipeline validation, not a chosen hypothesis — it is reported below
for completeness only, and is revisited once a real testing path is
decided; no conclusion should be drawn from "modified" as it currently
stands.

**Prior checkpoint (`π_β`).** Confirmed at iteration 282: tracking-seed
eval `success_rate=0.356±0.021`, independent confirmation-seed eval
`success_rate=0.354±0.021` — tight agreement, and `reached_near_target_zone=True`
confirms the near-target per-iteration checking (see `scripts/01_train_prior.py`)
did engage before the crossing was found. A *third* independent 500-episode
sample, taken later during final evaluation (`--eval-seed 999`, distinct
from both training-time seeds), came in at `success_rate=0.310±0.021` —
about 2.1 standard errors below the first two. This is exactly the kind of
sampling variation the whole tracking/confirmation design exists to make
visible rather than hide: even two independent samples agreeing tightly is
not a guarantee that a third will land in the same place, particularly for
a checkpoint selected *because* it crossed a threshold (a mild "selection
effect" — a checkpoint accepted for clearing a bar is somewhat more likely
to have its true rate sit at or below that bar than comfortably above it).
Read `target_success_rate` as "the threshold used to select a checkpoint
in roughly this performance range," not as a guaranteed exact figure under
every possible evaluation sample.

**Dataset D.** 4000 episodes, 626,133 transitions, state-action coverage
22.36% (351/900 states, 805/3600 state-action pairs touched). Notably,
this is essentially the *same* coverage the earlier (abandoned) 66%-prior
run reached at the same episode count (23.6%) — a much weaker, more
error-prone behavior policy did not translate into meaningfully broader
state-action coverage. The eval `mean_length=181.7` (out of `max_steps=200`)
for `π_β` suggests most of its episodes are running out the clock rather
than dying quickly, which is consistent with wandering within a limited
region rather than covering new ground faster.

**π_D\* (both definitions).** Empirical: `V(s0)=-0.1492`. True-restricted:
`V(s0)=-0.1540`. Close agreement,
`covered_states=351/900` for both (support is fixed by `D`, so this number
is identical by construction). One thing worth flagging explicitly: these
`V(s0)` values are *negative* discounted theoretical values, while the
live-rollout `mean_return` for empirical `π_D*` reported in the gap table
below is *positive* (`+0.1057`). This is not a contradiction — it is
precisely why the README's "Evaluation is always live-rollout" rule
exists. `V(s0)` discounts a reward arriving after a long, mostly-successful
path (with `γ=0.99`, a reward arriving ~150 steps out is worth roughly
`0.99^150 ≈ 0.22` of its face value), while the undiscounted live rollout
simply sums actual rewards received along whatever path was actually
taken (mean episode length in the live eval was much shorter, `88.6`
steps). The two numbers are answering different questions — only the
live-rollout number is used in any gap arithmetic.

**Live-rollout results** (500 episodes, `--eval-seed 999`, identical
protocol for every policy):

| policy | mean_return | success_rate | mean_length | gap vs π_D\*_empirical (return / success) |
|---|---|---|---|---|
| prior `π_β` | −1.544 ± 0.032 | 31.0% ± 2.1pp | 181.7 | +1.650 / +67.2pp |
| `π_D*` empirical | 0.106 ± 0.014 | 98.2% ± 0.6pp | 88.6 | — |
| `π_D*` true-restricted | 0.109 ± 0.013 | 98.4% ± 0.6pp | 88.5 | −0.003 / −0.2pp |
| standard PPO | −0.894 ± 0.039 | 65.0% ± 2.1pp | 151.3 | +1.000 / +33.2pp |
| modified PPO (epochs=30, a first try) | −0.883 ± 0.037 | 64.4% ± 2.1pp | 148.4 | +0.989 / +33.8pp |

**Interpretation.**

- **Coverage is trustworthy.** `uncovered_state_step_rate` for both `π_D*`
  variants is ≤0.07% over the full 500-episode eval — the reported ceiling
  is essentially never leaning on an uninformed default, so it can be read
  at face value.
- **The exploration/data term is large.** `π_D*` (98%) vastly exceeds `π_β`
  (31%) — `D`, generated by a policy that only reaches the goal about a
  third of the time, contains enough information to construct a
  near-perfect policy via exact DP. This is the qualitative pattern the
  motivating paper's core mechanism describes, now reproduced in this
  project's own controlled setting.
- **The exploitation term is also large, and this is the actual finding.**
  Standard PPO recovers 65.0% success against a 98.2% ceiling — it
  captures roughly two-thirds of what `D` supports and leaves about a
  third of it unexploited (33.2 percentage points). PPO is not failing to
  learn anything from `D` (65% is well above `π_β`'s own 31%), but it is
  leaving a substantial, measurable amount of value on the table. This is
  the exploitation gap this project set out to study, and at this prior
  strength it is large enough to be worth investigating.

**A note on eval seeds, relevant for everything that follows below**: the
reference lines and epoch-0 values in the next subsections
(`π_β ≈ 0.348`, `π_D* ≈ 0.992`) differ from the baseline table above
(`π_β = 0.310`, `π_D* = 0.982`) — this is deliberate, not an
inconsistency.
What's frozen across this whole project is the *artifacts* (`π_β`'s
weights, `D`'s contents, `π_D*`'s solved policy) — never regenerated.
What's reported for any given plot is always evaluated fresh, under
whatever eval seed that specific analysis uses, and every point *within*
one plot must share the same seed or the curve mixes incomparable
samples. `scripts/analyze_epochs.py` uses `eval_seed=24680` throughout
(see script docstring) rather than importing numbers computed under
script 05's `seed=999`, precisely to keep every point in a given figure
internally consistent. The ~2–4 percentage point gap between the two
seeds' estimates of the same frozen `π_β`/`π_D*` is itself expected
sampling variance (`stderr≈2.1pp` each, well under 2 standard errors
apart) — see "Baseline run" above for the same observation made the first
time it appeared.


### Epoch-count ceiling analysis (H4, clip_eps=0.2)

The 10-vs-30-epoch null result above only rules out a small range. To test
epochs on its own terms, `scripts/analyze_epochs.py` trains a single
300-epoch run (30–100× the conventional range — Schulman et al. 2017 use
K=3 for Atari-style discrete-action environments, K=10 for MuJoCo) on the
same frozen `D`/`π_β`, checkpointing live success_rate/mean_return every 5
epochs against 500 fixed held-out episodes never touched by training. At
`clip_eps=0.2`, the result is not a smooth plateau: from epoch ~35 onward,
the deterministic policy repeatedly alternates between two essentially
fixed configurations — `success_rate≈0.704` and `success_rate≈0.914` —
recurring bit-for-bit identically many times over the full 300 epochs
(the 0.914 policy alone recurs at 11 of 61 checkpoints, last seen at epoch
285), never settling on one or exceeding the higher of the two.
`clip_frac` climbs slowly and monotonically the entire run (0.0 → 0.266),
nowhere near saturation, so the clip mechanism itself still has headroom —
it is not obviously the wall. `entropy` falls over the first ~10 epochs,
stays flat until ~epoch 130, then climbs steadily back up through epoch
300, well after the oscillation had already begun.

This means epochs alone is a real but partial factor: more epochs clearly
help early on (0 → ~35), but past that point the fixed-D optimization
under `clip_eps=0.2` is not behaving like a well-behaved (concave-like)
climb toward a single optimum — it is cycling between at least two
attractors. A direct consequence worth taking seriously: since the
trajectory is non-monotonic, a *shorter* run (10 epochs, the standard
range) could just as easily land on an intermediate epoch that happens to
sit in the worse of the two regimes as the better one — the run never
"locks in" progress, so a fixed epoch count chosen a priori has no
guarantee of landing on the best policy the run ever produced, even if a
better one appeared earlier in training. This is a genuinely open question
about how the update mechanism could be made to preserve its own best
intermediate result, not just its final one — see "Further analyses"
below.

<table>
<tr>
<td width="50%"><img src="results/analysis/success_return_clip_0_2.svg" width="100%"><br><em>success_rate &amp; mean_return vs. epoch, clip_eps=0.2. Dashed/dotted/dash-dot lines mark the prior, the exact π_D* ceiling, and this run's best observed success_rate.</em></td>
<td width="50%"><img src="results/analysis/clip_entropy_clip_0_2.svg" width="100%"><br><em>clip_frac &amp; entropy vs. epoch, clip_eps=0.2. clip_frac never approaches saturation; entropy falls then climbs again well after the oscillation begins.</em></td>
</tr>
</table>

### Clip range sweep (H3, clip_eps ∈ {0.1, 0.2, 0.3, 0.4})

Since epochs alone doesn't explain the ceiling, the same 300-epoch sweep
was re-run at `clip_eps=0.1`, `0.3`, and `0.4` (in addition to `0.2`
above), everything else identical, to see how much of the ceiling
`clip_eps` itself is responsible for:

<table>
<tr>
<td width="50%"><img src="results/analysis/success_return_clip_0_1.svg" width="100%"><br><em>clip_eps=0.1 — best observed: 0.674 (epoch 15), mean over the run: 0.376</em></td>
<td width="50%"><img src="results/analysis/success_return_clip_0_2.svg" width="100%"><br><em>clip_eps=0.2 — best observed: 0.914 (epoch 35), mean over the run: 0.735</em></td>
</tr>
<tr>
<td width="50%"><img src="results/analysis/success_return_clip_0_3.svg" width="100%"><br><em>clip_eps=0.3 — best observed: 0.950 (epoch 190), mean over the run: 0.881</em></td>
<td width="50%"><img src="results/analysis/success_return_clip_0_4.svg" width="100%"><br><em>clip_eps=0.4 — best observed: 0.948 (epoch 40), mean over the run: 0.753</em></td>
</tr>
</table>

The oscillation between (at least) two recurring policies persists at
every clip_eps value tested — this is not specific to `0.2`. But the
*range* of that oscillation, and the best point it ever reaches, both
depend heavily on `clip_eps`:

- **Tighter (`0.1`) clearly limits the ceiling.** The best observed
  success_rate drops to 0.674 — well below every other setting, reached
  only once (epoch 15) — and the run spends nearly all of its time
  hovering near the prior (0.348; mean over the whole run: 0.376), with
  only three brief, narrow spikes escaping it. A trust region this tight
  does not give the policy enough room to move toward what `D` actually
  supports.
- **Looser (`0.4`) does not clearly help, and shows real instability.**
  Its best observed value (0.948) is close to `0.3`'s, but it is hit only
  **once**, at epoch 40, and never approached again for the remaining 260
  epochs — the run instead spends much of that time swinging as low as
  ~0.50–0.54 repeatedly (mean over the run: 0.753, well below `0.3`'s
  0.881). A wider trust region lets `θ` move further per epoch, but that
  extra freedom reads as drifting further into an imperfect fit of `D`'s
  finite, noisy sample rather than toward a better policy — consistent
  with overfitting to `D` rather than extracting more from it.
- **`0.3` is the best-behaved setting tested.** Highest best-observed
  success_rate (0.950, closest of any run to the 0.992 ceiling) *and*
  highest mean (0.881) — unlike `0.4` it keeps returning to its best
  region (6 of 61 checkpoints within 0.005 of its max, last at epoch 275)
  rather than spiking once and drifting away.

The practical reading: `clip_eps` is a real, first-order lever on how much
of `D` this mechanism can extract — worth a properly informed choice
rather than defaulting to the conventional `0.2`. It also suggests
scheduling `clip_eps` over the course of training (analogous to
learning-rate schedules) might be worth exploring, rather than treating it
as a single fixed value for an entire window — added to "Further
analyses" below.

<table>
<tr>
<td width="50%"><img src="results/analysis/clip_entropy_clip_0_1.svg" width="100%"><br><em>clip_frac &amp; entropy vs. epoch, clip_eps=0.1</em></td>
<td width="50%"><img src="results/analysis/clip_entropy_clip_0_2.svg" width="100%"><br><em>clip_frac &amp; entropy vs. epoch, clip_eps=0.2</em></td>
</tr>
<tr>
<td width="50%"><img src="results/analysis/clip_entropy_clip_0_3.svg" width="100%"><br><em>clip_frac &amp; entropy vs. epoch, clip_eps=0.3</em></td>
<td width="50%"><img src="results/analysis/clip_entropy_clip_0_4.svg" width="100%"><br><em>clip_frac &amp; entropy vs. epoch, clip_eps=0.4</em></td>
</tr>
</table>

These diagnostics mostly complement the success_rate story rather than
contradicting it. `clip_frac` is not directly comparable in absolute terms
across different `clip_eps` values (a narrower band is mechanically easier
to exceed, so a given amount of underlying policy drift produces a higher
`clip_frac` at `0.1` than the same drift would at `0.4` — the numbers
answer "how often was the band hit," not "how far did the policy move").
What is comparable is the *shape*: every setting shows the same
monotonically-climbing `clip_frac` and the same fall-then-rise `entropy`
pattern, and `0.4`'s `entropy` finishes lower (~0.41) than the other three
(~0.44–0.46) while its `clip_frac` plateaus earlier and at a lower level —
consistent with the wider band clipping less per step, letting `θ` settle
into a narrower, more confident (but, per the success_rate plot, less
reliable) policy rather than continuing to explore alternatives. Nothing
here points in a different direction from the success_rate/return
evidence — if anything it sharpens the "`0.4` drifts, `0.3` doesn't"
reading.

### Next steps

Since neither epochs nor `clip_eps` alone closes the gap to `π_D*`,
and the oscillation itself (not just its amplitude) persists across every
`clip_eps` tested, H5 (`entropy_coef`) — the one native PPO lever most
directly implicated by the entropy curves above — is the natural next
single-variable test on this same frozen `D`/`π_β`.

















## Project structure

```
configs/                        # every experimental knob lives here, not in code
  env_maze.yaml                  # maze layout, stochasticity, reward
  prior_training.yaml            # online PPO config for the checkpoint that generates D
  ppo_fixed_d_standard.yaml      # baseline fixed-D PPO hyperparameters
  ppo_fixed_d_modified.yaml      # example H4 ablation (edit/copy for other hypotheses)
  reference.yaml                 # pi_D* solver settings (gamma, unseen_penalty, VI tolerance)

src/ppo_exploitation/
  envs/stochastic_maze.py        # the custom discrete stochastic maze
  ppo/
    networks.py                  # actor-critic MLP
    buffer.py                    # Trajectory container + GAE
    online_agent.py              # standard online PPO (trains the prior only)
    fixed_d_trainer.py           # THE research instrument: one rigorous PPO
                                  # trust-region window on D, pi_old = pi_beta
  data/collect.py                # roll out the frozen prior to build D; save/load
  reference/experience_optimal.py# exact value iteration -> both pi_D* definitions
  eval/evaluate.py                # unified live-rollout scoring + gap-report table
  utils/config.py                 # YAML <-> dataclass config layer
  utils/seeding.py                # reproducibility

scripts/
  01_train_prior.py               # train the online PPO checkpoint to target success rate
  02_collect_dataset.py           # collect D from the frozen checkpoint
  03_compute_pi_d_star.py         # solve both pi_D* definitions exactly
  04_train_fixed_d_ppo.py         # standard OR modified PPO on D (same code, different YAML)
  05_evaluate_all.py              # evaluate everything, print/save the gap report
  run_pipeline.sh                 # runs 01-05 end to end

tests/
  test_env.py                     # Tier-1: maze connectivity, transition-kernel correctness
  test_reference.py               # Tier-1: value iteration vs. hand-derived closed forms
  test_fixed_d_trainer.py         # Tier-1: trainer smoke test (runs, stays finite, pi_old == pi_beta)
  tier0/
    frozen_lake_env.py             # gymnasium FrozenLake-v1 adapter (validation-only, not a src/ module)
    test_tier0_pipeline.py         # Tier-0: full pipeline exercised on FrozenLake instead of the maze

.github/workflows/
  tests.yml                       # runs `pytest tests/` (Tier-0 + Tier-1) on push and pull_request
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `ppo_exploitation` importable everywhere
```

## Running the tests

```bash
pytest tests/ -v
```

See "Two kinds of tests, kept deliberately separate" above.
`tests/test_reference.py` is the one worth reading closely if you want to
trust the ceiling numbers: it hand-derives the closed-form fixed point of a
tiny stochastic self-loop MDP (`V(0) = 0.45/0.55` for a specific
reward/gamma setup) and checks the value-iteration solver matches it to
`1e-4`. `tests/tier0/` is the one worth running first if you want to trust
the *pipeline as a whole* before spending time reading through the maze-
specific code — it validates the same D → π_D* → fixed-D-PPO → evaluate
chain against an environment neither of us built.

## Running the full pipeline

```bash
bash scripts/run_pipeline.sh
```

or step by step:

```bash
# 1. Train the online PPO prior to ~35% success rate (live environment)
python scripts/01_train_prior.py \
    --env-config configs/env_maze.yaml \
    --prior-config configs/prior_training.yaml \
    --out results/prior_checkpoint.pt

# 2. Freeze that checkpoint, collect the fixed dataset D from it
python scripts/02_collect_dataset.py \
    --env-config configs/env_maze.yaml \
    --checkpoint results/prior_checkpoint.pt \
    --n-episodes 4000 --seed 1 \
    --out results/dataset_D.pkl

# 3. Solve pi_D* exactly (both empirical and true-restricted definitions)
python scripts/03_compute_pi_d_star.py \
    --env-config configs/env_maze.yaml \
    --reference-config configs/reference.yaml \
    --dataset results/dataset_D.pkl \
    --out-empirical results/pi_d_star_empirical.pkl \
    --out-true-restricted results/pi_d_star_true_restricted.pkl

# 4. Train standard PPO and modified PPO on the SAME D and the SAME prior
#    checkpoint (pi_old = pi_beta = this checkpoint throughout -- see
#    "What fixed-D training means for PPO, operationally")
python scripts/04_train_fixed_d_ppo.py \
    --dataset results/dataset_D.pkl \
    --ppo-config configs/ppo_fixed_d_standard.yaml \
    --prior-checkpoint results/prior_checkpoint.pt \
    --out results/ppo_standard_on_D.pt \
    --history-out results/ppo_standard_on_D_history.csv

python scripts/04_train_fixed_d_ppo.py \
    --dataset results/dataset_D.pkl \
    --ppo-config configs/ppo_fixed_d_modified.yaml \
    --prior-checkpoint results/prior_checkpoint.pt \
    --out results/ppo_modified_on_D.pt \
    --history-out results/ppo_modified_on_D_history.csv

# 5. Evaluate everything under the identical live-rollout protocol
python scripts/05_evaluate_all.py \
    --env-config configs/env_maze.yaml \
    --prior-checkpoint results/prior_checkpoint.pt \
    --pi-d-star-empirical results/pi_d_star_empirical.pkl \
    --pi-d-star-true-restricted results/pi_d_star_true_restricted.pkl \
    --ppo-checkpoints standard=results/ppo_standard_on_D.pt modified=results/ppo_modified_on_D.pt \
    --n-episodes 500 --eval-seed 999 \
    --out results/gap_report.csv
```

`scripts/05_evaluate_all.py` accepts any number of `name=path` pairs after
`--ppo-checkpoints`, so once you've trained additional ablation configs
(H1–H7 or otherwise) from step 4, add them all to a single step-5 call to
compare every variant against the same `π_D*` reference in one table.

### Testing a new hypothesis

1. Copy `configs/ppo_fixed_d_modified.yaml` to a new file
   (e.g. `configs/ppo_fixed_d_h3_wide_clip.yaml`).
2. Revert the field(s) already changed back to the standard value, then
   change exactly one field to test your hypothesis (e.g. `clip_eps: 0.2 →
   0.4` for H3).
3. Re-run step 4 against the *same* `results/dataset_D.pkl` and the *same*
   `results/prior_checkpoint.pt` with the new config, then add it to step
   5's `--ppo-checkpoints` list.

Never regenerate `D` or the prior checkpoint between comparison runs — the
whole point of the design is that every variant sees byte-identical
experience and starts from a byte-identical `π_old`.

## What this codebase does *not* try to answer

- **Cross-algorithm comparison.** This is not "is PPO better than SAC/DQN
  at exploitation" — see the motivating paper for that broader comparison
  across algorithms and many environments. This project fixes PPO and
  varies only its own internals.
- **A principled offline-RL solution.** The fixed-D trainer here is one
  rigorous PPO trust-region window applied to static data — not a
  Conservative Q-Learning / decision-transformer-style offline method.
  Answering "how good could a *properly designed* offline method do on
  this same `D`" would be a natural and interesting extension of this
  codebase, but a different one.
- **Wall-clock/sample-efficiency claims.** All comparisons here are about
  final exploitation quality given a fixed `D`, not about how many epochs
  or how much compute each variant needed to get there (though
  `results/*_history.csv` records `approx_kl`, `clip_frac`, and both
  losses per epoch, enough to investigate that separately if useful).

## Relationship to the motivating paper

Berseth (2025) defines the experience-optimal policy as (for deterministic
environments) the single highest-return trajectory ever replayed from a
buffer, with two "softer" estimators for stochastic settings — the best 5%
and the most-recent-best-5% of collected trajectories by return — used
because a *sampled* trajectory can't be replayed deterministically to
reproduce its score in a stochastic environment. This project takes a
different, model-based approach precisely because the environment was
purpose-built to allow it: rather than estimating `π_D*` from top-quantile
trajectory returns, we solve for it exactly via tabular value iteration on
the (MLE or true-restricted) MDP implied by `D`. This is only possible
because the ground-truth state space here is small and fully enumerable —
it would not be a tractable substitute for the paper's estimator in the
large/continuous-state Atari- and MuJoCo-scale environments the paper
studies. The two approaches are answering the same underlying question
(how much of `D`'s latent value does the learned policy actually recover)
with different tools suited to different environment scales.

## Further analyses (not run by default)

These are real, separate questions this codebase could be extended to
answer, deliberately kept out of the primary standard-vs-modified
comparison so they don't get silently blended into one number. None of
this is implemented yet.

- **The multi-window (periodic-refresh) scheme.** Reusing `D` across many
  outer iterations, refreshing `π_old ← θ` periodically, does let PPO
  extract more from `D` cumulatively than a single window can — the
  question is whether that additional extraction is real (information
  `D` actually supports) or an artifact of the ratio's denominator drifting
  away from `π_β` (see above). Before trusting any result from that
  scheme, two passive diagnostics are cheap to add and would settle it:
  `KL(π_β ‖ π_θ)` on `D`'s own states, and the effective sample size of the
  importance ratio `π_θ/π_β` over `D` — both computable from quantities the
  codebase already has (`Trajectory.log_probs` is `π_β`'s real
  log-probability and is currently unused past the single-window trainer).
  Flat curves would mean the drift stayed small and the scheme's numbers
  can be trusted; growing curves would mean the extra extraction is
  confounded with drift and shouldn't be reported as a clean exploitation-
  gap number without correction.
- **An importance-weighted correction (H8).** A version of the multi-window
  scheme where the ratio is anchored to `π_β` instead of a periodically
  refreshed `π_old` — a real off-policy correction rather than PPO's native
  (single-window-only) trust region. Comparing its `J(π)` against the
  single-window result would give the actual size of the effect, not just
  its presence.
- **A properly designed offline-RL baseline**, as noted above — a different
  question from "what does PPO's own update rule extract," but a natural
  point of comparison once that number is established.
- **Preserving the best intermediate policy during a training window.**
  The epoch-count analysis (see "Results analysis") shows fixed-D
  optimization can be non-monotonic within a single window — a better
  policy can appear at an intermediate epoch and later be lost to
  continued training, not just plateau. A fixed epoch count chosen in
  advance has no mechanism to recover that best intermediate point. Worth
  exploring: tracking the best-observed checkpoint against a held-out
  signal during training itself (analogous to early stopping) — carefully,
  since using held-out performance to pick a checkpoint mid-training is
  itself a form of selection that would need the same scrutiny already
  applied elsewhere in this project (see the prior-checkpoint confirmation
  mechanism in `scripts/01_train_prior.py`).
- **Scheduling `clip_eps` over training**, analogous to learning-rate
  schedules, rather than treating it as a single fixed value for an
  entire window. The clip_eps sweep (see "Results analysis") shows the
  ceiling is sensitive to this value and that looser settings trade a
  higher ceiling for more instability — a schedule (e.g. loosening early,
  tightening late) might capture the reach of a wide clip without its
  instability. Not known whether this is already standard practice
  elsewhere; worth checking before implementing.
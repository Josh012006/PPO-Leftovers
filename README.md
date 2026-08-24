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
in `D` is routed, in both DP solves, to a dedicated absorbing state with a
large fixed negative reward (`unseen_penalty`, default `-50`, configured in
`configs/reference.yaml`). This must be set more negative than any
achievable real return so that greedy value iteration always prefers a
*visited* action at a state whenever one exists. The same convention is
applied at **live-rollout evaluation time**: if a tabular `π_D*` policy ever
lands on a state with literally zero visited actions in `D` (`.act()`
returns `None`), the evaluator (`eval/evaluate.py`) forcibly ends the
episode there and applies `unseen_penalty`, rather than letting the policy
take an arbitrary, evidence-free action in real dynamics. This is what
keeps `π_D*`'s live score from ever being silently inflated by undefined
behavior on states `D` never covered.

**Evaluation is always live-rollout, for every policy.** The DP solve gives
`π_D*` a closed-form `V(s0)` too, but that's a *theoretical* value under the
(possibly imperfect) solved model. The number that actually goes into the
exploitation-gap arithmetic is always the empirical mean return from live
rollouts in the real environment, using identical seeds/episode counts
across every policy — the prior, both `π_D*` variants, standard PPO, and
modified PPO. `scripts/03_compute_pi_d_star.py` prints both numbers so you
can see how close they are, but only `scripts/05_evaluate_all.py`'s output
is the headline result.

## What "fixed-D training" means for PPO, operationally

PPO's normal rhythm — collect a batch with `π_old` → update for `K` epochs →
set `π_old ← π_θ` → collect a *new* batch → repeat — has no "collect a new
batch" step once `D` is frozen. This project uses the scheme worked out in
design discussion ("option b"), implemented in
`src/ppo_exploitation/ppo/fixed_d_trainer.py`:

```
for outer_iter in range(n_outer_iterations):
    snapshot π_old <- current θ                      # refresh, even though
                                                       # no new rollouts happen
    recompute advantages/returns on D using the CURRENT critic
    for epoch in range(epochs_per_refresh):
        shuffle D into minibatches
        clipped PPO update of θ against the π_old snapshot
```

No call to `env.step()` ever happens in this trainer. `D` is read-only.
`epochs_per_refresh` is kept at a realistic value for real PPO usage (the
YAML default is 10); `n_outer_iterations` can be large, since refreshing
`π_old` and recomputing advantages is now cheap (no environment interaction
cost) — this is what "PPO on a fixed dataset" buys you: many more outer
iterations than a live-rollout budget would ever allow, while each
individual step stays a recognizable, conservative PPO update.

**Standard PPO and modified PPO are the same code path.** They differ only
in which YAML config is loaded (`configs/ppo_fixed_d_standard.yaml` vs.
`configs/ppo_fixed_d_modified.yaml`) — this guarantees any measured
difference in the resulting exploitation gap is attributable to the
hyperparameter/mechanism that changed, never to an accidental code
divergence.

### Open research fork: `init_from_checkpoint`

Should standard/modified PPO on `D` start from a **fresh random
initialization**, or from the **same checkpoint weights that generated
`D`**? This is a genuine, unsettled design choice, not a default to trust
blindly — it's exposed as `init_from_checkpoint` in `PPOHyperparams`:

- `false` (the shipped default): fresh random init. Scientifically
  conservative — isolates "how much can be extracted from `D` alone," with
  no information leaking in from the checkpoint that happened to produce
  `D`.
- `true`: start from the prior's weights. Closer to what PPO does in actual
  practice (it never resets), but confounds "started already good" with
  "extracted from `D`."

Our analyses will include both and compare to have an even larger idea of exploitation-gap .

### Theoretical caveat: PPO-on-fixed-D is not a proper offline-RL correction

The actions in `D` were sampled from the **original behavior policy**
`π_β` (the frozen prior checkpoint), never from any later `π_old` snapshot.
PPO's ratio `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` is a *local*
trust-region device relative to the most recent iterate — it is not an
importance-sampling correction back to `π_β`. As training proceeds, `π_old`
drifts further from `π_β` while the clip only bounds `θ`'s movement relative
to `π_old`, never the growing mismatch with the policy that actually
produced the stored actions. This is a real, unavoidable property of
replaying static, single-behavior-policy data through PPO's native update
rule — results from this trainer answer "how well can PPO's *update
mechanism* exploit `D` under its native (imperfect) treatment of that
mismatch," not "is PPO-on-fixed-D theoretically equivalent to a principled
offline RL method." A true offline-RL baseline (not implemented here) would
be answering a different, better-posed version of the question, and should
be understood as such if added later.

## Hypotheses (H1–H7)

The exploitation gap could come from the value/advantage-estimation side or
the policy-optimization side. Every field below is a YAML-reachable knob in
`configs/ppo_fixed_d_*.yaml` — no code changes needed to test any of these:

| # | Hypothesis | Config field(s) |
|---|---|---|
| H1 | Critic doesn't accurately represent the information in `D` | network capacity, `value_coef`, training length |
| H2 | Advantage estimates misrank actions | `gae_lambda`, GAE implementation |
| H3 | The clip range prevents sufficient policy improvement | `clip_eps` |
| H4 | Insufficient optimization epochs | `epochs_per_refresh`, `n_outer_iterations` |
| H5 | Entropy regularization prevents deterministic exploitation | `entropy_coef` |
| H6 | Policy network lacks capacity | `hidden_sizes` |
| H7 | Optimization hyperparameters (lr, minibatch size, grad clipping) limit convergence | `lr`, `minibatch_size`, `max_grad_norm` |

`configs/ppo_fixed_d_modified.yaml` ships as a worked example testing H4
(`epochs_per_refresh: 10 → 30`) with every other field held identical to
the standard config — copy it as a template for testing any other single
hypothesis (change exactly one field per copy, per the project's own "modify
ONE component at a time" rule).

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
matches it to `1e-4`. Fast (seconds), narrow, no claim about the real
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
    fixed_d_trainer.py           # THE research instrument: fixed-D PPO, option-(b) scheme
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
  test_fixed_d_trainer.py         # Tier-1: trainer smoke test (runs, stays finite, weights move)
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

<!-- See "Two kinds of tests, kept deliberately separate" above.
`tests/test_reference.py` is the one worth reading closely if you want to
trust the ceiling numbers: it hand-derives the closed-form fixed point of a
tiny stochastic self-loop MDP (`V(0) = 0.45/0.55` for a specific
reward/gamma setup) and checks the value-iteration solver matches it to
`1e-4`. `tests/tier0/` is the one worth running first if you want to trust
the *pipeline as a whole* before spending time reading through the maze-
specific code — it validates the same D → π_D* → fixed-D-PPO → evaluate
chain against an environment neither of us built. -->

## Running the full pipeline

```bash
bash scripts/run_pipeline.sh
```

or step by step:

```bash
# 1. Train the online PPO prior to ~60% success rate (live environment)
python scripts/01_train_prior.py \
    --env-config configs/env_maze.yaml \
    --prior-config configs/prior_training.yaml \
    --out results/prior_checkpoint.pt

# 2. Freeze that checkpoint, collect the fixed dataset D from it
python scripts/02_collect_dataset.py \
    --env-config configs/env_maze.yaml \
    --checkpoint results/prior_checkpoint.pt \
    --n-episodes 2000 --seed 1 \
    --out results/dataset_D.pkl

# 3. Solve pi_D* exactly (both empirical and true-restricted definitions)
python scripts/03_compute_pi_d_star.py \
    --env-config configs/env_maze.yaml \
    --reference-config configs/reference.yaml \
    --dataset results/dataset_D.pkl \
    --out-empirical results/pi_d_star_empirical.pkl \
    --out-true-restricted results/pi_d_star_true_restricted.pkl

# 4. Train standard PPO and modified PPO on the SAME D (no env interaction)
python scripts/04_train_fixed_d_ppo.py \
    --dataset results/dataset_D.pkl \
    --ppo-config configs/ppo_fixed_d_standard.yaml \
    --out results/ppo_standard_on_D.pt \
    --history-out results/ppo_standard_on_D_history.csv

python scripts/04_train_fixed_d_ppo.py \
    --dataset results/dataset_D.pkl \
    --ppo-config configs/ppo_fixed_d_modified.yaml \
    --out results/ppo_modified_on_D.pt \
    --history-out results/ppo_modified_on_D_history.csv

# 5. Evaluate everything under the identical live-rollout protocol
python scripts/05_evaluate_all.py \
    --env-config configs/env_maze.yaml \
    --reference-config configs/reference.yaml \
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
3. Re-run step 4 against the *same* `results/dataset_D.pkl` with the new
   config, then add it to step 5's `--ppo-checkpoints` list.

Never regenerate `D` between comparison runs — the whole point of the
design is that every variant sees byte-identical experience.

## What this codebase does *not* try to answer

- **Cross-algorithm comparison.** This is not "is PPO better than SAC/DQN
  at exploitation" — see the motivating paper for that broader comparison
  across algorithms and many environments. This project fixes PPO and
  varies only its own internals.
- **A principled offline-RL solution.** The fixed-D trainer here is PPO
  running on static data via its native (and, per the caveat above,
  imperfect) update rule — not a Conservative Q-Learning / decision-
  transformer-style offline method. Answering the question "how good could a
  *properly designed* offline method do on this same D", could be a natural and 
  interesting extension of this codebase.
- **Wall-clock/sample-efficiency claims.** All comparisons here are about
  final exploitation quality given a fixed `D`, not about how many outer
  iterations or how much compute each variant needed to get there (though
  `results/*_history.csv` records enough — `approx_kl`, `clip_frac`,
  losses over outer iterations — to investigate that separately if useful).

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
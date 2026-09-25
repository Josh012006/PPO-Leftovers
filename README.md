# PPO Exploitation Gap

**Research question.** Given a fixed amount of experience `D`, what prevents
PPO from reaching the best policy that can be extracted from `D`? Can that gap be reduced by a careful selection of hyperparameters or does it require a change to the optimization mechanism of PPO itself ? 

This is a study of PPO's *optimization/exploitation* behavior in isolation
from exploration: the environment, the data-collection policy, and the
dataset itself are all held fixed, so that any performance difference
between experiments is attributable to how well each one extracts
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

<div align="center">

| Term | Means | Does NOT mean |
|---|---|---|
| Exploitation *frequency* | how often the agent acts greedily w.r.t. its current policy (e.g. deterministic/mode action at eval time) | how *good* that greedy behavior actually is |
| Exploitation *quality* | how much of the value latent in `D` the learned policy actually recovers | how deterministic the policy is |
| Optimization | the mechanical process of updating θ to improve the PPO objective | value estimation |

</div>

A policy can be 100% "exploitative" in the frequency sense (always takes
its best-known action) while still being very bad at exploitation in the
quality sense (its best-known action is wrong, because the critic/advantage
estimate or the policy update failed to extract what `D` actually supports).
This project is about the latter.

## Phase 1: the stochastic maze (complete)

The project's first phase used a stochastic 30x30 maze with redundant
paths to study the PPO exploitation gap under fixed-D training.
Systematic hyperparameter tuning across all seven native PPO
hyperparameter groups closed most of the gap (standard PPO's `65.0%` to
a best configuration's `95.4%`, against a `99.2%` ceiling). The
disagreement investigation that followed found two real, independent,
partially-explanatory mechanisms (an asymmetry inherited from `π_β`'s own
behavior, and sparse pair-level data) -- but ultimately found that this
specific maze's redundant paths made it impossible to tell whether either
mechanism costs any real, measurable performance, since the states where
PPO and `π_D*` disagree are overwhelmingly off the path a real rollout
ever takes.

**Full write-up, every experiment, every result:**
[docs/PHASE1_STOCHASTIC_MAZE.md](docs/PHASE1_STOCHASTIC_MAZE.md).

## Phase 2: a more controlled experiment (complete)

Phase 1's redundant maze made it impossible to tell whether the
disagreements PPO showed with `pi_D*` cost anything real, since most of
them sat off any path a rollout ever took. Phase 2 redesigned the same
maze around three requirements phase 1's design lacked: a task hard
enough to require genuine learning, a small enough number of good
decisions that disagreement shows up in `success_rate` rather than being
absorbed by redundancy, and a start-state distribution that makes
overfitting to `D` directly observable rather than merely assumed absent
(covered vs. held-out starts, evaluated separately). This section
summarizes what that redesign found; the full write-up -- every sweep,
every figure, every number -- lives in
[docs/PHASE2_CONTROLLED_MAZE.md](docs/PHASE2_CONTROLLED_MAZE.md).

**Evaluation protocol.** Every result from here on is a single
`weighted_success_rate = 0.25*overall + 0.25*covered + 0.50*held_out`,
selected by its `mean` over the full training run (not a single best
epoch) specifically to resist peak-chasing.

**The systematic sweep.** All seven native PPO hyperparameter groups were
swept in sequence on the new task (clip range, GAE lambda, entropy
coefficient, value coefficient, max grad norm, epoch count, minibatch
size), each building on the previous winner. This alone found a
configuration (`clip_eps=0.55, gae_lambda=0.90, entropy_coef=0.0,
value_coef=1.0, max_grad_norm=0.1`) that closed most of the gap to the
`pi_D*` ceiling on this task, without touching PPO's own update rule.

**The disagreement investigation.** Comparing this best checkpoint's
greedy policy against `pi_D*` state by state, restricted to disagreements
both `pi_D*` references (empirical and true-restricted) agree with each
other on and where PPO does measurably worse, found two independent,
causally-validated mechanisms, confirmed by patching `pi_D*`'s action
into disagreement states and re-evaluating live:

1. **Genuinely close decisions** (~2/3 of disagreements): the true
   value gap between the top two actions is small enough that no amount
   of data resolves it -- a real but low-severity mechanism.
2. **A severe, topologically-isolated near-goal pocket** (a handful of
   states, 10-20 total visits despite geometric proximity to the goal):
   an exploration failure, not an exploitation one -- out of this
   project's stated scope (fixed `D`, no new data collection), and never
   resolved by any mechanism tried in this phase, regardless of
   hyperparameter or weighting strength.

**An attempt to close the gap: effective-sample weighting.** Borrowing
Cui et al. (2019)'s class-imbalance re-weighting idea (`w(n)/n`, an
"effective number of samples" correction), applied per `(state, action)`
pair using `D`'s own exact counts. A naive first sweep failed outright
(wrong scale relative to `D`'s actual count distribution); once properly
calibrated, a **KL-anchored decay** -- letting the weighting's strength
fade as the policy's cumulative KL-divergence from `pi_beta` grows, since
`pi_old` never moves in this fixed-`D` setting and `approx_kl` is
therefore already a cumulative measure, not a per-epoch one -- became the
first configuration to beat the un-weighted baseline on `mean`
(0.4289 vs. 0.4127), while cutting strict state-level disagreements from
66 to 51 (-23%), concentrated specifically in the "close decision"
mechanism above; the near-goal pocket was untouched, exactly as its
exploration-failure diagnosis predicts.

**A full `beta x k` cross-sweep** (36 combinations, resolving a gap left
by calibrating `beta` and `k` separately in sequence) confirmed
`beta=0.9995` as the right anchor -- its entire row stays in the
strongest, widest band of the whole grid, unlike isolated single-cell
spikes elsewhere -- while finding `k=0.30` edges out the previously
reported `k=0.10` on `mean` (0.4357 vs. 0.4289). Re-running the full
disagreement pipeline on `k=0.30`, however, found it fixes *zero* of
`k=0.10`'s 51 disagreements and adds 11 new ones (the tightest true
decisions in the dataset) -- its `mean` advantage comes entirely from a
smoother trajectory away from an otherwise identical peak, not from a
better policy. **`k=0.10` is what this project carries forward**: the two
peak checkpoints are functionally tied on every aggregate number, so
`k=0.30`'s only edge is bought at the cost of concrete, checkable
mistakes `k=0.10` never makes.

**Testing a second hypothesis: does this generalize past exact counts?**
The exact count this whole mechanism depends on has no equivalent in a
continuous action space. This phase searched hard for a substitute and
found a consistent, informative negative result rather than a working
one:

- **A small critic ensemble's disagreement** (variance across
  independently-initialized heads regressing the same GAE returns)
  correlates with the true count on 5 of 6 deliberately independent
  testbeds (this project's own maze, a fresh layout, FrozenLake, Taxi,
  Blackjack) -- but the *functional relationship* between that variance
  and the true count is not a stable law: an empirically fit power-law
  exponent ranged from `-0.55` to `-1.54` across three testbeds, an
  artifact of network capacity and training duration, not a fixed
  statistical property.
- **Random Network Distillation** was checked against the same testbeds
  and showed the identical instability, for the same underlying reason:
  the issue is intrinsic to any signal built by training a network via
  gradient descent to convergence on fixed data, not particular to
  ensembles.
- **Bellemare et al. (2016)'s density-based pseudo-count** -- which needs
  no sample count at all -- was implemented and tested; it correlates
  only weakly with the true count and is highly sensitive to an arbitrary
  "recoding" learning rate with no principled default, reproducing the
  same kind of instability under a different mechanism.
- **A direct reformulation abandoning the pseudo-count idea entirely** --
  `weight(s,a) = 1 - exp(-Variance(s,a)/tau)`, no beta, no count-shaped
  intermediate quantity -- was implemented last. Its first full-run test
  (constant `tau`) underperformed the established baseline outright; a
  KL-anchored decay for `tau` (shrinking rather than growing, since
  `weight` depends on its scale parameter in the opposite direction
  `w(n)/n` depends on `beta`'s) has been implemented but not yet evaluated
  on a full run.

Full detail on every one of these -- exact numbers, every figure, every
rejected intermediate idea -- is in
[docs/PHASE2_CONTROLLED_MAZE.md](docs/PHASE2_CONTROLLED_MAZE.md), which
closes with the same two open threads phase 3 picks up below.

## Phase 3: realistic training and continuous actions

Phase 3 has not started. It picks up two threads phase 2 closed without
resolving, and neither is a minor loose end -- each is a genuine
precondition for treating phase 2's central finding (the
effective-sample-weighting fix, KL-anchored, `beta=0.9995`, `k=0.10`) as
more than a result specific to phase 2's own deliberately extreme
protocol.

**1. Test on a full, realistic training run -- multiple short windows,
not one long one.** Every phase 2 result, without exception, comes from a
single trust-region window stretched to `epochs=300` against a `pi_old`
that never moves. Realistic PPO (Stable-Baselines3, CleanRL, and
effectively every published configuration) instead alternates short
windows of `E=3` to `10` epochs against a fresh rollout and a refreshed
`pi_old`, repeated many times over training. The exploitation-gap
mechanism phase 2 isolated -- an advantage estimate computed once and
then treated as equally reliable regardless of how many times a given
`(state, action)` pair was actually seen -- is structurally present in
that realistic loop too, just diluted across many small windows instead
of concentrated in one long one. Whether the KL-anchored fix transfers to
that regime, in what form, and at what strength (the `k` calibrated
against a 300-epoch window has no established relationship to whatever
would suit a 5-epoch one) is a real, unanswered empirical question. The
natural next experiment: instrument a standard multi-window training
loop, apply the same weighting mechanism at a similarly reduced strength,
and check whether the same qualitative benefit (fewer exploitable
state-level disagreements with `pi_D*`, a comparable or better `mean`
weighted_success_rate) survives the transfer.

**2. Extend the mechanism to continuous action spaces.** The exact
`(state, action)` count this fix depends on has no equivalent once
actions are continuous -- two continuous actions are essentially never
bit-identical, so "how many times was this pair observed" stops meaning
anything. Phase 2 searched hard for a substitute and came back with a
clear negative result rather than a working one:

- A small **critic ensemble's disagreement** correlates with the true
  count on 5 of 6 deliberately independent testbeds, but the *functional
  relationship* between variance and count is not a stable law -- a fitted
  power-law exponent ranged from `-0.55` to `-1.54` across three testbeds,
  an artifact of network capacity and training duration.
- **Random Network Distillation** showed the identical instability, for
  the same underlying reason: it is intrinsic to any signal built by
  training a network via gradient descent to convergence on fixed data.
- **Bellemare et al. (2016)'s density-based pseudo-count** correlates only
  weakly with the true count and is highly sensitive to an arbitrary
  "recoding" learning rate with no principled default.
- A **direct reformulation** dropping the pseudo-count idea entirely
  (`weight(s,a) = 1 - exp(-Variance(s,a)/tau)`, no beta) underperformed
  the baseline with constant `tau`; a KL-anchored decay for `tau` has been
  implemented but not yet evaluated on a full run.

Phase 3 either finds an approach that works reliably, or documents that
boundary as a genuine limit of this whole line of attack.

**For the complete record of everything established before phase 3
begins**, read
[docs/PHASE1_STOCHASTIC_MAZE.md](docs/PHASE1_STOCHASTIC_MAZE.md) (the
original stochastic maze, the first hyperparameter sweep, and the
disagreement investigation that motivated redesigning the environment)
together with
[docs/PHASE2_CONTROLLED_MAZE.md](docs/PHASE2_CONTROLLED_MAZE.md) (the
controlled maze, the full sweep and disagreement investigation on it, and
the effective-sample-weighting fix and its validation).

## Project structure

```
configs/                        # every experimental knob lives here, not in code
  phase1/                        # stochastic maze study (see docs/PHASE1_STOCHASTIC_MAZE.md)
    env_maze.yaml                  # maze layout, stochasticity, reward
    prior_training.yaml            # online PPO config for the checkpoint that generates D
    ppo_fixed_d_standard.yaml      # baseline fixed-D PPO hyperparameters
    ppo_fixed_d_modified.yaml      # example H4 ablation (edit/copy for other hypotheses)
    reference.yaml                 # pi_D* solver settings (gamma, unseen_penalty, VI tolerance)
    ...                            # (the rest of phase 1's sweep configs)
  phase2/                        # controlled-redundancy / variable-start study (see docs/PHASE2_CONTROLLED_MAZE.md)

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
    --env-config configs/phase1/env_maze.yaml \
    --prior-config configs/phase1/prior_training.yaml \
    --out results/phase1/prior_checkpoint.pt

# 2. Freeze that checkpoint, collect the fixed dataset D from it
python scripts/02_collect_dataset.py \
    --env-config configs/phase1/env_maze.yaml \
    --checkpoint results/phase1/prior_checkpoint.pt \
    --n-episodes 4000 --seed 1 \
    --out results/phase1/dataset_D.pkl

# 3. Solve pi_D* exactly (both empirical and true-restricted definitions)
python scripts/03_compute_pi_d_star.py \
    --env-config configs/phase1/env_maze.yaml \
    --reference-config configs/phase1/reference.yaml \
    --dataset results/phase1/dataset_D.pkl \
    --out-empirical results/phase1/pi_d_star_empirical.pkl \
    --out-true-restricted results/phase1/pi_d_star_true_restricted.pkl

# 4. Train standard PPO and modified PPO on the SAME D and the SAME prior
#    checkpoint (pi_old = pi_beta = this checkpoint throughout -- see
#    "What fixed-D training means for PPO, operationally")
python scripts/04_train_fixed_d_ppo.py \
    --dataset results/phase1/dataset_D.pkl \
    --ppo-config configs/phase1/ppo_fixed_d_standard.yaml \
    --prior-checkpoint results/phase1/prior_checkpoint.pt \
    --out results/phase1/ppo_standard_on_D.pt \
    --history-out results/phase1/ppo_standard_on_D_history.csv

python scripts/04_train_fixed_d_ppo.py \
    --dataset results/phase1/dataset_D.pkl \
    --ppo-config configs/phase1/ppo_fixed_d_modified.yaml \
    --prior-checkpoint results/phase1/prior_checkpoint.pt \
    --out results/phase1/ppo_modified_on_D.pt \
    --history-out results/phase1/ppo_modified_on_D_history.csv

# 5. Evaluate everything under the identical live-rollout protocol
python scripts/05_evaluate_all.py \
    --env-config configs/phase1/env_maze.yaml \
    --prior-checkpoint results/phase1/prior_checkpoint.pt \
    --pi-d-star-empirical results/phase1/pi_d_star_empirical.pkl \
    --pi-d-star-true-restricted results/phase1/pi_d_star_true_restricted.pkl \
    --ppo-checkpoints standard=results/phase1/ppo_standard_on_D.pt modified=results/phase1/ppo_modified_on_D.pt \
    --n-episodes 500 --eval-seed 999 \
    --out results/phase1/gap_report.csv
```

`scripts/05_evaluate_all.py` accepts any number of `name=path` pairs after
`--ppo-checkpoints`, so once you've trained additional ablation configs
(H1–H7 or otherwise) from step 4, add them all to a single step-5 call to
compare every variant against the same `π_D*` reference in one table.

### Testing a new hypothesis

1. Copy `configs/phase1/ppo_fixed_d_modified.yaml` to a new file
   (e.g. `configs/phase1/ppo_fixed_d_h3_wide_clip.yaml`).
2. Revert the field(s) already changed back to the standard value, then
   change exactly one field to test your hypothesis (e.g. `clip_eps: 0.2 →
   0.4` for H3).
3. Re-run step 4 against the *same* `results/phase1/dataset_D.pkl` and the *same*
   `results/phase1/prior_checkpoint.pt` with the new config, then add it to step
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
  away from `π_β` (see docs/PHASE1_STOCHASTIC_MAZE.md, "What 'fixed-D
training' means for PPO, operationally"). Before trusting any result from that
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
- **A properly designed offline-RL baseline** — a different
  question from "what does PPO's own update rule extract," but a natural
  point of comparison once that number is established.
- **Preserving the best intermediate policy during a training window.**
  The epoch-count analysis (see docs/PHASE1_STOCHASTIC_MAZE.md, "Results
  analysis") shows fixed-D
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
  entire window. The clip_eps sweep (see docs/PHASE1_STOCHASTIC_MAZE.md,
  "Results analysis") shows the
  ceiling is sensitive to this value and that looser settings trade a
  higher ceiling for more instability — a schedule (e.g. loosening early,
  tightening late) might capture the reach of a wide clip without its
  instability. Not known whether this is already standard practice
  elsewhere; worth checking before implementing.


## Literature

Papers this project's design or discussion draws on directly — not a
general reading list, only what's actually behind a specific decision or
claim made above.

**Motivating paper**

- Berseth, G. (2025). *Is Exploration or Optimization the Problem for Deep
  Reinforcement Learning?* arXiv:2508.01329.
  [arxiv.org/pdf/2508.01329](https://arxiv.org/pdf/2508.01329) — introduces
  the experience-optimal policy and practical sub-optimality concepts this
  project's entire decomposition is built on. See "Motivating paper" and
  "Relationship to the motivating paper" above for exactly how this
  project adapts it.

**Core algorithm and methods used directly**

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
  [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) — the
  algorithm this whole project studies. Its Atari/MuJoCo hyperparameter
  tables (`K=3` vs. `K=10` epochs) are the reference point for how far
  this project's epoch-count analysis (H4) pushes beyond conventional
  usage.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
  *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation.* arXiv:1506.02438 (ICLR 2016).
  [arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438) — defines
  the GAE formula (`gae_lambda`'s bias/variance trade-off) this project's
  `FixedDPPOTrainer` implements directly, and whose `λ→0` vs. `λ→1`
  behavior is the basis for the entire "GAE lambda sweep" analysis.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic
  Optimization.* arXiv:1412.6980 (ICLR 2015).
  [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) — the
  optimizer used throughout every trainer in this project
  (`torch.optim.Adam`); its per-parameter moment estimates were relevant
  to reasoning about the `value_coef` / shared-gradient-clipping question.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). *Class-Balanced 
  Loss Based on Effective Number of Samples.* arXiv:1901.05555 (CVPR 2019).
  [arxiv.org/abs/1901.05555](https://arxiv.org/abs/1901.05555) — the class-balancing 
  method used to account for imbalanced training data by re-weighting the loss 
  according to the effective number of samples in each class.
- Osband, I., Blundell, C., Pritzel, A., & Van Roy, B. (2016). *Deep
  Exploration via Bootstrapped DQN.* arXiv:1602.04621 (NeurIPS 2016).
  [arxiv.org/abs/1602.04621](https://arxiv.org/abs/1602.04621) — proposed
  alternative to RND for the same purpose: an ensemble of value heads
  whose disagreement, rather than a separate predictor network's error,
  serves as the confidence signal.

**Implementation verification reference**

- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., &
  Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning
  Implementations.* Journal of Machine Learning Research, 22(268), 1–8.
  [jmlr.org/papers/v22/20-1364.html](https://jmlr.org/papers/v22/20-1364.html)
  — this project's single combined loss / single optimizer / single
  global `clip_grad_norm_` pattern (see "Value coefficient sweep") was
  checked directly against SB3's reference PPO implementation, confirming
  it matches standard practice rather than being a project-specific
  deviation.
# Effective State-Action Counts for Local Coverage-Aware PPO Reweighting

## 1. Motivation: from sample frequency to effective coverage

Standard empirical objectives implicitly treat every occurrence in the dataset as an additional unit of information. This is problematic when many observations are highly redundant: repeatedly observing essentially the same region of the state-action space does not necessarily provide as much new information as observing a new, distinct region.

This motivates replacing the raw number of observations by an **effective count**.

The idea follows the effective-number formulation introduced by Cui et al. for long-tailed classification. Instead of assuming that $n$ samples provide $n$ independent units of information, the effective number is defined as

$$
E_\beta(n) =
\frac{1-\beta^n}{1-\beta},
\qquad
0 \leq \beta < 1.
$$

Equivalently,

$$
E_\beta(n)=
1+\beta+\beta^2+\cdots+\beta^{n-1}.
$$

The marginal contribution of the $(n+1)$-th observation is therefore

$$
E_\beta(n+1)-E_\beta(n)=
\beta^n,
$$

which decreases as more observations are accumulated.

Thus, the model explicitly captures **diminishing information returns due to redundancy**.

---

## 2. Extending effective counts to continuous state-action spaces

In classification, the count $n$ can simply be the number of samples belonging to a class. In reinforcement learning, however, the state-action space is generally continuous:

$$
x=(s,a)\in\mathcal S\times\mathcal A.
$$

Two observations will almost never have exactly the same continuous $(s,a)$. Therefore, an exact count is not meaningful.

Instead, we define a local neighborhood around each state-action point.

For a given observation

$$
x=(s,a),
$$

let

$$
B_\epsilon(x)=
\left\{
x'\in\mathcal S\times\mathcal A
:
d(x,x')\leq\epsilon
\right\}
$$

be a local neighborhood, where $d(\cdot,\cdot)$ is a distance on the state-action space.

For example, with an appropriately normalized Euclidean distance,

$$
d(x,x')=
\left\|
\begin{bmatrix}
s\\a
\end{bmatrix} -
\begin{bmatrix}
s'\\a'
\end{bmatrix}
\right\|_2.
$$

The local count around $x$ is then

$$
n_\epsilon(x) =
\sum_{i=1}^{N}
\mathbf 1
\left[
d(x,x_i)\leq\epsilon
\right].
$$

The effective local count becomes

$$
\boxed{
E_\beta(x) =
\frac{1-\beta^{n_\epsilon(x)}}{1-\beta}.
}
$$

Thus, two nearby observations are treated as partially redundant rather than as completely independent pieces of information.

### Gaussian-kernel formulation

The hard neighborhood above can also be replaced by a smooth kernel. For example, with a Gaussian kernel,

$$
K_h(x,x_i) =
\exp
\left(
-\frac{d(x,x_i)^2}{2h^2}
\right),
$$

we can define a soft local count

$$
n_h(x) =
\sum_{i=1}^{N}K_h(x,x_i).
$$

The same effective-count transformation can then be applied:

$$
\boxed{
E_\beta(x) =
\frac{1-\beta^{n_h(x)}}{1-\beta}.
}
$$

The important idea is not whether the neighborhood is implemented with a hard ball or a Gaussian kernel. The important idea is that **the count measures local coverage in state-action space rather than exact duplicate occurrences**.

### Separate bandwidths for state and action

The formulation above measures $d(x,x')$ as a single distance over the concatenated vector $[s;a]$, with one bandwidth $h$. This conflates two different notions of closeness: state and action spaces generally have different natural scales and dimensions, so "same state, different action" and "different state, same action" are not comparable through a single joint distance.

The kernel factorizes cleanly into a product over the state part and the action part:

$$
\boxed{
K(x,x_i) = K_s\!\big(d(s,s_i)\big)\cdot K_a\!\big(d(a,a_i)\big),
\qquad
K_s(r)=e^{-r^2/2h_s^2},
\quad
K_a(r)=e^{-r^2/2h_a^2},
}
$$

with independent bandwidths $h_s$, $h_a$, giving the same soft local count as before, $n_h(x)=\sum_i K(x,x_i)$. The single joint kernel above is the special case $h_s=h_a=h$ with $s$ and $a$ pre-concatenated. Letting either bandwidth go to zero recovers exact matching on that side alone while the other stays smoothed: $h_a\to0$ gives exact action matching with a smoothed state (the discrete-action case this project's own environments use), and symmetrically for $h_s\to0$.

None of Sections 5-15 depend on which distance or kernel produced $n_\epsilon(x)$ or $n_h(x)$ -- every subsequent property ($1\le E_\beta(n)\le n$, the bounded compression, the four-case analysis) holds for any valid local count, this factorized one included.

---

## 3. Standard empirical PPO distribution

Let the dataset contain $N$ state-action observations

$$
D=\{x_i=(s_i,a_i)\}_{i=1}^{N}.
$$

The empirical distribution used by standard PPO is

$$
p_D(x) =
\frac{n(x)}{N},
$$

where $n(x)$ is the number of observations associated with $x$.

For continuous spaces, this should be understood as the empirical distribution over the observed samples, or equivalently as a discretized/local representation when state-action neighborhoods are used.

The standard empirical policy objective can be written abstractly as

$$
J_D(\theta) =
\mathbb E_{x\sim p_D}
\left[
L_\theta(x)
\right],
$$

where $L_\theta(x)$ denotes the PPO surrogate contribution of a state-action sample.

Its gradient is

$$
\nabla_\theta J_D(\theta) =
\mathbb E_{x\sim p_D}
\left[
g_\theta(x)
\right],
$$

where

$$
g_\theta(x) =
\nabla_\theta L_\theta(x).
$$

The important point is that **the empirical frequency of a state-action region directly determines how much that region contributes to the objective**.

If one region occurs ten times more frequently than another, it receives approximately ten times as much total empirical weight.

---

## 4. The true RL visitation distribution

The empirical distribution $p_D$ is not the underlying RL distribution we ultimately care about.

For a policy $\pi_\theta$, the relevant state-action visitation distribution is

$$
d^{\pi_\theta}(s,a).
$$

The policy-gradient theorem gives, in its standard form,

$$
\boxed{
\nabla_\theta J(\theta) =
\mathbb E_{(s,a)\sim d^{\pi_\theta}}
\left[
\nabla_\theta
\log\pi_\theta(a\mid s)
A^{\pi_\theta}(s,a)
\right].
}
$$

Define

$$
g_\theta(s,a) =
\nabla_\theta
\log\pi_\theta(a\mid s)
A^{\pi_\theta}(s,a).
$$

Then

$$
\boxed{
\nabla_\theta J(\theta) =
\mathbb E_{d^{\pi_\theta}}
[g_\theta(s,a)].
}
$$

This gives us three conceptually distinct distributions:

$$
\boxed{
d^{\pi_\theta}
\longrightarrow
p_D
\longrightarrow
q_\beta
}
$$

with different meanings:

* $d^{\pi_\theta}$: the state-action distribution induced by the policy and associated with the true RL objective.
* $p_D$: the empirical distribution of the collected dataset, which standard PPO uses to approximate the relevant expectation.
* $q_\beta$: the effective-count distribution produced by our reweighting.

The goal is **not** to claim that $q_\beta$ is automatically equal to $d^{\pi_\theta}$. Rather, the theoretical question is whether replacing raw frequency by effective local coverage can produce a more useful approximation of the state-action structure relevant to the RL objective.

### 4.1 Closing the chain: what a departure from $p_D$ can still mean for $\pi_\theta$

The chain in the box above, $d^{\pi_\theta}\to p_D\to q_\beta$, is stated schematically but never closed: nothing so far relates $q_\beta$, a transform of the *fixed* dataset $D$, to $d^{\pi_\theta}$, the visitation distribution of the *current, still-changing* policy.

**A first attempt at closing it is wrong, and worth stating explicitly so it is not repeated.** $p_D$ is already the standard, consistent statistical estimator of $d^{\pi_{\mathrm{prior}}}$ -- by the law of large numbers, $p_D\to d^{\pi_{\mathrm{prior}}}$ as $N\to\infty$, exactly, with no correction needed. $q_\beta$ does not converge to $d^{\pi_{\mathrm{prior}}}$ even in that limit: since $E_\beta(n)\to\frac{1}{1-\beta}$ as $n\to\infty$ for any fixed $\beta<1$, a region's effective mass saturates at a finite constant no matter how much data accumulates there. $q_\beta$ therefore does not approximate $d^{\pi_{\mathrm{prior}}}$ better than $p_D$ does -- it departs from it, permanently and by construction, regardless of sample size. This is not a flaw to patch; it is exactly what Cui et al.'s class-balanced loss already does in classification: it never tries to better estimate the true class distribution (the raw empirical frequency already does that optimally) -- it deliberately reweights away from that empirical distribution because the distribution itself, however well estimated, is judged not to be the right thing to optimize against directly. $q_\beta$ makes the same kind of choice, not a better version of the same estimation problem $p_D$ already solves.

Given that, the trust-region bound below does not establish "$q_\beta$ is a better estimate of $d^{\pi_\theta}$." It establishes something more modest, but still useful: that the *geometric structure* $q_\beta$'s departure is built from -- which regions of $D$ are locally dense versus locally rare -- does not become a stale description of $\pi_\theta$'s own situation just because $\pi_\theta$ is no longer $\pi_{\mathrm{prior}}$. If

$$
\alpha=\max_s D_{TV}\!\left(\pi_\theta(\cdot\mid s),\ \pi_{\mathrm{prior}}(\cdot\mid s)\right)
$$

is bounded, then (Kakade & Langford, 2002; used in this form in Schulman et al., 2015, the TRPO paper) the two policies' visitation distributions are close:

$$
\boxed{
\left\|d^{\pi_\theta}-d^{\pi_{\mathrm{prior}}}\right\|_1
\ \le\
\frac{2\alpha}{1-\gamma}.
}
$$

A region that $D$ shows as locally dense under $\pi_{\mathrm{prior}}$ is therefore still, approximately, a locally dense region under $\pi_\theta$'s own visitation, and likewise for a locally rare one -- as long as $\alpha$ stays small. The reweighting choice $q_\beta$ makes about *which regions deserve relatively less or more influence* is a choice about that shared geometric structure, not about $\pi_{\mathrm{prior}}$ specifically -- and the bound is what keeps that structure from having quietly become a description of a policy $\pi_\theta$ no longer resembles. Whether departing from $p_D$ in the direction $q_\beta$ chooses is itself a *good* choice is a separate question, addressed empirically in Section 15 and its own limits, not something this bound speaks to.

**This closure is conditional, not unconditional**, and honestly so: the bound degrades as $\alpha$ grows, exactly when $\pi_\theta$ drifts away from $\pi_{\mathrm{prior}}$. This project's own empirical work gives a direct, observable proxy for when that is happening: a sustained rise in `clip_frac` signals that $\alpha$ is no longer small, since `clip_frac` measures the fraction of samples where the policy ratio has left the trust region clipping is meant to enforce. The chain above should therefore be read as holding *while `clip_frac` stays low* -- not as an unconditional guarantee, but as a precise statement of the one condition under which it is a guarantee at all.

---

## 5. The effective-count distribution

For each observation $x=(s,a)$, define the local effective count

$$
E_\beta(x) =
\frac{1-\beta^{n_\epsilon(x)}}{1-\beta}.
$$

Our reweighting assigns each occurrence a factor proportional to

$$
\frac{E_\beta(x)}{n_\epsilon(x)}.
$$

This is important because a local region containing $n_\epsilon(x)$ observations has total weight

$$
n_\epsilon(x)
\frac{E_\beta(x)}{n_\epsilon(x)} =
E_\beta(x).
$$

Therefore, after summing over all observations in a region, its **total contribution is proportional to its effective count rather than its raw count**.

The resulting effective distribution can be written schematically as

$$
\boxed{
q_\beta(x) =
\frac{p_D(x)f_\beta(x)}
{\mathbb E_{p_D}[f_\beta(x)]}
}
$$

where

$$
f_\beta(x) =
\frac{E_\beta(x)}{n_\epsilon(x)} =
\frac{1-\beta^{n_\epsilon(x)}}
{(1-\beta)n_\epsilon(x)}.
$$

Thus,

$$
q_\beta(x)
\propto
E_\beta(x).
$$

This is the key transformation:

$$
\boxed{
p_D(x)\propto n_\epsilon(x)
\quad\longrightarrow\quad
q_\beta(x)\propto E_\beta(x).
}
$$

We therefore replace **raw sampling frequency** by **effective local coverage**.

---

## 6. Why the state-action level matters in RL

A particularly important aspect of this construction is that the effective count is defined over the joint state-action space:

$$
n_\epsilon(s,a),
$$

rather than only over states:

$$
n_\epsilon(s).
$$

This distinction is fundamental for reinforcement learning.

For a fixed state $s$, the policy must learn a distribution over actions:

$$
\pi(a\mid s).
$$

Suppose the dataset contains two distinct action regions $a_1$ and $a_2$ for the same state:

$$
n_\epsilon(s,a_1)=100,
\qquad
n_\epsilon(s,a_2)=10.
$$

With raw frequency weighting, the first region contributes ten times as many samples.

But the 100 observations around $a_1$ do not necessarily represent ten times more independent information than the 10 observations around $a_2$. They may instead correspond to a highly redundant cluster.

For example, with $\beta=0.9$,

$$
E_{0.9}(100)\approx10,
$$

while

$$
E_{0.9}(10)\approx6.51.
$$

The effective ratio is therefore approximately

$$
\frac{E_{0.9}(100)}
{E_{0.9}(10)}
\approx1.54,
$$

rather than the raw-count ratio

$$
\frac{100}{10}=10.
$$

The reweighting therefore does **not** artificially force the two action regions to have equal importance. Instead, it compresses the influence of the highly redundant region.

This gives a useful interpretation:

> **The objective is encouraged to represent distinct local regions of the action space rather than simply reproducing the empirical frequency with which particular actions were sampled.**

This is especially relevant in RL because the policy is not merely learning how frequently states occur. It is learning **which actions to assign probability to within each state**.

A state-level count,

$$
n(s)=\sum_a n(s,a),
$$

would lose precisely this information.

If one action dominates the dataset for a particular state, a state-only effective count could conclude that the state is well represented, even if the action distribution within that state is extremely concentrated.

The state-action formulation instead asks:

$$
\boxed{
\text{How much effective information do we have locally around this particular decision }(s,a)?
}
$$

---

## 7. The connection to effective-number reweighting

The construction is directly inspired by the effective-number principle of Cui et al.

Their setting considers long-tailed classification, where different classes have different numbers of examples. Their observation is that the raw number of samples does not necessarily correspond to the amount of distinct information because examples can overlap in the underlying feature space.

The effective number

$$
E_\beta(n) =
\frac{1-\beta^n}{1-\beta}
$$

therefore grows sublinearly with $n$.

Our extension changes the object being counted.

Instead of asking

$$
\text{How many examples belong to class }c?
$$

we ask

$$
\boxed{
\text{How many effectively distinct observations cover the local neighborhood of }(s,a)?
}
$$

Thus the conceptual correspondence is:

| Long-tailed classification      | Our RL setting                               |
| ------------------------------- | -------------------------------------------- |
| Class $c$                     | Local state-action region $(s,a)$          |
| Number of class samples $n_c$ | Local count $n_\epsilon(s,a)$              |
| Class imbalance                 | State-action coverage imbalance              |
| Effective number $E(n_c)$     | Local effective count $E(n_\epsilon(s,a))$ |
| Class-balanced loss             | Coverage-aware PPO objective                 |

The important extension is therefore not merely applying Cui et al.'s formula to RL. It is **moving the notion of effective count from a class-level quantity to a local geometric quantity in state-action space**.

---

## 8. What the reweighting changes in the RL objective

Standard PPO approximately optimizes an empirical expectation of the form

$$
J_D(\theta) =
\mathbb E_{p_D}
[L_\theta(s,a)].
$$

Our method instead optimizes

$$
\boxed{
J_\beta(\theta) =
\mathbb E_{q_\beta}
[L_\theta(s,a)].
}
$$

The local PPO contribution $L_\theta(s,a)$ itself does not have to change.

What changes is the **measure under which the objective is integrated**:

$$
\boxed{
\text{Standard PPO:}
\qquad
\text{importance}\propto\text{raw frequency}
}
$$

versus

$$
\boxed{
\text{Effective-count PPO:}
\qquad
\text{importance}\propto\text{effective local coverage}.
}
$$

This distinction is important because the method is not simply a correction to an individual gradient term. It changes which regions of the state-action space dominate the empirical objective.

---

## 9. Why this can be useful for RL

The central hypothesis is that raw visitation frequency can be a poor proxy for useful information.

A dataset may contain many observations because the behavior policy repeatedly visits a narrow region:

$$
\underbrace{
x_1,x_2,\ldots,x_{100}
}_{\text{highly redundant}}
$$

while another region may contain only a few observations that are much more spatially separated:

$$
\underbrace{
y_1,y_2,\ldots,y_{10}
}_{\text{greater local diversity}}.
$$

Raw empirical weighting interprets the first region as ten times more important.

Effective-count weighting instead recognizes diminishing returns:

$$
100
\quad\longrightarrow\quad
E_\beta(100),
$$

while

$$
10
\quad\longrightarrow\quad
E_\beta(10).
$$

The resulting objective is therefore less dominated by repeated observations from already well-covered regions.

This suggests three main theoretical advantages.

---

## 10. Advantage 1: reducing domination by oversampled regions

Under standard empirical weighting,

$$
p_D(x)\propto n(x).
$$

Consequently, repeated observations from a dense region can dominate the objective.

Under effective-count weighting,

$$
q_\beta(x)\propto E_\beta(n_\epsilon(x)).
$$

Since

$$
E_\beta(n)
<
n
$$

for $0<\beta<1$ and $n>1$, the contribution of additional observations grows sublinearly.

Thus the method reduces the disproportionate influence of regions that are heavily revisited.

The important interpretation is not that frequent samples are "bad". Rather:

> **Repeated samples are assumed to have diminishing marginal informational value when they occupy the same local region.**

This is precisely the effective-number principle.

---

## 11. Advantage 2: preserving local action diversity

The second advantage is specific to RL.

For a given state $s$, the policy must learn the conditional action distribution

$$
\pi(a\mid s).
$$

If the dataset is heavily concentrated around one action region,

$$
p_D(a_1\mid s)
\gg
p_D(a_2\mid s),
$$

then standard empirical weighting causes the objective to receive many more contributions from the first region.

If this difference is primarily caused by repeated observations rather than genuinely greater coverage, the empirical objective can become dominated by a narrow portion of the action space.

The state-action effective count instead computes

$$
E_\beta(n_\epsilon(s,a))
$$

locally for each action region.

Therefore,

$$
\boxed{
\text{different action regions retain their distinct representation}
}
$$

while

$$
\boxed{
\text{redundant repetitions within one region have diminishing influence}.
}
$$

This is different from simply upweighting rare actions.

The method does not say that a rare action should automatically be preferred. It says that **additional observations of an already densely covered action region should contribute progressively less new information**.

This distinction is important because it avoids turning the method into a purely inverse-frequency heuristic.

---

## 12. Advantage 3: a controllable trade-off between frequency and coverage

The parameter $\beta$ determines how aggressively raw frequency is converted into effective coverage.

As

$$
\beta\rightarrow1,
$$

we have

$$
E_\beta(n)\rightarrow n,
$$

and therefore

$$
q_\beta\rightarrow p_D.
$$

The method becomes standard empirical weighting.

At the other extreme,

$$
\beta=0,
$$

gives

$$
E_0(n)=1.
$$

Every observed local region therefore receives the same effective count.

Hence,

$$
q_0
$$

approaches a uniform distribution over the distinct observed regions.

Intermediate values produce a continuum:

$$
\boxed{
\text{uniform coverage}
\quad
\longleftrightarrow
\quad
\text{empirical frequency}.
}
$$

Thus $\beta$ controls the amount of diminishing returns attributed to repeated local observations.

Small $\beta$ strongly emphasizes coverage diversity, whereas $\beta$ close to $1$ preserves more of the empirical visitation frequency.

---

## 13. The theoretical hypothesis for RL

The preceding construction leads to a precise hypothesis.

Let

$$
d^{\pi_\theta}(s,a)
$$

denote the state-action visitation distribution associated with the RL objective, and let

$$
p_D(s,a)
$$

be the empirical distribution generated by the available dataset.

Standard PPO uses the empirical frequency structure

$$
p_D(s,a)\propto n(s,a)
$$

Our method instead constructs

$$
q_\beta(s,a)
\propto
E_\beta(n_\epsilon(s,a)).
$$

The hypothesis is that, when raw visitation frequency contains substantial redundancy,

$$
\boxed{
q_\beta
\text{ can provide a more coverage-aware representation of the state-action space than }
p_D.
}
$$

This does **not** by itself prove that $q_\beta$ produces a lower-bias policy-gradient estimator. The gradient also depends on the local PPO quantities, including the advantage, policy ratio, clipping, and policy parameters.

Rather, the proposed mechanism is one level more fundamental:

$$
\boxed{
\text{raw frequency}
\rightarrow
\text{redundancy}
\rightarrow
\text{effective local coverage}
\rightarrow
q_\beta
\rightarrow
\text{coverage-aware RL objective}.
}
$$

The subsequent question is then whether this improved representation of state-action coverage translates into a better approximation of the policy-gradient objective and/or a more favorable bias-variance trade-off.

---

## 14. A useful decomposition for the next theoretical step

Let

$$
g_\theta(s,a)
$$

denote the local policy-gradient contribution.

The standard and reweighted gradient expectations are

$$
G_D = \mathbb E_{p_D}[g_\theta],
$$

and

$$
G_\beta = \mathbb E_{q_\beta}[g_\theta].
$$

Their difference is

$$
G_\beta-G_D = \mathbb E_{q_\beta}[g_\theta] - \mathbb E_{p_D}[g_\theta]
$$

Using

$$
q_\beta(x) = \frac{p_D(x)f_\beta(x)}{\mathbb E_{p_D}[f_\beta]}
$$

where

$$
f_\beta(x) = \frac{E_\beta(n_\epsilon(x))}{n_\epsilon(x)}
$$

we obtain

$$
\boxed{
G_\beta-G_D =
\frac{
\text{Cov}_{p_D}
\left(
f_\beta(x),g_\theta(x)
\right)
}{
\mathbb E_{p_D}[f_\beta(x)]
}
}
$$

This equation shows that the reweighting changes the gradient according to the relationship between **local effective coverage** and the local gradient contribution.

However, this should not be interpreted as saying that rare state-action regions necessarily have larger gradients. Such a claim would depend on the particular RL problem and on PPO quantities such as $A(s,a)$, the policy ratio, and the clipping parameter.

The more general and architecture-independent claim is that the reweighting changes the **measure over which the gradient is estimated**, from raw local frequency to effective local coverage.

That distinction provides a clean separation between:

1. the **geometric/statistical effect** of effective-count reweighting, and
2. the **optimization effect** of applying that new distribution to a PPO objective.

The first is what we just discussed independently of the details of PPO. The second is the next step of the theoretical analysis.


---

## 15. Low vs. High Coverage: Effect on Gradient Contributions

The effective-count reweighting does not simply mean "upweight rare samples and downweight frequent samples". More precisely, it **compresses the difference in influence between low- and high-coverage regions**.

To reason about the effect on the PPO gradient, consider a direction of reference $u$, for example the direction of the standard empirical gradient:

$$
u=\frac{G_D}{\|G_D\|}.
$$

For an observation $x=(s,a)$, define its projected gradient contribution as

$$
z(x)=u^\top g_\theta(x),
$$

where, at the beginning of a PPO update,

$$
g_\theta(s,a) =
A(s,a)\nabla_\theta\log\pi_\theta(a|s).
$$

Thus:

* $z(x)>0$: the observation pushes in the reference gradient direction;
* $z(x)<0$: it pushes against that direction;
* $|z(x)|$: magnitude of its contribution.

The effective-count reweighting uses

$$
f_\beta(n) =
\frac{E_\beta(n)}{n} =
\frac{1-\beta^n}{(1-\beta)n}.
$$

For $0<\beta<1$, $f_\beta(n)$ is decreasing in $n$. Therefore, each observation from a highly covered region receives less weight than an observation from a poorly covered region.

Importantly, this is a **relative** effect. The method does not create additional observations or arbitrarily amplify rare regions.

### High coverage + useful gradient contribution

Consider a highly covered region $H$ with

$$
n_H\gg1
$$

and a useful gradient contribution

$$
z_H>0.
$$

Under the raw empirical count, its total influence is proportional to

$$
W_{\mathrm{raw}}(H)=n_H.
$$

If the region is heavily represented, this contribution can become very large simply because the same state-action region appears many times.

With effective-count weighting,

$$
W_{\mathrm{eff}}(H) =
E_\beta(n_H) =
\frac{1-\beta^{n_H}}{1-\beta}.
$$

Since

$$
E_\beta(n_H)\leq n_H,
$$

the useful signal is still preserved, but its influence is compressed.

In particular, for large $n_H$,

$$
E_\beta(n_H)
\rightarrow
\frac{1}{1-\beta}.
$$

Thus, even when the region provides a genuinely useful gradient, additional repetitions cannot increase its influence indefinitely.

The method therefore does **not** assume that high-coverage regions are bad. It simply assumes that repeated observations should eventually provide diminishing additional evidence.

$$
\boxed{
\text{High coverage + useful gradient}
\Rightarrow
\text{useful signal preserved, redundancy compressed}
}
$$

### High coverage + weak or harmful gradient contribution

Now consider a highly covered region for which

$$
n_H\gg1
$$

but whose contribution is small, noisy, or points against the desired direction:

$$
z_H\approx0
\qquad\text{or}\qquad
z_H<0.
$$

Under raw counting, the region can nevertheless have a large aggregate influence because its contribution is repeated $n_H$ times.

The effective count reduces this influence to

$$
E_\beta(n_H)\ll n_H
$$

for sufficiently large $n_H$.

This is particularly useful when high coverage is caused by redundancy rather than by genuinely independent evidence. A weak or harmful signal repeated hundreds of times should not necessarily dominate a less-covered region merely because it has been observed more often.

Hence:

$$
\boxed{
\text{High coverage + weak/harmful gradient}
\Rightarrow
\text{strong compression of its influence}
}
$$

This is one of the main motivations for using coverage information in the first place.

### Low coverage + useful gradient contribution

Consider instead a low-coverage region $L$:

$$
n_L\ll n_H,
$$

but with a useful gradient contribution,

$$
z_L>0.
$$

Under raw counting, its aggregate influence is small because it has few observations:

$$
W_{\mathrm{raw}}(L)=n_L.
$$

The effective count gives

$$
W_{\mathrm{eff}}(L)=E_\beta(n_L).
$$

Since

$$
\frac{E_\beta(n)}{n}
$$

is larger for small $n$, the observations from this region receive a larger weight **relative to observations from highly covered regions**.

Thus, the useful signal becomes relatively more visible to the optimizer.

However, this should not be described as arbitrary amplification. We still have

$$
E_\beta(n_L)\leq n_L.
$$

The total effective mass of the region therefore never exceeds its original empirical mass.

What changes is its **relative share** compared with highly covered regions.

$$
\boxed{
\text{Low coverage + useful gradient}
\Rightarrow
\text{relative emphasis without absolute amplification}
}
$$

This distinction is important: the effective count does not manufacture importance for rare observations. It reduces the extent to which repeated observations elsewhere dominate them.

### Low coverage + weak or noisy gradient contribution

Finally, consider a low-coverage region with

$$
n_L\ll n_H
$$

but whose gradient contribution is weak, noisy, or potentially harmful:

$$
z_L\approx0,
\qquad
z_L<0,
$$

or has high variance.

This is the case where the reweighting can be less clearly beneficial.

Because the region is poorly covered, its relative weight increases compared with highly covered regions. Consequently, a noisy low-coverage signal can also receive greater relative influence.

However, this increase remains bounded by the original number of observations:

$$
1\leq E_\beta(n_L)\leq n_L.
$$

Even in the most aggressive case,

$$
\beta=0,
$$

we obtain

$$
E_0(n)=1.
$$

Thus, a region with $n_L$ observations is never turned into a region with an arbitrarily large effective mass.

The method therefore trades one potential problem for another in a controlled way: it reduces the dominance of highly covered regions, but it cannot guarantee that every low-coverage signal is useful.

$$
\boxed{
\text{Low coverage + weak/noisy gradient}
\Rightarrow
\text{relative emphasis, but bounded}
}
$$

### Summary of the four cases

| Coverage | Gradient contribution | Effect of effective count                                         |
| -------- | --------------------- | ----------------------------------------------------------------- |
| High     | Useful                | Useful signal is preserved, but redundant influence is compressed |
| High     | Weak / harmful        | Strong compression reduces its ability to dominate                |
| Low      | Useful                | Useful signal receives greater relative influence                 |
| Low      | Weak / noisy          | Relative influence also increases, but remains bounded            |

This gives a more precise interpretation of the reweighting.

The objective is **not** to identify which individual gradient contributions are good or bad. The count $n$ does not provide that information.

Instead, the method uses coverage to control how much repeated evidence is allowed to dominate the optimization:

$$
\boxed{
\text{high coverage}
\Rightarrow
\text{more compression}
}
$$

$$
\boxed{
\text{low coverage}
\Rightarrow
\text{less compression}
}
$$

The fundamental guarantee comes from

$$
1\leq E_\beta(n)\leq n.
$$

Therefore, the effective count always remains between counting a region only once and counting all of its observations independently.

This is why the correction is substantially more conservative than an inverse-frequency weighting scheme. It **compresses the coverage imbalance rather than inverting it**.

For two regions $H$ and $L$, with

$$
n_H>n_L,
$$

the raw ratio of their masses is

$$
R_{\mathrm{raw}} =
\frac{n_H}{n_L}.
$$

The effective-count ratio is

$$
R_{\mathrm{eff}} =
\frac{E_\beta(n_H)}{E_\beta(n_L)}.
$$

Because $E_\beta(n)$ is increasing and $E_\beta(n)/n$ is decreasing,

$$
\boxed{
1\leq R_{\mathrm{eff}}\leq R_{\mathrm{raw}}.
}
$$

Thus, the relative difference in influence is reduced, but the ordering is not reversed.

For example, if

$$
n_H=1000,
\qquad
n_L=10,
$$

then

$$
R_{\mathrm{raw}}=100.
$$

With $\beta=0.9$,

$$
E_{0.9}(1000)\approx10,
$$

while

$$
E_{0.9}(10)\approx6.51.
$$

Hence,

$$
R_{\mathrm{eff}}
\approx
\frac{10}{6.51}
\approx1.54.
$$

The imbalance has therefore been compressed from approximately $100:1$ to $1.54:1$, rather than being inverted.

This bounded compression is the key safety property of the approach: **we do not drastically amplify rare regions, while we prevent highly repeated regions from accumulating arbitrarily large influence.**

### Which case dominates in practice: bias versus noise in $z(x)$

The four-case table above is a taxonomy, not a proof that Case 3 (low coverage, useful gradient) dominates Case 4 (low coverage, noisy gradient) for RL problems in general -- nothing in the construction of $E_\beta$ argues for one case over the other, since $n_\epsilon(x)$ is a purely frequentist quantity, computed from how the data was collected, and $z(x)$ is a statement about optimization quality, computed from the current policy and critic. The two are related only through whatever the specific problem happens to do.

A distinction sharpens what actually separates the two cases: **whether the low-coverage error in $z(x)$ is noise or bias.**

- If $z(x)$ is a noisy but *unbiased* estimate of the true useful direction at low-coverage regions, upweighting it does not help on average -- it only adds variance to an estimate stochastic gradient descent already handles well through averaging across minibatches and epochs. This is Case 4's risk.
- If $z(x)$ is *biased* at low-coverage regions -- systematically wrong, not just uncertain -- then no amount of further SGD averaging removes that error on its own; more weight on the correction is exactly the lever that helps, since the error is not something the optimizer would otherwise wash out for free. This is Case 3.

This project's own disagreement-factor analysis (fixed-D PPO on the controlled maze; see the README) gives a direct, empirical answer for this project's own setting: `log_pair_min_samples` -- essentially $-\log n_\epsilon(s,a)$ -- is the dominant factor explaining *where* the trained policy disagrees with the oracle $\pi_{D^*}$, net of coverage and of the behavior policy's own preference. A factor that reliably predicts *where* an error occurs is evidence of a systematic relationship between coverage and correctness, not of unpredictable noise: pure noise would not produce a clean, dominant explanatory factor at all. In this project's own regime -- a single critic, frozen once per fixed window, its own generalization from better-covered neighboring states filling in for what a low-coverage region never got its own gradient signal to correct -- this is exactly the mechanism that would produce a systematic, not merely noisy, error at low coverage. That supports Case 3 as the regime this project's own results were obtained in, though it remains an empirical finding specific to this setting, not a claim that Case 3 dominates for every RL problem effective-count reweighting might be applied to.
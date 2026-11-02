# XGBoost

*Part IV · Trees & Ensembles. Chapter 9 moved predictions downhill using only the slope of the loss. XGBoost also uses how fast the slope changes, and it charges a price for complicated trees.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Gradient Boosting (Chapter 9)](chapter.html?c=09-gradient-boosting) | XGBoost keeps its loop: add one small tree per round to a running prediction. |
| Machine Learning | [Logistic Regression (Chapter 5)](chapter.html?c=05-logistic-regression#eq-newton) | Newton's method: use slope and curvature to jump to the bottom of a parabola. |
| Machine Learning | [Decision Trees (Chapter 7)](chapter.html?c=07-decision-trees#eq-reg-gain) | A regression tree and its split gain, which XGBoost replaces by a new formula. |
| Machine Learning | [Ridge Regression (Chapter 3)](chapter.html?c=03-ridge-lasso#eq-ridge) | The penalty on the leaf values is a Ridge penalty. |
| Calculus & Optimization | Second-order Taylor expansion, derivative of a quadratic | The whole derivation is a sequence of these. |

## 1. Everyday picture: walking downhill in fog

Imagine you are on a hillside in thick fog and want to reach the valley floor.

1. **First-order walker.** You feel only the **slope** under your feet. You know which way is downhill, but not how far to go. A fixed small step is a safe guess (this is gradient descent and Chapter 9).
2. **Second-order walker.** You also feel **how quickly the slope changes**, that is, whether the ground curves sharply or is nearly flat. On a gentle slope that curves slowly, you can stride far. In a tight bowl, you take a short step. This is Newton's method, and it reaches the bottom in far fewer steps.
3. **A careful second-order walker also carries a rule against over-complicated routes.** They refuse to add a new turn to their path unless it saves more than the turn costs.

XGBoost ("eXtreme Gradient Boosting") is the third walker. Compared with [Chapter 9](chapter.html?c=09-gradient-boosting):
- **Same loop:** each round adds one small tree to the current prediction.
- **New:** each tree is chosen using the slope **and** the curvature of the loss.
- **New:** an explicit **penalty on the size and the values of the tree** is part of the objective.
- Because of these two changes, **the best leaf values and the best splits come out of closed-form formulas.**

## 2. The objective for one round

**Step 1: the model so far.** After $t-1$ rounds the prediction for point $i$ is $F_{t-1}(\mathbf{x}_i)$. The round adds a new tree $f_t$, as in [Chapter 9](chapter.html?c=09-gradient-boosting#eq-fspace):

$$ F_t(\mathbf{x}_i)=F_{t-1}(\mathbf{x}_i)+f_t(\mathbf{x}_i) $$ {#eq-add}

(The learning rate $\eta$ is applied at the end of the round, in Section 9.)

**Step 2: what we minimize.** The loss on all training points after adding the tree, plus a penalty $\Omega$ on the new tree:

$$ \text{Obj}_t=\sum_{i=1}^{n}\ell\big(y_i,\;F_{t-1}(\mathbf{x}_i)+f_t(\mathbf{x}_i)\big)+\Omega(f_t) $$ {#eq-obj}

- $\ell$ is the loss for one point (squared error, logistic loss, and so on).
- $\Omega(f_t)$ measures how complicated the new tree is. Section 5 defines it.

:::box plain | What is this objective saying in plain words?
- The first part asks: **after adding this tree, how big is the loss?**
- The second part asks: **how much complexity did we buy to get that?**
- We want a tree that lowers the loss a lot **without** being needlessly large or extreme.
- Chapter 9 had only the first part, and relied on the learning rate and the number of rounds to control complexity.
:::

## 3. The second-order approximation

[[eq-obj]] contains the loss evaluated at $F_{t-1}+f_t$, which is hard to minimize over trees directly. We approximate it with a parabola.

**Step 1: Taylor-expand to second order.** For one point, expand the loss around the current prediction $F_{t-1}(\mathbf{x}_i)$, treating $f_t(\mathbf{x}_i)$ as the small change:

$$ \ell\big(y_i,F_{t-1}+f_t\big)\;\approx\;\ell\big(y_i,F_{t-1}\big)+g_i\,f_t(\mathbf{x}_i)+\tfrac12\,h_i\,f_t(\mathbf{x}_i)^2 $$ {#eq-taylor2}

Here the slope and curvature of the loss at the current prediction are

$$ g_i=\left.\frac{\partial\,\ell(y_i,F)}{\partial F}\right|_{F=F_{t-1}(\mathbf{x}_i)},\qquad h_i=\left.\frac{\partial^2\,\ell(y_i,F)}{\partial F^2}\right|_{F=F_{t-1}(\mathbf{x}_i)} $$ {#eq-gh}

:::box step | How we got here
- This is the usual "value now + slope × change + ½ × curvature × change²".
- Chapter 5 used the same expansion for [Newton's method in one variable](chapter.html?c=05-logistic-regression#eq-taylor). Chapter 9 stopped after the slope term.
- $g_i$ and $h_i$ are computed **once per round for each training point**, from the current predictions. They do not depend on the new tree.
:::

**Step 2: drop the constant.** The first term of [[eq-taylor2]] does not contain $f_t$, so it cannot change which tree is best. Remove it from [[eq-obj]]:

$$ \text{Obj}_t\;\approx\;\sum_{i=1}^{n}\Big[g_i\,f_t(\mathbf{x}_i)+\tfrac12\,h_i\,f_t(\mathbf{x}_i)^2\Big]+\Omega(f_t) $$ {#eq-obj-approx}

Now the **only** things the loss contributes are the numbers $g_i$ and $h_i$. Any twice-differentiable loss can be plugged in without changing what comes next. That is a large part of XGBoost's flexibility.

:::figure F11.1 | First vs second order | The loss of one point drawn against the prediction F. At the current F, the tangent line (first-order approximation) and the parabola (second-order approximation) are drawn. A slider moves the new prediction and shows where each approximation stays close to the true curve.

## 4. The slopes and curvatures for two common losses

**Squared error.** Take $\ell(y,F)=\tfrac12(y-F)^2$, the version used in [Chapter 9](chapter.html?c=09-gradient-boosting#eq-mse-deriv):

$$ g_i=F_{t-1}(\mathbf{x}_i)-y_i,\qquad h_i=1 $$ {#eq-gh-mse}

So $g_i$ is the **negative residual**, and the curvature is a constant. (Some texts use $\ell=(y-F)^2$, which gives $g_i=2(F_{t-1}-y_i)$ and $h_i=2$. That only doubles the loss, which is equivalent to halving $\lambda$ and $\gamma$ below.)

**Logistic loss.** With labels $y_i\in\{0,1\}$ and $p_i=\sigma\big(F_{t-1}(\mathbf{x}_i)\big)$, the gradient and Hessian of the cross-entropy are the ones from [Chapter 5](chapter.html?c=05-logistic-regression#eq-grad-lr):

$$ g_i=p_i-y_i,\qquad h_i=p_i\,(1-p_i) $$ {#eq-gh-logistic}

:::box plain | What are $g_i$ and $h_i$ saying in plain words?
- **$g_i$ is "how wrong, and in which direction"**. For classification it is predicted probability minus truth. It is large for badly wrong points.
- **$h_i$ is "how much the loss curves here"**, a measure of how **uncertain** the model currently is about point $i$. For the logistic loss it is largest when $p_i=\tfrac12$, and close to $0$ when the model is already confident.
- A confident point has small $g_i$ **and** small $h_i$, so it has little influence on the next tree.
:::

| Loss | $g_i$ | $h_i$ |
|---|---|---|
| Squared error $\tfrac12(y-F)^2$ | $F-y$ | $1$ |
| Logistic ($y\in\{0,1\}$) | $p-y$ | $p(1-p)$ |

## 5. Describing a tree and its penalty

**Step 1: a tree is a lookup of leaf values.** Suppose the tree has $T$ leaves, numbered $1,\dots,T$. Let $q(\mathbf{x})$ be the function that sends an input to the **index of the leaf it lands in**, and $w_j$ be the constant value stored in leaf $j$:

$$ f_t(\mathbf{x})=w_{q(\mathbf{x})},\qquad q:\mathbb{R}^d\to\{1,\dots,T\} $$ {#eq-tree}

**Step 2: the complexity penalty.**

$$ \Omega(f_t)=\gamma\,T+\tfrac12\,\lambda\sum_{j=1}^{T}w_j^2 $$ {#eq-omega}

- **$\gamma\,T$** charges $\gamma$ for every leaf, so **big trees cost more**.
- **$\tfrac12\lambda\sum_jw_j^2$** is an $\ell_2$ penalty on the leaf values, exactly the Ridge penalty of [Chapter 3](chapter.html?c=03-ridge-lasso#eq-ridge). It stops any leaf from making an extreme prediction.

## 6. The best leaf values, in closed form

This is the heart of the chapter. The aim is to turn [[eq-obj-approx]] into a sum of **independent** small problems, one per leaf.

**Step 1: group the data points by leaf.** Let $I_j=\{\,i:\,q(\mathbf{x}_i)=j\,\}$ be the set of training points that end up in leaf $j$. Every point in $I_j$ gets the same prediction $w_j$ by [[eq-tree]]. Define the **sums of slopes and curvatures** in leaf $j$:

$$ G_j=\sum_{i\in I_j}g_i,\qquad H_j=\sum_{i\in I_j}h_i $$ {#eq-GH}

**Step 2: rewrite the objective leaf by leaf.** Put [[eq-tree]] and [[eq-omega]] into [[eq-obj-approx]]. In the sum over points, replace $f_t(\mathbf{x}_i)$ by $w_j$ for each $i\in I_j$, and pull the common $w_j$ out of the inner sums:

$$ \text{Obj}_t\approx\sum_{j=1}^{T}\Big[\,G_j\,w_j+\tfrac12\,\big(H_j+\lambda\big)\,w_j^2\Big]+\gamma\,T $$ {#eq-obj-leaf}

:::box step | How we got here
- $\sum_{i\in I_j}g_i\,w_j=w_j\sum_{i\in I_j}g_i=G_j\,w_j$, and similarly $\sum_{i\in I_j}\tfrac12h_iw_j^2=\tfrac12H_jw_j^2$.
- The penalty $\tfrac12\lambda\sum_jw_j^2$ adds $\tfrac12\lambda\,w_j^2$ to each leaf's term, which turns $\tfrac12H_jw_j^2$ into $\tfrac12(H_j+\lambda)w_j^2$.
- The leaf count term $\gamma T$ is the sum of $\gamma$ over leaves.
:::

**What we have gained.** Each term in the bracket of [[eq-obj-leaf]] involves **only one leaf's value $w_j$**. A hard $T$-dimensional optimization has split into $T$ completely independent one-dimensional quadratic problems.

**Step 3: minimize one leaf.** Differentiate the bracket with respect to $w_j$ and set it to zero:

$$ G_j+(H_j+\lambda)\,w_j=0\;\Longrightarrow\;w_j^\ast=-\frac{G_j}{H_j+\lambda} $$ {#eq-wstar}

(It is a minimum because the coefficient $\tfrac12(H_j+\lambda)$ of $w_j^2$ is positive: $h_i\ge0$ for a convex loss and $\lambda>0$.)

:::box plain | What is this equation saying in plain words?
- **Numerator $-G_j$:** move the leaf **against** the total slope of its points. If the points' predictions are too high (positive $g_i$), the leaf value is pushed down.
- **Denominator $H_j+\lambda$:** the total curvature, plus the penalty $\lambda$. **High curvature or large $\lambda$ means a smaller, more cautious step.**
- This is Newton's step $-g/h$ from [Chapter 5](chapter.html?c=05-logistic-regression#eq-newton-1d), summed over the leaf, with $\lambda$ added to the denominator as a safety brake.
:::

:::tabs Three ways to read the $+\lambda$ in the denominator
--- Shrinkage
Without $\lambda$ the leaf value is $-G_j/H_j$. Adding $\lambda$ makes the denominator bigger, so **every leaf value is pulled towards zero**. The pull is strongest in leaves with little total curvature, that is, leaves with few or very confident points. Those are exactly the leaves whose values are least trustworthy.
--- Newton step with a brake
Newton's method divides the slope by the curvature. If the curvature is nearly zero the step becomes enormous and unstable. Adding $\lambda$ keeps the denominator away from zero, like a speed limit on the step.
--- Ridge on leaf values
Minimizing $G_jw+\tfrac12H_jw^2+\tfrac12\lambda w^2$ is a small Ridge regression with a single parameter. As in [Chapter 3](chapter.html?c=03-ridge-lasso#eq-ridge-sol), the penalty shrinks the solution by the factor $H_j/(H_j+\lambda)$.
:::

**Check against Chapter 9.** For squared error, [[eq-gh-mse]] gives $G_j=-\sum_{i\in I_j}r_i$ (minus the sum of residuals) and $H_j=\lvert I_j\rvert$ (the number of points). With $\lambda=0$,

$$ w_j^\ast=\frac{\sum_{i\in I_j}r_i}{\lvert I_j\rvert}=\text{mean residual in the leaf} $$

which is exactly the leaf value of Chapter 9. XGBoost reduces to Chapter 9 when $\lambda=0$ and the loss is squared error. With $\lambda>0$ the leaf value is multiplied by the shrinkage factor $\dfrac{H_j}{H_j+\lambda}$.

:::figure F11.3 | Leaf weight and lambda | A leaf containing a handful of points with their g and h values shown. A slider for lambda changes the optimal leaf value minus G over H plus lambda, drawn as a bar that shrinks towards zero, and the effect on the points' updated predictions.

**Step 4: the value of the best objective.** Substitute $w_j^\ast$ back into one bracket of [[eq-obj-leaf]]:

$$ G_jw_j^\ast+\tfrac12(H_j+\lambda)\,{w_j^\ast}^2=-\frac{G_j^2}{H_j+\lambda}+\frac12\frac{G_j^2}{H_j+\lambda}=-\frac12\,\frac{G_j^2}{H_j+\lambda} $$

Summing over leaves gives the **score of a tree structure**, the lowest the objective can be once the leaves are fixed:

$$ \text{Obj}_t^\ast=-\frac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma\,T $$ {#eq-objstar}

:::box plain | What is this equation saying in plain words?
- Each leaf contributes a **quality score** $\dfrac{G_j^2}{H_j+\lambda}$. It is large when the points in the leaf all want to move in the **same direction** (so $G_j$ is big in size), and small when their pushes cancel.
- The minus sign means **a bigger total score gives a lower objective**, and lower is better.
- $\gamma T$ is the price of having $T$ leaves.
- So a good tree has leaves full of points that **agree** about which way the prediction should move.
:::

## 7. The split-gain formula

[[eq-objstar]] scores a **complete** tree. To grow a tree one split at a time, we compare the score **before** and **after** splitting a leaf.

**Step 1: set up.** A leaf with slope and curvature totals $G=\sum_{i}g_i$ and $H=\sum_ih_i$ is split into a left child $(G_L,H_L)$ and a right child $(G_R,H_R)$, where

$$ G_L+G_R=G,\qquad H_L+H_R=H $$ {#eq-conservation}

**Step 2: the score before and after.** From [[eq-objstar]], with one leaf versus two:

$$ \text{before}=-\frac12\frac{G^2}{H+\lambda}+\gamma,\qquad\text{after}=-\frac12\frac{G_L^2}{H_L+\lambda}-\frac12\frac{G_R^2}{H_R+\lambda}+2\gamma $$

(All other leaves are unchanged, so they cancel in the comparison. The split adds one extra leaf, which costs one extra $\gamma$.)

**Step 3: gain is the improvement.** Gain is how much the objective drops, which is **before minus after**:

$$ \text{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}\right]-\gamma $$ {#eq-gain}

:::box step | How we got here
- Subtract: the $-\tfrac12\tfrac{G_L^2}{H_L+\lambda}-\tfrac12\tfrac{G_R^2}{H_R+\lambda}$ terms of "after" flip sign into the positive bracket, and the $-\tfrac12\tfrac{G^2}{H+\lambda}$ from "before" stays with a minus sign.
- The constants: $\gamma-2\gamma=-\gamma$.
:::

:::box plain | What is this equation saying in plain words?
- **The bracket** compares the quality of the **two children** with the quality of the **parent**. It is positive when splitting makes the points within each child agree more about their direction.
- **$-\gamma$** is the entry fee for the extra leaf.
- **Rule: accept a split only if $\text{Gain}>0$**, and otherwise leave the node as a leaf. This is built-in **pruning**: a larger $\gamma$ blocks splits that improve things only slightly.
- Compare with [Chapter 7's gain](chapter.html?c=07-decision-trees#eq-reg-gain): the same "parent versus children" idea, but with the quality score taken from [[eq-objstar]] instead of from variance or entropy.
:::

:::figure F11.2 | Split gain explorer | Training points sorted along one feature, each shown with its g and h. Click a threshold to see G_L, H_L, G_R, H_R and the gain, with the best threshold marked. A slider for gamma moves the zero line, and a slider for lambda changes the scores.

:::figure F11.4 | Gain and pruning | A small tree with the gain of every split written on its branches. A slider for gamma turns branches grey, from the bottom up, as their gain falls below zero.

### A worked example by hand

Use the data of [Chapter 9's example](chapter.html?c=09-gradient-boosting#eq-f0-mse): four points $x=1,2,3,4$ with targets $y=1,3,7,9$, squared error, starting prediction $F_0=\bar y=5$. Take $\lambda=1$, $\gamma=0$ and $\eta=0.5$.

**Slopes and curvatures** by [[eq-gh-mse]]: $g_i=F_0-y_i$ and $h_i=1$.

| $x$ | $y$ | $g_i=5-y$ | $h_i$ |
|---|---|---|---|
| 1 | 1 | $+4$ | 1 |
| 2 | 3 | $+2$ | 1 |
| 3 | 7 | $-2$ | 1 |
| 4 | 9 | $-4$ | 1 |

The root has $G=0$ and $H=4$, so its score term is $\dfrac{G^2}{H+\lambda}=0$.

**Three candidate splits.** Use [[eq-gain]] with $\gamma=0$:

| Split | $(G_L,H_L)$ | $(G_R,H_R)$ | $\dfrac{G_L^2}{H_L+1}$ | $\dfrac{G_R^2}{H_R+1}$ | Gain |
|---|---|---|---|---|---|
| between $x=1$ and $x=2$ | $(4,\,1)$ | $(-4,\,3)$ | $8$ | $4$ | $\tfrac12(8+4-0)=6$ |
| **between $x=2$ and $x=3$** | $(6,\,2)$ | $(-6,\,2)$ | $12$ | $12$ | $\tfrac12(12+12-0)=\mathbf{12}$ |
| between $x=3$ and $x=4$ | $(4,\,3)$ | $(-4,\,1)$ | $4$ | $8$ | $\tfrac12(4+8-0)=6$ |

The middle split wins with a gain of $12$. With a penalty $\gamma\ge12$ it would be **rejected**, and the tree would stay a single leaf.

**Leaf values** by [[eq-wstar]]:
- left leaf: $w^\ast=-\dfrac{6}{2+1}=-2$,
- right leaf: $w^\ast=-\dfrac{-6}{2+1}=+2$.

**Updating the predictions** with $\eta=0.5$: left points $5+0.5\times(-2)=4$, right points $5+0.5\times2=6$.

**Comparison with Chapter 9.** Chapter 9 used the leaf means $-3$ and $+3$ (this is $\lambda=0$ in [[eq-wstar]]: $-6/2=-3$). Here $\lambda=1$ shrinks them by the factor $\dfrac{H}{H+\lambda}=\dfrac23$, giving $-2$ and $+2$. XGBoost takes smaller, more cautious steps.

## 8. Using second-order information: why $h_i$ is a sample weight

There is a neat way to see what the curvature $h_i$ is doing. Take one term of [[eq-obj-approx]] and **complete the square**:

$$ g_i\,f+\tfrac12\,h_i\,f^2=\tfrac12\,h_i\Big(f+\frac{g_i}{h_i}\Big)^2-\frac{g_i^2}{2h_i} $$ {#eq-complete-square}

The last term does not depend on the tree, so up to a constant

$$ \text{Obj}_t\;\simeq\;\sum_{i=1}^{n}\tfrac12\,h_i\Big(\underbrace{-\frac{g_i}{h_i}}_{\text{target}}-f_t(\mathbf{x}_i)\Big)^2+\Omega(f_t) $$ {#eq-weighted-ls}

:::box plain | What is this equation saying in plain words?
- XGBoost is fitting a tree by **weighted least squares**, as in [Chapter 1](chapter.html?c=01-orthogonal-projections) and [Chapter 7](chapter.html?c=07-decision-trees#eq-reg-imp).
- The **target** of point $i$ is $-g_i/h_i$, the Newton step for that point alone.
- The **weight** of point $i$ is its curvature $h_i$.
- For squared error ($h_i=1$) the weights are all equal and the target is the ordinary residual, which is exactly [Chapter 9](chapter.html?c=09-gradient-boosting#eq-fit).
- For the logistic loss the weight $p_i(1-p_i)$ is **largest for uncertain points**, so the next tree concentrates on points the model is unsure about. This resembles the reweighting in AdaBoost and the $\mathbf{R}$ matrix in [IRLS](chapter.html?c=05-logistic-regression#eq-irls).
:::

**One consequence for the leaf values.** A leaf of the weighted-least-squares fit (with $\lambda=0$) is the **weighted mean** of the targets $-g_i/h_i$, which equals $-\dfrac{\sum g_i}{\sum h_i}=-\dfrac{G_j}{H_j}$, matching [[eq-wstar]]. This is the proper leaf rule for the logistic loss that [Chapter 9's warning](chapter.html?c=09-gradient-boosting#eq-neggrad-logistic) promised: use $\sum(y_i-p_i)\big/\sum p_i(1-p_i)$ in place of a plain mean.

## 9. Finding the splits, and the full round

### Exact greedy search

For one feature, the best threshold can be found in a single sweep.

1. **Sort** the $n$ points by the feature value.
2. **Sweep** from left to right, keeping running totals $G_L$ and $H_L$ of the points passed so far. At each gap between two different values, the right-hand totals are $G_R=G-G_L$ and $H_R=H-H_L$ by [[eq-conservation]].
3. **Score** each of the at most $n-1$ thresholds with [[eq-gain]].
4. **Repeat** for every feature and keep the best split overall.

With $d$ features the cost of a pass over all points is about $O\big(d\,n\log n\big)$, dominated by sorting. This is the same greedy strategy as in [Chapter 7](chapter.html?c=07-decision-trees), and it has the same weakness of not looking ahead. Ensembles are the cure.

### Approximate search for large data

When $n$ is huge, sorting and testing every threshold is too slow. The approximate algorithm:
1. Proposes only a **small set of candidate thresholds** for each feature, taken at **quantiles** of the feature.
2. Uses **weighted** quantiles, with weight $h_i$ for point $i$.
3. Scores only those candidates with [[eq-gain]].

**Why weight the quantiles by $h_i$?** By [[eq-weighted-ls]], $h_i$ is exactly the weight of point $i$ in the objective. Spacing the candidates so that each bucket contains about the same **total weight** means every bucket matters equally to the objective. Uncertain points then get a finer grid of thresholds than confident points.

### The whole round, and the learning rate

For each round $t=1,\dots,T$:
1. Compute $g_i$ and $h_i$ for every training point from the current predictions, as in [[eq-gh]].
2. Grow a tree by repeatedly splitting a leaf with the best positive gain from [[eq-gain]], stopping when no split has $\text{Gain}>0$ (or a depth limit is reached).
3. Set each leaf to $w_j^\ast$ from [[eq-wstar]].
4. Update the prediction with a shrunken step:

$$ F_t(\mathbf{x})=F_{t-1}(\mathbf{x})+\eta\,f_t(\mathbf{x}) $$ {#eq-update-xgb}

As in [Chapter 9](chapter.html?c=09-gradient-boosting#eq-fspace), a small $\eta$ makes the model move slowly. It makes each tree less decisive, smooths the overall function and reduces the risk of overfitting to the noisy gradient of any one round.

:::figure F11.5 | Gradient vs Newton boosting | The same noisy data fitted by first-order boosting (Chapter 9) and by XGBoost. Training and test error are plotted against the number of rounds, together with the tree built in the first round by each method.

## 10. The knobs, and what each one controls

| Knob | Where it appears | Effect | Direction for less overfitting |
|---|---|---|---|
| $\eta$ (learning rate) | [[eq-update-xgb]] | size of each round's step | smaller (and use more rounds) |
| $\lambda$ | leaf penalty in [[eq-omega]] | shrinks leaf values: $-G/(H+\lambda)$ | larger |
| $\gamma$ | per-leaf cost in [[eq-omega]] | a split must gain more than $\gamma$ | larger |
| Tree depth | stopping rule | how many features can interact in one tree | smaller |
| Minimum $H_j$ per leaf | stopping rule ("minimum child weight") | blocks leaves with too little total curvature | larger |
| Row and column subsampling | each round | decorrelates trees, as in [Chapter 8](chapter.html?c=08-random-forest#eq-var-final) | smaller fractions |
| Number of rounds $T$ | outer loop | more rounds means lower bias, eventually higher variance | stop early on a validation set |

**Why a minimum on $H_j$ and not on the number of points?** $H_j=\sum_{i\in I_j}h_i$ counts points **by how uncertain they are**. A leaf with many confidently predicted points can have a small $H_j$, and then its value would be estimated from little real information.

**What is new compared with plain gradient boosting:**

| | Gradient boosting (Chapter 9) | XGBoost |
|---|---|---|
| Information used about the loss | slope $g_i$ | slope $g_i$ and curvature $h_i$ |
| Leaf values | mean of pseudo-residuals (approximate for logistic loss) | $-G_j/(H_j+\lambda)$, exact for the quadratic approximation, [[eq-wstar]] |
| Split quality | variance reduction of residuals | the gain formula, [[eq-gain]] |
| Complexity control | only $\eta$, depth and the number of rounds | also built into the objective through $\gamma$ and $\lambda$, [[eq-omega]] |
| Pruning | none during growth | splits with $\text{Gain}\le0$ are refused |
| Large data | exact search | quantile-based approximate search, weighted by $h_i$ |

:::box warn | Common misunderstandings
- XGBoost is **not** a different kind of model. The final predictor is still a sum of small trees, as in [Chapter 9's Equation](chapter.html?c=09-gradient-boosting#eq-final). Only the way each tree is chosen has changed.
- The formulas rely on the **second-order approximation** [[eq-taylor2]], which is only accurate for a small step. This is another reason to use a small $\eta$.
- The loss must be twice differentiable with $h_i\ge0$ (a convex loss) for [[eq-wstar]] to be a minimum. Losses with negative curvature need special handling.
- A negative gain means "do not split", but a **later** split below it might have been good. This is the greedy limitation again, and it is the same one seen in [Chapter 7's XOR example](chapter.html?c=07-decision-trees).
:::

## 11. Bridge cheat-sheet

| Question | Answer | Where |
|---|---|---|
| What is minimized each round? | Loss after adding a tree, plus a complexity penalty | [[eq-obj]] |
| How is the loss simplified? | Second-order Taylor expansion at the current prediction | [[eq-taylor2]], [[eq-obj-approx]] |
| What are $g_i$ and $h_i$? | Slope and curvature of the loss per point | [[eq-gh]], [[eq-gh-mse]], [[eq-gh-logistic]] |
| How is a tree described? | Leaf values $w_j$, leaf assignment $q(\mathbf{x})$ | [[eq-tree]] |
| What is the penalty? | $\gamma T+\tfrac12\lambda\sum w_j^2$ | [[eq-omega]] |
| Best leaf value? | $-G_j/(H_j+\lambda)$ | [[eq-wstar]] |
| Score of a tree structure? | $-\tfrac12\sum G_j^2/(H_j+\lambda)+\gamma T$ | [[eq-objstar]] |
| When to split? | If $\text{Gain}>0$ | [[eq-gain]] |
| What does $h_i$ do? | Acts as a sample weight in a weighted least-squares fit | [[eq-weighted-ls]] |
| Final update | $F_t=F_{t-1}+\eta f_t$ | [[eq-update-xgb]] |

## 12. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | LightGBM, CatBoost, histogram-based gradient boosting | They keep the objective of [[eq-obj-leaf]] and the gain of [[eq-gain]], and change how thresholds are found and how categorical features are handled. |
| Machine Learning | Custom losses, ranking (LambdaMART), quantile and Poisson regression | You only supply $g_i$ and $h_i$, as in [[eq-gh]]. Everything else follows from the formulas in this chapter. |
| Machine Learning | Model selection and tuning | The table in Section 10 tells you which knob moves bias and which moves variance, linking to [Chapter 4](chapter.html?c=04-bias-variance). |
| Optimization | Newton and quasi-Newton methods, trust regions, functional optimization | The step $-G/(H+\lambda)$ is a damped Newton step, a close relative of Levenberg-Marquardt damping. |
| Probability & Statistics | Regularization as prior belief, additive models | $\lambda$ in [[eq-omega]] is a Gaussian prior on leaf values, as in [Chapter 3](chapter.html?c=03-ridge-lasso#eq-gprior). |
| Computer Systems | Parallel and distributed training, sparse data handling | The gradient and Hessian statistics $G$ and $H$ are simple sums, so they can be computed in parallel and combined across machines. |
| Applications | Tabular data in finance, insurance, ad click prediction, healthcare risk scoring | Boosted trees of this kind are among the strongest models for data in tables, and are common in competition winning solutions. |
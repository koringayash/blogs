# Logistic Regression

*Part III · Classification. The same recipe as linear regression (likelihood, gradient, iterate), applied to yes/no answers.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Linear Regression (Chapter 2)](chapter.html?c=02-linear-regression) | We reuse its recipe: model, likelihood, gradient, then an update rule. |
| Probability & Statistics | Bernoulli distribution | The label $y\in\{0,1\}$ is a coin flip with an unknown head probability. |
| Probability & Statistics | [Maximum likelihood (Chapter 2)](chapter.html?c=02-linear-regression#eq-loglik) | The loss function will come out of a log-likelihood. |
| Calculus & Optimization | Chain rule, gradient, Hessian | Needed to differentiate the loss and to run Newton's method. |
| Linear Algebra | Positive semi-definite matrices | Used to prove that the loss is a bowl. See the [Chapter 2 version](chapter.html?c=02-linear-regression#eq-hess). |

## 1. Everyday picture: a dimmer switch for "yes"

Imagine an email filter. For each email it adds up weighted evidence: suspicious words push the total up, familiar senders push it down.

1. The total can be any number, from very negative ("clearly fine") to very positive ("clearly spam").
2. We want a **probability** of spam, which must sit between $0$ and $1$.
3. So we pass the total through a **dimmer**: a smooth S-shaped curve that squeezes every real number into the range from $0$ to $1$.
4. Very negative totals turn the dimmer almost fully off ($\approx0$), very positive totals turn it almost fully on ($\approx1$), and a total of $0$ gives exactly $\tfrac12$.

That S-shaped dimmer is the **sigmoid**, and the weighted total is a linear model just like [Chapter 2](chapter.html?c=02-linear-regression#eq-pred). The rest of this chapter answers three questions: why this particular dimmer, what loss to train it with, and how to minimize that loss when there is no formula.

## 2. Why not just use linear regression?

We want to model the probability that the label is $1$, for a label $y\in\{0,1\}$. The tempting idea is to set it equal to the linear prediction:

$$ P(y=1\mid\mathbf{x})=\mathbf{w}^\top\mathbf{x} \qquad\text{(this does not work)} $$ {#eq-bad}

Reasons it fails:

1. **Range.** A probability must lie in $[0,1]$, but $\mathbf{w}^\top\mathbf{x}$ can be any real number, including $-3$ or $7$.
2. **Far-away points distort the line.** A straight line keeps rising, so an extremely confident example (a very long email full of spam words) pulls the whole line towards itself, even though it is already correctly classified.
3. **Wrong noise picture.** The Gaussian noise of [Chapter 2](chapter.html?c=02-linear-regression#eq-noise) allows any real value. A label can only be $0$ or $1$.

:::figure F5.1 | Why not a straight line | A straight line and a sigmoid fitted to 0/1 data. A button adds an outlier far to the right. The line tilts to chase it and starts misclassifying; the sigmoid barely moves.

## 3. The general recipe: generalized linear models

The fix is to keep the **linear** part but change what it predicts. A generalized linear model (GLM) has three ingredients:

1. Choose a probability distribution suited to the target $y$ (here, Bernoulli, because $y$ is $0$ or $1$).
2. Choose a **transformation** $g$ of the expected target.
3. Make that transformed quantity a linear function of the features:

$$ g\big(\mathbb{E}[y\mid\mathbf{x}]\big)=\mathbf{w}^\top\mathbf{x} $$ {#eq-glm}

:::box plain | What is this equation saying in plain words?
- We no longer say "the prediction is the linear score". We say "**the linear score is a transformed version of the prediction**".
- The transformation $g$ is called the **link function**. Linear regression is the special case where $g$ does nothing and $y$ is Gaussian.
- For a Bernoulli target, $\mathbb{E}[y\mid\mathbf{x}]=P(y=1\mid\mathbf{x})$, which we write as $p$.
:::

## 4. From probability to odds to log-odds

We need a $g$ that stretches the interval $(0,1)$ onto the **whole** real line, so that it matches the range of $\mathbf{w}^\top\mathbf{x}$. We get there in two stretches.

| Quantity | Formula | Range | Is it enough? |
|---|---|---|---|
| Probability | $p$ | $(0,1)$ | No: bounded on both sides |
| Odds | $\dfrac{p}{1-p}$ | $(0,\infty)$ | No: fixes the top side only, and can never be negative |
| **Log-odds** | $\log\dfrac{p}{1-p}$ | $(-\infty,\infty)$ | **Yes: any real number is allowed** |

**Step 1: the odds.** Dividing $p$ by $1-p$ stretches the upper end: as $p\to1$ the odds grow without limit. But odds are never negative, so a linear score such as $-2$ would still have no meaning.

**Step 2: the log-odds** (also called the **logit**). A logarithm turns the interval $(0,\infty)$ into the whole real line, with odds $=1$ (a fair coin, $p=\tfrac12$) mapped to $0$.

$$ g(p)=\log\frac{p}{1-p}=\mathbf{w}^\top\mathbf{x} $$ {#eq-logit}

**Step 3: undo the link to get the probability.** Let $z=\mathbf{w}^\top\mathbf{x}$ and solve [[eq-logit]] for $p$:

1. Remove the log: $\dfrac{p}{1-p}=e^{z}$.
2. Multiply out: $p=e^{z}(1-p)=e^{z}-e^{z}p$.
3. Collect the $p$ terms: $p\,(1+e^{z})=e^{z}$.
4. Divide, then divide top and bottom by $e^{z}$:

$$ p=P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}),\qquad \sigma(z)=\frac{1}{1+e^{-z}} $$ {#eq-sigmoid}

This is the S-shaped dimmer from Section 1. Because the link in [[eq-logit]] is linear in $\mathbf{x}$, every extra unit of a feature **adds** $w_j$ to the log-odds, which means it **multiplies the odds** by $e^{w_j}$.

:::tabs Three ways to understand the log-odds
--- Range
Linear scores live on the whole real line. Probabilities live in $(0,1)$. The log-odds is a smooth, one-to-one map from $(0,1)$ onto the real line, so every score corresponds to exactly one probability, and the other way round.
--- Betting
Odds are how gamblers speak: "3 to 1" means $p=0.75$. Taking the log makes odds additive. Each feature adds or subtracts evidence in the same units, which is exactly what a weighted sum $\sum_j w_jx_j$ does.
--- Symmetry
Swapping the two classes turns $p$ into $1-p$ and flips the sign of the log-odds. So $\sigma(-z)=1-\sigma(z)$. Calling the classes "spam" or "not spam" cannot change anything except the sign of $\mathbf{w}$.
:::

**A property we will need later.** The sigmoid has a very neat derivative:

$$ \sigma'(z)=\sigma(z)\big(1-\sigma(z)\big) $$ {#eq-sigderiv}

:::box step | How we got here
- Write $\sigma(z)=(1+e^{-z})^{-1}$ and differentiate with the chain rule: $\sigma'(z)=\dfrac{e^{-z}}{(1+e^{-z})^2}$.
- Split this into two fractions: $\dfrac{1}{1+e^{-z}}\cdot\dfrac{e^{-z}}{1+e^{-z}}$.
- The first factor is $\sigma(z)$. The second is $1-\sigma(z)$, since $1-\dfrac{1}{1+e^{-z}}=\dfrac{e^{-z}}{1+e^{-z}}$.
:::

:::figure F5.2 | Probability, odds, log-odds | A slider for p between 0 and 1 that shows the odds and the log-odds as live numbers. Beside it, the sigmoid curve with the point (z, p) marked, so the stretching of (0,1) onto the whole line is visible.

:::box warn | A correction to a common claim
It is often said that "log-odds is the only link that works". That is too strong. Other links (for example the probit link, which uses the Gaussian cumulative distribution) also map $(0,1)$ onto the real line and are used in practice. The log-odds is special for a different reason: it is the **natural parameter** of the Bernoulli distribution (next section), and this makes the gradient and Hessian particularly clean.
:::

## 5. Where the log-odds really comes from: the Bernoulli distribution

**Step 1: write a Bernoulli probability in one line.** For $y\in\{0,1\}$ and head probability $p$:

$$ P(y;p)=p^{\,y}(1-p)^{1-y} $$ {#eq-bern}

When $y=1$ this equals $p$, and when $y=0$ it equals $1-p$.

**Step 2: rewrite it with an exponential.** Use $a=e^{\log a}$ and expand:

$$ P(y;p)=\exp\!\Big[\,y\log p+(1-y)\log(1-p)\,\Big]=\exp\!\Big[\,y\,\underbrace{\log\frac{p}{1-p}}_{\eta}+\log(1-p)\,\Big] $$ {#eq-expfam}

:::box step | How we got here
- Take the logarithm of [[eq-bern]]: $\log P=y\log p+(1-y)\log(1-p)$, then re-exponentiate.
- Open the bracket $(1-y)\log(1-p)=\log(1-p)-y\log(1-p)$.
- Collect everything that multiplies $y$: $y\big[\log p-\log(1-p)\big]=y\log\dfrac{p}{1-p}$.
:::

**Step 3: read off the pieces.** [[eq-expfam]] has the general shape of the **exponential family**,
$P(y\mid\eta)=h(y)\exp\big(\eta\,T(y)-A(\eta)\big)$, with:

- $T(y)=y$, the **sufficient statistic** (all the data tells us about $p$ is the value of $y$).
- $\eta=\log\dfrac{p}{1-p}$, the **natural parameter**.
- $A(\eta)=-\log(1-p)=\log(1+e^{\eta})$, the log-partition function.

**So for a Bernoulli variable, the natural parameter is the log-odds.** That is why the GLM recipe of [[eq-glm]] makes the log-odds the thing we model linearly: $\eta=\mathbf{w}^\top\mathbf{x}$.

## 6. From likelihood to the cross-entropy loss

Now the loss. We follow the exact path used for [least squares in Chapter 2](chapter.html?c=02-linear-regression#eq-loglik): likelihood, then log, then flip the sign.

**Step 1: one data point.** Let $p_i=\sigma(\mathbf{w}^\top\mathbf{x}_i)$. By [[eq-bern]]:

$$ P(y_i\mid\mathbf{x}_i)=p_i^{\,y_i}(1-p_i)^{1-y_i} $$ {#eq-point}

**Step 2: all $n$ independent points.**

$$ L(\mathbf{w})=\prod_{i=1}^{n}p_i^{\,y_i}(1-p_i)^{1-y_i} $$ {#eq-lik-lr}

**Step 3: take the logarithm.**

$$ \log L(\mathbf{w})=\sum_{i=1}^{n}\Big[\,y_i\log p_i+(1-y_i)\log(1-p_i)\,\Big] $$ {#eq-loglik-lr}

:::box step | How we got here
- Multiplying independent probabilities is the same as in [Chapter 2's likelihood](chapter.html?c=02-linear-regression#eq-lik): the joint probability is a product.
- The logarithm turns the product of [[eq-lik-lr]] into a sum, and brings each exponent $y_i$ and $1-y_i$ down in front.
:::

**Step 4: flip the sign to get something to minimize.**

$$ J(\mathbf{w})=-\log L(\mathbf{w})=-\sum_{i=1}^{n}\Big[\,y_i\log p_i+(1-y_i)\log(1-p_i)\,\Big] $$ {#eq-bce}

This is the **binary cross-entropy (BCE)** loss. Maximizing [[eq-loglik-lr]] and minimizing [[eq-bce]] are the same problem.

:::box plain | What is the loss saying in plain words?
- For a data point with $y_i=1$ only the term $-\log p_i$ survives: **the loss is small when you gave the true class a high probability**, and it blows up as $p_i\to0$.
- For $y_i=0$ only $-\log(1-p_i)$ survives: the loss is small when you gave the *other* class a high probability.
- So the loss does not just ask "was the answer right?" It asks "**how much probability did you put on what actually happened?**", and punishes confident mistakes very hard.
:::

## 7. Gradient, Hessian, and why there is no formula

**Step 1: the gradient for one data point.** Let $z_i=\mathbf{w}^\top\mathbf{x}_i$ and $J_i=-\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]$. By the chain rule,

$$ \frac{\partial J_i}{\partial\mathbf{w}}=\frac{\partial J_i}{\partial p_i}\cdot\frac{\partial p_i}{\partial z_i}\cdot\frac{\partial z_i}{\partial\mathbf{w}} $$ {#eq-chain}

The three pieces are:

1. $\dfrac{\partial J_i}{\partial p_i}=\dfrac{p_i-y_i}{p_i(1-p_i)}$ (differentiate [[eq-bce]]; the two fractions combine over a common denominator).
2. $\dfrac{\partial p_i}{\partial z_i}=p_i(1-p_i)$, by [[eq-sigderiv]].
3. $\dfrac{\partial z_i}{\partial\mathbf{w}}=\mathbf{x}_i$.

The factor $p_i(1-p_i)$ appears once downstairs and once upstairs, so it cancels:

$$ \frac{\partial J_i}{\partial\mathbf{w}}=(p_i-y_i)\,\mathbf{x}_i $$ {#eq-grad-i}

**Step 2: sum over all points.** With $\mathbf{p}=(p_1,\dots,p_n)^\top$ and the data matrix $\mathbf{X}$ from [Chapter 2](chapter.html?c=02-linear-regression#eq-model):

$$ \nabla J(\mathbf{w})=\mathbf{X}^\top(\mathbf{p}-\mathbf{y}) $$ {#eq-grad-lr}

:::box plain | Look how familiar this is
- Chapter 2's gradient was $-\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w})=\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})$, see [its gradient](chapter.html?c=02-linear-regression#eq-grad).
- Here it is the **same shape**: $\mathbf{X}^\top\times(\text{prediction}-\text{truth})$. The only change is that the prediction is $\sigma(\mathbf{w}^\top\mathbf{x})$ instead of $\mathbf{w}^\top\mathbf{x}$.
- This neat cancellation is the payoff of using the log-odds (the canonical link) together with the Bernoulli likelihood.
:::

**Step 3: the Hessian.** Differentiate [[eq-grad-i]] once more with respect to $\mathbf{w}$. Only $p_i$ depends on $\mathbf{w}$, and $\partial p_i/\partial\mathbf{w}=p_i(1-p_i)\mathbf{x}_i$, so each point contributes $p_i(1-p_i)\,\mathbf{x}_i\mathbf{x}_i^\top$. Summing over points:

$$ \mathbf{H}=\nabla^2J(\mathbf{w})=\sum_{i}p_i(1-p_i)\,\mathbf{x}_i\mathbf{x}_i^\top=\mathbf{X}^\top\mathbf{R}\,\mathbf{X},\qquad \mathbf{R}=\operatorname{diag}\big(p_1(1-p_1),\dots,p_n(1-p_n)\big) $$ {#eq-hess-lr}

**Step 4: the loss is convex.** For any vector $\mathbf{v}$, put $\mathbf{u}=\mathbf{X}\mathbf{v}$:

$$ \mathbf{v}^\top\mathbf{H}\mathbf{v}=(\mathbf{X}\mathbf{v})^\top\mathbf{R}(\mathbf{X}\mathbf{v})=\sum_i p_i(1-p_i)\,u_i^2\;\ge\;0 $$ {#eq-convex}

:::box step | How we got here
- Take [[eq-hess-lr]] and sandwich it between $\mathbf{v}^\top$ and $\mathbf{v}$.
- Every $p_i(1-p_i)$ lies between $0$ and $\tfrac14$, so each term is non-negative.
- This is the same argument as for the [Chapter 2 Hessian](chapter.html?c=02-linear-regression#eq-hess): a positive semi-definite Hessian means a bowl-shaped loss with no bad local minima. So any point where the gradient is zero is a **global minimum**.
:::

**Step 5: why we still cannot write a formula.** Setting [[eq-grad-lr]] to zero gives

$$ \mathbf{X}^\top\big(\sigma(\mathbf{X}\mathbf{w})-\mathbf{y}\big)=\mathbf{0} $$ {#eq-nofix}

The unknown $\mathbf{w}$ sits **inside the sigmoid**, so we cannot isolate it the way we did in [Chapter 2's normal equation](chapter.html?c=02-linear-regression#eq-normal-lr). There is no closed-form solution, but the loss being convex tells us that an iterative method will find the answer.

:::figure F5.3 | Cross-entropy loss surface | A 3D or contour view of the loss over two weights. It is a single smooth bowl with no other dips, and a draggable starting point shows the gradient descent path rolling to the bottom.

:::box warn | When the minimum does not exist
- If the two classes can be **perfectly separated** by a line, the loss keeps decreasing as the weights grow larger, because the sigmoid can get ever closer to $0$ and $1$. The minimizer is at infinity.
- In practice the optimizer just stops at some large weights, which means over-confident probabilities.
- The remedy is the same penalty idea as in [Chapter 3](chapter.html?c=03-ridge-lasso#eq-ridge): add $\lambda\lVert\mathbf{w}\rVert^2$ (or an $\ell_1$ penalty) to [[eq-bce]]. This is MAP estimation with the same Gaussian or Laplace priors.
:::

## 8. Iterative solutions

### Method 1: gradient descent

We use the same rule as [Chapter 2](chapter.html?c=02-linear-regression#eq-gd), now with the gradient [[eq-grad-lr]]:

$$ \mathbf{w}_{t+1}=\mathbf{w}_t-\eta\,\mathbf{X}^\top(\mathbf{p}_t-\mathbf{y}) $$ {#eq-gd-lr}

Because the loss is convex, a small enough step size $\eta$ is guaranteed to reach the global minimum. It uses only the slope, so many small steps may be needed.

### Method 2: Newton's method

Gradient descent knows only which way is downhill. Newton's method also knows **how the slope is changing** (the curvature), so it can take a smarter step.

**Step 1: one weight first.** Approximate the loss near the current point $w_t$ by a parabola (a second-order Taylor expansion):

$$ f(w)\approx f(w_t)+f'(w_t)(w-w_t)+\tfrac12f''(w_t)(w-w_t)^2 $$ {#eq-taylor}

**Step 2: jump to the bottom of the parabola.** The minimum of a parabola is where its derivative is zero:

$$ f'(w_t)+f''(w_t)(w-w_t)=0\;\Longrightarrow\;w_{t+1}=w_t-\frac{f'(w_t)}{f''(w_t)} $$ {#eq-newton-1d}

:::box step | How we got here
- Differentiate [[eq-taylor]] with respect to $w$: the constant disappears, the linear term gives $f'(w_t)$, and the squared term gives $f''(w_t)(w-w_t)$.
- Set that to zero and solve for $w$.
- **Plain meaning:** the slope divided by the curvature. Where the loss is steep but only slightly curved, take a long step. Where it is highly curved, take a short step.
:::

**Step 3: many weights.** Replace the slope by the gradient $\nabla J$ and the curvature by the Hessian $\mathbf{H}$. Division becomes multiplication by the inverse matrix:

$$ \mathbf{w}_{t+1}=\mathbf{w}_t-\mathbf{H}^{-1}\nabla J=\mathbf{w}_t-(\mathbf{X}^\top\mathbf{R}\mathbf{X})^{-1}\mathbf{X}^\top(\mathbf{p}-\mathbf{y}) $$ {#eq-newton}

The second form substitutes [[eq-hess-lr]] and [[eq-grad-lr]], where $\mathbf{R}$ and $\mathbf{p}$ are evaluated at $\mathbf{w}_t$.

### Method 3: IRLS (iteratively reweighted least squares)

[[eq-newton]] can be rewritten as a *weighted* version of the least-squares formula from [Chapter 2](chapter.html?c=02-linear-regression#eq-ols).

**Step 1: put everything over the same matrix.** Write $\mathbf{w}_t=(\mathbf{X}^\top\mathbf{R}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{R}\mathbf{X}\,\mathbf{w}_t$ and combine with the second term of [[eq-newton]]:

$$ \mathbf{w}_{t+1}=(\mathbf{X}^\top\mathbf{R}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{R}\,\mathbf{z},\qquad \mathbf{z}=\mathbf{X}\mathbf{w}_t+\mathbf{R}^{-1}(\mathbf{y}-\mathbf{p}) $$ {#eq-irls}

:::box step | How we got here
- Factor out $(\mathbf{X}^\top\mathbf{R}\mathbf{X})^{-1}\mathbf{X}^\top$ from both terms of [[eq-newton]]. What is left inside is $\mathbf{R}\mathbf{X}\mathbf{w}_t-(\mathbf{p}-\mathbf{y})$.
- Pull out one $\mathbf{R}$: that is $\mathbf{R}\big[\mathbf{X}\mathbf{w}_t+\mathbf{R}^{-1}(\mathbf{y}-\mathbf{p})\big]=\mathbf{R}\mathbf{z}$.
- This needs $\mathbf{R}^{-1}$, which exists as long as no $p_i$ is exactly $0$ or $1$.
:::

:::box plain | Why is this called "iteratively reweighted least squares"?
- [[eq-irls]] has the same shape as weighted least squares: $(\mathbf{X}^\top\mathbf{R}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{R}\mathbf{z}$.
- $\mathbf{z}$ is a "working target": the current score plus the scaled error.
- $\mathbf{R}$ holds the weights. Points with $p_i$ close to $0$ or $1$ (already confident) get small weight $p_i(1-p_i)$. Points near $p_i=\tfrac12$ (uncertain) get the most weight.
- Each iteration solves one weighted least-squares problem, then recomputes $\mathbf{p}$, $\mathbf{R}$ and $\mathbf{z}$.
:::

| | Gradient descent | Newton / IRLS |
|---|---|---|
| Uses | slope only | slope and curvature |
| Cost per step | cheap, $O(np)$ | expensive: forms and solves with a $p\times p$ matrix |
| Number of steps | many | very few (it converges quickly near the answer) |
| Best for | very many features or data points | moderate number of features |

:::figure F5.4 | Gradient descent vs Newton | The same logistic loss contours with two paths drawn: gradient descent taking many small steps and Newton arriving in a handful. An iteration counter and a step-size slider for gradient descent are included.

## 9. Geometry: the decision boundary is a hyperplane

To classify, we pick class $1$ whenever $p\ge\tfrac12$.

$$ \sigma(\mathbf{w}^\top\mathbf{x})\ge\tfrac12\;\Longleftrightarrow\;\mathbf{w}^\top\mathbf{x}\ge0 $$ {#eq-boundary}

:::box step | How we got here
- $\sigma(z)=\tfrac12$ exactly when $e^{-z}=1$, which means $z=0$, and the sigmoid is increasing.
- So the dividing line between the two decisions is the set of points with $\mathbf{w}^\top\mathbf{x}=0$. This is a **hyperplane** (a line in 2D, a plane in 3D), and $\mathbf{w}$ is perpendicular to it.
:::

**What this tells us:**
- Logistic regression, despite its curved sigmoid, produces a **linear** decision boundary in the feature space.
- The sigmoid does not bend the boundary. It only turns the signed distance from the boundary into a **confidence**: the further from the line, the closer $p$ is to $0$ or $1$.
- The distance from a point to this plane uses the same perpendicular-drop idea as [Chapter 1](chapter.html?c=01-orthogonal-projections#eq-p).

:::figure F5.5 | Decision boundary | Two classes of draggable points. The fitted line w^T x = 0 and shaded probability bands (10%, 50%, 90%) update live as points move. A toggle shows how a strong penalty makes the bands wider.

**The link to SVM.** The [Support Vector Machine chapter](chapter.html?c=06-svm) produces a boundary of exactly the same algebraic form $\mathbf{w}^\top\mathbf{x}+b=0$. Both models draw a straight boundary. They differ in **which loss they minimize to choose $\mathbf{w}$**:

| | Logistic regression | SVM |
|---|---|---|
| Boundary | $\mathbf{w}^\top\mathbf{x}=0$, linear | $\mathbf{w}^\top\mathbf{x}+b=0$, linear |
| Labels used | $y\in\{0,1\}$ | $y\in\{-1,+1\}$ |
| Loss | Cross-entropy, [[eq-bce]] | Hinge loss |
| Output | A probability | A signed distance (a class) |
| Which points matter | All points, weighted by how uncertain they are | Only the points close to the boundary (support vectors) |

## 10. Every loss function is a negative log-likelihood

The path of this chapter was: assume a distribution for the target, write its likelihood, take the log, flip the sign. We have now done it twice, and the pattern is general:

| Assumed distribution of the target | Loss obtained | Where |
|---|---|---|
| Gaussian | Squared error | [Chapter 2](chapter.html?c=02-linear-regression#eq-loglik) |
| Laplace | Absolute error | [Chapter 2](chapter.html?c=02-linear-regression#eq-loglik) |
| Bernoulli | Binary cross-entropy | [[eq-bce]] |
| Categorical (several classes) | Softmax cross-entropy | extends [[eq-bce]] |
| Gaussian target plus a Gaussian prior on $\mathbf{w}$ | Squared error plus an $\ell_2$ penalty | [Chapter 3](chapter.html?c=03-ridge-lasso#eq-ridge) |

So choosing a loss function is the same as choosing an assumption about how the data was generated.

## 11. Bridge cheat-sheet: linear versus logistic regression

| | Linear regression (Chapter 2) | Logistic regression |
|---|---|---|
| Target | Real number | Label $0$ or $1$ |
| Distribution | Gaussian | Bernoulli |
| Link | none | log-odds, [[eq-logit]] |
| Prediction | $\mathbf{w}^\top\mathbf{x}$ | $\sigma(\mathbf{w}^\top\mathbf{x})$, [[eq-sigmoid]] |
| Loss | squared error | cross-entropy, [[eq-bce]] |
| Gradient | $\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})$ | $\mathbf{X}^\top(\mathbf{p}-\mathbf{y})$, [[eq-grad-lr]] |
| Hessian | $2\mathbf{X}^\top\mathbf{X}$ | $\mathbf{X}^\top\mathbf{R}\mathbf{X}$, [[eq-hess-lr]] |
| Convex? | yes | yes, [[eq-convex]] |
| Closed form? | yes | no, [[eq-nofix]] |
| How to solve | normal equation | Newton / IRLS, [[eq-irls]] |

## 12. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [Support Vector Machines](chapter.html?c=06-svm) | Same linear boundary, different loss. Comparing cross-entropy with the hinge loss is the key to the next chapter. |
| Machine Learning | [Gradient Boosting](chapter.html?c=09-gradient-boosting) | Its classification version fits trees to the negative gradient of the logistic loss, which is $y-p$ up to scale. |
| Machine Learning | [XGBoost](chapter.html?c=11-xgboost) | It uses the same $p_i-y_i$ gradient and the $p_i(1-p_i)$ curvature from [[eq-hess-lr]], with Newton's idea of [[eq-newton]]. |
| Deep Learning | Neural network classifiers, softmax | The last layer of a classifier is logistic (or softmax) regression on learned features, trained with cross-entropy. |
| Probability & Statistics | Generalized linear models (Poisson, binomial), odds ratios | [[eq-glm]] is the general recipe. $e^{w_j}$ is an odds ratio, widely reported in medicine and social science. |
| Optimization | Newton and quasi-Newton methods (BFGS), IRLS | [[eq-newton]] and [[eq-irls]] are the classic examples. |
| Evaluation | ROC curves, calibration, class imbalance | The output is a probability, so you can move the $0.5$ cut-off and study the trade-offs. |
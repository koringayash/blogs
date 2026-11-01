# MLE vs MAP, Ridge & Lasso

*Part II · Regression. Regularization is not a trick: it is a prior belief written as a penalty.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Linear Regression (Chapter 2)](chapter.html?c=02-linear-regression) | We start from its loss, its normal equation, and its Gauss-Markov result. |
| Linear Algebra | [Orthogonal projections (Chapter 1)](chapter.html?c=01-orthogonal-projections) | The "residual is perpendicular" fact gives a Pythagoras identity used for Lasso. |
| Probability & Statistics | Bayes' rule, prior, posterior | MAP estimation is Bayes' rule applied to the weights. |
| Probability & Statistics | Gaussian and Laplace distributions | They are the two priors behind Ridge and Lasso. |
| Probability & Statistics | Bias and variance of an estimator | We compute both for Ridge and compare with OLS. |
| Calculus & Optimization | Gradients, absolute value near $0$ | Ridge is smooth; Lasso has a kink at $0$. |

## 1. Everyday picture: data against a rubber band

Imagine you pin a weight on each knob of the regression model, and attach a **rubber band** from every knob back to the value $0$.

1. The data pulls each knob towards the value that fits the training points best (the OLS weights).
2. The rubber band pulls each knob back towards $0$, like a modest guess made *before* seeing any data.
3. The knob settles where the two pulls balance.
4. A stiffer band means weights stay smaller. A weaker band means we trust the data more.

Both Ridge and Lasso are this tug-of-war. They differ only in the *shape* of the rubber band.

## 2. Two ways to pick a weight: MLE and MAP

**Step 1: maximum likelihood (MLE).**

$$ \hat{\mathbf{w}}_{\text{MLE}}=\arg\max_{\mathbf{w}}\; P(\mathbf{y}\mid\mathbf{X},\mathbf{w}) $$ {#eq-mle}

Pick the weights that make the observed data **most likely**. This is what we did in [Chapter 2](chapter.html?c=02-linear-regression#eq-lik), and it gave least squares.

**Step 2: Bayes' rule for the weights.**

$$ P(\mathbf{w}\mid\mathbf{X},\mathbf{y}) = \frac{P(\mathbf{y}\mid\mathbf{X},\mathbf{w})\,P(\mathbf{w})}{P(\mathbf{y}\mid\mathbf{X})}\;\propto\;P(\mathbf{y}\mid\mathbf{X},\mathbf{w})\,P(\mathbf{w}) $$ {#eq-bayes}

:::box step | How we got here
- $P(\mathbf{w})$ is the **prior**: what we believe about the weights *before* seeing data.
- $P(\mathbf{y}\mid\mathbf{X},\mathbf{w})$ is the **likelihood**, the same quantity MLE maximizes.
- $P(\mathbf{w}\mid\mathbf{X},\mathbf{y})$ is the **posterior**: our belief *after* seeing data.
- The denominator does not contain $\mathbf{w}$, so it cannot change which $\mathbf{w}$ is best. That is why we may write "$\propto$" (proportional to).
:::

**Step 3: maximum a posteriori (MAP).**

$$ \hat{\mathbf{w}}_{\text{MAP}}=\arg\max_{\mathbf{w}}\; P(\mathbf{y}\mid\mathbf{X},\mathbf{w})\,P(\mathbf{w}) $$ {#eq-map}

Pick the weights that are **most probable after combining the data with the prior**. [[eq-map]] is [[eq-mle]] with one extra factor, the prior.

| | MLE | MAP |
|---|---|---|
| Question asked | Which $\mathbf{w}$ makes the data most likely? | Which $\mathbf{w}$ is most probable given data **and** prior belief? |
| Uses a prior? | No | Yes |
| With a flat prior | Same answer as MAP | Same answer as MLE |
| Result for regression | Least squares | A *penalized* least squares |

:::figure F3.1 | MLE vs MAP | A likelihood curve and a prior curve for one weight, with their product (the posterior) drawn below. A slider changes the prior width, and the MAP peak slides between the MLE peak and zero.

## 3. Ridge regression from a Gaussian prior

**Step 1: choose the prior.** Before seeing any data we believe each weight is probably small and centered on $0$:

$$ \mathbf{w}\sim\mathcal{N}(\mathbf{0},\,\tau^2\mathbf{I})\quad\Longrightarrow\quad P(\mathbf{w})\propto\exp\!\left(-\frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2\right) $$ {#eq-gprior}

A small $\tau$ means a strong belief that weights are tiny.

**Step 2: multiply likelihood and prior.**

$$ P(\mathbf{w}\mid\mathbf{X},\mathbf{y}) \;\propto\; \exp\!\left(-\frac{1}{2\sigma^2}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2-\frac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2\right) $$ {#eq-post}

:::box step | How we got here
- The Gaussian likelihood from [Chapter 2](chapter.html?c=02-linear-regression#eq-lik) is proportional to $\exp\!\big(-\tfrac{1}{2\sigma^2}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2\big)$.
- Multiply it by [[eq-gprior]], as [[eq-bayes]] says.
- Two exponentials multiply by adding their exponents, which gives [[eq-post]].
:::

**Step 3: take the logarithm and flip the sign.**

$$ \max_{\mathbf{w}}\log P(\mathbf{w}\mid\mathbf{X},\mathbf{y}) \;\Longleftrightarrow\; \min_{\mathbf{w}}\;\; \lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2+\frac{\sigma^2}{\tau^2}\lVert\mathbf{w}\rVert^2 $$ {#eq-logpost}

:::box step | How we got here
- The log cancels the $\exp$ in [[eq-post]], leaving $-\tfrac{1}{2\sigma^2}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2-\tfrac{1}{2\tau^2}\lVert\mathbf{w}\rVert^2$ plus a constant that does not involve $\mathbf{w}$.
- Maximizing a negative quantity is the same as minimizing its positive version.
- Multiply everything by $2\sigma^2$. That does not change the best $\mathbf{w}$, and it turns the second term's coefficient into $\sigma^2/\tau^2$.
:::

**Step 4: name the ratio.** With $\lambda=\sigma^2/\tau^2$ we get the **Ridge objective**:

$$ J_{\text{ridge}}(\mathbf{w}) = \underbrace{\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2}_{\text{fit the data}} + \lambda\underbrace{\lVert\mathbf{w}\rVert^2}_{\text{keep weights small}} $$ {#eq-ridge}

:::box plain | What is this objective saying in plain words?
- **First term:** the same squared error as [Chapter 2](chapter.html?c=02-linear-regression#eq-loss). It is the data pulling.
- **Second term:** the rubber band. It grows with the squared size of the weights.
- **$\lambda=\sigma^2/\tau^2$:** a *noisy* data set (large $\sigma$) or a *confident* prior (small $\tau$) makes $\lambda$ large, so the band wins.
- **$\lambda=0$:** no band, so we are back to ordinary least squares.
:::

**Step 5: solve it.**

$$ \frac{\partial J_{\text{ridge}}}{\partial\mathbf{w}} = -2\,\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w})+2\lambda\mathbf{w}=\mathbf{0}\;\Longrightarrow\;(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})\,\mathbf{w}=\mathbf{X}^\top\mathbf{y} $$ {#eq-ridge-normal}

$$ \hat{\mathbf{w}}_{\text{ridge}}=(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y} $$ {#eq-ridge-sol}

:::box step | How we got here
- The first part of the gradient is the OLS gradient from [Chapter 2](chapter.html?c=02-linear-regression#eq-grad). The penalty $\lambda\lVert\mathbf{w}\rVert^2$ adds $2\lambda\mathbf{w}$.
- Collect the $\mathbf{w}$ terms: $\mathbf{X}^\top\mathbf{X}\mathbf{w}+\lambda\mathbf{w}=(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})\mathbf{w}$. This gives [[eq-ridge-normal]].
- Multiply on the left by the inverse to get [[eq-ridge-sol]]. Setting $\lambda=0$ recovers [Chapter 2's solution](chapter.html?c=02-linear-regression#eq-ols).
:::

**A bonus of [[eq-ridge-sol]]:**
- OLS needs the columns of $\mathbf{X}$ to be independent so that $\mathbf{X}^\top\mathbf{X}$ can be inverted.
- For any $\lambda>0$, every eigenvalue of $\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}$ is at least $\lambda$, so it is **always invertible**.
- Ridge therefore also works when features are duplicated or nearly duplicated.

## 4. Lasso from a Laplace prior

**Step 1: change only the prior.** Replace the Gaussian bell by a Laplace curve, which has a sharp peak at $0$ and fatter tails:

$$ w_j\sim\text{Laplace}(0,b)\quad\Longrightarrow\quad P(\mathbf{w})\propto\exp\!\left(-\frac{1}{b}\sum_{j}\lvert w_j\rvert\right) $$ {#eq-lprior}

**Step 2: repeat Steps 2 and 3 of the previous section.**

$$ J_{\text{lasso}}(\mathbf{w}) = \lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2+\lambda\sum_{j}\lvert w_j\rvert,\qquad \lambda=\frac{2\sigma^2}{b} $$ {#eq-lasso}

:::box step | How we got here
- Multiply the Gaussian likelihood by [[eq-lprior]] instead of [[eq-gprior]].
- Take the log and flip the sign. The data term stays $\tfrac{1}{2\sigma^2}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2$, and the prior term becomes $\tfrac{1}{b}\sum_j\lvert w_j\rvert$.
- Multiply by $2\sigma^2$: the data term is now $\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2$ and the penalty coefficient is $2\sigma^2/b$.
:::

**What we have learned so far:**
- Ridge and Lasso are the **same idea**: MAP estimation with a shrinkage prior.
- They differ only in which prior is assumed: Gaussian (squared penalty, [[eq-ridge]]) or Laplace (absolute-value penalty, [[eq-lasso]]).
- Ridge is smooth, so it has the closed form [[eq-ridge-sol]]. Lasso has a kink at $w_j=0$, so there is **no closed form** in general and we use iterative solvers.

| | Ridge ($\ell_2$) | Lasso ($\ell_1$) |
|---|---|---|
| Prior | Gaussian | Laplace |
| Penalty | $\lambda\sum_j w_j^2$ | $\lambda\sum_j\lvert w_j\rvert$ |
| Closed form | Yes, [[eq-ridge-sol]] | No |
| Exact zeros | No | Yes (Section 6) |

## 5. Bias and variance of Ridge

[Chapter 2](chapter.html?c=02-linear-regression#eq-unbiased) showed OLS has zero bias and the smallest variance among unbiased linear estimators. Ridge is **biased on purpose**. Here is how much, and what it buys.

**Step 1: write the estimator in terms of the noise.**

$$ \hat{\mathbf{w}}_{\text{ridge}}=(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}\,\mathbf{w}+(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\boldsymbol\epsilon $$ {#eq-ridge-split}

Substitute $\mathbf{y}=\mathbf{X}\mathbf{w}+\boldsymbol\epsilon$ from [Chapter 2](chapter.html?c=02-linear-regression#eq-model) into [[eq-ridge-sol]]. The first term is fixed (given $\mathbf{X}$) and the second is random.

**Step 2: the average estimate.**

$$ \mathbb{E}[\hat{\mathbf{w}}_{\text{ridge}}\mid\mathbf{X}]=(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}\,\mathbf{w} $$ {#eq-ridge-mean}

The noise term averages to $\mathbf{0}$ because $\mathbb{E}[\boldsymbol\epsilon]=\mathbf{0}$.

**Step 3: the bias.** Bias means expected estimate minus the truth:

$$ \operatorname{Bias}(\hat{\mathbf{w}}_{\text{ridge}})=\mathbb{E}[\hat{\mathbf{w}}_{\text{ridge}}]-\mathbf{w}=-\lambda\,(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{w} $$ {#eq-ridge-bias}

:::box step | How we got here
- Subtract $\mathbf{w}$ from [[eq-ridge-mean]] and write $\mathbf{w}=(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})\,\mathbf{w}$.
- Factor out $(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}$ and $\mathbf{w}$. What is left inside the bracket is $\mathbf{X}^\top\mathbf{X}-(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})=-\lambda\mathbf{I}$.
- **Your question from the notes, answered:** with $\lambda=0$ the factor $\lambda$ in [[eq-ridge-bias]] kills the whole expression, so the bias is exactly $0$ and Ridge becomes OLS.
:::

**Step 4: the variance.** Only the noise term of [[eq-ridge-split]] is random. Call $\mathbf{M}=(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top$:

$$ \operatorname{Var}(\hat{\mathbf{w}}_{\text{ridge}}\mid\mathbf{X})=\sigma^2\,\mathbf{M}\mathbf{M}^\top=\sigma^2(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\,\mathbf{X}^\top\mathbf{X}\,(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1} $$ {#eq-ridge-var}

:::box step | How we got here
- For a linear map, $\operatorname{Var}(\mathbf{M}\boldsymbol\epsilon)=\mathbf{M}\operatorname{Var}(\boldsymbol\epsilon)\mathbf{M}^\top$, and $\operatorname{Var}(\boldsymbol\epsilon)=\sigma^2\mathbf{I}$ by [Chapter 2](chapter.html?c=02-linear-regression#eq-noise). This is the same step as in the Gauss-Markov proof.
- Multiply out $\mathbf{M}\mathbf{M}^\top$. The inverse matrix is symmetric, so its transpose is itself, and the middle becomes $\mathbf{X}^\top\mathbf{X}$.
- Setting $\lambda=0$ gives $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$, the OLS variance from [Chapter 2](chapter.html?c=02-linear-regression#eq-var).
:::

**A one-feature example to see the trade.** Take a single feature with $\mathbf{X}^\top\mathbf{X}=d>0$. Then [[eq-ridge-bias]] and [[eq-ridge-var]] become:

$$ \operatorname{Bias}=-\frac{\lambda\,w}{d+\lambda},\qquad \operatorname{Var}=\frac{\sigma^2\,d}{(d+\lambda)^2} $$ {#eq-1d}

- As $\lambda$ grows, the bias magnitude $\tfrac{\lambda\lvert w\rvert}{d+\lambda}$ **rises** from $0$ towards $\lvert w\rvert$.
- As $\lambda$ grows, the variance $\tfrac{\sigma^2 d}{(d+\lambda)^2}$ **falls** from $\sigma^2/d$ towards $0$.
- For several features the same thing happens along each direction of the data. [Chapter 4](chapter.html?c=04-bias-variance) proves it with eigenvalues.

:::box plain | What is the trade-off saying in plain words?
- Ridge **pulls every estimate a little towards zero**, which is a systematic error (bias).
- In return, the estimate becomes **much steadier** from one training set to the next (less variance).
- Total error is bias squared plus variance (plus noise). If the variance saved is bigger than the bias squared added, Ridge **beats OLS**. A classical result says such a $\lambda>0$ always exists.
- This does not contradict Gauss-Markov: that theorem only compares *unbiased* estimators, and Ridge is not one.
:::

:::figure F3.3 | Coefficient paths | A lambda slider shrinks all weights. Ridge weights shrink smoothly towards zero without reaching it, while Lasso weights hit exactly zero one by one. A second panel shows the variance falling and the bias rising as lambda grows.

## 6. Why Lasso gives zeros and Ridge does not

### Picture 1: the constraint view

Both penalties can be written as "minimize the error, **but keep the weights inside a budget** $t$":

$$ \text{Ridge: }\min_{\mathbf{w}}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2\ \text{ s.t. }\ \lVert\mathbf{w}\rVert_2^2\le t \qquad\qquad \text{Lasso: }\min_{\mathbf{w}}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2\ \text{ s.t. }\ \lVert\mathbf{w}\rVert_1\le t $$ {#eq-constraint}

Take two weights $(w_1,w_2)$:

1. The error $\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2$ has **elliptical contour lines** around the OLS solution.
2. The Ridge budget region is a **disc**. The Lasso budget region is a **diamond** with sharp corners on the axes.
3. The solution is where the growing ellipse first **touches** the region.
4. A disc is smooth, so the touching point is almost never exactly on an axis. Both weights stay non-zero, only smaller.
5. A diamond has corners on the axes, and an ellipse very often meets a corner first. At a corner one weight is **exactly zero**.
6. In many dimensions the diamond has many corners and edges, so many weights become exactly zero. The model is **sparse**.

:::figure F3.2 | Ridge vs Lasso constraint | Elliptical loss contours around a draggable OLS solution, with a disc for Ridge and a diamond for Lasso. A toggle switches between them, and the touching point is marked. Drag the OLS point to see Lasso land on an axis again and again.

### Picture 2: a formula that shows it exactly

To see it in one line, take the simplest design: **orthonormal columns**, $\mathbf{X}^\top\mathbf{X}=\mathbf{I}$. That is the nice basis from [Chapter 1](chapter.html?c=01-orthogonal-projections#eq-onb).

**Step 1: split the error using Pythagoras.**

$$ \lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2=\lVert\mathbf{y}-\mathbf{X}\hat{\mathbf{w}}\rVert^2+(\mathbf{w}-\hat{\mathbf{w}})^\top\mathbf{X}^\top\mathbf{X}\,(\mathbf{w}-\hat{\mathbf{w}}) $$ {#eq-pyth}

:::box step | How we got here
- Write $\mathbf{y}-\mathbf{X}\mathbf{w}=(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}})+\mathbf{X}(\hat{\mathbf{w}}-\mathbf{w})$, where $\hat{\mathbf{w}}$ is the OLS solution.
- The first piece is the OLS residual, which is **perpendicular to every column of $\mathbf{X}$** by [Chapter 2](chapter.html?c=02-linear-regression#eq-resid-orth). So the two pieces are perpendicular, and the squared length of the sum is the sum of the squared lengths (Pythagoras).
- The squared length of the second piece is $(\hat{\mathbf{w}}-\mathbf{w})^\top\mathbf{X}^\top\mathbf{X}(\hat{\mathbf{w}}-\mathbf{w})$.
:::

**Step 2: simplify using $\mathbf{X}^\top\mathbf{X}=\mathbf{I}$.** The first term of [[eq-pyth]] does not contain $\mathbf{w}$, so only $\sum_j(w_j-\hat w_j)^2$ matters. Each weight can then be optimized **on its own**:

$$ \text{Ridge: }\min_{w_j}\ (w_j-\hat w_j)^2+\lambda w_j^2 \qquad\qquad \text{Lasso: }\min_{w_j}\ (w_j-\hat w_j)^2+\lambda\lvert w_j\rvert $$ {#eq-per-weight}

**Step 3: solve each one.**

$$ \text{Ridge: }\ w_j=\frac{\hat w_j}{1+\lambda} \qquad\qquad \text{Lasso: }\ w_j=\operatorname{sign}(\hat w_j)\,\max\!\Big(0,\ \lvert\hat w_j\rvert-\frac{\lambda}{2}\Big) $$ {#eq-shrink}

:::box step | How we got here
- **Ridge:** set the derivative of the Ridge part of [[eq-per-weight]] to zero: $2(w_j-\hat w_j)+2\lambda w_j=0$, so $w_j=\hat w_j/(1+\lambda)$.
- **Lasso, away from zero:** for $w_j\ne0$ the derivative is $2(w_j-\hat w_j)+\lambda\operatorname{sign}(w_j)=0$, so $w_j=\hat w_j-\tfrac{\lambda}{2}\operatorname{sign}(w_j)$. This is only consistent when $\lvert\hat w_j\rvert>\lambda/2$.
- **Lasso, when $\lvert\hat w_j\rvert\le\lambda/2$:** the pull towards zero from the kink is stronger than the pull from the data, so the best value is exactly $w_j=0$.
:::

:::box plain | What is this saying in plain words?
- **Ridge** multiplies every OLS weight by the same factor $\tfrac{1}{1+\lambda}$. Everything shrinks proportionally, and nothing becomes exactly zero (unless it was zero already).
- **Lasso** subtracts a fixed amount $\lambda/2$ from every weight's size. Weights smaller than that amount are **cut to exactly zero**. Bigger weights survive, shifted down.
- This is the "kink at zero" in action: the absolute value has a sharp point at $0$ that can hold a weight there, while the square has a flat bottom that cannot.
:::

:::figure F3.4 | Shrink vs threshold | A one-dimensional plot with the OLS weight on the horizontal axis and the penalized weight on the vertical axis. Ridge is a straight line with a smaller slope. Lasso is a line with a flat segment at zero (the dead zone) that then rises with the same slope as OLS. A lambda slider widens the dead zone.

:::box warn | Two caveats
- [[eq-shrink]] needs orthonormal columns. With correlated features Lasso still gives zeros, but the simple formula is no longer exact, and Lasso may pick one of several correlated features somewhat arbitrarily.
- Lasso zeroing a weight means "this feature is not needed to fit the training data well". It is a convenient built-in feature selection, not a proof that the feature is unrelated to the target.
:::

## 7. Bridge cheat-sheet: OLS, Ridge and Lasso side by side

| | OLS (Chapter 2) | Ridge | Lasso |
|---|---|---|---|
| Objective | [squared error](chapter.html?c=02-linear-regression#eq-loss) | [[eq-ridge]] | [[eq-lasso]] |
| Probabilistic view | MLE | MAP with Gaussian prior | MAP with Laplace prior |
| Solution | [closed form](chapter.html?c=02-linear-regression#eq-ols) | [[eq-ridge-sol]] | iterative, no closed form |
| Bias | zero | [[eq-ridge-bias]] | non-zero |
| Variance | [Chapter 2](chapter.html?c=02-linear-regression#eq-var) | [[eq-ridge-var]] (smaller) | smaller |
| Orthonormal case | $\hat w_j$ | $\hat w_j/(1+\lambda)$ | soft-threshold, [[eq-shrink]] |
| Exact zeros | no | no | yes |
| Needs $\mathbf{X}^\top\mathbf{X}$ invertible | yes | no | no |

## 8. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [Bias-variance trade-off (Chapter 4)](chapter.html?c=04-bias-variance) | Ridge is the cleanest worked example of paying bias to reduce variance. The next chapter generalizes it. |
| Machine Learning | SVMs and boosting | The soft-margin SVM and XGBoost both add a penalty to the objective, just like [[eq-ridge]]. |
| Machine Learning | Logistic regression with $\ell_1$ or $\ell_2$ | The same priors can be put on logistic weights. |
| Probability & Statistics | Bayesian inference, empirical Bayes | MAP is the first step. Full Bayesian regression keeps the whole posterior of [[eq-post]]. |
| Optimization | Coordinate descent, proximal gradient (ISTA) | These are the algorithms that solve Lasso. The soft-threshold in [[eq-shrink]] is their core step. |
| Deep Learning | Weight decay, sparsity, dropout | Weight decay is Ridge's penalty on network weights. |
| Signal Processing | Compressed sensing, sparse coding | They use $\ell_1$ penalties because of the corner geometry in Section 6. |
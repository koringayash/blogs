# Linear Regression

*Part II · Regression. The first place where the geometry of Chapter 1 meets real data.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Linear Algebra | [Orthogonal projections (Chapter 1)](chapter.html?c=01-orthogonal-projections) | The best fit turns out to be a projection onto the column space of $\mathbf{X}$. |
| Linear Algebra | Matrix transpose, inverse, positive semi-definite matrices | Needed for the normal equations and for the minimum test. |
| Calculus & Optimization | Gradient and Hessian of a function of a vector | Used to find the weights that minimize the error. |
| Probability & Statistics | Expectation, variance, covariance matrix | Used to show that the estimator is unbiased and to compare variances. |
| Probability & Statistics | Gaussian distribution, likelihood | Used to show that least squares is maximum likelihood. |
| Machine Learning | None required | This is the starting point of the regression block. |

## 1. Everyday picture: turning knobs to fit a cloud of points

Imagine a scatter plot where each dot is one flat: size on the horizontal axis, rent on the vertical axis.

1. You want one straight line that "goes through the middle" of the cloud, so you can predict the rent of a flat you have not seen.
2. The line has adjustable **knobs**: its slope and its starting height. With more information (rooms, floor, age) you get more knobs.
3. For every dot, measure the **vertical miss**: how far the dot is above or below the line.
4. Turn the knobs until the total of these misses is as small as possible.

Everything below is this picture written in symbols: the knobs are the weights $\mathbf{w}$, and "total miss" is the loss.

:::figure F2.1 | Fit as projection | 3D view with the vector y, the plane spanned by the columns of X, the fitted vector (the shadow of y on that plane) and the residual. Drag y and watch the residual stay perpendicular to the plane.

## 2. The model

**Step 1: write the model for all data points at once.**

$$ \mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol\epsilon $$ {#eq-model}

- $\mathbf{y}\in\mathbb{R}^{n}$: the $n$ observed targets (the rents).
- $\mathbf{X}\in\mathbb{R}^{n\times p}$: one row per data point, one column per feature (a column of ones gives the starting height).
- $\mathbf{w}\in\mathbb{R}^{p}$: the unknown weights, the knobs.
- $\boldsymbol\epsilon\in\mathbb{R}^{n}$: noise, everything the features cannot explain.

**Step 2: say what we assume about the noise.**

$$ \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0},\,\sigma^2\mathbf{I}) $$ {#eq-noise}

- Each noise value has mean $0$ (no systematic push up or down).
- Every point has the same noise level $\sigma^2$.
- Different points have independent noise (that is what the identity matrix $\mathbf{I}$ says).

:::box plain | What are these two equations saying in plain words?
- The target is a **straight-line signal** $\mathbf{X}\mathbf{w}$ plus **random wobble** $\boldsymbol\epsilon$.
- We never see $\mathbf{w}$ or $\boldsymbol\epsilon$. We see only $\mathbf{X}$ and $\mathbf{y}$, and we try to recover $\mathbf{w}$.
- Whenever we later take an expectation, the only random thing is $\boldsymbol\epsilon$, because we treat $\mathbf{X}$ as fixed.
:::

## 3. Prediction and error

**Step 1: the prediction for any candidate weights.**

$$ \hat{\mathbf{y}} = \mathbf{X}\mathbf{w} $$ {#eq-pred}

**Step 2: the error is what is left over.**

$$ \mathbf{r} = \mathbf{y} - \mathbf{X}\mathbf{w} $$ {#eq-resid}

:::box step | How we got here
- [[eq-pred]] is [[eq-model]] with the unknown noise removed: we predict using only the signal part.
- [[eq-resid]] is the vertical miss from the picture in Section 1, written for all $n$ points at once.
:::

**Step 3: turn all the misses into one number to minimize.**

$$ L(\mathbf{w}) = \lVert \mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2 = (\mathbf{y}-\mathbf{X}\mathbf{w})^\top(\mathbf{y}-\mathbf{X}\mathbf{w}) $$ {#eq-loss}

**Why squares?**
- Squaring stops positive and negative misses from cancelling.
- Big misses are punished much more than small ones.
- The result is smooth, so we can use calculus. It is also a plain squared length, so Chapter 1 geometry applies.
- Section 6 gives a deeper reason: it is exactly what the Gaussian noise of [[eq-noise]] asks for.

## 4. Deriving the normal equations

We minimize [[eq-loss]] over $\mathbf{w}$ in four steps.

**Step 1: expand the loss.**

$$ \begin{aligned} L(\mathbf{w}) &= \mathbf{y}^\top\mathbf{y} - \mathbf{y}^\top\mathbf{X}\mathbf{w} - \mathbf{w}^\top\mathbf{X}^\top\mathbf{y} + \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} \\ &= \mathbf{y}^\top\mathbf{y} - 2\,\mathbf{w}^\top\mathbf{X}^\top\mathbf{y} + \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} \end{aligned} $$ {#eq-expand}

:::box step | How we got here
- Multiply out the brackets in [[eq-loss]], using $(\mathbf{a}-\mathbf{b})^\top=\mathbf{a}^\top-\mathbf{b}^\top$ and $(\mathbf{X}\mathbf{w})^\top=\mathbf{w}^\top\mathbf{X}^\top$.
- The two middle terms are equal. Each is a single number (a $1\times1$ matrix), and a number equals its own transpose, so $\mathbf{y}^\top\mathbf{X}\mathbf{w}=\mathbf{w}^\top\mathbf{X}^\top\mathbf{y}$. That is why they merge into $-2\,\mathbf{w}^\top\mathbf{X}^\top\mathbf{y}$.
:::

**Step 2: take the gradient with respect to $\mathbf{w}$.**

$$ \frac{\partial L}{\partial \mathbf{w}} = -2\,\mathbf{X}^\top\mathbf{y} + 2\,\mathbf{X}^\top\mathbf{X}\mathbf{w} $$ {#eq-grad}

:::box step | How we got here
- Differentiate [[eq-expand]] term by term.
- $\mathbf{y}^\top\mathbf{y}$ has no $\mathbf{w}$ in it, so its derivative is $0$.
- $\mathbf{w}^\top\mathbf{c}$ differentiates to $\mathbf{c}$, here with $\mathbf{c}=\mathbf{X}^\top\mathbf{y}$.
- $\mathbf{w}^\top\mathbf{M}\mathbf{w}$ differentiates to $2\mathbf{M}\mathbf{w}$ when $\mathbf{M}$ is symmetric, and $\mathbf{M}=\mathbf{X}^\top\mathbf{X}$ is symmetric.
:::

**Step 3: set the gradient to zero.**

$$ \mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y} $$ {#eq-normal-lr}

This is the **normal equation**. It comes from [[eq-grad]] $=\mathbf{0}$, then dividing by $2$ and moving one term across.

**Step 4: solve for $\mathbf{w}$.**

$$ \hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} $$ {#eq-ols}

:::box step | How we got here
- Multiply [[eq-normal-lr]] on the left by $(\mathbf{X}^\top\mathbf{X})^{-1}$.
- This inverse exists only if $\mathbf{X}^\top\mathbf{X}$ is invertible, which means **the columns of $\mathbf{X}$ are linearly independent** (no feature is an exact combination of the others).
:::

:::box plain | What is this solution saying in plain words?
- **$\mathbf{X}^\top\mathbf{y}$:** how strongly each feature lines up with the target.
- **$(\mathbf{X}^\top\mathbf{X})^{-1}$:** corrects for features that are large or that overlap with each other.
- Compare with the 2D projection formula of Chapter 1: the top measures agreement and the bottom removes the size of the reference vectors.
:::

### Is this a minimum, not a maximum?

$$ \frac{\partial^2 L}{\partial\mathbf{w}^2} = 2\,\mathbf{X}^\top\mathbf{X} $$ {#eq-hess}

:::box step | How we got here
- Differentiate [[eq-grad]] once more with respect to $\mathbf{w}$. Only the second term depends on $\mathbf{w}$.
- For any vector $\mathbf{v}$: $\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v}=\lVert\mathbf{X}\mathbf{v}\rVert^2\ge 0$. So the Hessian is **positive semi-definite**, and the loss is a bowl that never curves downwards.
- A bowl has one lowest point, so the stationary point of [[eq-normal-lr]] is a global minimum, not a maximum or a saddle.
- If the columns of $\mathbf{X}$ are independent, $\mathbf{X}\mathbf{v}=\mathbf{0}$ only for $\mathbf{v}=\mathbf{0}$, so the bowl is strictly curved and the minimum is unique.
:::

## 5. Geometric meaning: the fit is a projection

**Step 1: spot the Chapter 1 recipe.**

The columns of $\mathbf{X}$ span a subspace $S=\operatorname{col}(\mathbf{X})$. Every possible prediction $\mathbf{X}\mathbf{w}$ lies in $S$, so minimizing [[eq-loss]] means *finding the point of $S$ closest to $\mathbf{y}$*. In [Chapter 1](chapter.html?c=01-orthogonal-projections#eq-subspace) that was exactly the projection problem, with $\mathbf{B}=\mathbf{X}$ and $\mathbf{x}=\mathbf{y}$.

**Step 2: the residual is perpendicular to every feature.**

$$ \mathbf{X}^\top(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}}) = \mathbf{0} $$ {#eq-resid-orth}

:::box step | How we got here
- Rearranging [[eq-normal-lr]] gives [[eq-resid-orth]]. It is word for word the [orthogonality condition of Chapter 1](chapter.html?c=01-orthogonal-projections#eq-normal).
- So the "normal" in *normal equation* means *perpendicular*: the leftover error is at $90^\circ$ to every column of $\mathbf{X}$.
:::

**Step 3: the fitted values are the shadow of $\mathbf{y}$.**

$$ \hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}} = \underbrace{\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top}_{\mathbf{H}}\,\mathbf{y} $$ {#eq-hat}

:::box step | How we got here
- Put [[eq-ols]] into [[eq-pred]].
- The matrix in front of $\mathbf{y}$ is the [projection matrix of Chapter 1](chapter.html?c=01-orthogonal-projections#eq-P) with $\mathbf{B}=\mathbf{X}$. In statistics it is called the **hat matrix**, because it puts a hat on $\mathbf{y}$.
- It inherits both properties from Chapter 1: $\mathbf{H}^2=\mathbf{H}$ (the shadow of a shadow is itself) and $\mathbf{H}^\top=\mathbf{H}$.
:::

**What this picture tells us:**
- $\hat{\mathbf{y}}$ lies in the plane $S$, and $\mathbf{y}-\hat{\mathbf{y}}$ points straight out of it.
- Fitting a regression is nothing more than dropping a perpendicular from the target vector $\mathbf{y}$ onto the span of the features.

:::tabs Why more features can never increase the training error
--- Geometry
Adding a feature adds a column to $\mathbf{X}$, so the plane $S$ becomes a bigger space that **contains the old one**. The closest point in a bigger space can only be equally close or closer to $\mathbf{y}$. It can never be farther.
--- Set the new weight to zero
Take the old best fit and give the new feature the weight $0$. Nothing changes, so the old fit is still available in the new model. The new minimum is therefore at most as large as the old error.
--- Bigger search set
Minimizing over more choices can only give a smaller (or equal) minimum. The old weights are a special case of the new weights, so the new search space contains them.
:::

$$ \min_{\mathbf{w}\in\mathbb{R}^{p+1}} L(\mathbf{w}) \;\le\; \min_{\mathbf{w}\in\mathbb{R}^{p}} L(\mathbf{w}) $$ {#eq-more-features}

:::box warn | A warning that comes with [[eq-more-features]]
- Training error always drops as you add features, even useless ones. With $p=n$ independent features the fit passes through every training point.
- That is the geometric root of **overfitting**. A lower training error does not mean better predictions on new data. [Chapter 4](chapter.html?c=04-bias-variance) studies exactly this.
:::

:::figure F2.2 | Adding features | A slider adds feature columns to X, even random ones. The residual length can only shrink or stay equal, while a held-out test error curve eventually rises.

## 6. Least squares is maximum likelihood under Gaussian noise

So far squares were a choice. Here is why they are the *right* choice when the noise is Gaussian.

**Step 1: what [[eq-noise]] says about one data point.**

$$ y_i\mid\mathbf{x}_i \;\sim\; \mathcal{N}(\mathbf{x}_i^\top\mathbf{w},\,\sigma^2) $$ {#eq-cond}

Here $\mathbf{x}_i$ is the $i$-th row of $\mathbf{X}$ written as a column. Each observation is Normally spread around the straight-line prediction.

**Step 2: the likelihood of the whole data set.**

$$ \mathcal{L}(\mathbf{w},\sigma)=\prod_{i=1}^{n}\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(y_i-\mathbf{x}_i^\top\mathbf{w})^2}{2\sigma^2}\right) $$ {#eq-lik}

:::box step | How we got here
- Write the Normal density from [[eq-cond]] for each point.
- The points are independent, so the joint probability is the product of the individual densities.
:::

**Step 3: take the logarithm.**

$$ \log\mathcal{L}(\mathbf{w},\sigma) = -n\log\sigma - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i-\mathbf{x}_i^\top\mathbf{w})^2 $$ {#eq-loglik}

:::box step | How we got here
- The log turns the product of [[eq-lik]] into a sum, and it cancels the $\exp$.
- The log is increasing, so the $\mathbf{w}$ that maximizes $\log\mathcal{L}$ also maximizes $\mathcal{L}$.
:::

**Step 4: see what depends on $\mathbf{w}$.**

- The first two terms of [[eq-loglik]] do not contain $\mathbf{w}$, so they do not affect the best $\mathbf{w}$.
- The last term has a minus sign. Maximizing $\log\mathcal{L}$ is therefore the same as **minimizing** $\sum_i (y_i-\mathbf{x}_i^\top\mathbf{w})^2$.
- That sum is exactly [[eq-loss]]. So the maximum likelihood weights equal the least squares weights of [[eq-ols]].

**Other noise, other loss:**

| Noise assumption | Loss that maximum likelihood gives |
|---|---|
| Gaussian | Squared error (this chapter) |
| Laplace | Absolute error |
| Bernoulli (labels 0/1) | Cross-entropy ([Logistic Regression](chapter.html?c=05-logistic-regression)) |

Every common loss function is a negative log-likelihood under some assumption about the noise.

:::figure F2.3 | Gaussian noise gives squared loss | A line through points with a vertical Gaussian bump at each point. A toggle switches the noise to Laplace and shows the loss changing from squared to absolute.

## 7. Gradient descent: the iterative alternative

Computing [[eq-ols]] requires inverting a $p\times p$ matrix, which costs roughly $O(p^3)$ operations. For many features that is too slow, so we walk downhill instead.

**Step 1: the rule.**

$$ \mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,\frac{\partial L}{\partial\mathbf{w}}\bigg|_{\mathbf{w}_t} $$ {#eq-gd}

**Step 2: the rule for squared loss.**

$$ \mathbf{w}_{t+1} = \mathbf{w}_t + \eta\,\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w}_t) $$ {#eq-gd-ls}

:::box step | How we got here
- Use the loss $L(\mathbf{w})=\tfrac12\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2$. The extra $\tfrac12$ only cancels the factor $2$ in [[eq-grad]], so the gradient becomes $-\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w})$.
- Put that into [[eq-gd]]. The two minus signs give a plus.
- Without the $\tfrac12$ you get the same rule with $2\eta$ in place of $\eta$.
:::

- $\eta$ is the **step size** (learning rate).
- The loss is a bowl by [[eq-hess]], so there is one global minimum and no local traps.
- Gradient descent reaches it if $\eta$ is small enough: for this loss, any $\eta<2/\lambda_{\max}(\mathbf{X}^\top\mathbf{X})$ works.
- At the point where [[eq-gd-ls]] stops moving, $\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w})=\mathbf{0}$, which is exactly [[eq-resid-orth]].

:::figure F2.4 | Gradient descent on a bowl | Contour plot of the loss with a draggable start point. A step-size slider shows smooth convergence, zig-zagging, and divergence when eta is too large.

## 8. Gauss-Markov: why least squares is "best"

**Claim.** Among all *linear, unbiased* estimators of $\mathbf{w}$, ordinary least squares has the smallest variance. It is the **BLUE**: Best Linear Unbiased Estimator.

**Assumptions the theorem needs:**
- The model is linear, as in [[eq-model]].
- The noise has mean $\mathbf{0}$ and covariance $\sigma^2\mathbf{I}$ (equal spread, uncorrelated). **Gaussian noise is not required.**
- $\mathbf{X}$ is fixed and has independent columns.

**Step 1: OLS is unbiased.**

$$ \mathbb{E}[\hat{\mathbf{w}}] = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\,\mathbb{E}[\mathbf{y}] = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{w} $$ {#eq-unbiased}

:::box step | How we got here
- Start from [[eq-ols]]. Only $\mathbf{y}$ is random, so the expectation passes through the fixed matrices.
- $\mathbb{E}[\mathbf{y}]=\mathbf{X}\mathbf{w}$ because $\mathbb{E}[\boldsymbol\epsilon]=\mathbf{0}$ in [[eq-model]].
- The matrices then cancel. In words: if you collected many fresh data sets and ran OLS on each, the average estimate would be the true $\mathbf{w}$.
:::

**Step 2: write any other linear estimator in the same shape.**

$$ \tilde{\mathbf{w}} = \mathbf{C}\mathbf{y},\qquad \mathbf{C} = \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top}_{\mathbf{A}} + \mathbf{D} $$ {#eq-other}

OLS itself is $\hat{\mathbf{w}}=\mathbf{A}\mathbf{y}$. The matrix $\mathbf{D}$ measures how much a competitor differs from OLS.

**Step 3: unbiasedness forces a condition on $\mathbf{D}$.**

$$ \mathbb{E}[\tilde{\mathbf{w}}]=\mathbf{C}\mathbf{X}\mathbf{w}=\mathbf{w}\ \text{ for all } \mathbf{w} \;\Longrightarrow\; \mathbf{C}\mathbf{X}=\mathbf{I} \;\Longrightarrow\; \mathbf{D}\mathbf{X}=\mathbf{0} $$ {#eq-dx}

:::box step | How we got here
- Compute the expectation of [[eq-other]] exactly as in [[eq-unbiased]]. It equals $\mathbf{C}\mathbf{X}\mathbf{w}$.
- For this to equal $\mathbf{w}$ whatever $\mathbf{w}$ is, $\mathbf{C}\mathbf{X}$ must be the identity.
- Expand: $(\mathbf{A}+\mathbf{D})\mathbf{X}=\mathbf{A}\mathbf{X}+\mathbf{D}\mathbf{X}$. Since $\mathbf{A}\mathbf{X}=\mathbf{I}$ already, we need $\mathbf{D}\mathbf{X}=\mathbf{0}$.
:::

**Step 4: compare the variances.**

$$ \operatorname{Var}(\tilde{\mathbf{w}}) = \mathbf{C}\,\operatorname{Var}(\mathbf{y})\,\mathbf{C}^\top = \sigma^2(\mathbf{A}+\mathbf{D})(\mathbf{A}+\mathbf{D})^\top = \sigma^2\Big[(\mathbf{X}^\top\mathbf{X})^{-1} + \mathbf{D}\mathbf{D}^\top\Big] $$ {#eq-var}

:::box step | How we got here
- For a linear map, $\operatorname{Var}(\mathbf{C}\mathbf{y})=\mathbf{C}\operatorname{Var}(\mathbf{y})\mathbf{C}^\top$, and $\operatorname{Var}(\mathbf{y})=\sigma^2\mathbf{I}$.
- Expand the product into four terms: $\mathbf{A}\mathbf{A}^\top+\mathbf{A}\mathbf{D}^\top+\mathbf{D}\mathbf{A}^\top+\mathbf{D}\mathbf{D}^\top$.
- $\mathbf{A}\mathbf{A}^\top=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}=(\mathbf{X}^\top\mathbf{X})^{-1}$.
- The two cross terms vanish by [[eq-dx]]: $\mathbf{A}\mathbf{D}^\top=(\mathbf{X}^\top\mathbf{X})^{-1}(\mathbf{D}\mathbf{X})^\top=\mathbf{0}$, and $\mathbf{D}\mathbf{A}^\top=\mathbf{D}\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}=\mathbf{0}$.
:::

**Step 5: read the result.**

- Setting $\mathbf{D}=\mathbf{0}$ in [[eq-var]] gives the variance of OLS: $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$.
- The competitor's variance is that **plus** $\sigma^2\mathbf{D}\mathbf{D}^\top$.
- That extra part is never negative, since $\mathbf{v}^\top\mathbf{D}\mathbf{D}^\top\mathbf{v}=\lVert\mathbf{D}^\top\mathbf{v}\rVert^2\ge0$. This is the same trick as in [[eq-hess]].
- So every other linear unbiased estimator has OLS's variance plus some extra, which proves the claim.

:::box warn | What Gauss-Markov does NOT say
- It does **not** say OLS is the best possible estimator. It only beats other estimators that are both **linear and unbiased**.
- A **biased** estimator can have a smaller total error: it gives up a little bias to cut variance a lot. That is exactly what Ridge regression does in [Chapter 3](chapter.html?c=03-ridge-lasso).
:::

:::figure F2.5 | Gauss-Markov | Two clouds of repeated estimates around the true weights: OLS (tight) and another unbiased estimator (wider). Both are centered on the truth, which makes "same bias, different variance" visible.

## 9. Bridge cheat-sheet: Chapter 1 geometry and regression

| Chapter 1 (projection) | Chapter 2 (regression) |
|---|---|
| Vector to project, $\mathbf{x}$ | Target vector, $\mathbf{y}$ |
| Basis matrix, $\mathbf{B}$ | Feature matrix, $\mathbf{X}$ |
| Weights, $\boldsymbol\lambda$ | Regression weights, $\hat{\mathbf{w}}$ |
| Perpendicular error, [[eq-normal]] | Residual orthogonal to features, [[eq-resid-orth]] |
| Solution, [[eq-lambda-general]] | Normal equation solution, [[eq-ols]] |
| Projection matrix, [[eq-P]] | Hat matrix, [[eq-hat]] |

Note: the equation links in the first column point into Chapter 1, so on this page they will report as unknown. Use the plain-text names there, or open the linked equations in [Chapter 1](chapter.html?c=01-orthogonal-projections).

## 10. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [Ridge and Lasso](chapter.html?c=03-ridge-lasso) | They change [[eq-loss]] by adding a penalty. Without Section 4 and Section 8 you cannot see what the penalty trades away. |
| Machine Learning | [Logistic Regression](chapter.html?c=05-logistic-regression) | Same recipe: define a likelihood (Section 6), take the gradient, then iterate as in Section 7. |
| Probability & Statistics | Bias-variance trade-off | Section 8 computed the variance of an estimator. The next chapters add bias and compare. |
| Probability & Statistics | Confidence intervals, hypothesis tests on weights | They use [[eq-var]], the variance of $\hat{\mathbf{w}}$. |
| Optimization | Stochastic and mini-batch gradient descent | They are [[eq-gd-ls]] computed on a small random subset of rows. |
| Deep Learning | Neural networks | A network's last layer is often a linear regression on learned features. |
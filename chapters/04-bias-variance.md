# Bias–Variance Tradeoff & the Eigenvalue View

*Part II · Regression. The idea that explains why simple models underfit, complex models overfit, and why shrinking weights can help.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Linear Regression (Chapter 2)](chapter.html?c=02-linear-regression) | The OLS estimator, its variance, and the hat matrix. |
| Machine Learning | [Ridge Regression (Chapter 3)](chapter.html?c=03-ridge-lasso) | Its bias and variance are examined direction by direction. |
| Probability & Statistics | Expectation and variance; independence | The whole decomposition is a calculation with expectations. |
| Linear Algebra | Eigenvalues and eigenvectors of a symmetric matrix | They split the estimate into independent directions. |
| Linear Algebra | [Projection matrices (Chapter 1)](chapter.html?c=01-orthogonal-projections) | Idempotent and symmetric properties give the training-error formula. |

## 1. Everyday picture: an archer and a dartboard

Imagine many archers, each given a *different* random sample of practice data and asked to hit the bullseye (the true answer). Each archer's final shot is one fitted model.

1. **Bias** is how far the *centre* of all the shots is from the bullseye. A badly aimed sight pushes every shot the same wrong way.
2. **Variance** is how *spread out* the shots are. A shaky hand scatters them around their own centre.
3. **Noise** is a gust of wind nobody can control. Even a perfect archer cannot remove it.

This gives four cases:

| | Low variance (tight shots) | High variance (scattered shots) |
|---|---|---|
| **Low bias** (centred) | **Ideal** | Overfitting: right on average, wild each time |
| **High bias** (off-centre) | Underfitting: consistently wrong | Worst of both |

:::figure F4.1 | Dartboard | Four dartboards for the four cases. A button resamples the training set, and each click adds one cross showing the fitted weights from that sample. The mean of the crosses is drawn as a dot.

## 2. Setting up the question

**Step 1: the truth plus noise.** At a fixed test point $\mathbf{x}$, the target is generated as

$$ y=f(\mathbf{x})+\epsilon,\qquad \mathbb{E}[\epsilon]=0,\quad \operatorname{Var}(\epsilon)=\sigma^2 $$ {#eq-truth}

- $f(\mathbf{x})$ is the true relationship, which we never see.
- $\epsilon$ is irreducible noise. It is the same kind of noise as [Chapter 2's model](chapter.html?c=02-linear-regression#eq-model), but here the true $f$ may not be exactly linear.

**Step 2: the estimator is itself random.** We fit a model $\hat f$ on a training set $D$ that was drawn at random. Another draw of $D$ gives a slightly different $\hat f$. So $\hat f(\mathbf{x})$ is a **random number**, and we give its average a name:

$$ \bar f(\mathbf{x})=\mathbb{E}_D\big[\hat f(\mathbf{x})\big] $$ {#eq-fbar}

**Step 3: what we want to make small.** Keep $\mathbf{x}$ fixed. Average over every possible training set $D$ and every possible noise value $\epsilon$:

$$ \text{Err}(\mathbf{x})=\mathbb{E}_{D,\epsilon}\Big[\big(y-\hat f(\mathbf{x})\big)^2\Big] $$ {#eq-err}

:::box plain | What is this definition saying in plain words?
- Pick a test point $\mathbf{x}$ and freeze it.
- Repeat the whole experiment many times: draw fresh training data, train the model, draw a fresh test target $y$, record the squared miss.
- The **average** of those squared misses is $\text{Err}(\mathbf{x})$. It is the quantity we actually care about for new data.
:::

## 3. Deriving the decomposition

We split [[eq-err]] into three meaningful pieces, one step at a time.

**Step 1: write the miss as a sum.** Let $g=f(\mathbf{x})-\hat f(\mathbf{x})$ be the model's error against the *truth*. Then $y-\hat f(\mathbf{x})=g+\epsilon$, and

$$ \big(y-\hat f(\mathbf{x})\big)^2=g^2+2g\,\epsilon+\epsilon^2 $$ {#eq-square}

:::box step | How we got here
- Substitute $y=f(\mathbf{x})+\epsilon$ from [[eq-truth]]: $y-\hat f=(f-\hat f)+\epsilon=g+\epsilon$.
- Expand the square $(g+\epsilon)^2$.
:::

**Step 2: average over the test noise.**

$$ \mathbb{E}_\epsilon\big[g^2+2g\,\epsilon+\epsilon^2\big]=g^2+\sigma^2 \quad\Longrightarrow\quad \text{Err}(\mathbf{x})=\mathbb{E}_D\big[(f-\hat f)^2\big]+\sigma^2 $$ {#eq-noise-out}

:::box step | How we got here
- $g$ depends only on the training set, while the test noise $\epsilon$ is fresh and independent of it. So $\mathbb{E}[2g\epsilon]=2g\,\mathbb{E}[\epsilon]=0$ by [[eq-truth]].
- $\mathbb{E}[\epsilon^2]=\operatorname{Var}(\epsilon)+(\mathbb{E}\epsilon)^2=\sigma^2$.
- Now average over $D$ as well, which gives the second form. The noise has been split off as a separate $\sigma^2$.
:::

**Step 3: add and subtract the average prediction.** To open up $\mathbb{E}_D[(f-\hat f)^2]$, insert $\bar f$ from [[eq-fbar]]:

$$ f-\hat f=\big(f-\bar f\big)+\big(\bar f-\hat f\big) $$ {#eq-addsub}

**Step 4: expand the square and average over $D$.**

$$ \mathbb{E}_D\big[(f-\hat f)^2\big]=\underbrace{(f-\bar f)^2}_{\text{constant in }D}+\mathbb{E}_D\big[(\bar f-\hat f)^2\big]+2\,(f-\bar f)\,\mathbb{E}_D\big[\bar f-\hat f\big] $$ {#eq-expand-bv}

:::box step | How we got here
- Square [[eq-addsub]]. It gives three terms: the first squared, the second squared, and twice their product.
- $f(\mathbf{x})$ and $\bar f(\mathbf{x})$ do not depend on the training set, so they come out of $\mathbb{E}_D$ unchanged.
:::

**Step 5: the cross term vanishes.**

$$ \mathbb{E}_D\big[\bar f-\hat f\big]=\bar f-\mathbb{E}_D[\hat f]=\bar f-\bar f=0 $$ {#eq-cross}

By the definition [[eq-fbar]], the *average* deviation from the average is zero. So the last term of [[eq-expand-bv]] disappears.

**Step 6: put everything together.**

$$ \text{Err}(\mathbf{x})=\underbrace{\big(f(\mathbf{x})-\bar f(\mathbf{x})\big)^2}_{\text{Bias}^2}+\underbrace{\mathbb{E}_D\Big[\big(\hat f(\mathbf{x})-\bar f(\mathbf{x})\big)^2\Big]}_{\text{Variance}}+\underbrace{\sigma^2}_{\text{Noise}} $$ {#eq-bv}

:::box step | How we got here
- Start from the second form of [[eq-noise-out]].
- Replace $\mathbb{E}_D[(f-\hat f)^2]$ by [[eq-expand-bv]], where [[eq-cross]] removes the cross term.
- Note that $(\bar f-\hat f)^2=(\hat f-\bar f)^2$, which is the usual way variance is written.
:::

:::box plain | What is this decomposition saying in plain words?
- **Bias$^2$:** the gap between the truth and the **average** prediction. High bias means the model is *systematically wrong* whatever training data it sees, because its form is too rigid to represent $f$.
- **Variance:** how much the prediction **moves around** between training sets. High variance means the model is very sensitive to *which* points happened to be in the sample.
- **$\sigma^2$:** noise in the target itself. No model, however good, can remove it.
- To reduce test error we may only change the first two terms.
:::

:::tabs Three ways to see why the two terms pull against each other
--- Wiggly vs stiff curve
A very flexible curve can bend through every training point, so on average it matches the truth (low bias). But move one training point and the whole curve wobbles (high variance). A stiff straight line barely moves when the data changes (low variance), but it cannot follow a curved truth (high bias).
--- Dartboard
A complex model is a skilled but jittery archer: centred on average, scattered each time. A simple model is a steady archer with a misaligned sight: tightly clustered, but in the wrong place. Making the model more flexible slides you from the second picture towards the first.
--- Parameters and data
Each parameter you add lets the model fit more detail (less bias), but you now estimate more numbers from the same $n$ data points, so each estimate is noisier (more variance). More data reduces variance without changing bias, which is why big data sets allow bigger models.
:::

:::figure F4.2 | Many fits, one truth | Draw many training sets from a noisy curve. Fit a stiff line and a high-degree polynomial to each, showing every fit as a faint line. Overlay the true function and the average fit, so bias (average vs truth) and variance (spread of faint lines) can be read off directly.

## 4. Two worked examples

### Example 1: k-nearest neighbours

A $k$-NN regressor predicts by averaging the targets of the $k$ closest training points.

$$ \text{Err}(k)\;\approx\;\text{Bias}(k)^2+\frac{\sigma^2}{k}+\sigma^2 $$ {#eq-knn}

- **Variance is $\sigma^2/k$:** an average of $k$ independent noisy numbers has variance $\sigma^2/k$. Larger $k$ means smoother, steadier predictions.
- **Bias grows with $k$:** the more distant neighbours you include, the more the average blurs the true shape of $f$.
- Small $k$ is flexible (low bias, high variance), and large $k$ is stiff (high bias, low variance). The best $k$ sits in between.

### Example 2: training error versus test error for linear regression

Assume the true model really is linear, $\mathbf{y}=\mathbf{X}\mathbf{w}+\boldsymbol\epsilon$, with $p$ parameters and $n$ training points. Take $\mathbf{X}$ fixed, and test on **fresh** targets at the same $\mathbf{X}$.

$$ \mathbb{E}[\text{training error}]=\sigma^2\Big(1-\frac{p}{n}\Big),\qquad \mathbb{E}[\text{test error}]=\sigma^2\Big(1+\frac{p}{n}\Big) $$ {#eq-train-test}

Both are per data point. We prove it with the hat matrix $\mathbf{H}$ from [Chapter 2](chapter.html?c=02-linear-regression#eq-hat).

**Step 1: the key trace fact.** For a projection matrix, $\operatorname{tr}(\mathbf{H})=p$.

:::box step | How we got here
- $\mathbf{H}$ is the [projection onto the $p$-dimensional column space](chapter.html?c=01-orthogonal-projections#eq-P). Its eigenvalues are $1$ (exactly $p$ times) and $0$ (the other $n-p$ times).
- The trace is the sum of the eigenvalues, so it equals $p$. Likewise $\operatorname{tr}(\mathbf{I}-\mathbf{H})=n-p$.
- We also use $\mathbf{H}^\top\mathbf{H}=\mathbf{H}$, from the [symmetric](chapter.html?c=01-orthogonal-projections#eq-sym) and [idempotent](chapter.html?c=01-orthogonal-projections#eq-idem) properties.
:::

**Step 2: training error.**

- The residual is $\mathbf{y}-\hat{\mathbf{y}}=(\mathbf{I}-\mathbf{H})\mathbf{y}=(\mathbf{I}-\mathbf{H})\boldsymbol\epsilon$, because $(\mathbf{I}-\mathbf{H})\mathbf{X}\mathbf{w}=\mathbf{0}$ (the signal lies inside the column space).
- Its expected squared length is $\sigma^2\operatorname{tr}\big((\mathbf{I}-\mathbf{H})^\top(\mathbf{I}-\mathbf{H})\big)=\sigma^2(n-p)$.
- Dividing by $n$ gives $\sigma^2(1-p/n)$.

**Step 3: test error on fresh noise $\boldsymbol\epsilon^\ast$.**

- The new targets are $\mathbf{y}^\ast=\mathbf{X}\mathbf{w}+\boldsymbol\epsilon^\ast$, so $\mathbf{y}^\ast-\hat{\mathbf{y}}=\boldsymbol\epsilon^\ast-\mathbf{H}\boldsymbol\epsilon$.
- The two noises are independent, so $\mathbb{E}\lVert\boldsymbol\epsilon^\ast-\mathbf{H}\boldsymbol\epsilon\rVert^2=n\sigma^2+\sigma^2\operatorname{tr}(\mathbf{H})=n\sigma^2+p\sigma^2$.
- Dividing by $n$ gives $\sigma^2(1+p/n)$.

:::box plain | What is this saying in plain words?
- The model has $p$ knobs, and it uses them to chase the noise in the training data. That makes training error look **better** than the truth by about $\sigma^2 p/n$.
- On new data the same knobs add error, so test error is **worse** than the noise level by the same amount.
- The gap between the two is $2\sigma^2 p/n$. It grows with the number of parameters and shrinks with more data.
- Always $\mathbb{E}[\text{training error}]\le\mathbb{E}[\text{test error}]$. This is why you must **never judge a model by its training error**.
:::

:::figure F4.6 | Training vs test error | Plot both errors against the ratio p/n. The training curve falls as sigma squared times (1 minus p/n), the test curve rises as sigma squared times (1 plus p/n), and they open up like a pair of scissors.

:::figure F4.3 | Model complexity curve | A slider for polynomial degree. Training error keeps falling, test error falls and then rises to give the classic U-shape, with the best degree marked.

## 5. The eigenvalue view of OLS

Now we look *inside* the estimator and see where its variance comes from.

**Step 1: recall the variance of OLS.** From [Chapter 2](chapter.html?c=02-linear-regression#eq-var) (with $\mathbf{D}=\mathbf{0}$):

$$ \operatorname{Var}(\hat{\mathbf{w}}_{\text{OLS}}\mid\mathbf{X})=\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1} $$ {#eq-var-ols}

**Step 2: split $\mathbf{X}^\top\mathbf{X}$ into its eigen-directions.** The matrix is symmetric and positive semi-definite (as shown for the [Hessian](chapter.html?c=02-linear-regression#eq-hess)). So the spectral theorem gives:

$$ \mathbf{X}^\top\mathbf{X}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top,\qquad \boldsymbol\Lambda=\operatorname{diag}(d_1,\dots,d_p),\quad \mathbf{Q}^\top\mathbf{Q}=\mathbf{I} $$ {#eq-eig}

- The columns $\mathbf{q}_i$ of $\mathbf{Q}$ are **eigenvectors**: perpendicular unit directions in weight space (an orthonormal basis, as in [Chapter 1](chapter.html?c=01-orthogonal-projections#eq-onb)).
- The eigenvalue $d_i\ge0$ says **how much information the data has about the direction $\mathbf{q}_i$**. If the data points are spread widely along that direction, $d_i$ is large.

**Step 3: invert it direction by direction.**

$$ (\mathbf{X}^\top\mathbf{X})^{-1}=\mathbf{Q}\boldsymbol\Lambda^{-1}\mathbf{Q}^\top,\qquad \boldsymbol\Lambda^{-1}=\operatorname{diag}\!\Big(\frac1{d_1},\dots,\frac1{d_p}\Big) $$ {#eq-inv-eig}

:::box step | How we got here
- Take the inverse of [[eq-eig]]. Since $\mathbf{Q}^{-1}=\mathbf{Q}^\top$, we get $(\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top)^{-1}=\mathbf{Q}\boldsymbol\Lambda^{-1}\mathbf{Q}^\top$.
- The inverse of a diagonal matrix just inverts each diagonal entry.
:::

**Step 4: put it into the variance, and read off one direction.**

$$ \operatorname{Var}(\hat{\mathbf{w}}_{\text{OLS}})=\sigma^2\,\mathbf{Q}\boldsymbol\Lambda^{-1}\mathbf{Q}^\top \qquad\Longrightarrow\qquad \operatorname{Var}\big(\mathbf{q}_i^\top\hat{\mathbf{w}}_{\text{OLS}}\big)=\frac{\sigma^2}{d_i} $$ {#eq-var-dir}

:::box step | How we got here
- Substitute [[eq-inv-eig]] into [[eq-var-ols]] for the first form.
- For the second, the variance along a unit direction $\mathbf{q}_i$ is $\mathbf{q}_i^\top\operatorname{Var}(\hat{\mathbf{w}})\,\mathbf{q}_i=\sigma^2\mathbf{q}_i^\top\mathbf{Q}\boldsymbol\Lambda^{-1}\mathbf{Q}^\top\mathbf{q}_i$.
- Since $\mathbf{Q}^\top\mathbf{q}_i$ is the $i$-th unit vector, only the $i$-th diagonal entry survives, leaving $\sigma^2/d_i$.
:::

:::box plain | What is this saying in plain words?
- **Large $d_i$** (lots of information in that direction): the estimate along $\mathbf{q}_i$ is very steady, with small variance $\sigma^2/d_i$.
- **Small $d_i$** (little information, for example two nearly identical features): the variance $\sigma^2/d_i$ **explodes**. OLS becomes extremely unstable in exactly the directions the data barely informs.
- The total variance is dominated by the *worst-informed* directions.
:::

:::figure F4.4 | Eigen-directions | A cloud of data points with the ellipse of X^T X and its two axes drawn. Next to it, a bar for each axis shows the variance sigma squared over d. Drag the points to flatten the cloud and watch the bar of the thin direction explode.

## 6. The same view for Ridge

**Step 1: add $\lambda\mathbf{I}$ in the eigenbasis.** Since $\mathbf{I}=\mathbf{Q}\mathbf{Q}^\top$, using [[eq-eig]],

$$ \mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}=\mathbf{Q}(\boldsymbol\Lambda+\lambda\mathbf{I})\mathbf{Q}^\top $$ {#eq-ridge-eig}

The eigenvectors do not change. **Every eigenvalue simply becomes $d_i+\lambda$.**

**Step 2: variance of Ridge in each direction.** Put [[eq-ridge-eig]] into the Ridge variance from [Chapter 3](chapter.html?c=03-ridge-lasso#eq-ridge-var) and repeat Step 4 of the previous section:

$$ \operatorname{Var}\big(\mathbf{q}_i^\top\hat{\mathbf{w}}_{\text{ridge}}\big)=\sigma^2\,\frac{d_i}{(d_i+\lambda)^2}\;<\;\frac{\sigma^2}{d_i}=\operatorname{Var}\big(\mathbf{q}_i^\top\hat{\mathbf{w}}_{\text{OLS}}\big) $$ {#eq-ridge-var-dir}

The inequality holds for every $\lambda>0$, because $d_i^2<(d_i+\lambda)^2$.

**Step 3: how Ridge changes the estimate itself.** Write $\mathbf{w}_{\text{OLS}}$ for the OLS estimate. Using $\mathbf{X}^\top\mathbf{y}=\mathbf{X}^\top\mathbf{X}\,\hat{\mathbf{w}}_{\text{OLS}}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top\hat{\mathbf{w}}_{\text{OLS}}$ in [Chapter 3's solution](chapter.html?c=03-ridge-lasso#eq-ridge-sol):

$$ \hat{\mathbf{w}}_{\text{ridge}}=\mathbf{Q}(\boldsymbol\Lambda+\lambda\mathbf{I})^{-1}\boldsymbol\Lambda\,\mathbf{Q}^\top\hat{\mathbf{w}}_{\text{OLS}}\quad\Longrightarrow\quad \mathbf{q}_i^\top\hat{\mathbf{w}}_{\text{ridge}}=\frac{d_i}{d_i+\lambda}\,\mathbf{q}_i^\top\hat{\mathbf{w}}_{\text{OLS}} $$ {#eq-shrink-dir}

:::box plain | What is this saying in plain words?
- Ridge **multiplies the OLS estimate in direction $\mathbf{q}_i$ by the factor $\tfrac{d_i}{d_i+\lambda}$**, which always lies between $0$ and $1$.
- **Large $d_i$** (a well-informed direction): the factor is close to $1$, so Ridge barely touches it.
- **Small $d_i$** (a poorly informed direction): the factor is close to $0$, so Ridge shrinks it hard. That is exactly where OLS was most unstable.
- Ridge is therefore **adaptive**: it shrinks most where the data is least trustworthy.
:::

**Step 4: bias and variance in one direction.** Write $v_i=\mathbf{q}_i^\top\mathbf{w}$ for the true weight along $\mathbf{q}_i$. From [[eq-shrink-dir]] and [[eq-ridge-var-dir]], and from the fact that OLS is unbiased:

$$ \text{Bias}_i=-\frac{\lambda}{d_i+\lambda}\,v_i,\qquad \text{Var}_i=\frac{\sigma^2 d_i}{(d_i+\lambda)^2},\qquad \text{MSE}_i(\lambda)=\frac{\lambda^2v_i^2+\sigma^2d_i}{(d_i+\lambda)^2} $$ {#eq-dir-mse}

This is the multi-feature version of the [one-feature example in Chapter 3](chapter.html?c=03-ridge-lasso#eq-1d). Differentiating with respect to $\lambda$ proves what Chapter 3 only claimed:

1. $\text{Var}_i$ has derivative $-2\sigma^2d_i/(d_i+\lambda)^3<0$, so **variance falls steadily as $\lambda$ grows**.
2. $\lvert\text{Bias}_i\rvert=\lambda\lvert v_i\rvert/(d_i+\lambda)$ has derivative $d_i\lvert v_i\rvert/(d_i+\lambda)^2>0$, so **bias grows steadily**.
3. At $\lambda=0$, bias$^2$ has derivative $0$ (it grows like $\lambda^2$), while variance has derivative $-2\sigma^2/d_i^2<0$. So **a small positive $\lambda$ always lowers the total error** compared with OLS.
4. Setting the derivative of $\text{MSE}_i$ to zero gives the best value for this direction: $\lambda_i^\ast=\sigma^2/v_i^2$.

:::box step | How we got here
- For point 4: differentiating $\text{MSE}_i$ gives numerator $2\lambda v_i^2(d_i+\lambda)-2(\lambda^2v_i^2+\sigma^2d_i)$. The $\lambda^2v_i^2$ terms cancel, leaving $2d_i(\lambda v_i^2-\sigma^2)$, which is zero at $\lambda=\sigma^2/v_i^2$.
- This matches [Chapter 3's $\lambda=\sigma^2/\tau^2$](chapter.html?c=03-ridge-lasso#eq-ridge) with $\tau^2$ playing the role of the true weight squared. A bigger true weight means a smaller best $\lambda$.
- The best value depends on the unknown $v_i$, so in practice we choose $\lambda$ from data, as in the next section.
:::

:::figure F4.5 | Ridge per direction | A lambda slider with two panels, one for a large eigenvalue and one for a small one. Each shows bias squared rising, variance falling, and their sum with its minimum marked. The small-eigenvalue panel has a much lower and earlier minimum.

## 7. Choosing the sweet spot from data

[[eq-bv]] contains the true $f$, which we do not know, so we cannot compute the bias directly. Instead we estimate the test error itself:

1. **Hold out** part of the data. Train on the rest and measure error on the held-out part.
2. **K-fold cross-validation:** split the data into $K$ parts. Train on $K-1$ of them, test on the remaining one, rotate, and average the $K$ errors. This is an empirical estimate of the expected test error.
3. Do this for each candidate (a value of $\lambda$, a polynomial degree, a number $k$ of neighbours) and **pick the one with the smallest cross-validated error**.

:::box warn | Common mistakes
- Choosing the model by **training error**. It always improves with complexity, as [[eq-train-test]] and [Chapter 2](chapter.html?c=02-linear-regression#eq-more-features) show.
- Using the **test set** to choose $\lambda$ and then reporting the same test score. Choose with validation or cross-validation, and keep a final test set untouched.
- Thinking more data always fixes it. More data lowers **variance**, but it does not remove **bias** from a model that is too rigid.
:::

## 8. Bridge cheat-sheet

| Question | Answer | Where |
|---|---|---|
| What does test error consist of? | Bias$^2$ + variance + noise | [[eq-bv]] |
| Which term can we reduce? | The first two. Noise is fixed. | [[eq-bv]] |
| What happens to a very flexible model? | Low bias, high variance | Sections 3 and 4 |
| What happens to a very rigid model? | High bias, low variance | Sections 3 and 4 |
| Training vs test error for OLS | $\sigma^2(1\mp p/n)$ | [[eq-train-test]] |
| Where does OLS variance come from? | Directions with small $d_i$ | [[eq-var-dir]] |
| What does Ridge do per direction? | Multiplies by $d_i/(d_i+\lambda)$ | [[eq-shrink-dir]] |
| Does a small $\lambda$ always help? | Yes, it lowers total error | [[eq-dir-mse]] |
| How do we pick $\lambda$ in practice? | Cross-validation | Section 7 |

## 9. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [Decision Trees](chapter.html?c=07-decision-trees) | A deep tree has near-zero bias and high variance. Understanding [[eq-bv]] explains why. |
| Machine Learning | [Random Forest](chapter.html?c=08-random-forest) | Averaging many trees lowers variance without raising bias, a direct use of this decomposition. |
| Machine Learning | Boosting and XGBoost | Boosting attacks **bias** by adding weak learners one at a time, and then regularization controls variance. |
| Machine Learning | SVM with the parameter $C$ | $C$ is a bias-variance knob: large $C$ means a flexible fit, small $C$ a smoother, more rigid one. |
| Probability & Statistics | Model selection (AIC, BIC, cross-validation) | They are different ways to estimate the test error that [[eq-train-test]] shows training error underestimates. |
| Linear Algebra | PCA, SVD, ill-conditioned problems | Small eigenvalues of $\mathbf{X}^\top\mathbf{X}$ cause unstable estimates, and PCA regression simply drops those directions. |
| Deep Learning | Double descent, early stopping, dropout | Modern models go beyond the classic U-curve, but each of these techniques is explained as controlling variance or bias. |
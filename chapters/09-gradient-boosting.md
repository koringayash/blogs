# Gradient Boosting

*Part IV · Trees & Ensembles. Chapter 8 made a model better by averaging many independent trees. This chapter builds trees one after another, each correcting the last, and the recipe is gradient descent in disguise.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Linear Regression (Chapter 2)](chapter.html?c=02-linear-regression#eq-gd) | Gradient descent on a weight vector. We repeat the same idea on a function. |
| Machine Learning | [Logistic Regression (Chapter 5)](chapter.html?c=05-logistic-regression#eq-grad-lr) | The logistic-loss gradient $\mathbf{p}-\mathbf{y}$ reappears as the classification version. |
| Machine Learning | [Decision Trees (Chapter 7)](chapter.html?c=07-decision-trees) | A small regression tree is the building block we add at every round. |
| Machine Learning | [Bias and variance (Chapter 4)](chapter.html?c=04-bias-variance) | Boosting mainly reduces bias; the regularization knobs control variance. |
| Calculus & Optimization | Gradient, first-order Taylor approximation | They show why stepping against the gradient lowers the loss. |
| Linear Algebra | [Projections (Chapter 1)](chapter.html?c=01-orthogonal-projections) | Fitting a tree to the gradient is a least-squares projection. |

## 1. Everyday picture: a golfer who corrects each shot

Imagine a golfer who is far from the hole.

1. The **first shot** is a rough guess in the right direction.
2. For the **second shot** the golfer does not start over. They look at **how far and in which direction the ball still is from the hole**, and hit a short shot to fix exactly that.
3. For the **third shot** they again look at the remaining gap and correct *that*.
4. Each shot is small and simple. Together they reach the hole.

Gradient boosting works the same way:
- The **current prediction** is where the ball is.
- The **gap to the target** (the error the model still makes) tells us what the next shot must fix.
- Each **new tree** is one small corrective shot.
- The final model is the **sum** of all the shots.

Compare with [Chapter 8](chapter.html?c=08-random-forest): a random forest asks many people at once and averages. Boosting asks one person at a time, and each person studies the mistakes of those before.

## 2. Gradient descent, once more

**In Chapter 2** we minimized a loss $J(\mathbf{w})$ over a **vector of weights** by repeatedly stepping against the slope:

$$ \mathbf{w}_{t+1}=\mathbf{w}_t-\eta\,\nabla J(\mathbf{w}_t) $$ {#eq-gd-w}

This is the rule from [Chapter 2](chapter.html?c=02-linear-regression#eq-gd), written for a general loss. The pieces:
1. $\nabla J$ points **uphill**, so $-\nabla J$ points **downhill**.
2. $\eta>0$ is the step size.
3. After many steps, $\mathbf{w}$ settles near a minimum.

**A change of viewpoint.** Instead of thinking of the **weights** as the thing we adjust, think of the **predictions themselves** as the thing we adjust.
- Our training set has $n$ points. A model's output on them is a list of $n$ numbers: $F(\mathbf{x}_1),\dots,F(\mathbf{x}_n)$.
- Moving these $n$ numbers up or down changes the loss.
- Boosting says: **"move the predictions downhill, and let a small tree decide how."**

:::figure F9.3 | Parameter space vs function space | Two panels. On the left, gradient descent moves a weight vector down a bowl. On the right, the same loss is viewed as a function of the vector of predictions, and boosting moves that vector one tree-step at a time.

## 3. The model: a sum of small trees

**Step 1: the additive form.** Start from a simple constant $F_0$ and add a small correction at every round:

$$ F_{t+1}(\mathbf{x})=F_t(\mathbf{x})+\eta\,h_t(\mathbf{x}) $$ {#eq-fspace}

Here $h_t$ is a small decision tree (the "weak learner") and $\eta$ is a small number called the **learning rate** or **shrinkage**.

**Step 2: unroll it.** Applying [[eq-fspace]] repeatedly from $F_0$ gives the final model after $T$ rounds:

$$ F_T(\mathbf{x})=F_0(\mathbf{x})+\eta\sum_{t=1}^{T}h_t(\mathbf{x}) $$ {#eq-final}

:::box plain | What is this model saying in plain words?
- $F_0$ is the first rough guess, the same for every input (for example, the average of the targets).
- Each $h_t$ is a small tree that adds a little correction wherever the model so far is wrong.
- $\eta$ keeps each correction **small**, so no single tree can do too much damage. Many small, careful steps beat a few big ones.
:::

**The key question:** what should $h_t$ be? The answer comes from a gradient.

## 4. Why the negative gradient

**Step 1: the loss on the training set.** Let $\ell(y,F)$ be the loss of predicting $F$ when the truth is $y$ (squared error, logistic loss, and so on). The total loss is $\sum_i\ell\big(y_i,F(\mathbf{x}_i)\big)$.

**Step 2: how the loss responds to a small change.** Suppose we nudge each prediction by a small amount $\delta_i$. A first-order Taylor expansion says

$$ \sum_i\ell\big(y_i,F(\mathbf{x}_i)+\delta_i\big)\;\approx\;\sum_i\ell\big(y_i,F(\mathbf{x}_i)\big)+\sum_ig_i\,\delta_i,\qquad g_i=\frac{\partial\,\ell\big(y_i,F(\mathbf{x}_i)\big)}{\partial F(\mathbf{x}_i)} $$ {#eq-taylor-f}

:::box step | How we got here
- This is the usual "new value ≈ old value + slope × change", applied to each of the $n$ predictions separately.
- $g_i$ is the slope of the loss for point $i$ with respect to **its own prediction**. It tells us whether raising $F(\mathbf{x}_i)$ would increase or decrease the loss.
:::

**Step 3: choose the nudge that lowers the loss fastest.** The change in loss is $\sum_ig_i\delta_i$. For a fixed total size of nudge, this is as negative as possible when each nudge points **against** its slope:

$$ \delta_i=-\eta\,g_i $$ {#eq-steepest}

This is the same reasoning as steepest descent in [[eq-gd-w]]. We define the **pseudo-residual** of point $i$ at round $t$ as the negative slope:

$$ r_{it}=-\left[\frac{\partial\,\ell\big(y_i,F(\mathbf{x}_i)\big)}{\partial F(\mathbf{x}_i)}\right]_{F=F_t} $$ {#eq-pseudo}

So the ideal correction at point $i$ is $\eta\,r_{it}$.

**Step 4: the catch.** [[eq-steepest]] tells us what to add at the $n$ training points only. A model must give an answer at **every** input, including ones it has never seen. Simply storing the $r_{it}$ values would memorize the training set.

**Step 5: the fix.** Train a tree to **imitate** the pseudo-residuals, so that it learns a smooth rule that generalizes:

$$ h_t=\arg\min_{h}\;\sum_{i=1}^{n}\big(r_{it}-h(\mathbf{x}_i)\big)^2 $$ {#eq-fit}

The new tree is an ordinary **regression tree** ([Chapter 7](chapter.html?c=07-decision-trees#eq-reg-imp)) whose targets are $r_{it}$ instead of the original labels $y_i$.

:::box plain | What are Steps 1 to 5 saying in plain words?
- The pseudo-residual $r_{it}$ says, for each training point, **which way and how hard the prediction should move** to lower the loss.
- A tree is fitted to those directions so that the correction also makes sense for **new inputs**.
- Fitting a tree by least squares is a **projection** of the vector $(r_{1t},\dots,r_{nt})$ onto the set of functions a tree can express, the same "closest point" idea as [Chapter 1](chapter.html?c=01-orthogonal-projections).
- So each round is a gradient-descent step, taken in the space of functions and approximated by the closest tree.
:::

**The whole algorithm:**
1. Start with a constant $F_0$ that minimizes the loss (Section 5 gives examples).
2. For $t=1,\dots,T$:
   1. Compute the pseudo-residuals $r_{it}$ for every training point using [[eq-pseudo]].
   2. Fit a small regression tree $h_t$ to them, using [[eq-fit]].
   3. Update $F_{t+1}=F_t+\eta\,h_t$, as in [[eq-fspace]].
3. Output $F_T$.

## 5. Special case: squared error (regression)

**Step 1: the loss.** Use $\ell(y,F)=\tfrac12(y-F)^2$. (The $\tfrac12$ only cancels a factor of $2$, as in [Chapter 2's gradient descent](chapter.html?c=02-linear-regression#eq-gd-ls). Without it, the same algorithm results with $2\eta$ in place of $\eta$.)

**Step 2: its derivative.**

$$ \frac{\partial\ell}{\partial F}=-(y-F) $$ {#eq-mse-deriv}

**Step 3: the pseudo-residual.** Put [[eq-mse-deriv]] into [[eq-pseudo]]:

$$ r_{it}=y_i-F_t(\mathbf{x}_i) $$ {#eq-resid-gb}

:::box plain | What is this saying in plain words?
- For squared error, **the negative gradient is exactly the ordinary residual**: truth minus current prediction.
- So "fit a tree to the negative gradient" becomes **"fit a tree to the mistakes the model still makes"**. This is the golfer of Section 1.
- Gradient boosting with squared error is therefore the plain, intuitive procedure: predict, measure the leftover error, fit a tree to the leftover, add it, repeat.
:::

**The starting point $F_0$.** With a single constant $c$ for every point, minimize $\sum_i\tfrac12(y_i-c)^2$. Setting the derivative to zero gives $\sum_i(y_i-c)=0$, so

$$ F_0=\bar y=\frac1n\sum_{i=1}^{n}y_i $$ {#eq-f0-mse}

This is the same "projection onto the all-ones vector" fact used for a leaf in [Chapter 7](chapter.html?c=07-decision-trees#eq-reg-gain).

### A worked example by hand

Four points with $x=1,2,3,4$ and targets $y=1,\,3,\,7,\,9$. Use $\eta=0.5$ and a stump (a tree with one split, here between $x=2$ and $x=3$).

**Round 0.** $F_0=\bar y=5$ for every point. Residuals: $-4,\,-2,\,+2,\,+4$.

**Round 1.** The stump fits the residuals by their leaf means: left leaf $\{-4,-2\}$ gives $-3$, right leaf $\{2,4\}$ gives $+3$. Update with $\eta=0.5$:
- left points: $5+0.5\times(-3)=3.5$,
- right points: $5+0.5\times3=6.5$.

New residuals: $1-3.5=-2.5,\;3-3.5=-0.5,\;7-6.5=0.5,\;9-6.5=2.5$.

**Round 2.** Leaf means: left $-1.5$, right $+1.5$. Update:
- left points: $3.5-0.75=2.75$,
- right points: $6.5+0.75=7.25$.

New residuals: $-1.75,\;0.25,\;-0.25,\;1.75$.

| After round | Predictions $(x=1,2,3,4)$ | Sum of squared residuals |
|---|---|---|
| 0 | $5,\,5,\,5,\,5$ | $16+4+4+16=40$ |
| 1 | $3.5,\,3.5,\,6.5,\,6.5$ | $6.25+0.25+0.25+6.25=13$ |
| 2 | $2.75,\,2.75,\,7.25,\,7.25$ | $3.0625+0.0625+0.0625+3.0625=6.25$ |

**What we see:**
- The error falls every round. Within each group the average residual shrinks by the factor $1-\eta=0.5$.
- A single stump cannot separate $x=1$ from $x=2$, so the remaining error (the $-1.75$ versus $+0.25$ gap) is not removable by this split. Later rounds, where residuals differ more *inside* a group, will pick a different split and fix it. This is why boosting uses **many** different small trees.

:::figure F9.1 | Boosting round by round | A noisy curve with points. A slider for the number of rounds shows the fitted function as a staircase that gets closer to the curve each round, with the residuals drawn as vertical bars from each point to the current fit.

:::figure F9.2 | Residual equals negative gradient | The squared-error loss of one point drawn against the prediction F. A marker sits at the current prediction, the tangent line shows the slope, and a horizontal arrow shows that the downhill direction and size equal the residual y minus F.

## 6. Classification with the logistic loss

For classification the recipe stays the same. Only the **loss**, and therefore the pseudo-residual, changes.

**Step 1: labels and loss.** Use labels $y\in\{-1,+1\}$ and a real-valued score $F(\mathbf{x})$ (the model's raw output, before any probability). Use the logistic loss:

$$ \ell(y,F)=\log\big(1+e^{-yF}\big) $$ {#eq-logloss}

This is the cross-entropy of [Chapter 5](chapter.html?c=05-logistic-regression#eq-bce) written with $\pm1$ labels, as shown in [Chapter 6's comparison](chapter.html?c=06-svm#eq-hinge-obj).

**Step 2: differentiate with respect to $F$.** By the chain rule,

$$ \frac{\partial\ell}{\partial F}=\frac{-y\,e^{-yF}}{1+e^{-yF}}=\frac{-y}{1+e^{yF}} $$ {#eq-logloss-deriv}

:::box step | How we got here
- The derivative of $\log(1+u)$ with $u=e^{-yF}$ is $\dfrac{u'}{1+u}$, and $u'=-y\,e^{-yF}$.
- Divide the top and bottom of $\dfrac{-ye^{-yF}}{1+e^{-yF}}$ by $e^{-yF}$ to get $\dfrac{-y}{e^{yF}+1}$.
:::

**Step 3: the pseudo-residual.** Put [[eq-logloss-deriv]] into [[eq-pseudo]]:

$$ r_{it}=\frac{y_i}{1+e^{\,y_iF_t(\mathbf{x}_i)}} $$ {#eq-neggrad-logistic}

The tree $h_t$ is now fitted to this quantity **instead of** to $y_i-F_t(\mathbf{x}_i)$.

**Step 4: what does this quantity mean?** Let $p_i=\sigma\big(F_t(\mathbf{x}_i)\big)$ be the predicted probability of class $+1$, using the sigmoid from [Chapter 5](chapter.html?c=05-logistic-regression#eq-sigmoid). Use the symmetry $\sigma(-z)=1-\sigma(z)$:

1. **If $y_i=+1$:** $r_{it}=\dfrac{1}{1+e^{F}}=1-\sigma(F)=1-p_i$.
2. **If $y_i=-1$:** $r_{it}=\dfrac{-1}{1+e^{-F}}=-\sigma(F)=-p_i$.

In both cases, writing $\tilde y_i\in\{0,1\}$ for the label in $0/1$ form ($\tilde y_i=1$ when $y_i=+1$ and $0$ otherwise):

$$ r_{it}=\tilde y_i-p_i $$ {#eq-01form}

:::box plain | What is this saying in plain words?
- For classification, **the negative gradient is "the true label minus the predicted probability"**.
- Compare with regression: there it was "the true value minus the predicted value". The structure is identical, only the prediction is now a probability.
- A point already classified with high confidence has $r_{it}$ close to $0$ and is **ignored** by the next tree. A badly misclassified point has $r_{it}$ close to $\pm1$ and gets **most of the attention**.
- This is also the gradient from [Chapter 5](chapter.html?c=05-logistic-regression#eq-grad-lr), $\mathbf{p}-\mathbf{y}$, with the sign flipped to point downhill.
:::

**The starting point $F_0$ for classification.** For one constant score $c$ on all points, the loss gradient from Chapter 5 vanishes when $\sum_i(p-\tilde y_i)=0$, that is, when $p=\bar{\tilde y}$ is the share of positive examples. Therefore the best constant is the **log-odds** of that share:

$$ F_0=\log\frac{\bar{\tilde y}}{1-\bar{\tilde y}} $$ {#eq-f0-logit}

(For example, if $80\%$ of the points are positive, $F_0=\log4\approx1.39$.)

:::box warn | One honest simplification
Fitting a tree to [[eq-neggrad-logistic]] and then using plain leaf **means**, as in regression, is only an approximation for the logistic loss. Careful implementations set each leaf's value with a better rule, based on the curvature as well as the slope, so that the leaf value truly lowers the loss. That refinement is a Newton-type step, and it is exactly the idea behind [XGBoost](chapter.html?c=11-xgboost).
:::

:::figure F9.5 | Classification version | Two classes of points with the decision boundary after each boosting round. At every point an arrow shows the pseudo-residual y minus p. Arrows are long for badly classified points and almost vanish for confident ones, so attention visibly moves towards the hard cases.

## 7. One family, many losses

The only thing that changes between tasks is the loss. The recipe in Section 4 stays fixed.

| Task | Loss $\ell(y,F)$ | Pseudo-residual $r_{it}$ | Starting value $F_0$ |
|---|---|---|---|
| Regression, squared error | $\tfrac12(y-F)^2$ | $y-F$, [[eq-resid-gb]] | mean of $y$, [[eq-f0-mse]] |
| Classification, logistic | $\log(1+e^{-yF})$ | $\tilde y-p$, [[eq-01form]] | log-odds of the base rate, [[eq-f0-logit]] |
| Regression, absolute error | $\lvert y-F\rvert$ | $\operatorname{sign}(y-F)$ | median of $y$ |
| AdaBoost, exponential loss | $e^{-yF}$ | $y\,e^{-yF}$ | $0$ |

The last row is the topic of [Chapter 10](chapter.html?c=10-adaboost): AdaBoost is gradient boosting with the exponential loss and a one-split tree as its weak learner. Seen this way, boosting is **one idea**: pick a loss, take its negative gradient at the current predictions, and fit a small tree to it.

## 8. Controlling the algorithm

### Shrinkage: the learning rate $\eta$

- A small $\eta$ (for example $0.05$ to $0.1$) makes each tree contribute only a little. The model moves slowly along the descent path and needs **more rounds**, but it usually reaches a **better test error**.
- A large $\eta$ races ahead, can overshoot the minimum, and overfits faster.
- This is the same step-size trade-off as in [Chapter 2](chapter.html?c=02-linear-regression#eq-gd-ls), and acts as regularization.

### The number of rounds $T$

- Training error keeps falling as $T$ grows. Test error falls and then **rises**, because late trees begin to fit noise.
- Pick $T$ by **early stopping**: watch the error on held-out data and stop when it no longer improves.
- **$\eta$ and $T$ trade off.** Halving $\eta$ roughly means doubling $T$ for similar results.

### The size of each tree

- Boosted trees are **shallow** (depth $1$ to $8$, often $3$ to $6$). Each tree is a deliberately **weak** model.
- Depth controls how many features can interact in one correction. A stump captures effects of single features, depth $2$ allows pairs, and so on.

### Subsampling

- Using a random fraction of the rows for each tree (**stochastic gradient boosting**), and optionally a random subset of the features, adds randomness like [Chapter 8](chapter.html?c=08-random-forest#eq-var-final). It lowers the correlation between consecutive trees and acts as extra regularization.

:::figure F9.4 | Learning rate and rounds | Sliders for eta and for the number of rounds T. Training and test error are plotted against T. A small eta gives a slow smooth descent with a lower final test error, while a large eta gives a fast descent that turns upward early.

## 9. Bias, variance and the contrast with bagging

By [Chapter 4](chapter.html?c=04-bias-variance#eq-bv), test error is $\text{Bias}^2+\text{Variance}+\sigma^2$.

| | Random forest ([Chapter 8](chapter.html?c=08-random-forest)) | Gradient boosting |
|---|---|---|
| How trees are built | independently, in parallel | one after another, each depending on the last |
| Each tree is | deep: low bias, high variance | shallow: **high bias**, low variance |
| What the ensemble does | **averages out variance** | **adds up corrections to remove bias** |
| Bias of the ensemble | same as one tree | falls as rounds are added |
| Variance of the ensemble | falls, with a floor $\rho\sigma^2$ | can rise if you run too many rounds or use big steps |
| Main risk | too few features per split or too few trees | overfitting from too many rounds |
| Parallel training | easy | hard, because each tree needs the previous residuals |

:::box plain | The contrast in plain words
- A forest begins with **experts who are each too jumpy** and calms them by averaging.
- Boosting begins with **beginners who are each too simple** and makes them capable by stacking corrections.
- That is why one wants **deep** trees for bagging and **shallow** trees for boosting.
:::

## 10. Bridge cheat-sheet: from weights to functions

| Gradient descent for linear regression (Chapter 2) | Gradient boosting |
|---|---|
| Thing being adjusted: weights $\mathbf{w}$ | Thing being adjusted: the predictions $F(\mathbf{x}_i)$ |
| Rule: [[eq-gd-w]] | Rule: [[eq-fspace]] |
| Direction: $-\nabla J$ | Direction: pseudo-residuals, [[eq-pseudo]] |
| Step size $\eta$ | Learning rate $\eta$ |
| Squared loss gives $\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w})$ | Squared loss gives residuals, [[eq-resid-gb]] |
| Update uses the residual through $\mathbf{X}^\top$ | Update fits a tree to the residual, [[eq-fit]] |
| Stops after enough steps | Stops after $T$ rounds, [[eq-final]] |

For squared loss the two sides use the **same residual** $\mathbf{y}-\hat{\mathbf{y}}$. Linear regression turns it into a weight update through $\mathbf{X}^\top$, while boosting turns it into a new tree.

## 11. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [AdaBoost](chapter.html?c=10-adaboost) | It is gradient boosting with the exponential loss. The last row of the table in Section 7 is its starting point. |
| Machine Learning | [XGBoost](chapter.html?c=11-xgboost) | It replaces the slope-only step of [[eq-pseudo]] with a slope-and-curvature step (Newton) and adds an explicit penalty on the trees. |
| Machine Learning | LightGBM, CatBoost, histogram-based boosting | Engineering variations of the same loop in Section 4, made faster or better with categorical features. |
| Machine Learning | Learning to rank, quantile regression, survival analysis | You only supply a different loss; the same recipe gives its pseudo-residuals. |
| Optimization | Functional gradient descent, coordinate descent, mirror descent | Section 4 is the textbook example of descent in a space of functions. |
| Probability & Statistics | Additive models, generalized additive models | [[eq-final]] is an additive model built greedily from small pieces. |
| Applications | Tabular data in finance, ad click prediction, search ranking, fraud detection | Boosted trees are often the strongest model for data in tables, and are frequent winners in modelling competitions. |
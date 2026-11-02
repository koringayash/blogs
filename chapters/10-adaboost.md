# AdaBoost

*Part IV · Trees & Ensembles. The first boosting algorithm that worked in practice. It is gradient boosting (Chapter 9) with one particular loss, and for that loss every formula comes out in closed form.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Gradient Boosting (Chapter 9)](chapter.html?c=09-gradient-boosting) | AdaBoost is shown to be a special case of its recipe. |
| Machine Learning | [Decision Trees (Chapter 7)](chapter.html?c=07-decision-trees) | The weak learner is a **stump**: a tree with a single split. |
| Machine Learning | [Support Vector Machines (Chapter 6)](chapter.html?c=06-svm) | Labels are $\pm1$ and the margin $y\,f(\mathbf{x})$ is used again. |
| Calculus & Optimization | Exponentials, logarithms, setting a derivative to zero | The optimal vote strength comes from one derivative. |
| Probability & Statistics | Weighted averages | The "weighted error" of a stump is a weighted average. |

## 1. Everyday picture: a study group that focuses on what it got wrong

Imagine a student preparing for an exam with a pile of practice questions.

1. **First pass.** The student tries every question once and gets some wrong.
2. **Second pass.** They put extra effort into the questions they got wrong, and spend less time on those they already know.
3. **Third pass.** Again, whatever is still wrong gets more attention.
4. At the end, the student trusts a **method more if it worked well** on the questions that mattered.

AdaBoost ("Adaptive Boosting") is exactly this:
- The **practice questions** are the training points.
- The **attention** given to a question is its **weight**.
- Each **pass** trains one very simple classifier (a stump) with those weights.
- The final answer is a **vote** in which better classifiers get a bigger say.

The chapter answers three questions in turn: how to measure "how wrong" (the loss), how strong each vote should be, and how the weights should change.

## 2. Setup and the exponential loss

**Step 1: labels and the weak learner.** For two classes use $y\in\{-1,+1\}$. Each weak learner $h(\mathbf{x})$ outputs $-1$ or $+1$. A **stump** asks a single question such as "is feature 3 below 0.7?".

**Step 2: the model is a weighted vote.**

$$ F(\mathbf{x})=\sum_{t=1}^{T}\alpha_t\,h_t(\mathbf{x}),\qquad\hat y(\mathbf{x})=\operatorname{sign}\big(F(\mathbf{x})\big) $$ {#eq-ada-model}

Each $h_t$ is a weak learner and each $\alpha_t$ is its **vote strength**, a number we will compute. This is the additive form of [Chapter 9](chapter.html?c=09-gradient-boosting#eq-final), with the learning rate folded into the $\alpha_t$.

**Step 3: the margin.** For a point $(\mathbf{x},y)$, the product $y\,F(\mathbf{x})$ is its **margin**:
- positive means the vote is **correct**,
- negative means the vote is **wrong**,
- large means it is correct with **high confidence**.

**Step 4: the loss.** AdaBoost uses the **exponential loss**:

$$ \ell\big(y,F(\mathbf{x})\big)=e^{-y\,F(\mathbf{x})} $$ {#eq-exp-loss}

:::box plain | What is this loss saying in plain words?
- **Margin large and positive** (correct and confident): $e^{-yF}$ is tiny, close to $0$.
- **Margin near $0$** (a coin flip): the loss is about $1$.
- **Margin large and negative** (confidently wrong): the loss becomes **huge**, because the exponential grows very fast.
- So confident mistakes are punished extremely hard, much harder than in the logistic or hinge loss (Section 8).
:::

**Why not just count mistakes?** The mistake count, $\mathbf{1}[y\ne\operatorname{sign}F]$, jumps from $0$ to $1$ and has zero slope almost everywhere, so it cannot guide an optimizer. The exponential loss is **smooth, convex and differentiable everywhere**, and it always lies **above** the mistake count:

$$ \mathbf{1}\big[y\ne\operatorname{sign}(F)\big]\;\le\;e^{-yF} $$ {#eq-upper}

:::box step | How we got here
- When the vote is correct, the left side is $0$, and the right side is positive, so the inequality holds.
- When the vote is wrong, $yF\le0$, so $e^{-yF}\ge e^{0}=1$, and the left side is exactly $1$.
- So lowering the exponential loss pushes down an upper bound on the number of mistakes. Section 6 uses this.
:::

:::figure F10.4 | Loss functions compared | Four curves against the margin y F(x): the 0-1 loss, the exponential loss, the logistic loss, and the hinge loss. A draggable marker shows how each loss scores the same margin, and the exponential curve visibly shoots up for negative margins.

## 3. Adding one learner at a time

We cannot optimize all the $\alpha_t$ and $h_t$ at once, so we proceed in rounds. This is **stagewise** fitting, as in Chapter 9.

**Step 1: the model so far.** After $t-1$ rounds we have

$$ F_{t-1}(\mathbf{x})=\sum_{s=1}^{t-1}\alpha_s\,h_s(\mathbf{x}) $$ {#eq-so-far}

**Step 2: add one more learner with strength $\alpha$.** Choose $(\alpha_t,h_t)$ to minimize the total exponential loss:

$$ (\alpha_t,h_t)=\arg\min_{\alpha,\,h}\;\sum_{i=1}^{n}\exp\!\Big(-y_i\big[F_{t-1}(\mathbf{x}_i)+\alpha\,h(\mathbf{x}_i)\big]\Big) $$ {#eq-stage}

**Step 3: split the exponential into a product.** Since $e^{a+b}=e^ae^b$,

$$ \sum_{i}\exp\!\big(-y_iF_{t-1}(\mathbf{x}_i)\big)\cdot\exp\!\big(-\alpha\,y_i\,h(\mathbf{x}_i)\big) $$ {#eq-split-exp}

**Step 4: name the first factor, the weight of point $i$.**

$$ w_i^{(t)}=\exp\!\big(-y_i\,F_{t-1}(\mathbf{x}_i)\big) $$ {#eq-weights}

Then [[eq-stage]] becomes

$$ (\alpha_t,h_t)=\arg\min_{\alpha,\,h}\;\sum_{i=1}^{n}w_i^{(t)}\,\exp\!\big(-\alpha\,y_i\,h(\mathbf{x}_i)\big) $$ {#eq-weighted-obj}

:::box step | How we got here
- The first factor in [[eq-split-exp]] does not depend on the new learner $(\alpha,h)$. It depends only on the model so far, so it is a **fixed number for each point**. We call it $w_i^{(t)}$.
- **What the weight means:** it is the exponential loss that the current ensemble already suffers on point $i$. A point the ensemble gets wrong with confidence has a **large** weight. A point it gets right with confidence has a **small** weight.
- Therefore the weights are not an extra idea bolted on. They fall out of the loss.
:::

**Step 5: the weights at the start.** Before any learner exists, $F_0=0$, so every weight is $e^{0}=1$. All points start equally important.

## 4. The best learner and the best vote strength

Fix a candidate stump $h$ with outputs $\pm1$. For each point, $y_ih(\mathbf{x}_i)$ is $+1$ if the stump is right and $-1$ if it is wrong. Split the sum in [[eq-weighted-obj]] by whether the stump is right:

$$ \sum_iw_i^{(t)}e^{-\alpha y_ih(\mathbf{x}_i)}=e^{-\alpha}\sum_{i:\ \text{correct}}w_i^{(t)}+e^{\alpha}\sum_{i:\ \text{wrong}}w_i^{(t)} $$ {#eq-split-cw}

Let $W_t=\sum_iw_i^{(t)}$ be the total weight, and define the **weighted error** of the stump:

$$ \text{err}_t=\frac{\sum_{i:\ h(\mathbf{x}_i)\ne y_i}w_i^{(t)}}{W_t} $$ {#eq-err}

Then the sum of the correct weights is $W_t(1-\text{err}_t)$ and the sum of the wrong weights is $W_t\,\text{err}_t$, so [[eq-split-cw]] equals

$$ W_t\Big[e^{-\alpha}\,(1-\text{err}_t)+e^{\alpha}\,\text{err}_t\Big] $$ {#eq-obj-err}

### Choosing the stump

Rewrite the bracket in [[eq-obj-err]] as $e^{-\alpha}+\text{err}_t\,(e^{\alpha}-e^{-\alpha})$. For any positive $\alpha$ the factor $e^{\alpha}-e^{-\alpha}$ is positive, so **the objective gets smaller when $\text{err}_t$ gets smaller**. Hence:

> **The best stump is the one with the smallest weighted error $\text{err}_t$.**

This is the usual weighted training of a classifier: points with large weight count more when choosing the split.

### Choosing the vote strength $\alpha$

**Step 1: differentiate [[eq-obj-err]] with respect to $\alpha$ and set it to zero.**

$$ -e^{-\alpha}(1-\text{err}_t)+e^{\alpha}\,\text{err}_t=0 $$ {#eq-dalpha}

**Step 2: solve.** Multiply by $e^{\alpha}$ and rearrange: $e^{2\alpha}\,\text{err}_t=1-\text{err}_t$, so $e^{2\alpha}=\dfrac{1-\text{err}_t}{\text{err}_t}$. Taking logarithms:

$$ \alpha_t=\frac12\ln\!\Big(\frac{1-\text{err}_t}{\text{err}_t}\Big) $$ {#eq-alpha}

:::box step | How we got here
- Differentiate $e^{-\alpha}(1-\text{err}_t)$ to get $-e^{-\alpha}(1-\text{err}_t)$, and $e^{\alpha}\text{err}_t$ to get $e^{\alpha}\text{err}_t$. This gives [[eq-dalpha]].
- The second derivative of [[eq-obj-err]] is $e^{-\alpha}(1-\text{err}_t)+e^{\alpha}\text{err}_t>0$, so this stationary point is a **minimum**.
- We assumed $\alpha>0$ while choosing the stump. [[eq-alpha]] is positive exactly when $\text{err}_t<\tfrac12$, which is the case for any stump that beats a coin flip.
:::

:::box plain | What is this equation saying in plain words?
- The fraction $\dfrac{1-\text{err}}{\text{err}}$ compares the weight of **right** answers with the weight of **wrong** answers. It is an "odds of being right".
- The logarithm turns those odds into a **vote strength**, and the $\tfrac12$ is a convention that falls out of the algebra.
- A learner that is almost always right gets a **huge** vote. A learner that is barely better than guessing gets a **tiny** vote.
:::

| Weighted error $\text{err}_t$ | Vote strength $\alpha_t$ | Meaning |
|---|---|---|
| $0.1$ | $\tfrac12\ln9=1.099$ | strong learner, large vote |
| $0.3$ | $\tfrac12\ln\tfrac{7}{3}=0.424$ | modest vote |
| $0.5$ | $0$ | no better than a coin flip, no vote |
| $0.7$ | $-0.424$ | worse than a coin flip, so **flip its answer** |
| $\to0$ | $\to+\infty$ | perfect on the weighted data. In practice cap $\alpha$. |

:::figure F10.2 | Vote strength alpha vs error | The curve of alpha against the weighted error. A slider on err marks the point on the curve and shows the vote strength as a bar. The curve crosses zero at err equal to one half and goes negative beyond it.

**The value of the objective at the optimum.** Put $e^{\alpha_t}=\sqrt{(1-\text{err}_t)/\text{err}_t}$ into [[eq-obj-err]]:
- $e^{-\alpha_t}(1-\text{err}_t)=\sqrt{\dfrac{\text{err}_t}{1-\text{err}_t}}\,(1-\text{err}_t)=\sqrt{\text{err}_t(1-\text{err}_t)}$,
- $e^{\alpha_t}\,\text{err}_t=\sqrt{\dfrac{1-\text{err}_t}{\text{err}_t}}\;\text{err}_t=\sqrt{\text{err}_t(1-\text{err}_t)}$.

The two terms are equal, so the minimum value is

$$ \min_{\alpha}\sum_iw_i^{(t)}e^{-\alpha y_ih(\mathbf{x}_i)}=2\,W_t\sqrt{\text{err}_t\,(1-\text{err}_t)} $$ {#eq-min-value}

This neat result is the key to the bound in Section 6.

## 5. Updating the weights

After round $t$ the model becomes $F_t=F_{t-1}+\alpha_th_t$. By the definition [[eq-weights]], the next weights are

$$ w_i^{(t+1)}=e^{-y_iF_t(\mathbf{x}_i)}=e^{-y_iF_{t-1}(\mathbf{x}_i)}\cdot e^{-\alpha_ty_ih_t(\mathbf{x}_i)}=w_i^{(t)}\,e^{-\alpha_t\,y_i\,h_t(\mathbf{x}_i)} $$ {#eq-update}

Since $y_ih_t(\mathbf{x}_i)$ is $+1$ for a correct point and $-1$ for a wrong one:

$$ w_i^{(t+1)}=\begin{cases}w_i^{(t)}\,e^{-\alpha_t}&\text{if }h_t(\mathbf{x}_i)=y_i\ \text{(correct: weight shrinks)}\\[2pt]w_i^{(t)}\,e^{+\alpha_t}&\text{if }h_t(\mathbf{x}_i)\ne y_i\ \text{(wrong: weight grows)}\end{cases} $$ {#eq-update-cases}

:::box plain | What is this saying in plain words?
- **Points the new stump got right** are multiplied by $e^{-\alpha_t}<1$. They matter less from now on.
- **Points the new stump got wrong** are multiplied by $e^{\alpha_t}>1$. They matter more.
- The better the stump (larger $\alpha_t$), the stronger the shift of attention towards its mistakes.
- In practice the weights are **renormalized** to sum to $1$ after each round. This changes nothing about which stump or vote is best, because [[eq-err]] already divides by $W_t$.
:::

### A worked example by hand

Ten points, all with weight $0.1$ (normalized). The first stump misclassifies $3$ of them, so $\text{err}_1=0.3$.

**Vote strength.** $\alpha_1=\tfrac12\ln\tfrac{0.7}{0.3}=\tfrac12\ln2.333=0.424$.

**Update.** The multiplication factors are $e^{-\alpha_1}=\sqrt{0.3/0.7}=0.655$ for correct points and $e^{\alpha_1}=\sqrt{0.7/0.3}=1.528$ for wrong ones.

| Group | Number of points | Weight each, before | After the update | Total, after |
|---|---|---|---|---|
| Correct | 7 | $0.1$ | $0.1\times0.655=0.0655$ | $0.4583$ |
| Wrong | 3 | $0.1$ | $0.1\times1.528=0.1528$ | $0.4583$ |

The new total is $0.9165$. After dividing by it, each correct point has weight $\tfrac1{14}=0.0714$ and each wrong point has weight $\tfrac16=0.1667$.

**A remarkable fact is visible here:** the correct group and the wrong group each carry **exactly half** of the total weight. In general,

$$ \sum_{i:\ \text{correct}}w_i^{(t+1)}=e^{-\alpha_t}W_t(1-\text{err}_t)=W_t\sqrt{\text{err}_t(1-\text{err}_t)}=e^{\alpha_t}W_t\,\text{err}_t=\sum_{i:\ \text{wrong}}w_i^{(t+1)} $$ {#eq-half}

using the two equal terms found just before [[eq-min-value]].

:::box plain | What does this mean in plain words?
- Under the **new** weights, the stump we just trained has weighted error **exactly $\tfrac12$**: no better than a coin flip.
- So the next round cannot reuse the same stump. It is forced to find a **different** question that is good on the points the previous stump got wrong.
- This is how AdaBoost keeps its learners complementary.
:::

:::figure F10.1 | Sample weights round by round | Two-class points drawn with radius proportional to their weight. A button runs one round: the current stump is drawn, misclassified points grow, correct points shrink, and a readout shows err, alpha, and the two half-weights from the table above.

## 6. Why AdaBoost works: the training-error bound

**The claim.** If every weak learner is only slightly better than random, the training error of the whole ensemble falls **exponentially fast** in the number of rounds.

**Step 1: the mistake count is bounded by the exponential loss.** Take [[eq-upper]] at the final model $F_T$ and average over the $n$ training points:

$$ \text{TrainErr}(F_T)=\frac1n\sum_{i}\mathbf{1}\big[y_i\ne\operatorname{sign}F_T(\mathbf{x}_i)\big]\;\le\;\frac1n\sum_ie^{-y_iF_T(\mathbf{x}_i)} $$ {#eq-bound-1}

**Step 2: the right side is a total weight.** By the definition [[eq-weights]] at round $T+1$, the sum on the right is $W_{T+1}=\sum_iw_i^{(T+1)}$. So

$$ \text{TrainErr}(F_T)\le\frac{W_{T+1}}{n} $$ {#eq-bound-2}

**Step 3: how does the total weight change each round?** Sum the cases of [[eq-update-cases]]:

$$ W_{t+1}=e^{-\alpha_t}W_t(1-\text{err}_t)+e^{\alpha_t}W_t\,\text{err}_t=2\,W_t\sqrt{\text{err}_t(1-\text{err}_t)} $$ {#eq-w-recursion}

This is the same quantity as [[eq-min-value]]: the minimum value of the objective at round $t$ equals the total weight at round $t+1$.

**Step 4: unroll the recursion from $W_1=n$.** At the start all weights equal $1$, so $W_1=n$. Applying [[eq-w-recursion]] $T$ times:

$$ W_{T+1}=n\prod_{t=1}^{T}2\sqrt{\text{err}_t(1-\text{err}_t)} $$ {#eq-w-product}

**Step 5: combine.** Put [[eq-w-product]] into [[eq-bound-2]]:

$$ \text{TrainErr}(F_T)\le\prod_{t=1}^{T}2\sqrt{\text{err}_t\,(1-\text{err}_t)} $$ {#eq-bound-prod}

**Step 6: rewrite with the "edge".** Write $\text{err}_t=\tfrac12-\gamma_t$, where $\gamma_t>0$ is how much better than a coin flip the stump is. Then

$$ 4\,\text{err}_t(1-\text{err}_t)=4\Big(\tfrac12-\gamma_t\Big)\Big(\tfrac12+\gamma_t\Big)=1-4\gamma_t^2 $$

so each factor is $2\sqrt{\text{err}_t(1-\text{err}_t)}=\sqrt{1-4\gamma_t^2}$, and

$$ \text{TrainErr}(F_T)\le\prod_{t=1}^{T}\sqrt{1-4\gamma_t^2}\;\le\;\exp\!\Big(-2\sum_{t=1}^{T}\gamma_t^2\Big) $$ {#eq-bound-gamma}

:::box step | How we got here
- The first inequality in [[eq-bound-gamma]] is [[eq-bound-prod]] with the algebra above.
- For the second, use $1-x\le e^{-x}$ for every $x$. Then $\sqrt{1-4\gamma^2}\le\sqrt{e^{-4\gamma^2}}=e^{-2\gamma^2}$, and a product of exponentials is the exponential of the sum.
:::

:::box plain | What is this bound saying in plain words?
- Every round multiplies the bound by a factor $\sqrt{1-4\gamma_t^2}$, which is **strictly less than $1$** as soon as the stump has any edge at all.
- Even tiny edges add up: with a constant edge $\gamma$, the bound is at most $e^{-2T\gamma^2}$, which **decays exponentially in $T$**.
- AdaBoost does not need strong learners. It turns many learners that are **barely better than guessing** into one that is very accurate on the training set.
:::

**A numerical example.** Let each stump have edge $\gamma=0.1$, that is, error $0.4$.

| Number of rounds $T$ | Product bound $(\sqrt{1-0.04})^T$ | Simplified bound $e^{-0.02\,T}$ |
|---|---|---|
| $10$ | $0.816$ | $0.819$ |
| $100$ | $0.130$ | $0.135$ |
| $346$ | $0.00085$ | $0.00099$ |

With $n=1000$ training points, the training error is a multiple of $\tfrac1{1000}$. A bound below $0.001$ therefore means **zero training mistakes**, which this bound guarantees after $346$ rounds.

:::box warn | What the bound does and does not say
- It controls the **training** error only. It says nothing directly about the **test** error. Boosting for too many rounds, or on noisy labels, can still overfit.
- It needs every $\text{err}_t<\tfrac12$ (a positive edge). If a round finds no stump better than chance, the edge is $0$ and the bound stops improving.
- With noisy labels the exponential loss puts **huge weight** on mislabelled points, so AdaBoost can become unstable. Section 8 explains why.
:::

:::figure F10.5 | Training-error bound | A plot of the actual training error and the product bound against the number of rounds, as AdaBoost runs on a small data set. A slider on the edge gamma shows how the bound falls faster for larger edges.

## 7. AdaBoost is gradient boosting with the exponential loss

[Chapter 9](chapter.html?c=09-gradient-boosting#eq-pseudo) fits each new tree to the **negative gradient** of the loss. Let us compute that gradient for AdaBoost's loss.

**Step 1: differentiate the exponential loss.**

$$ \frac{\partial\,e^{-yF}}{\partial F}=-y\,e^{-yF} $$ {#eq-ada-deriv}

**Step 2: the pseudo-residual at round $t$.** Evaluate the negative of [[eq-ada-deriv]] at $F_{t-1}$ and use [[eq-weights]]:

$$ r_{it}=y_i\,e^{-y_iF_{t-1}(\mathbf{x}_i)}=y_i\,w_i^{(t)} $$ {#eq-ada-resid}

So the direction we should move at point $i$ is its label $y_i$, **scaled by its weight**.

**Step 3: how does choosing the stump relate to this direction?** A natural way to pick $h$ is to make it **align** with the negative gradient, that is, to maximize $\sum_ir_{it}\,h(\mathbf{x}_i)$. With $h=\pm1$ and the definition [[eq-err]]:

$$ \sum_iw_i^{(t)}\,y_i\,h(\mathbf{x}_i)=\sum_{\text{correct}}w_i^{(t)}-\sum_{\text{wrong}}w_i^{(t)}=W_t\,(1-2\,\text{err}_t) $$ {#eq-align}

:::box step | How we got here
- A correct point contributes $+w_i$ (because $y_ih=+1$), and a wrong point contributes $-w_i$.
- The correct weights total $W_t(1-\text{err}_t)$ and the wrong weights total $W_t\,\text{err}_t$. Subtract.
- The expression is largest exactly when $\text{err}_t$ is smallest. **Picking the stump most aligned with the negative gradient is the same as picking the stump with the smallest weighted error**, which is what Section 4 derived directly.
:::

**What differs from Chapter 9:**

| | Gradient boosting (Chapter 9) | AdaBoost |
|---|---|---|
| Loss | any differentiable loss | exponential loss, [[eq-exp-loss]] |
| Weak learner $h_t$ | a regression tree fitted by least squares | a classifier with outputs $\pm1$ (a stump) |
| Direction | pseudo-residual $r_{it}$ | $r_{it}=y_iw_i^{(t)}$, [[eq-ada-resid]] |
| Step size | a fixed learning rate $\eta$ | **chosen exactly** by minimizing the loss along the step: $\alpha_t$ in [[eq-alpha]] |
| Final model | $F_0+\eta\sum h_t$ | $\operatorname{sign}\big(\sum\alpha_th_t\big)$ |

:::box plain | The whole story in plain words
- AdaBoost is not a separate idea from gradient boosting. It is **gradient boosting for a classification problem, with the exponential loss and a perfectly tuned step size**.
- The "weights on the data points" that make AdaBoost look unusual are just the **size of the exponential loss on each point**, which is also the size of the gradient.
- The choice of loss is the only thing that changes the algorithm. Chapter 9's last table already listed this row.
:::

## 8. The exponential loss compared with the others

All the losses below are functions of the margin $m=y\,F(\mathbf{x})$.

| Loss | Formula | Behaviour for $m\to-\infty$ (confident mistakes) | Behaviour for $m\to+\infty$ | Used by |
|---|---|---|---|---|
| 0-1 loss | $\mathbf{1}[m\le0]$ | constant $1$ | $0$ | what we really want to minimize |
| Exponential | $e^{-m}$ | **grows exponentially** | tends to $0$ | AdaBoost |
| Logistic | $\log(1+e^{-m})$ | grows **linearly** | tends to $0$ | [Logistic regression](chapter.html?c=05-logistic-regression#eq-bce), gradient boosting with logistic loss |
| Hinge | $\max(0,1-m)$ | grows linearly | exactly $0$ for $m\ge1$ | [SVM](chapter.html?c=06-svm#eq-hinge-obj) |

**Consequences of the exponential growth:**
1. A single mislabelled or very unusual point gets an enormous weight $e^{-yF}$, and later rounds spend their effort on it.
2. So AdaBoost is **sensitive to label noise and outliers**. Logistic-loss boosting is more forgiving because its penalty grows only linearly.
3. On clean data, the exponential loss's aggressive focus on hard points is exactly what produces the fast training-error decay of Section 6.

## 9. Bias, variance and practical notes

1. **AdaBoost mainly reduces bias.** A stump is a very simple model (high bias, low variance). Adding weighted stumps makes the ensemble progressively more flexible, as in [Chapter 9's comparison with bagging](chapter.html?c=09-gradient-boosting#eq-final).
2. **It can overfit**, especially with noisy labels or very many rounds. Choose the number of rounds with a validation set, and optionally add a shrinkage factor $\eta<1$ in front of each $\alpha_t$.
3. **Weak learners other than stumps** work too (small trees). Deeper trees capture interactions between features, but are less "weak".
4. **Handling the degenerate cases.** If $\text{err}_t=0$, $\alpha_t$ would be infinite, so implementations cap it or stop. If $\text{err}_t\ge\tfrac12$, no useful stump exists, so training stops.
5. **Sampling version.** Instead of passing weights to the learner, one can draw a resampled data set with probabilities proportional to the weights. This is useful for learners that cannot accept weights.
6. **More than two classes.** There are multi-class extensions such as SAMME, which keep the same weight-update idea.

## 10. Bridge cheat-sheet

| Question | Answer | Where |
|---|---|---|
| What is the final model? | A weighted vote of $\pm1$ learners | [[eq-ada-model]] |
| What loss does it minimize? | $e^{-yF}$, stagewise | [[eq-exp-loss]], [[eq-stage]] |
| Where do the weights come from? | The loss already suffered by the current ensemble | [[eq-weights]] |
| Which stump is chosen? | The one with the smallest weighted error | [[eq-err]] |
| How strong is its vote? | $\tfrac12\ln\frac{1-\text{err}}{\text{err}}$ | [[eq-alpha]] |
| How do weights change? | Wrong points $\times e^{\alpha}$, right points $\times e^{-\alpha}$ | [[eq-update-cases]] |
| Why can't it reuse the stump? | Under new weights that stump has error exactly $\tfrac12$ | [[eq-half]] |
| What does it guarantee? | Training error at most $\exp(-2\sum\gamma_t^2)$ | [[eq-bound-gamma]] |
| How is it related to Chapter 9? | Gradient boosting with exponential loss and exact step size | [[eq-ada-resid]] |

## 11. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [XGBoost](chapter.html?c=11-xgboost) | The next chapter keeps the loop of this chapter (compute slope, fit a tree) and adds the curvature of the loss and an explicit penalty on the trees. |
| Machine Learning | LogitBoost, GentleBoost, cost-sensitive boosting | Variants that swap the exponential loss for a gentler or weighted one, using the recipe of [[eq-ada-resid]]. |
| Machine Learning | Margin theory, generalization of boosting | The quantity $y\,F(\mathbf{x})$ and the exponential loss are the starting point of margin-based explanations of why boosting often resists overfitting. |
| Probability & Statistics | Additive logistic regression | The population minimizer of the exponential loss is half the log-odds, which links AdaBoost to the log-odds model of [Chapter 5](chapter.html?c=05-logistic-regression#eq-logit). |
| Optimization | Coordinate descent, mirror descent, online learning | Multiplying weights by $e^{\mp\alpha}$ is the classic **multiplicative-weights update** used throughout online learning. |
| Computer Vision | Viola-Jones face detector | A famous real-world system built on AdaBoost with very cheap stump-like features. |
| Applications | Spam filtering, text categorization, ranking | Boosted stumps give strong, fast classifiers when labels are reasonably clean. |
# Bagging & Random Forest

*Part IV · Trees & Ensembles. Chapter 7 ended with a problem: one deep tree has almost no bias but a lot of variance. This chapter fixes the variance by averaging many trees.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Decision Trees (Chapter 7)](chapter.html?c=07-decision-trees) | The base model. We need to know why a deep tree has high variance. |
| Machine Learning | [Bias and variance (Chapter 4)](chapter.html?c=04-bias-variance) | The goal is to cut the variance term of the decomposition. |
| Probability & Statistics | Variance and covariance of a sum | The whole derivation is one calculation with these. |
| Probability & Statistics | Correlation coefficient | It measures how similar two trees' predictions are. |
| Probability & Statistics | Sampling with replacement | This is what a bootstrap sample is. |

## 1. Everyday picture: ask a crowd, not one expert

Imagine guessing the weight of an ox at a fair.

1. One person's guess is noisy: sometimes too high, sometimes too low.
2. Ask a hundred people and take the **average**. The errors point in different directions, so they largely **cancel**.
3. But there is a catch. If everyone first looked at the same newspaper article and copied its number, the errors would all point the **same way**, and averaging a hundred copies would help almost nothing.

So the crowd helps only as much as the people **think differently from each other**. A random forest is built on exactly this idea:
- the "people" are decision trees,
- we **average** their predictions,
- and we go out of our way to make the trees **different from each other**.

## 2. Bagging: train each tree on a different resample

**Step 1: the problem.** We have only one training set of $n$ points, but we want many different trees. We cannot go and collect new data sets, so we **resample** the one we have.

**Step 2: the bootstrap sample.** To build one bootstrap sample:
1. Draw one point at random from the $n$ training points and write it down.
2. **Put it back**, so it can be drawn again.
3. Repeat $n$ times.

The result is a data set of size $n$ in which some points appear **twice or more** and some do **not appear at all**.

**Step 3: how many points are left out?** Look at one particular training point.
- In a single draw it is **not** chosen with probability $1-\tfrac1n$.
- The $n$ draws are independent, so it is never chosen with probability

$$ \Big(1-\frac1n\Big)^{n}\;\longrightarrow\;e^{-1}\approx0.368\qquad(n\to\infty) $$ {#eq-oob-prob}

So each bootstrap sample contains about $63.2\%$ of the distinct training points, and leaves out about $36.8\%$.

**Step 4: bagging ("bootstrap aggregating").** Repeat Steps 2 and 3 to make $B$ bootstrap samples, train one tree $T_b$ on each, and average:

$$ \hat f(\mathbf{x})=\frac1B\sum_{b=1}^{B}T_b(\mathbf{x}) $$ {#eq-forest}

For classification, the trees vote, or we average their class probabilities.

:::box plain | What is this equation saying in plain words?
- Each tree $T_b(\mathbf{x})$ gives its own prediction at the point $\mathbf{x}$.
- The forest's prediction is simply the **average** of those $B$ predictions.
- Every tree saw a slightly different version of the data, so each makes slightly different mistakes, and the average smooths them out.
:::

:::figure F8.1 | Bootstrap sampling | A row of numbered training points. A button draws a bootstrap sample: repeated points are drawn thicker, points left out are greyed, and a counter shows the share of distinct points settling near 63 percent as n grows.

:::figure F8.2 | Averaging trees | A noisy curve with training points. Many trees are fitted to bootstrap samples and drawn as faint step functions, with their average in bold. A slider for B shows the bold curve becoming smoother and more stable.

## 3. How much does averaging help? The variance formula

This is the central calculation of the chapter. We keep the test point $\mathbf{x}$ fixed and ask: **what is the variance of the forest's prediction $\hat f(\mathbf{x})$?** The randomness comes from the training data and from the random choices made while building the trees.

**Step 1: what we assume about the trees.**
1. Every tree has the **same variance**: $\operatorname{Var}\big(T_b(\mathbf{x})\big)=\sigma^2$.
2. The trees are **not independent**, because they are trained on overlapping resamples of the same data.
3. Any two different trees have the same correlation $\rho$, with $0\le\rho\le1$. Since covariance equals correlation times the two standard deviations,

$$ \operatorname{Cov}\big(T_b,T_{b'}\big)=\rho\,\sigma\cdot\sigma=\rho\,\sigma^2\qquad(b\ne b') $$ {#eq-cov}

(This "same $\sigma^2$ and same $\rho$ for every pair" is a simplification. It is a good mental model, and the conclusions below do not depend on its details.)

**Step 2: the variance of an average.** Pull the constant $\tfrac1B$ out of [[eq-forest]]. A constant comes out of a variance **squared**:

$$ \operatorname{Var}\big(\hat f(\mathbf{x})\big)=\frac1{B^2}\operatorname{Var}\Big(\sum_{b=1}^{B}T_b(\mathbf{x})\Big) $$ {#eq-var-avg}

**Step 3: the variance of a sum.** The variance of a sum is the sum of all the variances **plus the sum of all the pairwise covariances**:

$$ \operatorname{Var}\Big(\sum_{b=1}^{B}T_b\Big)=\sum_{b=1}^{B}\operatorname{Var}(T_b)+\sum_{b\ne b'}\operatorname{Cov}(T_b,T_{b'}) $$ {#eq-var-sum}

**Step 4: count the terms and plug in.**
- There are $B$ variance terms, each equal to $\sigma^2$.
- There are $B(B-1)$ ordered pairs $(b,b')$ with $b\ne b'$, each with covariance $\rho\sigma^2$ by [[eq-cov]].

$$ \operatorname{Var}\Big(\sum_{b=1}^{B}T_b\Big)=B\sigma^2+B(B-1)\,\rho\,\sigma^2 $$ {#eq-sum-count}

**Step 5: divide by $B^2$ and simplify.** Put [[eq-sum-count]] into [[eq-var-avg]]:

$$ \operatorname{Var}\big(\hat f(\mathbf{x})\big)=\frac{B\sigma^2+B(B-1)\rho\sigma^2}{B^2}=\frac{\sigma^2}{B}+\frac{(B-1)}{B}\rho\sigma^2 $$ {#eq-intermediate}

Write $\dfrac{B-1}{B}=1-\dfrac1B$ and collect the $\rho\sigma^2$ terms:

$$ \operatorname{Var}\big(\hat f(\mathbf{x})\big)=\sigma^2\Big[\rho+\frac{1-\rho}{B}\Big] $$ {#eq-var-final}

:::box step | How we got here
- Start from [[eq-intermediate]]: $\dfrac{\sigma^2}{B}+\rho\sigma^2-\dfrac{\rho\sigma^2}{B}$.
- Group the two terms that contain $\tfrac1B$: $\dfrac{\sigma^2(1-\rho)}{B}$.
- What is left is $\rho\sigma^2$. Together they give [[eq-var-final]].
:::

:::box plain | What is this equation saying in plain words?
- **The term $\dfrac{1-\rho}{B}$:** shrinks as you add trees. This is the part of the variance that comes from the trees' **independent** quirks, and averaging removes it.
- **The term $\rho$:** does **not** depend on $B$ at all. This is the part the trees have **in common**. They all make the same mistakes together, and averaging cannot cancel a mistake that everyone shares.
- **So the variance never drops below $\rho\sigma^2$, however many trees you add.** That floor is the point of the whole chapter.
:::

**Four sanity checks:**

| Situation | What [[eq-var-final]] gives | Meaning |
|---|---|---|
| $B=1$ | $\sigma^2$ | One tree, no averaging |
| $\rho=0$ (independent trees) | $\sigma^2/B$ | The classic "average of $B$ independent numbers" |
| $\rho=1$ (identical trees) | $\sigma^2$ | A hundred copies of the same tree: no gain at all |
| $B\to\infty$ | $\rho\sigma^2$ | The floor |

**A numerical example.** Let $\sigma^2=1$.

| $\rho$ | $B=1$ | $B=10$ | $B=100$ | $B=\infty$ |
|---|---|---|---|---|
| $0.9$ | $1.000$ | $0.910$ | $0.901$ | $0.900$ |
| $0.5$ | $1.000$ | $0.550$ | $0.505$ | $0.500$ |
| $0.1$ | $1.000$ | $0.190$ | $0.109$ | $0.100$ |

With strongly correlated trees ($\rho=0.9$), a hundred trees remove only $10\%$ of the variance. With weakly correlated trees ($\rho=0.1$) they remove almost $90\%$. **Decorrelating the trees matters more than adding more of them.**

:::figure F8.3 | Variance vs B | A plot of the variance of the average against the number of trees B. A slider for rho moves the curve, which always falls quickly at first and then flattens at the floor rho times sigma squared, drawn as a dashed line.

:::tabs Three ways to see why a floor exists
--- Shared mistakes
Every tree was trained on resamples of the **same** data set. If that data set happens to contain a misleading cluster of points, every tree gets fooled in the same direction. Averaging over trees cannot undo a mistake that all of them share. It can only cancel their private mistakes.
--- Algebra
In [[eq-var-sum]] the $B$ variance terms get divided by $B^2$, so their total shrinks like $1/B$. But the $B(B-1)$ covariance terms grow like $B^2$, and dividing by $B^2$ leaves a constant. That constant is the floor.
--- Independent copies
Imagine splitting each tree's prediction into a **shared part** (the same for every tree, with variance $\rho\sigma^2$) and a **private part** (independent, with variance $(1-\rho)\sigma^2$). Averaging $B$ trees averages the private parts, dividing their variance by $B$, and leaves the shared part untouched. This gives exactly [[eq-var-final]].
:::

## 4. What averaging does to the bias

Variance is only one half of [Chapter 4's decomposition](chapter.html?c=04-bias-variance#eq-bv). What happens to the bias?

**Step 1: the trees are identically distributed.** Each bootstrap sample is drawn the same way, so every $T_b$ has the same expected value:

$$ \mathbb{E}\big[T_b(\mathbf{x})\big]=\mu(\mathbf{x})\quad\text{for every }b $$ {#eq-same-mean}

**Step 2: the average has that same expectation.** The expected value of an average is the average of the expected values:

$$ \mathbb{E}\big[\hat f(\mathbf{x})\big]=\frac1B\sum_{b=1}^{B}\mathbb{E}\big[T_b(\mathbf{x})\big]=\mu(\mathbf{x})=\mathbb{E}\big[T_b(\mathbf{x})\big] $$ {#eq-same-bias}

:::box step | How we got here
- Insert [[eq-same-mean]] into each term of the sum. All $B$ terms are equal to $\mu(\mathbf{x})$, so they average to $\mu(\mathbf{x})$.
- The **bias** is the gap between this expectation and the truth $f(\mathbf{x})$. Since the expectation is unchanged, **the forest has the same bias as a single tree**.
:::

**What this tells us:**
1. Averaging **cannot lower the bias**. It can only lower the variance.
2. So the base trees should be ones that **already have low bias**, even if they have high variance.
3. Deep trees fit the training data almost perfectly, so they have near-zero bias, as [Chapter 7](chapter.html?c=07-decision-trees) explained. **That is exactly why random forests use deep, unpruned trees**: the bias is already tiny, and the averaging takes care of the variance.

| | One deep tree | Random forest |
|---|---|---|
| Bias | near zero | same as one tree, near zero |
| Variance | high | $\sigma^2[\rho+(1-\rho)/B]$, much lower |
| Test error | high, from the variance | much lower |

:::box warn | Bagging helps most when the base model is unstable
If a model hardly changes when the training set changes (a straight-line regression, for example), its variance $\sigma^2$ is already small, so there is little to remove. Bagging pays off for **unstable, high-variance** models. Trees are the classic example.
:::

## 5. Lowering the correlation: random feature subsets

Bootstrap samples already make the trees different, but often not different enough. [[eq-var-final]] says the remaining enemy is $\rho$.

**Why bagged trees stay correlated.** Suppose one feature is far stronger than the rest.
1. In **every** bootstrap sample, that feature wins the first split, because its gain is the largest.
2. So all the trees start with the same question, and look alike near the top.
3. Similar trees make similar mistakes, so $\rho$ is large and the floor $\rho\sigma^2$ is high.

**The fix.** At **each split** of each tree, do not look at all $d$ features. Look only at a **random subset of $m$ features** (chosen fresh at every split) and pick the best split among those.

| Task | Common default for $m$ |
|---|---|
| Classification | about $\sqrt d$ |
| Regression | about $d/3$ |

**What this does.**
1. A strong feature is **not even offered** at many splits, so other features get to be used.
2. Different trees therefore ask **different questions**, so they are less alike and $\rho$ falls.
3. The price: each individual tree is a little worse, because it is sometimes denied the best split. Its **bias rises slightly**, and its own variance $\sigma^2$ can change.

**The overall trade.** By [[eq-var-final]], a drop in $\rho$ lowers the floor $\rho\sigma^2$ directly. In practice this gain outweighs the small increase in bias. That is the entire difference between **bagged trees** (all $d$ features at each split) and a **random forest** ($m<d$ features at each split).

:::box plain | What are the two sources of randomness in plain words?
- **Bootstrap sampling:** each tree is trained on a different random resample of the *rows* (data points).
- **Feature subsampling:** each split can look at only a random subset of the *columns* (features).
- Both exist for one reason: to make the trees **disagree with each other**, which is what makes averaging effective.
:::

:::figure F8.4 | Feature subsampling | A data set with one dominant feature and several weaker ones. A bar chart shows how often each feature is chosen for the first split across many trees. With m equal to d, the dominant feature takes almost every bar; reducing m spreads the choices out, and a readout shows the measured correlation between trees falling.

## 6. Out-of-bag error: a free test set

**Step 1: every tree has a leftover set.** By [[eq-oob-prob]], each tree leaves out about $36.8\%$ of the training points. These are its **out-of-bag (OOB)** points. The tree never saw them during training.

**Step 2: predict each point using only the trees that did not see it.** For a training point $\mathbf{x}_i$, average the predictions of just those trees for which $\mathbf{x}_i$ was out-of-bag:

$$ \hat f_{\text{oob}}(\mathbf{x}_i)=\frac1{\lvert\mathcal{B}_i\rvert}\sum_{b\in\mathcal{B}_i}T_b(\mathbf{x}_i),\qquad\mathcal{B}_i=\{\,b:\mathbf{x}_i\text{ not in bootstrap sample }b\,\} $$ {#eq-oob}

**Step 3: score it.** Compare $\hat f_{\text{oob}}(\mathbf{x}_i)$ with the true $y_i$ for every $i$ and average the errors.

:::box plain | What is this saying in plain words?
- Each point gets a prediction from trees that **never trained on it**, so the prediction is honest.
- The resulting error is an estimate of **test error**, obtained without setting aside any data and without retraining.
- It plays the same role as the cross-validation of [Chapter 4](chapter.html?c=04-bias-variance), and is built into the algorithm for free.
:::

:::figure F8.5 | Out-of-bag error | A grid with one row per tree and one column per training point, each cell marked in-bag or out-of-bag. A line plot shows the out-of-bag error settling to a stable value as trees are added.

## 7. Practical notes and common mistakes

1. **More trees never hurt in the usual sense.** By [[eq-var-final]] the variance can only fall as $B$ grows and then flatten, so adding trees does not cause overfitting the way a more flexible model would. It only costs time and memory.
2. **The important knobs** are $m$ (features per split), the minimum leaf size, and the depth. They control the bias of each tree and the correlation $\rho$. $B$ only needs to be large enough for the error to plateau.
3. **A forest is much harder to read than one tree.** You trade the explainability of Chapter 7 for accuracy. Feature importance scores (based on total impurity decrease, or on how much OOB error rises when a feature is shuffled) partly make up for it.
4. **A forest cannot predict outside the range of its training targets**, because it averages leaf values, which are themselves means of training targets.
5. **Out-of-bag estimates assume the rows are independent.** If your data has strong groups or time order, use a split that respects that structure.

## 8. Bridge cheat-sheet

| Question | Answer | Where |
|---|---|---|
| How do we get many different trees from one data set? | Bootstrap resampling | Section 2 |
| How much of the data does each tree see? | About $63.2\%$ distinct points | [[eq-oob-prob]] |
| How is the forest's prediction formed? | Average of the trees | [[eq-forest]] |
| What is its variance? | $\sigma^2[\rho+(1-\rho)/B]$ | [[eq-var-final]] |
| Can $B\to\infty$ remove all variance? | No, a floor of $\rho\sigma^2$ remains | [[eq-var-final]] |
| What happens to the bias? | Unchanged from one tree | [[eq-same-bias]] |
| Why deep trees? | Their bias is already near zero | Section 4 |
| How do we lower $\rho$? | A random feature subset at each split | Section 5 |
| How do we estimate test error for free? | Out-of-bag predictions | [[eq-oob]] |

## 9. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [Gradient Boosting](chapter.html?c=09-gradient-boosting) | It takes the opposite route: shallow, high-bias trees that are added one at a time to reduce **bias**. Comparing [[eq-same-bias]] with boosting makes the contrast clear. |
| Machine Learning | [XGBoost](chapter.html?c=11-xgboost) | Row and column subsampling in XGBoost come straight from this chapter. |
| Machine Learning | Extremely randomized trees, isolation forests | Variations that push the randomness even further, using the same variance-reduction logic. |
| Probability & Statistics | The bootstrap, jackknife, resampling inference | Section 2 is the simplest use of the bootstrap. The same resampling gives confidence intervals. |
| Probability & Statistics | Variance of correlated sums, portfolio theory | [[eq-var-final]] has the same shape as the variance of a diversified portfolio, where the floor is the market-wide risk you cannot diversify away. |
| Deep Learning | Model ensembles, dropout, snapshot averaging | They average several models for the same reason, and the same floor applies when the models make shared mistakes. |
| Applications | Credit scoring, fraud detection, bioinformatics, tabular-data competitions | Random forests are a strong default for table-shaped data, needing little tuning. |
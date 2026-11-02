# Support Vector Machines

*Part III · Classification. Logistic regression asks "how likely is each class?". An SVM asks a different question: "which dividing line leaves the widest safety gap?"*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Linear Algebra | [Orthogonal projections (Chapter 1)](chapter.html?c=01-orthogonal-projections) | The shortest path from a point to a plane is perpendicular. This gives the distance formula. |
| Machine Learning | [Logistic Regression (Chapter 5)](chapter.html?c=05-logistic-regression) | Same linear boundary $\mathbf{w}^\top\mathbf{x}+b=0$, different loss. |
| Calculus & Optimization | Constrained optimization, Lagrange multipliers | The dual problem is built from them. |
| Calculus & Optimization | Convexity | It guarantees that the primal and dual problems have the same answer. |
| Linear Algebra | Positive semi-definite matrices | They decide which functions are valid kernels. |
| Machine Learning | [Bias and variance (Chapter 4)](chapter.html?c=04-bias-variance) | The parameter $C$ and the kernel width $\gamma$ are bias-variance knobs. |

## 1. Everyday picture: the widest street between two neighbourhoods

Imagine two neighbourhoods, red houses and blue houses, and you must build a straight road between them.

1. Many roads separate the two neighbourhoods without touching a house.
2. A careful planner picks the road with the **widest empty strip on both sides**. If a few houses are later built slightly differently, the road still separates the two groups.
3. The width of that strip is decided only by the houses **closest to the road**. Houses far away could be moved around freely and the best road would not change.
4. Those closest houses are the **support vectors**: they "hold up" the road.

This chapter turns the picture into formulas in a fixed order:
1. measure the distance from a point to a line (Section 2),
2. maximize the smallest distance (Sections 3 and 4),
3. solve it through a second, equivalent problem called the dual (Section 5),
4. allow a few mistakes (Section 7),
5. bend the road with kernels (Section 8),
6. and use the same ideas for regression (Section 9).

## 2. Distance from a point to a hyperplane

**Step 1: the dividing line.** In two features this is a line, in three a plane, and in general a **hyperplane**:

$$ \mathbf{w}^\top\mathbf{x}+b=0 $$ {#eq-hyper}

The vector $\mathbf{w}$ is **perpendicular** to the hyperplane. To see why, take two points $\mathbf{x}_a,\mathbf{x}_b$ on it. Subtracting their equations gives $\mathbf{w}^\top(\mathbf{x}_a-\mathbf{x}_b)=0$, and $\mathbf{x}_a-\mathbf{x}_b$ can point along any direction inside the hyperplane.

**Step 2: split any point into a foot point and a step.** Let $\mathbf{x}_0$ be any point and $\mathbf{x}_p$ the point of the hyperplane **closest** to it. As in [Chapter 1](chapter.html?c=01-orthogonal-projections), the shortest path to a flat surface is perpendicular to it. So the step from $\mathbf{x}_p$ to $\mathbf{x}_0$ points along $\mathbf{w}$:

$$ \mathbf{x}_0=\mathbf{x}_p+\rho\,\frac{\mathbf{w}}{\lVert\mathbf{w}\rVert} $$ {#eq-split}

Here $\rho$ is the **signed distance**. Dividing by $\lVert\mathbf{w}\rVert$ makes $\mathbf{w}/\lVert\mathbf{w}\rVert$ a unit vector, so $\rho$ is a true length. It is positive on the side $\mathbf{w}$ points towards and negative on the other.

**Step 3: apply the hyperplane equation to both sides.**

$$ \mathbf{w}^\top\mathbf{x}_0+b=\underbrace{(\mathbf{w}^\top\mathbf{x}_p+b)}_{=\,0}+\rho\,\frac{\mathbf{w}^\top\mathbf{w}}{\lVert\mathbf{w}\rVert}=\rho\,\lVert\mathbf{w}\rVert $$ {#eq-derive-dist}

:::box step | How we got here
- Multiply [[eq-split]] on the left by $\mathbf{w}^\top$ and add $b$.
- $\mathbf{x}_p$ lies on the hyperplane, so $\mathbf{w}^\top\mathbf{x}_p+b=0$ by [[eq-hyper]].
- $\mathbf{w}^\top\mathbf{w}=\lVert\mathbf{w}\rVert^2$, so $\dfrac{\mathbf{w}^\top\mathbf{w}}{\lVert\mathbf{w}\rVert}=\lVert\mathbf{w}\rVert$.
:::

**Step 4: solve for the distance.**

$$ \rho=\frac{\mathbf{w}^\top\mathbf{x}_0+b}{\lVert\mathbf{w}\rVert} $$ {#eq-dist}

:::box plain | What is this equation saying in plain words?
- **Top, $\mathbf{w}^\top\mathbf{x}_0+b$:** a "raw score" that is zero on the line and grows as you move away from it. Its size depends on how long $\mathbf{w}$ happens to be.
- **Bottom, $\lVert\mathbf{w}\rVert$:** divides out that length, turning the raw score into an actual distance. This mirrors the denominator in [Chapter 1's projection formula](chapter.html?c=01-orthogonal-projections#eq-lambda).
- **Sign:** tells you which side of the road the point is on.
:::

**A fact we will use heavily: the hyperplane does not care about scale.** The pairs $(\mathbf{w},b)$ and $(c\mathbf{w},cb)$ with any $c>0$ describe the same hyperplane, and the distance is unchanged:

$$ \frac{(c\mathbf{w})^\top\mathbf{x}_0+cb}{\lVert c\mathbf{w}\rVert}=\frac{c\,(\mathbf{w}^\top\mathbf{x}_0+b)}{c\,\lVert\mathbf{w}\rVert}=\rho $$ {#eq-scale}

:::figure F6.1 | Distance to a hyperplane | A line with its normal vector w. Drag a point x0 and see its foot point x_p on the line, the signed distance rho, and the raw score w^T x0 + b. A slider multiplies w and b by c to show that the distance does not change.

## 3. From "correct" to "as far as possible"

**Step 1: labels.** In this chapter the two classes are labelled $y_i\in\{-1,+1\}$. Chapter 5 used $\{0,1\}$. The change makes one formula cover both classes.

**Step 2: a point is classified correctly when its score has the same sign as its label.**

$$ y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)>0\qquad\text{for every }i $$ {#eq-correct}

**Step 3: distance on the correct side.** Multiplying the signed distance [[eq-dist]] by $y_i$ gives a number that is positive for correct points and negative for mistakes. It is the **geometric margin** of point $i$:

$$ \gamma_i=\frac{y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)}{\lVert\mathbf{w}\rVert} $$ {#eq-gmargin}

**Step 4: the SVM goal.** Classifying every point correctly is not enough. We want the line whose **closest** point is as far as possible:

$$ \max_{\mathbf{w},b}\;\Big[\min_i\;\frac{y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)}{\lVert\mathbf{w}\rVert}\Big] $$ {#eq-maxmin}

:::box plain | What is this objective saying in plain words?
- The **inner** $\min_i$ finds the point nearest to the line. Its distance is the width of the empty strip on each side.
- The **outer** $\max$ wiggles the line, over all choices of $\mathbf{w}$ and $b$, to make that smallest distance as big as possible.
- So the road is placed to maximize the room around the *worst-off* house.
:::

## 4. The hard-margin problem

[[eq-maxmin]] is awkward to solve. [[eq-scale]] gives us a free choice that simplifies it.

**Step 1: fix the scale.** Since rescaling $(\mathbf{w},b)$ changes nothing, choose the scale so that the **closest point has score exactly $1$**:

$$ \min_i\;y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)=1 $$ {#eq-normalize}

**Step 2: every point obeys an inequality.** Since the closest point has score $1$, all the others have at least $1$:

$$ y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)\ge 1\qquad\text{for all }i $$ {#eq-constraint}

**Step 3: the margin becomes simple.** Put [[eq-normalize]] into [[eq-gmargin]]. The geometric margin of the closest point is

$$ \gamma=\frac{1}{\lVert\mathbf{w}\rVert} $$ {#eq-margin}

**Step 4: turn "maximize the margin" into "minimize the length of $\mathbf{w}$".** Maximizing $1/\lVert\mathbf{w}\rVert$ is the same as minimizing $\lVert\mathbf{w}\rVert$, and also the same as minimizing $\tfrac12\lVert\mathbf{w}\rVert^2$. We use the last form because it is smooth, convex, and has a neat derivative. The result is the **hard-margin primal problem**:

$$ \min_{\mathbf{w},b}\;\tfrac12\lVert\mathbf{w}\rVert^2\qquad\text{subject to}\quad y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)\ge1\;\;\text{for all }i $$ {#eq-primal}

:::box step | How we got here
- Start from [[eq-maxmin]]. Use the scale freedom of [[eq-scale]] to impose [[eq-normalize]]; this turns the inner minimum into the constraints [[eq-constraint]].
- What remains of the objective is the outer maximum of $1/\lVert\mathbf{w}\rVert$ from [[eq-margin]].
- The square and the factor $\tfrac12$ do not change which $\mathbf{w}$ is best. They only make the objective a smooth quadratic, so this is a convex quadratic program.
:::

**Support vectors.** The points that satisfy [[eq-constraint]] with **equality**, $y_i(\mathbf{w}^\top\mathbf{x}_i+b)=1$, sit exactly on the edge of the empty strip. These are the support vectors. Moving any other point (slightly) does not change the answer.

:::figure F6.2 | Maximum margin | Two separable classes with a draggable, rotatable separating line. The two margin edges w^T x + b = +1 and -1 are drawn, the support vectors are highlighted, and a live readout shows the width 2/||w||. A button jumps to the maximum-margin line.

## 5. The Lagrangian dual

**Why bother?** The primal problem [[eq-primal]] works with the weights directly. Its twin, the dual, works with one number $\alpha_i$ per training point and reveals two things: the answer depends on the data **only through inner products** $\mathbf{x}_i^\top\mathbf{x}_j$ (the door to kernels), and **only the support vectors matter**.

**Step 1: build the Lagrangian.** Attach a multiplier $\alpha_i\ge0$ to each constraint of [[eq-primal]] and subtract it:

$$ \mathcal{L}(\mathbf{w},b,\boldsymbol\alpha)=\tfrac12\lVert\mathbf{w}\rVert^2-\sum_{i=1}^{n}\alpha_i\Big[y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)-1\Big],\qquad\alpha_i\ge0 $$ {#eq-lagr}

**Step 2: why this recovers the primal.** Look at the largest value $\mathcal{L}$ can take over $\boldsymbol\alpha\ge0$:

1. If some constraint is **violated**, the bracket is negative, so $-\alpha_i[\cdot]$ is positive and grows without limit as $\alpha_i\to\infty$. The maximum is $+\infty$.
2. If **all constraints hold**, every bracket is $\ge0$, so the best choice is $\alpha_i[\cdot]=0$ (take $\alpha_i=0$), leaving $\tfrac12\lVert\mathbf{w}\rVert^2$.

So minimizing this maximum over $(\mathbf{w},b)$ is exactly the primal problem:

$$ p^\ast=\min_{\mathbf{w},b}\;\max_{\boldsymbol\alpha\ge0}\;\mathcal{L}(\mathbf{w},b,\boldsymbol\alpha) $$ {#eq-primal-opt}

**Step 3: swap the order to get the dual.**

$$ d^\ast=\max_{\boldsymbol\alpha\ge0}\;\min_{\mathbf{w},b}\;\mathcal{L}(\mathbf{w},b,\boldsymbol\alpha) $$ {#eq-dual-opt}

**Weak and strong duality:**
- A maximum of minima can never exceed a minimum of maxima, so $d^\ast\le p^\ast$ always (**weak duality**).
- If the primal problem is **convex** and has at least one strictly feasible point, then $d^\ast=p^\ast$ (**strong duality**). Both hold here whenever the data can be separated by a line, so solving the dual solves the primal.

**Step 4: do the inner minimization.** Set the derivatives of [[eq-lagr]] to zero (the **stationarity conditions**):

$$ \frac{\partial\mathcal{L}}{\partial\mathbf{w}}=\mathbf{w}-\sum_i\alpha_iy_i\mathbf{x}_i=\mathbf{0}\;\Rightarrow\;\mathbf{w}=\sum_{i}\alpha_iy_i\mathbf{x}_i $$ {#eq-stat-w}

$$ \frac{\partial\mathcal{L}}{\partial b}=-\sum_i\alpha_iy_i=0\;\Rightarrow\;\sum_i\alpha_iy_i=0 $$ {#eq-stat-b}

**Step 5: substitute back to remove $\mathbf{w}$ and $b$.** Write $S=\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j$. Then:

1. By [[eq-stat-w]], $\tfrac12\lVert\mathbf{w}\rVert^2=\tfrac12S$.
2. The term $\sum_i\alpha_iy_i\mathbf{w}^\top\mathbf{x}_i=\mathbf{w}^\top\mathbf{w}=S$.
3. The term $b\sum_i\alpha_iy_i=0$ by [[eq-stat-b]].
4. The remaining term is $\sum_i\alpha_i$.

So $\mathcal{L}=\tfrac12S-S-0+\sum_i\alpha_i$, which is the **dual problem**:

$$ \max_{\boldsymbol\alpha}\;\sum_{i}\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_j\,y_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j\qquad\text{subject to}\quad\alpha_i\ge0,\;\;\sum_i\alpha_iy_i=0 $$ {#eq-dual}

:::box plain | What is the dual saying in plain words?
- There is one weight $\alpha_i$ per training point. It measures "how much this point pushes on the road".
- The data enters **only** through the dot products $\mathbf{x}_i^\top\mathbf{x}_j$: how similar two points are. Remember this; it is the whole reason kernels work.
- The condition $\sum_i\alpha_iy_i=0$ says the pushes from the two classes balance, like a road held in equilibrium.
:::

**Step 6: the complementary slackness condition.** At the optimum, the multiplier and its constraint cannot both be "slack":

$$ \alpha_i\Big[y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)-1\Big]=0\qquad\text{for every }i $$ {#eq-cs}

So for each training point exactly one of these holds:
1. $\alpha_i=0$: the point is **not on the margin** and does **not contribute** to $\mathbf{w}$ in [[eq-stat-w]].
2. $y_i(\mathbf{w}^\top\mathbf{x}_i+b)=1$: the point lies **exactly on the margin**. These are the **support vectors**.

:::box step | How we got here
- This is one of the KKT conditions, which extend Lagrange multipliers to inequality constraints. The four conditions are stationarity ([[eq-stat-w]], [[eq-stat-b]]), the original constraints [[eq-constraint]], $\alpha_i\ge0$, and [[eq-cs]].
- Intuition: in step 2 above, at the best $\boldsymbol\alpha$ a constraint with room to spare ("slack") must have $\alpha_i=0$; otherwise $\alpha_i$ could be reduced to improve $\mathcal{L}$.
:::

## 6. Making predictions

**Step 1: the weights.** From [[eq-stat-w]], with the dual solution $\alpha_i^\ast$:

$$ \mathbf{w}^\ast=\sum_{i}\alpha_i^\ast y_i\mathbf{x}_i $$ {#eq-wstar}

By [[eq-cs]] only support vectors have $\alpha_i^\ast\ne0$, so $\mathbf{w}^\ast$ is a weighted sum of the support vectors alone.

**Step 2: the bias.** Take any support vector $\mathbf{x}_k$. It sits on the margin, so $y_k(\mathbf{w}^{\ast\top}\mathbf{x}_k+b^\ast)=1$. Because $y_k=\pm1$, we have $1/y_k=y_k$, so

$$ b^\ast=y_k-\mathbf{w}^{\ast\top}\mathbf{x}_k $$ {#eq-bstar}

In practice average this over the set $S$ of all support vectors, $b^\ast=\frac1{\lvert S\rvert}\sum_{k\in S}\big(y_k-\mathbf{w}^{\ast\top}\mathbf{x}_k\big)$, to reduce rounding errors.

**Step 3: classify a new point.**

$$ \hat y(\mathbf{x})=\operatorname{sign}\big(\mathbf{w}^{\ast\top}\mathbf{x}+b^\ast\big)=\operatorname{sign}\Big(\sum_{i\in S}\alpha_i^\ast y_i\,\mathbf{x}_i^\top\mathbf{x}+b^\ast\Big) $$ {#eq-predict}

:::box plain | What is this saying in plain words?
- To classify a new point, compare it (by dot product) **only with the support vectors**, let each vote for its own class with strength $\alpha_i^\ast$, add the offset, and take the sign.
- Everything else in the training set can be thrown away after training.
- Once more, the data appears only through dot products $\mathbf{x}_i^\top\mathbf{x}$.
:::

## 7. Soft margin and the hinge loss

Real data is rarely perfectly separable, and a single odd point can force a hard-margin road to be very narrow, or make the problem infeasible. We let some points break the rules, **for a price**.

**Step 1: add a slack variable $\xi_i\ge0$ to each constraint.**

$$ y_i\,(\mathbf{w}^\top\mathbf{x}_i+b)\ge1-\xi_i,\qquad\xi_i\ge0 $$ {#eq-slack}

Here $\xi_i$ is how far inside the strip (or on the wrong side) point $i$ is allowed to sit. Hard margin is the special case $\xi_i=0$.

**Step 2: charge a price for slack.**

$$ \min_{\mathbf{w},b,\boldsymbol\xi}\;\tfrac12\lVert\mathbf{w}\rVert^2+C\sum_i\xi_i\qquad\text{subject to [[eq-slack]]} $$ {#eq-primal-soft}

The constant $C>0$ is the price per unit of slack.

**Step 3: how little slack is needed?** Write $m_i=y_i(\mathbf{w}^\top\mathbf{x}_i+b)$. The constraints say $\xi_i\ge1-m_i$ and $\xi_i\ge0$. Since slack costs money, the best choice is the smallest valid value:

$$ \xi_i=\max\big(0,\;1-m_i\big) $$ {#eq-xi}

| Where the point is | Value of $m_i$ | Slack $\xi_i$ |
|---|---|---|
| Correct and outside the margin | $m_i\ge1$ | $0$ |
| Correct but inside the margin | $0<m_i<1$ | between $0$ and $1$ |
| Misclassified | $m_i<0$ | greater than $1$ |

**Step 4: put [[eq-xi]] into [[eq-primal-soft]] to remove the constraints.**

$$ \min_{\mathbf{w},b}\;\tfrac12\lVert\mathbf{w}\rVert^2+C\sum_{i}\max\big(0,\;1-y_i(\mathbf{w}^\top\mathbf{x}_i+b)\big) $$ {#eq-hinge-obj}

The function $\max(0,1-m)$ is the **hinge loss**. So a soft-margin SVM is "**a penalty on $\lVert\mathbf{w}\rVert^2$ plus a hinge loss**".

**Compare with what you already know.** Divide [[eq-hinge-obj]] by $C$. It becomes $\sum_i\text{hinge}_i+\frac{1}{2C}\lVert\mathbf{w}\rVert^2$: **a data-fit term plus a Ridge-style penalty**, as in [Chapter 3](chapter.html?c=03-ridge-lasso#eq-ridge), with the penalty strength proportional to $1/C$.

**Hinge versus the logistic loss of Chapter 5.** With labels $y\in\{-1,+1\}$ and score $z=\mathbf{w}^\top\mathbf{x}$, the cross-entropy of [Chapter 5](chapter.html?c=05-logistic-regression#eq-bce) becomes $\log\big(1+e^{-m}\big)$ with $m=yz$. (For $y=+1$ it is $-\log\sigma(z)$. For $y=-1$ it is $-\log\big(1-\sigma(z)\big)=-\log\sigma(-z)$, using $\sigma(-z)=1-\sigma(z)$.)

| Margin $m=y\,(\mathbf{w}^\top\mathbf{x}+b)$ | Hinge loss | Logistic loss |
|---|---|---|
| Large positive (confidently correct) | exactly $0$ | small but never exactly $0$ |
| Near $0$ | about $1$ | about $\log 2$ |
| Large negative (confidently wrong) | grows linearly | grows linearly |
| Smooth everywhere? | no, a kink at $m=1$ | yes |
| Gives probabilities? | no | yes |

The hinge loss is **exactly zero** once a point is correct with room to spare. That is why distant points have no influence and only support vectors matter. The logistic loss keeps pulling on every point a little.

:::figure F6.4 | Hinge vs logistic loss | The two loss curves plotted against the margin m = y f(x), together with the 0-1 loss for reference. A marker follows a draggable point so its loss under each rule is read off live.

**The dual of the soft-margin problem.** Repeat the recipe of Section 5 with extra multipliers $\mu_i\ge0$ for the constraints $\xi_i\ge0$. The new derivative is

$$ \frac{\partial\mathcal{L}}{\partial\xi_i}=C-\alpha_i-\mu_i=0\;\Rightarrow\;\alpha_i=C-\mu_i\le C $$ {#eq-box-c}

The $\xi$-terms cancel in the substitution, so the dual objective is **identical** to [[eq-dual]]. Only the constraint changes, to the **box constraint**:

$$ \max_{\boldsymbol\alpha}\;\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j\qquad\text{subject to}\quad0\le\alpha_i\le C,\;\;\sum_i\alpha_iy_i=0 $$ {#eq-dual-soft}

| Value of $\alpha_i$ | Where point $i$ lies | Role |
|---|---|---|
| $\alpha_i=0$ | outside the margin, correct | no influence |
| $0<\alpha_i<C$ | exactly on the margin edge | free support vector |
| $\alpha_i=C$ | inside the margin or misclassified | bounded support vector |

:::box step | How we got here
- Two more KKT conditions apply: $\mu_i\xi_i=0$ and $\alpha_i\big[y_i(\mathbf{w}^\top\mathbf{x}_i+b)-1+\xi_i\big]=0$.
- If $0<\alpha_i<C$, then $\mu_i=C-\alpha_i>0$, so $\xi_i=0$; and since $\alpha_i>0$ the point is exactly on the margin.
- If $\alpha_i=C$, then $\mu_i=0$, so $\xi_i$ is allowed to be positive.
- **Consequence for the bias:** use [[eq-bstar]] only with support vectors satisfying $0<\alpha_i<C$, because only they are guaranteed to sit exactly on the margin.
:::

**What $C$ does:**
- **Large $C$:** violations are expensive. The road hugs the data tightly: low bias, high variance, risk of overfitting.
- **Small $C$:** violations are cheap. The road is wide and smooth: higher bias, lower variance. This is the trade-off of [Chapter 4](chapter.html?c=04-bias-variance#eq-bv).

:::figure F6.3 | Slack variables and C | Overlapping classes with a fitted soft-margin line. A slider for C widens or narrows the street. Points inside the margin or misclassified are highlighted with their slack drawn as a short arrow.

## 8. The kernel trick and Mercer's condition

A straight road cannot separate every data set. A classical remedy is to **lift** the points into a space with extra, nonlinear features $\boldsymbol\phi(\mathbf{x})$ and draw a straight road *there*. In the original space the road becomes curved.

**Step 1: the dual only needs inner products.** Both [[eq-dual-soft]] and the predictor [[eq-predict]] use the data only through $\mathbf{x}_i^\top\mathbf{x}_j$. To work in the lifted space, replace each $\mathbf{x}$ by $\boldsymbol\phi(\mathbf{x})$, so we need $\boldsymbol\phi(\mathbf{x}_i)^\top\boldsymbol\phi(\mathbf{x}_j)$.

**Step 2: the problem.** If $\boldsymbol\phi$ has thousands (or infinitely many) coordinates, computing it explicitly is slow or impossible.

**Step 3: the trick.** Find a function that gives the inner product **directly from the original inputs**, without ever building $\boldsymbol\phi$:

$$ K(\mathbf{x},\mathbf{z})=\boldsymbol\phi(\mathbf{x})^\top\boldsymbol\phi(\mathbf{z}) $$ {#eq-kernel}

**A concrete example.** Take two features, $\mathbf{x}=(x_1,x_2)$, and $\boldsymbol\phi(\mathbf{x})=\big(x_1^2,\;\sqrt2\,x_1x_2,\;x_2^2\big)$. Then

$$ \boldsymbol\phi(\mathbf{x})^\top\boldsymbol\phi(\mathbf{z})=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2=(x_1z_1+x_2z_2)^2=(\mathbf{x}^\top\mathbf{z})^2 $$ {#eq-poly-example}

Computing the left side needs a 3-dimensional lift. The right side is one dot product and a square, in the original 2D space.

**Step 4: the kernelized dual and predictor.** Replace every $\mathbf{x}_i^\top\mathbf{x}_j$ by $K(\mathbf{x}_i,\mathbf{x}_j)$:

$$ \max_{\boldsymbol\alpha}\;\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,K(\mathbf{x}_i,\mathbf{x}_j)\quad\text{s.t. }0\le\alpha_i\le C,\;\sum_i\alpha_iy_i=0 $$ {#eq-dual-k}

$$ \hat y(\mathbf{x})=\operatorname{sign}\Big(\sum_{i\in S}\alpha_i^\ast y_i\,K(\mathbf{x}_i,\mathbf{x})+b^\ast\Big),\qquad b^\ast=y_k-\sum_{i\in S}\alpha_i^\ast y_iK(\mathbf{x}_i,\mathbf{x}_k) $$ {#eq-pred-k}

(The bias formula is [[eq-bstar]] with the same replacement, using a support vector $k$ with $0<\alpha_k<C$.)

**Popular kernels:**

| Kernel | Formula | What it does |
|---|---|---|
| Linear | $\mathbf{x}^\top\mathbf{z}$ | the ordinary SVM |
| Polynomial | $(\mathbf{x}^\top\mathbf{z}+c)^d$ | all feature products up to degree $d$, as in [[eq-poly-example]] |
| RBF (Gaussian) | $\exp\big(-\gamma\lVert\mathbf{x}-\mathbf{z}\rVert^2\big)$, $\gamma>0$ | a similarity score: $1$ when $\mathbf{x}=\mathbf{z}$, fading smoothly towards $0$ with distance |

:::box plain | What is the RBF kernel saying in plain words?
- $K(\mathbf{x},\mathbf{z})$ answers "how similar are these two points?". Nearby points score close to $1$, far-apart points close to $0$.
- The prediction [[eq-pred-k]] is then a **weighted vote of the support vectors, where closer ones count more**.
- $\gamma$ controls how fast similarity fades. **Large $\gamma$:** only very near neighbours count, so the boundary is wiggly (low bias, high variance). **Small $\gamma$:** similarity spreads far, so the boundary is smooth (higher bias, lower variance).
- Behind the scenes the RBF kernel corresponds to an **infinite-dimensional** $\boldsymbol\phi$. We get its benefits at the cost of one distance computation.
:::

:::figure F6.5 | Kernel trick | One-dimensional points where one class sits in the middle, so no single threshold works. A toggle lifts them to the plane with phi(x) = (x, x^2), where a straight line separates them, then maps the line back as two thresholds.

:::figure F6.6 | RBF kernel and gamma | A two-class data set with a nonlinear boundary. A slider for gamma changes the boundary from a smooth blob to many tiny islands, with support vectors highlighted. A second slider for C shows the combined effect.

### Which functions are valid kernels? Mercer's condition

Not every function $K$ equals an inner product $\boldsymbol\phi(\mathbf{x})^\top\boldsymbol\phi(\mathbf{z})$ for some $\boldsymbol\phi$. A symmetric function $K$ is a valid kernel **if and only if** for every finite set of points $\mathbf{x}_1,\dots,\mathbf{x}_n$ the **kernel matrix** is positive semi-definite:

$$ \mathbf{K}=\begin{pmatrix}K(\mathbf{x}_1,\mathbf{x}_1)&\cdots&K(\mathbf{x}_1,\mathbf{x}_n)\\\vdots&\ddots&\vdots\\K(\mathbf{x}_n,\mathbf{x}_1)&\cdots&K(\mathbf{x}_n,\mathbf{x}_n)\end{pmatrix}\succeq0 $$ {#eq-mercer}

**Why a real kernel must satisfy this (the easy direction):**

1. **Symmetric:** $K(\mathbf{x},\mathbf{z})=\boldsymbol\phi(\mathbf{x})^\top\boldsymbol\phi(\mathbf{z})=\boldsymbol\phi(\mathbf{z})^\top\boldsymbol\phi(\mathbf{x})=K(\mathbf{z},\mathbf{x})$, because a dot product is symmetric.
2. **Positive semi-definite:** for any vector $\mathbf{v}$,

$$ \mathbf{v}^\top\mathbf{K}\mathbf{v}=\sum_{i,j}v_iv_j\,\boldsymbol\phi(\mathbf{x}_i)^\top\boldsymbol\phi(\mathbf{x}_j)=\Big\lVert\sum_iv_i\boldsymbol\phi(\mathbf{x}_i)\Big\rVert^2\ge0 $$ {#eq-psd-proof}

:::box step | How we got here
- Expand $\mathbf{v}^\top\mathbf{K}\mathbf{v}$ entry by entry, then insert [[eq-kernel]] for each $K(\mathbf{x}_i,\mathbf{x}_j)$.
- A double sum of $v_iv_j\,\mathbf{a}_i^\top\mathbf{a}_j$ is the squared length of $\sum_iv_i\mathbf{a}_i$. A squared length cannot be negative.
:::

**The converse is Mercer's theorem:** any symmetric function that passes the test [[eq-mercer]] *does* come from some $\boldsymbol\phi$ (possibly infinite-dimensional). So you can **check a kernel without ever finding $\boldsymbol\phi$**. The condition also keeps the dual [[eq-dual-k]] a well-behaved concave problem, so it has no bad local optima.

## 9. Support Vector Regression (SVR)

The same machinery works for predicting numbers.

**Step 1: a loss that forgives small errors.** Predict with $f(\mathbf{x})=\mathbf{w}^\top\mathbf{x}+b$ and use the **$\varepsilon$-insensitive loss**:

$$ L_\varepsilon\big(y,f(\mathbf{x})\big)=\max\big(0,\;\lvert y-f(\mathbf{x})\rvert-\varepsilon\big) $$ {#eq-eps-loss}

If the prediction is within $\varepsilon$ of the truth the loss is **exactly zero**. Only errors larger than $\varepsilon$ are charged. Picture a "tube" of half-width $\varepsilon$ around the fitted line: points inside it are free.

**Step 2: two slacks per point**, one for each side of the tube:

$$ \min_{\mathbf{w},b,\boldsymbol\xi,\boldsymbol\xi^\ast}\;\tfrac12\lVert\mathbf{w}\rVert^2+C\sum_i(\xi_i+\xi_i^\ast)\quad\text{s.t.}\quad\begin{cases}y_i-\mathbf{w}^\top\mathbf{x}_i-b\le\varepsilon+\xi_i&\text{(point too far above)}\\\mathbf{w}^\top\mathbf{x}_i+b-y_i\le\varepsilon+\xi_i^\ast&\text{(point too far below)}\\\xi_i,\xi_i^\ast\ge0\end{cases} $$ {#eq-svr-primal}

:::box plain | What is this saying in plain words?
- The term $\tfrac12\lVert\mathbf{w}\rVert^2$ keeps the line **flat** (the SVM's usual preference for a simple, wide-margin solution).
- A point costs nothing while it stays inside the $\varepsilon$-tube. Outside the tube, it costs $C$ times how far outside.
- Same recipe as before: a regularizer plus a loss that is exactly zero over a range.
:::

**Step 3: the dual.** Attach multipliers $\alpha_i$ to the first constraint and $\alpha_i^\ast$ to the second, and repeat Section 5 (the $\partial\mathcal{L}/\partial\mathbf{w}=\mathbf{0}$ and $\partial\mathcal{L}/\partial b=0$ conditions). Writing $\beta_i=\alpha_i-\alpha_i^\ast$:

$$ \mathbf{w}=\sum_i(\alpha_i-\alpha_i^\ast)\,\mathbf{x}_i $$ {#eq-svr-w}

$$ \max_{\boldsymbol\alpha,\boldsymbol\alpha^\ast}\;-\frac12\sum_{i,j}(\alpha_i-\alpha_i^\ast)(\alpha_j-\alpha_j^\ast)\,\mathbf{x}_i^\top\mathbf{x}_j-\varepsilon\sum_i(\alpha_i+\alpha_i^\ast)+\sum_iy_i(\alpha_i-\alpha_i^\ast) $$ {#eq-svr-dual}

subject to $0\le\alpha_i,\alpha_i^\ast\le C$ and $\sum_i(\alpha_i-\alpha_i^\ast)=0$.

**Step 4: prediction, with kernels.** Replace $\mathbf{x}_i^\top\mathbf{x}$ by $K(\mathbf{x}_i,\mathbf{x})$ just as in [[eq-pred-k]]:

$$ f(\mathbf{x})=\sum_i(\alpha_i-\alpha_i^\ast)\,K(\mathbf{x}_i,\mathbf{x})+b $$ {#eq-svr-pred}

- Points strictly **inside** the tube have $\alpha_i=\alpha_i^\ast=0$, so they do not influence the fit. This is the regression twin of "only support vectors matter".
- The bias comes from any point with $0<\alpha_i<C$: such a point sits exactly on the upper tube edge, so $b=y_i-\mathbf{w}^\top\mathbf{x}_i-\varepsilon$ (and similarly for a point with $0<\alpha_i^\ast<C$ on the lower edge).

:::figure F6.7 | Support vector regression | A noisy curve with a fitted line and a shaded tube of half-width epsilon. Points inside the tube are grey, points on or outside it are highlighted, and short arrows show the slacks. Sliders for epsilon and C change which points become support vectors.

## 10. Bridge cheat-sheet

| | Hard margin | Soft margin | Kernel SVM | SVR |
|---|---|---|---|---|
| Idea | widest empty street | allow paid mistakes | bend the street | a tube around a line |
| Primal | [[eq-primal]] | [[eq-primal-soft]] | same, in lifted space | [[eq-svr-primal]] |
| Loss in disguise | zero or infinite | hinge, [[eq-hinge-obj]] | hinge | $\varepsilon$-insensitive, [[eq-eps-loss]] |
| Dual constraint | $\alpha_i\ge0$ | $0\le\alpha_i\le C$, [[eq-dual-soft]] | with $K$, [[eq-dual-k]] | box constraint on $\alpha_i,\alpha_i^\ast$ |
| Who is a support vector | points on the margin | on, inside or beyond it | same | on or outside the tube |
| Prediction | [[eq-predict]] | same | [[eq-pred-k]] | [[eq-svr-pred]] |

**Logistic regression versus SVM, in one line:** both draw the boundary $\mathbf{w}^\top\mathbf{x}+b=0$ ([Chapter 5](chapter.html?c=05-logistic-regression#eq-boundary)). Logistic regression minimizes cross-entropy and lets every point pull a little. The SVM minimizes hinge loss plus $\lVert\mathbf{w}\rVert^2$, and only the points near the road matter.

## 11. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | Kernel methods: kernel ridge regression, kernel PCA, Gaussian processes | They all use the "only inner products matter" idea of [[eq-dual]] and the validity test [[eq-mercer]]. |
| Machine Learning | Regularization and model selection | $C$, $\gamma$ and $\varepsilon$ are bias-variance knobs, as in [Chapter 4](chapter.html?c=04-bias-variance). They are chosen by cross-validation. |
| Machine Learning | [Decision trees and ensembles](chapter.html?c=07-decision-trees) | A completely different way of drawing boundaries, whose weaknesses (variance) the next part studies. |
| Optimization | Duality, KKT conditions, quadratic programming, SMO | Sections 5 and 7 are the standard worked example of duality and complementary slackness. SMO is the algorithm that solves [[eq-dual-soft]] fast. |
| Deep Learning | Margin-based losses, contrastive and triplet losses | They are hinge-style losses: zero once a margin is achieved. |
| Probability & Statistics | Reproducing kernel Hilbert spaces, nonparametric methods | Mercer's theorem is the doorway to them. |
| Applications | Text classification, bioinformatics, handwriting recognition, anomaly detection (one-class SVM) | Classic uses where data has many features and few examples, and kernels encode problem-specific similarity. |
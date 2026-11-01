# Orthogonal Projections & Gram-Schmidt

*Part I · Geometry. This one idea quietly powers regression, SVMs and clustering.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Linear Algebra | Dot product | "Perpendicular" simply means the dot product is $0$. |
| Linear Algebra | Transpose, inverse, matrix product | The general formula is $\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top$. |
| Linear Algebra | Span, basis, subspace | A subspace is the "wall" we project onto. |
| Calculus & Optimization | Minimizing by setting a derivative to $0$ | Used in the third intuition tab below. |
| Machine Learning | None required | Linear regression, the next chapter, is built directly on this one. |

## 1. Everyday picture: a shadow on a wall

Imagine standing in a room, facing a flat white wall, with an object floating in front of it.

1. Shine a big flat light **straight at the wall**, so the rays hit it at $90^\circ$.
2. The object casts a flat 2D shadow on the wall.
3. In math, that shadow is the **orthogonal projection** of the object onto the wall. "Orthogonal" is a fancy word for perpendicular ($90^\circ$).

Keep this picture in mind. Every formula below is just this shadow, written in symbols.

:::figure F1.1 | Shadow simulator | 3D room with an object, a wall and a light. Sliders for light tilt and object distance. A toggle compares a parallel 90° beam (true projection) with a point light (not a true projection).

## 2. The simplest case: one vector onto a line (2D)

**Goal.** Given a vector $\mathbf{x}$ and a line along the vector $\mathbf{b}$, find the shadow $\mathbf{p}$ of $\mathbf{x}$ on that line.

:::widget projection2d | Drag the red tip of x. The shadow p (green) and the error e (dashed blue) update live.

We derive the formula in six small steps. Each step says how we got there from the previous ones.

**Step 1: the shadow lies on the line.**

$$ \mathbf{p} = \lambda\,\mathbf{b} $$ {#eq-proj}

:::box step | How we got here
- $\mathbf{p}$ sits on the line through $\mathbf{b}$, so it points along $\mathbf{b}$.
- That means it is $\mathbf{b}$ stretched or shrunk by some unknown number, which we call $\lambda$ (lambda).
- Finding $\mathbf{p}$ is now the same as finding one number, $\lambda$.
:::

**Step 2: name the gap between $\mathbf{x}$ and its shadow.**

$$ \mathbf{e} = \mathbf{x} - \mathbf{p} $$ {#eq-error}

This is the light beam falling from the tip of $\mathbf{x}$ straight down onto the line.

**Step 3: that gap is perpendicular to the line.**

$$ \mathbf{e}\cdot\mathbf{b} = 0 $$ {#eq-orth}

:::box step | How we got here
- The beam falls at $90^\circ$ to the line, so $\mathbf{e}\perp\mathbf{b}$.
- Two perpendicular vectors have dot product zero, which gives [[eq-orth]].
- Why must it be $90^\circ$? If $\mathbf{e}$ were tilted, it would still cast its own small shadow on the line, so $\mathbf{p}$ would not yet be the *full* shadow.
:::

**Step 4: substitute everything in terms of $\lambda$.**

$$ (\mathbf{x} - \lambda\mathbf{b})\cdot\mathbf{b} = 0 $$ {#eq-sub}

:::box step | How we got here
- Put $\mathbf{p}=\lambda\mathbf{b}$ from [[eq-proj]] into $\mathbf{e}=\mathbf{x}-\mathbf{p}$ from [[eq-error]]. This gives $\mathbf{e}=\mathbf{x}-\lambda\mathbf{b}$.
- Put that $\mathbf{e}$ into [[eq-orth]]. The result is [[eq-sub]].
- The only unknown left is $\lambda$.
:::

**Step 5: expand and solve for $\lambda$.**

$$ \mathbf{x}\cdot\mathbf{b} - \lambda\,(\mathbf{b}\cdot\mathbf{b}) = 0 \quad\Longrightarrow\quad \lambda = \frac{\mathbf{x}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}} $$ {#eq-lambda}

:::box step | How we got here
- The dot product distributes over subtraction, so [[eq-sub]] splits into $\mathbf{x}\cdot\mathbf{b}-\lambda(\mathbf{b}\cdot\mathbf{b})$.
- Move the $\lambda$ term to the other side: $\lambda(\mathbf{b}\cdot\mathbf{b})=\mathbf{x}\cdot\mathbf{b}$.
- Divide by $\mathbf{b}\cdot\mathbf{b}$, which is a plain number (and non-zero as long as $\mathbf{b}\neq\mathbf{0}$).
:::

**Step 6: put $\lambda$ back into the shadow.**

$$ \mathbf{p} = \left(\frac{\mathbf{x}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\right)\mathbf{b} $$ {#eq-p}

:::box step | How we got here
- Take the unknown $\lambda$ in [[eq-proj]] and replace it with the value found in [[eq-lambda]].
:::

:::box plain | What is this equation saying in plain words?
- **Top, $\mathbf{x}\cdot\mathbf{b}$:** how much $\mathbf{x}$ and $\mathbf{b}$ point in the same direction.
- **Bottom, $\mathbf{b}\cdot\mathbf{b}$:** the length of $\mathbf{b}$ squared. Dividing by it cancels the size of $\mathbf{b}$, because making the reference vector longer must not change the shadow of $\mathbf{x}$.
- **Whole fraction:** one number, $\lambda$. Multiplying $\mathbf{b}$ by it gives a vector of exactly the shadow's length.
:::

**Quick sanity checks** (try each with the preset buttons above):
- $\mathbf{x}\perp\mathbf{b}$: the top of [[eq-lambda]] is $0$, so $\lambda=0$ and the shadow is a single point.
- $\mathbf{x}$ along $\mathbf{b}$: the shadow is $\mathbf{x}$ itself, and $\mathbf{e}=\mathbf{0}$.
- $\lambda<0$: $\mathbf{x}$ leans away from $\mathbf{b}$, so the shadow points the opposite way.

:::tabs Three ways to see why the error must be perpendicular
--- Shortest walk
A house sits beside a straight road. The closest point on the road to the house is straight across, not diagonally. Any other point is farther, by Pythagoras. The "closest point on the line to $\mathbf{x}$" is therefore the one where $\mathbf{e}\perp\mathbf{b}$.
--- Shadow
If $\mathbf{e}$ made any angle other than $90^\circ$ with the line, $\mathbf{e}$ itself would cast a shadow on the line. Then $\mathbf{p}$ would be missing a piece, so it would not be the complete shadow of $\mathbf{x}$.
--- Calculus
Minimize the squared distance $f(\lambda)=\|\mathbf{x}-\lambda\mathbf{b}\|^2$:
- Differentiate: $f'(\lambda)=-2\,\mathbf{b}\cdot(\mathbf{x}-\lambda\mathbf{b})$.
- Set $f'(\lambda)=0$. This is exactly [[eq-sub]], the perpendicularity condition.
- So "closest point" and "perpendicular error" are the same statement.
:::

## 3. The general case: a whole subspace

Now replace the line by a subspace $U\subseteq\mathbb{R}^n$ of dimension $m\ge1$ (a plane if $m=2$). **We repeat the same recipe.**

**Step 1: the shadow is a combination of the basis vectors.**

Let $\mathbf{b}_1,\dots,\mathbf{b}_m$ be a basis of $U$, stacked as the columns of $\mathbf{B}\in\mathbb{R}^{n\times m}$. Let $\boldsymbol\lambda=[\lambda_1,\dots,\lambda_m]^\top$.

$$ \pi_U(\mathbf{x}) = \sum_{i=1}^{m}\lambda_i\,\mathbf{b}_i = \mathbf{B}\boldsymbol\lambda $$ {#eq-subspace}

:::box step | How we got here
- This is [[eq-proj]] with one stretch factor replaced by $m$ of them, one per basis vector.
- The single number $\lambda$ became a vector $\boldsymbol\lambda$ of $m$ numbers.
:::

**Step 2: the error is perpendicular to every basis vector.**

$$ \mathbf{b}_i^\top(\mathbf{x}-\mathbf{B}\boldsymbol\lambda) = 0 \qquad\text{for } i = 1,\dots,m $$ {#eq-orth-all}

:::box step | How we got here
- $\pi_U(\mathbf{x})$ is the closest point of $U$ to $\mathbf{x}$, so the error $\mathbf{x}-\pi_U(\mathbf{x})$ must be perpendicular to all of $U$.
- Being perpendicular to every basis vector is enough, because every vector in $U$ is a combination of them.
- This is [[eq-orth]] repeated $m$ times, with $\mathbf{e}=\mathbf{x}-\mathbf{B}\boldsymbol\lambda$ from [[eq-subspace]].
:::

**Step 3: stack the $m$ equations into one.**

$$ \mathbf{B}^\top(\mathbf{x}-\mathbf{B}\boldsymbol\lambda)=\mathbf{0} \quad\Longleftrightarrow\quad \mathbf{B}^\top\mathbf{B}\,\boldsymbol\lambda=\mathbf{B}^\top\mathbf{x} $$ {#eq-normal}

:::box step | How we got here
- The rows $\mathbf{b}_i^\top$ of $\mathbf{B}^\top$ are exactly the left-hand sides of [[eq-orth-all]], so one matrix equation holds them all.
- Multiply out the bracket and move $\mathbf{B}^\top\mathbf{B}\boldsymbol\lambda$ to the other side.
- This is called the **normal equation**. It reappears in linear regression.
:::

**Step 4: solve for $\boldsymbol\lambda$.**

$$ \boldsymbol\lambda = (\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{x} $$ {#eq-lambda-general}

:::box step | How we got here
- In [[eq-lambda]] we divided by $\mathbf{b}\cdot\mathbf{b}$. A matrix cannot be divided, but it can be inverted.
- Multiply [[eq-normal]] on the left by $(\mathbf{B}^\top\mathbf{B})^{-1}$. This inverse exists because the columns of $\mathbf{B}$ are independent (they form a basis).
:::

**Step 5: put it back to get the projection matrix.**

$$ \pi_U(\mathbf{x}) = \underbrace{\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top}_{\mathbf{P}_\pi}\,\mathbf{x} $$ {#eq-P}

:::box step | How we got here
- Substitute [[eq-lambda-general]] into [[eq-subspace]]. The matrix in front of $\mathbf{x}$ is named $\mathbf{P}_\pi$, the **projection matrix**.
- **Check against the 2D case:** take one basis vector, $\mathbf{B}=\mathbf{b}$. Then $\mathbf{B}^\top\mathbf{B}=\mathbf{b}\cdot\mathbf{b}$ is a single number, and [[eq-P]] becomes [[eq-p]] exactly.
:::

:::box plain | What is Equation (14) saying in plain words?
- **$\mathbf{B}^\top\mathbf{x}$:** measures how much $\mathbf{x}$ agrees with each basis vector (the "top" of the 2D fraction).
- **$(\mathbf{B}^\top\mathbf{B})^{-1}$:** corrects for basis vectors that are long or not perpendicular to each other (the "bottom" of the 2D fraction).
- **Leading $\mathbf{B}$:** rebuilds the shadow from these weights, just as $\lambda\mathbf{b}$ did.
:::

:::figure F1.2 | 3D subspace projector | A plane U with basis b1 and b2. Drag x in 3D, orbit the camera, switch between a general and an orthonormal basis. Button "project twice" shows that P applied twice equals P.

### Two properties of the projection matrix

$$ \mathbf{P}_\pi^2=\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\underbrace{\mathbf{B}^\top\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}}_{=\,\mathbf{I}}\mathbf{B}^\top=\mathbf{P}_\pi $$ {#eq-idem}

$$ \mathbf{P}_\pi^\top=\mathbf{B}\big((\mathbf{B}^\top\mathbf{B})^{-1}\big)^\top\mathbf{B}^\top=\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top=\mathbf{P}_\pi $$ {#eq-sym}

- **Idempotent, [[eq-idem]]:** the middle of $\mathbf{P}_\pi\mathbf{P}_\pi$ cancels to $\mathbf{I}$. In words: *the shadow of a shadow is the same shadow.*
- **Symmetric, [[eq-sym]]:** the matrix $\mathbf{B}^\top\mathbf{B}$ equals its own transpose, so its inverse does too.

### A much simpler formula with an orthonormal basis

If the basis vectors are perpendicular, each of length 1, then $\mathbf{B}^\top\mathbf{B}=\mathbf{I}$. The inverse in [[eq-P]] disappears:

$$ \pi_U(\mathbf{x}) = \mathbf{B}\mathbf{B}^\top\mathbf{x} $$ {#eq-onb}

That raises a question: can we turn *any* basis into an orthonormal one? Yes, and that is Gram-Schmidt.

## 4. Gram-Schmidt: building an orthonormal basis

**Idea.** Go through the basis one vector at a time. From each new vector, subtract its shadow on the vectors already built. What remains is perpendicular to all of them.

**Step 1: keep the first vector.**

$$ \mathbf{u}_1=\mathbf{b}_1 $$ {#eq-gs1}

**Step 2: remove the shadow from every later vector.**

$$ \mathbf{u}_k=\mathbf{b}_k-\sum_{j=1}^{k-1}\frac{\langle\mathbf{b}_k,\mathbf{u}_j\rangle}{\langle\mathbf{u}_j,\mathbf{u}_j\rangle}\,\mathbf{u}_j \qquad (k=2,\dots,n) $$ {#eq-gs}

:::box step | How we got here
- Each fraction times $\mathbf{u}_j$ is [[eq-p]] with $\mathbf{x}=\mathbf{b}_k$ and $\mathbf{b}=\mathbf{u}_j$: the shadow of $\mathbf{b}_k$ on $\mathbf{u}_j$.
- The sum is the total shadow of $\mathbf{b}_k$ on the span of $\mathbf{u}_1,\dots,\mathbf{u}_{k-1}$, which is [[eq-P]] for an orthogonal set.
- Subtracting a shadow leaves the error vector $\mathbf{e}$ of [[eq-error]], which is perpendicular to what it was projected onto.
:::

**Step 3: shrink every vector to length 1.**

$$ \mathbf{e}_i=\frac{\mathbf{u}_i}{\|\mathbf{u}_i\|} $$ {#eq-gs-norm}

:::box warn | Careful with the index
The shadows are taken on the **new** vectors $\mathbf{u}_j$, not on the original $\mathbf{b}_j$. The two agree only when the $\mathbf{b}$'s are already perpendicular.
:::

:::figure F1.3 | Gram-Schmidt stepper | Three vectors in 3D. A "next step" button projects, subtracts and normalizes one vector at a time, showing u1, u2, u3 appear.

## 5. Bridge cheat-sheet: simple case to general case

| | One vector (2D recipe) | Subspace (matrix recipe) |
|---|---|---|
| Unknown | number $\lambda$, [[eq-proj]] | vector $\boldsymbol\lambda$, [[eq-subspace]] |
| Perpendicular condition | $\mathbf{e}\cdot\mathbf{b}=0$, [[eq-orth]] | $\mathbf{B}^\top\mathbf{e}=\mathbf{0}$, [[eq-normal]] |
| Solution | $\lambda=\frac{\mathbf{x}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}$, [[eq-lambda]] | $\boldsymbol\lambda=(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{x}$, [[eq-lambda-general]] |
| Projection | [[eq-p]] | [[eq-P]] |
| With perpendicular unit basis | not needed | [[eq-onb]] |
| Building such a basis | not needed | Gram-Schmidt, [[eq-gs]] |

## 6. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | Linear regression, Ridge, Lasso | The fitted values $\mathbf{X}\hat{\mathbf{w}}$ are the projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$. The normal equations are [[eq-normal]]. |
| Machine Learning | SVM | The distance from a point to a hyperplane uses the same perpendicular drop. |
| Machine Learning | K-Means | A cluster mean is the point closest (in squared distance) to its members. |
| Probability & Statistics | Least squares as MLE, Gauss-Markov, bias-variance | These are statements about this projection when noise is added. |
| Linear Algebra | QR decomposition, PCA, SVD | QR is Gram-Schmidt in matrix form. PCA projects onto the best subspace. |
| Signal Processing / Computer Vision | Fourier and wavelet transforms, denoising, compression | Each is a projection onto an orthonormal basis, as in [[eq-onb]]. |
| Deep Learning | Orthogonal weight initialization, projected gradient methods | Both rely on orthonormal bases and projection matrices. |
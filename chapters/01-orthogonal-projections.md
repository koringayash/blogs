# Orthogonal Projections & Gram-Schmidt

*Part I · Geometry — the idea that quietly powers regression, SVMs and clustering.*

## Before you start

| Topic | Why you need it here |
|---|---|
| Dot product | "Perpendicular" simply means dot product $=0$. |
| Matrix transpose & inverse | The general formula is $\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top$. |
| Span / subspace | The thing we project *onto*. |

## 1. The shadow on the wall

Stand in a room, shine a flat light straight at a wall so rays hit it at $90^\circ$. An object floating in front casts a **shadow**. That shadow is the *orthogonal projection* ("orthogonal" = perpendicular).

:::figure F1.1 | Shadow simulator | 3D room: object, wall, parallel vs point light; sliders for tilt and distance. Toggle shows that only a 90° beam gives the true projection.

## 2. One vector onto a line (2D)

We want the shadow $\mathbf{p}$ of $\mathbf{x}$ on the line along $\mathbf{b}$. Since $\mathbf{p}$ lies on that line, it is a stretched copy of $\mathbf{b}$:

$$\mathbf{p}=\lambda\mathbf{b}$$

The leftover $\mathbf{e}=\mathbf{x}-\mathbf{p}$ is the light beam dropping onto the line, so $\mathbf{e}\perp\mathbf{b}$. Perpendicular means dot product zero:

$$(\mathbf{x}-\lambda\mathbf{b})\cdot\mathbf{b}=0\;\Rightarrow\;\lambda=\frac{\mathbf{x}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}},\qquad \mathbf{p}=\frac{\mathbf{x}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}$$

:::widget projection2d | Drag the red tip of x — watch e·b stay at 0

> **In plain words.** Top: how much $\mathbf{x}$ and $\mathbf{b}$ agree in direction. Bottom: $\|\mathbf{b}\|^2$, which cancels the length of $\mathbf{b}$ (a longer $\mathbf{b}$ must not change the shadow). The ratio is one number, $\lambda$.

**Three ways to see why $\mathbf{e}\perp\mathbf{b}$:** (1) *Shortest walk* — the nearest point on a road to a house is straight across. (2) *Shadow* — if $\mathbf{e}$ were tilted, it would still cast its own shadow on $\mathbf{b}$, so $\mathbf{p}$ wasn't the full shadow. (3) *Calculus* — minimizing $\|\mathbf{x}-\lambda\mathbf{b}\|^2$ over $\lambda$ gives the same equation.

:::figure F1.2 | Three-tab intuition box | Tabs: road / shadow / calculus. Calculus tab plots distance² vs λ with the minimum marked.

## 3. A whole subspace (general case)

Let $U\subseteq\mathbb{R}^n$ have basis $\mathbf{b}_1,\dots,\mathbf{b}_m$, stacked as columns of $\mathbf{B}\in\mathbb{R}^{n\times m}$. The projection is some combination $\pi_U(\mathbf{x})=\mathbf{B}\boldsymbol\lambda$. It is the **closest point of $U$ to $\mathbf{x}$**, so the error $\mathbf{x}-\pi_U(\mathbf{x})$ must be orthogonal to *every* basis vector (m equations at once):

$$\mathbf{B}^\top(\mathbf{x}-\mathbf{B}\boldsymbol\lambda)=\mathbf{0}\;\Rightarrow\;\boldsymbol\lambda=(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{x}$$

$$\pi_U(\mathbf{x})=\underbrace{\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top}_{\mathbf{P}_\pi}\mathbf{x}$$

Properties of $\mathbf{P}_\pi$: **idempotent** ($\mathbf{P}^2=\mathbf{P}$: projecting a shadow changes nothing) and **symmetric** ($\mathbf{P}^\top=\mathbf{P}$).

:::figure F1.3 | 3D subspace projector | Plane U with b₁, b₂; drag x in 3D, orbit camera; button "project twice" shows P²=P.

If the basis is **orthonormal**, $\mathbf{B}^\top\mathbf{B}=\mathbf{I}$ and everything collapses to $\pi_U(\mathbf{x})=\mathbf{B}\mathbf{B}^\top\mathbf{x}$ — no inverse.

## 4. Can we always get an orthonormal basis? Gram-Schmidt

Yes. Walk through the basis; from each $\mathbf{b}_k$ subtract its shadow on the vectors already built:

$$\mathbf{u}_1=\mathbf{b}_1,\qquad \mathbf{u}_k=\mathbf{b}_k-\sum_{j=1}^{k-1}\frac{\langle\mathbf{b}_k,\mathbf{u}_j\rangle}{\langle\mathbf{u}_j,\mathbf{u}_j\rangle}\mathbf{u}_j,\qquad \mathbf{e}_k=\frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

That is just Section 2 repeated: each $\mathbf{u}_k$ is an *error vector*.

:::figure F1.4 | Gram-Schmidt stepper | 3 vectors in 3D; "next step" button projects, subtracts, normalizes.

> **Fixes made to the handwritten notes.** (a) The text said "$\mathbf{x}$ must be orthogonal to all basis vectors"; it is the *error* $\mathbf{x}-\pi_U(\mathbf{x})$. (b) In Gram-Schmidt the sum projects onto the **new** vectors $\mathbf{u}_j$ (not the original $\mathbf{b}_j$) — the two agree only if the $\mathbf{b}$'s are already orthogonal.

## Cheat-sheet: simple → general

| | One vector | Subspace |
|---|---|---|
| Unknown | scalar $\lambda$ | vector $\boldsymbol\lambda$ |
| Condition | $\mathbf{e}\cdot\mathbf{b}=0$ | $\mathbf{B}^\top\mathbf{e}=\mathbf{0}$ |
| Projection | $\frac{\mathbf{b}\mathbf{b}^\top}{\mathbf{b}^\top\mathbf{b}}\mathbf{x}$ | $\mathbf{B}(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{x}$ |
| Orthonormal basis | — | $\mathbf{B}\mathbf{B}^\top\mathbf{x}$ |

## Where this unlocks

- **Linear regression (next chapter):** $\mathbf{X}\hat{\mathbf{w}}$ *is* the projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$; the normal equations are Section 3.
- **SVM:** distance from a point to a hyperplane reuses the same perpendicular-drop idea.
- **K-Means:** the cluster mean is the point minimizing squared distance to its members.
- **Beyond:** PCA, QR decomposition, Fourier/wavelet bases, least-squares solvers, Kalman filters.

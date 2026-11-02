# K-Means

*Part V · Unsupervised. Until now every data point came with a label or a target. Here there are no labels at all, and the machine has to find the groups by itself.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Linear Algebra | [Orthogonal projections (Chapter 1)](chapter.html?c=01-orthogonal-projections) | The point closest to a set of data points, in squared distance, is their mean. |
| Machine Learning | [Decision Trees (Chapter 7)](chapter.html?c=07-decision-trees#eq-reg-imp) | A regression leaf also predicts the mean, for the same reason. |
| Calculus & Optimization | Setting a derivative to zero, checking it is a minimum | Used to derive the mean update. |
| Optimization | Alternating (coordinate) minimization | The algorithm minimizes one set of unknowns at a time. |
| Machine Learning | [Bias and variance (Chapter 4)](chapter.html?c=04-bias-variance) | Choosing $K$ is a complexity choice, like choosing the polynomial degree. |
| Probability & Statistics | Nothing new | This chapter prepares the ground for [EM](chapter.html?c=13-em) and [GMM](chapter.html?c=14-gmm). |

## 1. Everyday picture: where to open the pizza shops

Imagine a city with thousands of houses and a plan to open exactly $K$ pizza shops.

1. **Guess** where the $K$ shops go.
2. **Every house orders from its nearest shop.** That splits the city into $K$ delivery areas.
3. Each shop owner looks at the houses they serve and **moves to the middle of that area**, the spot that is closest, on average, to their customers.
4. Moving a shop changes which shop is nearest for some houses, so **go back to step 2** and repeat.
5. Stop when nobody switches shops any more.

This is the whole of K-Means:
- the **shops** are the cluster centres $\boldsymbol\mu_k$,
- the **houses** are the data points $\mathbf{x}_n$,
- "order from the nearest shop" is the **assignment step**,
- "move to the middle of your customers" is the **update step**.

The rest of the chapter explains why these two simple steps work, why they always stop, and where they can go wrong.

## 2. The clustering problem

**The data.** We have $N$ observations, each a $D$-dimensional vector:

$$ \{\mathbf{x}_1,\dots,\mathbf{x}_N\},\qquad\mathbf{x}_n\in\mathbb{R}^D $$ {#eq-data}

**The goal.** Split them into $K$ groups (**clusters**), where $K$ is given for now (Section 9 shows how to choose it), so that:
1. points in the **same** cluster are close to each other,
2. points in **different** clusters are far apart.

**What "close" means.** "Close" needs a definition of distance, and a clustering of a point cloud also needs a way to say where a group *is*. K-Means makes two choices:
1. Distance is **squared Euclidean distance**, $\lVert\mathbf{x}-\mathbf{y}\rVert^2$.
2. Each cluster is represented by one point, its **prototype** (or centre) $\boldsymbol\mu_k$. A point belongs to the cluster whose prototype is nearest.

**The difficulty.** We do not know the prototypes, and we do not know which point belongs to which cluster. Each unknown would be easy to find *if the other were given*. That chicken-and-egg structure is what the algorithm exploits.

## 3. What is a latent variable?

A **latent variable** is a quantity that **exists in our model, but is never recorded in the data**. We cannot see it, but we believe it helps explain what we do see.

| | Observed | Latent (hidden) |
|---|---|---|
| In this chapter | the position $\mathbf{x}_n$ of each point | **which cluster** each point belongs to |
| A medical example | blood pressure, age, test results | the disease a patient actually has |
| A text example | the words in a document | the topic of the document |

**How K-Means writes the hidden cluster label.** For every point $n$ we introduce $K$ binary **indicator variables**:

$$ r_{nk}\in\{0,1\},\qquad\sum_{k=1}^{K}r_{nk}=1 $$ {#eq-onehot}

- $r_{nk}=1$ means "point $n$ belongs to cluster $k$", and $r_{nk}=0$ otherwise.
- The sum is $1$, so every point belongs to **exactly one** cluster. This is called **1-of-$K$ coding** (or a one-hot vector).
- Example with $K=3$: a point in the second cluster has $(r_{n1},r_{n2},r_{n3})=(0,1,0)$.

:::box plain | Why call these "latent" variables?
- Nobody told us the values of $r_{nk}$. They are **unknown, and we have to infer them** from the positions of the points.
- Together with the prototypes $\boldsymbol\mu_k$, they form everything the model needs to explain where the points are.
- Because each point's label is a *hard* choice (exactly one $1$), K-Means is called **hard clustering**. The [next chapters](chapter.html?c=13-em) replace the hard choice by probabilities (**soft clustering**).
:::

## 4. The objective: the distortion measure

**Step 1: the cost of one point.** If point $n$ belongs to cluster $k$, its cost is the squared distance to that cluster's centre:

$$ \lVert\mathbf{x}_n-\boldsymbol\mu_k\rVert^2 $$

**Step 2: pick out the right cluster with the indicators.** Since $r_{nk}$ is $1$ only for the assigned cluster, the cost of point $n$ is the sum $\sum_kr_{nk}\lVert\mathbf{x}_n-\boldsymbol\mu_k\rVert^2$. All other terms are multiplied by $0$.

**Step 3: add up over all points.**

$$ J(\{r_{nk}\},\{\boldsymbol\mu_k\})=\sum_{n=1}^{N}\sum_{k=1}^{K}r_{nk}\,\lVert\mathbf{x}_n-\boldsymbol\mu_k\rVert^2 $$ {#eq-J}

This $J$ is called the **distortion measure** (or **within-cluster sum of squares**).

:::box plain | What is this objective saying in plain words?
- For each house, measure the squared distance to the shop it uses, then add up over all houses.
- **A small $J$ means every point is close to its centre**, so the clusters are tight.
- Our goal is to find the indicators $\{r_{nk}\}$ **and** the centres $\{\boldsymbol\mu_k\}$ that make $J$ as small as possible.
- Equivalently, $J=\sum_kN_k\operatorname{Var}_k$: the number of points in each cluster times that cluster's spread, added up.
:::

**The optimization problem:**

$$ \min_{\{r_{nk}\},\,\{\boldsymbol\mu_k\}}\;J\qquad\text{subject to [[eq-onehot]]} $$ {#eq-problem}

This is hard to solve all at once, because $\{r_{nk}\}$ takes discrete values (which of $K^N$ possible labelings is best?). But it becomes easy if we fix one set of unknowns and optimize the other, which is the next section.

## 5. The two steps

The algorithm picks starting centres $\boldsymbol\mu_k^{(0)}$ and then alternates between two minimizations. In each, one group of unknowns is held fixed.

### Step A: assignment (fix the centres, find the labels)

With the centres fixed, look at how $J$ in [[eq-J]] is built:

$$ J=\sum_{n=1}^{N}\Big[\sum_{k=1}^{K}r_{nk}\,\lVert\mathbf{x}_n-\boldsymbol\mu_k\rVert^2\Big] $$ {#eq-J-per-point}

**Each point $n$ appears in its own bracket, and the brackets do not share any unknowns.** So we can minimize $J$ by minimizing **each bracket separately**.

For one point, the bracket equals the squared distance to whichever centre it is assigned to. It is smallest if the point is assigned to the **nearest centre**:

$$ r_{nk}=\begin{cases}1&\text{if }k=\arg\min_{j}\lVert\mathbf{x}_n-\boldsymbol\mu_j\rVert^2\\0&\text{otherwise}\end{cases} $$ {#eq-assign}

(If two centres tie, pick the one with the smaller index.)

:::box plain | What is this saying in plain words?
- **Every house orders from its nearest shop.**
- Because the cost of each point is independent of all the others, this simple rule is **exactly optimal** for the given centres, not just a good guess.
:::

### Step B: update (fix the labels, find the centres)

With the labels fixed, minimize $J$ over each centre $\boldsymbol\mu_k$. Only the terms of [[eq-J]] containing $\boldsymbol\mu_k$ matter.

**Step 1: differentiate with respect to $\boldsymbol\mu_k$.**

$$ \frac{\partial J}{\partial\boldsymbol\mu_k}=-2\sum_{n=1}^{N}r_{nk}\,(\mathbf{x}_n-\boldsymbol\mu_k) $$ {#eq-dJ}

:::box step | How we got here
- Only the terms with $r_{nk}=1$ involve $\boldsymbol\mu_k$.
- The gradient of $\lVert\mathbf{x}-\boldsymbol\mu\rVert^2$ with respect to $\boldsymbol\mu$ is $-2(\mathbf{x}-\boldsymbol\mu)$, the same kind of derivative as in [Chapter 2](chapter.html?c=02-linear-regression#eq-grad).
:::

**Step 2: set it to zero and solve.**

$$ \sum_nr_{nk}\,\mathbf{x}_n=\boldsymbol\mu_k\sum_nr_{nk}\;\Longrightarrow\;\boldsymbol\mu_k=\frac{\sum_{n=1}^{N}r_{nk}\,\mathbf{x}_n}{\sum_{n=1}^{N}r_{nk}} $$ {#eq-mean}

**Step 3: check it is a minimum.** The second derivative is $2\sum_nr_{nk}=2N_k>0$, where $N_k$ is the number of points in cluster $k$. A positive second derivative means a minimum, not a maximum.

:::box plain | What is this saying in plain words?
- The numerator adds up the positions of all points in cluster $k$. The denominator counts them. So **$\boldsymbol\mu_k$ is simply the average of the points in cluster $k$**. This is where the name "K-**Means**" comes from.
- **Each shop moves to the middle of its own customers.**
- The same fact appeared in [Chapter 7](chapter.html?c=07-decision-trees#eq-reg-imp): the single number closest, in squared distance, to a list of numbers is their mean.
:::

:::figure F12.1 | K-Means step by step | Points on a plane with K movable centres. An Assign button colours every point by its nearest centre, an Update button moves each centre to the mean of its points, and a readout shows J after each half-step so the decrease can be followed.

## 6. The algorithm, with a worked example

**The algorithm:**
1. **Initialize** the centres $\boldsymbol\mu_1,\dots,\boldsymbol\mu_K$ (Section 8 discusses how).
2. **Assign:** give every point to its nearest centre, using [[eq-assign]].
3. **Update:** move every centre to the mean of its points, using [[eq-mean]].
4. **Repeat** steps 2 and 3 until the assignments stop changing.

**Cost per round.** The assignment step compares each of the $N$ points with each of the $K$ centres in $D$ dimensions, so it costs about $O(NKD)$. The update step costs $O(ND)$.

### A worked example by hand (one dimension)

Six points: $x=1,\,2,\,3,\,10,\,11,\,12$. Choose $K=2$ and start with $\mu_1=1,\ \mu_2=2$. (This is a deliberately unlucky start: both centres sit in the left group.)

| Stage | Centres | Clusters | $J$ |
|---|---|---|---|
| Assign (start) | $\mu_1=1,\ \mu_2=2$ | $\{1\}$ and $\{2,3,10,11,12\}$ | $0+0+1+64+81+100=246$ |
| Update | $\mu_1=1,\ \mu_2=\tfrac{2+3+10+11+12}{5}=7.6$ | same | $0+31.36+21.16+5.76+11.56+19.36=89.2$ |
| Assign | $\mu_1=1,\ \mu_2=7.6$ | $\{1,2,3\}$ and $\{10,11,12\}$ | $0+1+4+5.76+11.56+19.36=41.68$ |
| Update | $\mu_1=2,\ \mu_2=11$ | same | $1+0+1+1+0+1=4$ |
| Assign | $\mu_1=2,\ \mu_2=11$ | no change | $4$, **stop** |

**What we see:**
- $J$ goes $246\to89.2\to41.68\to4$. **It never goes up.** Each of the two steps can only lower it. Section 7 proves this.
- The algorithm found the obvious grouping despite a poor start.
- After the final assignment nothing changes, so the algorithm has reached a fixed point.

## 7. Why it always stops

**Claim 1: the assignment step never increases $J$.** For fixed centres, [[eq-assign]] picks, for every point, the smallest of its $K$ possible costs. Any other assignment costs at least as much.

**Claim 2: the update step never increases $J$.** For fixed labels, [[eq-mean]] is the minimizer of $J$ over each centre (Step 3 above confirmed it is a minimum). Any other centre positions cost at least as much.

**Putting the two claims together:**

$$ J^{(0)}\;\ge\;J^{(1)}\;\ge\;J^{(2)}\;\ge\;\cdots\;\ge\;0 $$ {#eq-monotone}

:::box step | How we got here
- Each half-step of the algorithm is an exact minimization over one group of variables, so it can only keep $J$ the same or lower it.
- $J$ is a sum of squares, so it can never fall below $0$.
- There are only finitely many ways to assign $N$ points to $K$ clusters (exactly $K^N$), and the algorithm never repeats an assignment unless it has stopped: if the assignment is unchanged, so is $J$, and the process has reached a fixed point.
- **Consequence: the algorithm stops after a finite number of rounds.**
:::

**What does it stop at?** A point where neither step can improve $J$. That is a **local minimum** (strictly, a fixed point of the algorithm). It is **not guaranteed to be the global minimum** of [[eq-problem]]: finding the global minimum is a hard (NP-hard) problem, so K-Means is a heuristic that works very well in practice but offers no such guarantee.

:::box plain | The pattern behind this proof
- This **"alternate between two easy exact minimizations"** idea is called **alternating minimization** (or coordinate descent).
- It is the same logic you will see again in [EM](chapter.html?c=13-em), where the proof also relies on each step never making the objective worse.
:::

## 8. Initialization: the result depends on where you start

Because K-Means only finds a local minimum, **different starting centres can lead to different final clusterings.**

### A concrete trap

Four points at the corners of a rectangle: $(0,0)$, $(4,0)$, $(0,2)$ and $(4,2)$. Take $K=2$.

| Clustering | Centres | Cost $J$ |
|---|---|---|
| **Left and right** pairs | $(0,1)$ and $(4,1)$ | each point is $1$ away, so $J=4\times1^2=4$ |
| **Bottom and top** pairs | $(2,0)$ and $(2,2)$ | each point is $2$ away, so $J=4\times2^2=16$ |

**Both are stable.** For the bottom/top clustering, the point $(0,0)$ is at distance $2$ from $(2,0)$ and at distance $\sqrt8\approx2.83$ from $(2,2)$, so it stays where it is, and the same holds for the other three points. The algorithm stops, yet $J=16$ is four times worse than the best value $4$.

### Remedies

1. **Run it many times** from different random starts and keep the clustering with the smallest final $J$. This is the standard practice. Because $J$ always decreases within a run, comparing runs by their final $J$ is meaningful.
2. **Smarter seeding (K-Means++).** Choose the first centre at random from the data. Then pick each following centre from the data points, with probability proportional to the **squared distance to the nearest centre already chosen**. This spreads the starting centres out, so two of them rarely land in the same group. It is the default in most software, and comes with a guarantee that the expected cost is within a factor of order $\log K$ of the best possible.
3. **Handle empty clusters.** If a cluster ends up with no points, the mean in [[eq-mean]] divides by zero. Typical fixes: re-seed that centre at the point farthest from its current centre, or at a random data point.

:::figure F12.2 | Same data, different starts | The same data set clustered from several random initializations, each shown with its final J. Some runs find the good clustering and some stop in a worse one, and the best run is highlighted. A toggle switches to K-Means++ seeding.

## 9. Choosing the number of clusters $K$

So far $K$ was given. In real problems it is not, so we have to choose it.

**A tempting but useless idea: pick the $K$ with the smallest $J$.** Increasing $K$ **always** lowers the best achievable $J$, for the same reason that [more features always lower the training error in Chapter 2](chapter.html?c=02-linear-regression#eq-more-features): a larger family of solutions contains the smaller one. At the extreme $K=N$, every point is its own cluster and $J=0$, which is useless.

**The elbow method.**
1. Run K-Means for $K=1,2,3,\dots$ and record the best final $J$ for each.
2. Plot $J$ against $K$.
3. The curve drops steeply while $K$ is smaller than the true number of groups, since each new centre splits a group that really is two, and then flattens once extra centres only chop real groups into smaller pieces.
4. Choose $K$ at the **elbow**: the bend where the sharp drop turns into a slow decline.

:::box warn | Honest limitations of the elbow method
- The bend is often **not clear**, and different people may pick different $K$ from the same plot.
- It uses only the training distortion. It does not measure how well the clusters would hold up on new data.
- **Other approaches:** the *silhouette score* (how much closer a point is to its own cluster than to the next nearest one), the *gap statistic* (compare $J$ with what random data would give), stability under resampling, and, for the probabilistic version in [Chapter 14](chapter.html?c=14-gmm), likelihood-based criteria such as BIC.
- Often the real answer comes from the **purpose** of the clustering: how many customer groups can the business actually act on?
:::

:::figure F12.3 | Elbow plot | A data set with a slider for K. The clustering is drawn on the left and the curve of the best J against K on the right, with the current K marked. The bend of the curve is highlighted.

## 10. What K-Means assumes, and where it breaks

Each assumption is built into the objective [[eq-J]].

| Hidden assumption | Where it comes from | What goes wrong when it is false |
|---|---|---|
| Clusters are **round (spherical)** | distance is the same in every direction | elongated or curved clusters get cut in the wrong place |
| Clusters have **similar sizes** | each point goes to the nearest centre | a big, spread-out cluster is split, and a small one swallows neighbours |
| Clusters have **similar spread** | one shared notion of distance | a tight cluster beside a wide one gives a poor boundary between them |
| **Euclidean distance** is meaningful | squared distance in [[eq-J]] | needs scaling when features have different units, since the feature with the largest range dominates |
| Every point belongs to **exactly one** cluster | $r_{nk}\in\{0,1\}$ | a point halfway between two groups is forced into one |
| The **mean** is a good summary | [[eq-mean]] | outliers drag the mean away from the bulk of the cluster |

**Practical advice:**
1. **Standardize** features (subtract the mean and divide by the standard deviation) before clustering.
2. **Look at the results.** If you can plot or inspect them, check that the clusters make sense.
3. **Boundaries are straight.** A point is assigned to the nearer of two centres, so the dividing line between two clusters is the straight line (a hyperplane) halfway between the two centres. The plane is cut into convex regions called **Voronoi cells**.

:::figure F12.4 | Voronoi view | The plane coloured by nearest centre for movable centres, showing the straight boundaries between regions. Dragging a centre moves the boundaries, and the points it captures change colour.

:::figure F12.5 | When K-Means fails | Three data sets: two elongated clusters, one large and one small cluster, and two concentric rings. For each, the K-Means result is shown beside the true grouping.

## 11. Looking ahead: K-Means is a special case of something bigger

The two steps have the same shape as the two steps of the algorithm in the next chapter.

| K-Means | [EM algorithm (Chapter 13)](chapter.html?c=13-em) |
|---|---|
| Latent variable: a **hard** label $r_{nk}\in\{0,1\}$ | Latent variable: a **soft** label, a probability |
| Assignment step: nearest centre, [[eq-assign]] | **E-step:** compute the probability that each point belongs to each cluster |
| Update step: mean of the assigned points, [[eq-mean]] | **M-step:** re-estimate the parameters using those probabilities as weights |
| Objective: $J$, always decreases | Objective: the likelihood, always increases |
| Clusters are round and equal | Clusters can have different shapes and sizes ([GMM, Chapter 14](chapter.html?c=14-gmm)) |

**Preview of the idea.** If you replace "every point belongs entirely to one cluster" by "every point belongs to each cluster with some probability", and let each cluster have its own shape and size, you get a **Gaussian mixture model** trained with **EM**. K-Means is what it becomes when the probabilities are forced to be exactly $0$ or $1$ and all clusters are round and equal.

## 12. Bridge cheat-sheet

| Question | Answer | Where |
|---|---|---|
| What is the data? | $N$ vectors in $\mathbb{R}^D$ | [[eq-data]] |
| What is latent? | The cluster of each point | [[eq-onehot]] |
| What is minimized? | Total squared distance to the assigned centres | [[eq-J]] |
| Best labels for fixed centres? | Nearest centre | [[eq-assign]] |
| Best centres for fixed labels? | Mean of the cluster | [[eq-mean]] |
| Does $J$ ever go up? | No | [[eq-monotone]] |
| Does it find the global best? | Not guaranteed | Sections 7 and 8 |
| How to start? | Many random starts, or K-Means++ | Section 8 |
| How to choose $K$? | Elbow, silhouette, and so on | Section 9 |
| Cost per round? | $O(NKD)$ | Section 6 |
| Weakness? | Round, similar-size clusters only | Section 10 |

## 13. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [EM algorithm](chapter.html?c=13-em) and [Gaussian mixture models](chapter.html?c=14-gmm) | The two-step structure of Section 5 is exactly the E-step and M-step. K-Means is the hard, round-cluster special case. |
| Machine Learning | Hierarchical clustering, DBSCAN, spectral clustering | They relax the assumptions in Section 10 (shape, size, need for $K$). |
| Machine Learning | Vector quantization, codebooks, bag-of-visual-words | The centres are a compressed dictionary, and $J$ is the loss caused by replacing each point with its centre. |
| Optimization | Alternating minimization, coordinate descent, local minima | Sections 5 to 8 are the standard small example, including the guarantee that the objective never goes up. |
| Linear Algebra | [Projections](chapter.html?c=01-orthogonal-projections), PCA | The mean update is a projection, and there is a known link between K-Means and PCA through the cluster indicator matrix. |
| Computer Vision | Image compression by colour quantization, image segmentation | Cluster the pixel colours, then replace each pixel by its cluster's centre. |
| Applications | Customer segmentation, document grouping, anomaly detection, initial values for other models | A fast, simple first look at unlabelled data. |
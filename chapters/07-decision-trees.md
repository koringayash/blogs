# Decision Trees

*Part IV · Trees & Ensembles. A model that works like a game of twenty questions, and the first one in this book whose main weakness is variance.*

## Prerequisites

| Domain | Topic | How it is used in this chapter |
|---|---|---|
| Machine Learning | [Bias and variance (Chapter 4)](chapter.html?c=04-bias-variance) | Explains why a fully grown tree overfits. |
| Probability & Statistics | Expectation, class proportions | Entropy and Gini are both expected values over the classes in a node. |
| Calculus & Optimization | Logarithms, concave functions, Jensen's inequality | Used to prove that a split never makes things worse on average. |
| Linear Algebra | [Orthogonal projections (Chapter 1)](chapter.html?c=01-orthogonal-projections) | Used once, to see that a regression leaf predicts a mean. |
| Machine Learning | [Logistic Regression (Chapter 5)](chapter.html?c=05-logistic-regression) | A useful contrast: it draws one smooth line, a tree draws many axis-aligned cuts. |

## 1. Everyday picture: twenty questions

Think of the game where someone picks an animal and you ask yes/no questions.

1. A good first question splits the possibilities into two groups that are each **easier to guess** ("Does it live in water?").
2. A bad question leaves both groups as mixed as before ("Is it alive?").
3. You keep asking, one question per branch, until you can name the answer.

A decision tree is this game played by a computer on your data:
- each **question** is a test on one feature (for example "is the size below 80?"),
- each **answer** sends a data point down the left or right branch,
- at the end of a path, a **leaf** gives the prediction.

The whole chapter answers one question: **what makes a question good, and how does a machine find one?**

## 2. Anatomy of a tree

| Part | What it is | Example |
|---|---|---|
| Root node | All training data, before any question | 10 emails, 6 spam and 4 not spam |
| Split | A test on one feature and a threshold | "number of links $\le 2$?" |
| Child nodes | The two groups created by a split | left: 5 emails, right: 5 emails |
| Leaf | A node we stop at; it makes the prediction | "predict spam" |

**A leaf's prediction.** If a leaf holds $n$ training points and $n_k$ of them belong to class $k$:
1. The predicted class is the **majority** class in the leaf.
2. The predicted probability of class $k$ is the proportion $n_k/n$.

**What each node stores.** A node holds a *distribution of classes*, which means the proportions $P=(p_1,\dots,p_K)$ of each class among the points that reached it. A split is good when it makes the distributions in the two children **purer** than the distribution in the parent. Purity is what we measure next.

## 3. Measuring impurity with entropy

**Step 1: surprise.** How surprised are you when an event of probability $p$ happens?
- A certain event ($p=1$) is not surprising at all.
- A rare event ($p$ close to $0$) is very surprising.
- A function with exactly these properties is $-\log_2 p$, measured in **bits**.

**Step 2: average the surprise.** For a node with class proportions $P$, the *expected* surprise of a randomly drawn point's class is the **entropy**:

$$ H(P)=-\sum_{k=1}^{K}p_k\log_2 p_k $$ {#eq-entropy}

(By convention $0\log_20=0$.)

:::box plain | What is this equation saying in plain words?
- Each class $k$ contributes its surprise $-\log_2p_k$, **weighted by how often it occurs**, $p_k$.
- **Pure node** (one class has $p=1$): entropy is $0$. No surprise, nothing left to ask.
- **Most mixed node** (all $K$ classes equally likely): entropy is $\log_2K$, the maximum. For two classes that is $1$ bit.
- So **lower entropy means a purer node**, and we want splits that lower it.
:::

**Entropy for two classes, with $p$ the proportion of the first class:**

| $p$ | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|
| $H$ in bits | 0 | 0.469 | 0.811 | 1 | 0.811 | 0.469 | 0 |

:::tabs Three ways to read entropy
--- Surprise
Entropy is the average surprise you feel when someone reveals the class of a random point. A node that is $99\%$ one class reveals almost nothing new each time, so entropy is low.
--- Yes/no questions
Entropy is roughly the **average number of yes/no questions** you would need to ask, with a well-designed question list, to find out a point's class. A $50/50$ node needs one question. A $99/1$ node needs, on average, very few.
--- Mixing
Imagine pouring the classes together. Entropy is zero for one pure colour, and grows as more colours are mixed in equal amounts. It measures how "stirred up" the node is.
:::

:::figure F7.1 | Entropy and Gini curves | The entropy curve and the Gini curve for two classes, plotted against the class proportion p. A draggable marker on p shows both values live, and a toggle rescales Gini to compare shapes.

## 4. Information gain: how good is a split?

**Step 1: the idea.** Compare the node before the split with the **average** of the two children after it. The children are averaged **weighted by their size**, because a child with most of the points matters more.

**Step 2: the formula.** Let a split send $n_L$ of the node's $n$ points to the left child and $n_R=n-n_L$ to the right:

$$ \text{IG}=H(N)-\Big[\frac{n_L}{n}\,H(N_L)+\frac{n_R}{n}\,H(N_R)\Big] $$ {#eq-gain}

:::box plain | What is this equation saying in plain words?
- $H(N)$ is how mixed the node was **before** asking the question.
- The bracket is how mixed it is **on average after**, weighting each child by its share of the points.
- The difference is the **reduction in uncertainty** bought by this question. We choose the question with the largest information gain (IG).
:::

### A worked example

A node has $10$ points: $6$ positive and $4$ negative. A candidate split sends **5 points left (all 5 positive)** and **5 points right (1 positive, 4 negative)**.

| Node | Positives and negatives | Entropy |
|---|---|---|
| Parent | 6 and 4, so $p=0.6$ | $-0.6\log_20.6-0.4\log_20.4=0.971$ |
| Left child | 5 and 0 | $0$ |
| Right child | 1 and 4, so $p=0.2$ | $-0.2\log_20.2-0.8\log_20.8=0.722$ |

Weighted average after the split: $\tfrac{5}{10}\cdot0+\tfrac{5}{10}\cdot0.722=0.361$. Information gain by [[eq-gain]]:

$$ \text{IG}=0.971-0.361=0.610\ \text{bits} $$

A split that produced two children with the same mixture as the parent would give $\text{IG}=0$. This one removed about $63\%$ of the uncertainty.

:::figure F7.2 | Split explorer | Points on a line coloured by class. Click anywhere to place a threshold; the two child nodes appear with their class counts, their entropies and the resulting information gain, and the best threshold is marked automatically.

## 5. Why information gain is never negative

It would be alarming if a split could make the average uncertainty **worse**. It cannot, and the proof uses one idea, **Jensen's inequality**.

**Step 1: the parent is a mixture of the children.** The class counts add up: $n_k(N)=n_k(N_L)+n_k(N_R)$. Dividing by $n$ gives the class proportions:

$$ P_N=\frac{n_L}{n}\,P_L+\frac{n_R}{n}\,P_R $$ {#eq-mixture}

Here $P_N,P_L,P_R$ are the class-proportion vectors of the parent and the two children. The parent's distribution is a **weighted average** of the children's distributions, with the weights $\tfrac{n_L}{n}$ and $\tfrac{n_R}{n}$ that already appear in [[eq-gain]].

**Step 2: entropy is a concave function.** Write $H(P)=\sum_k\varphi(p_k)$ with $\varphi(p)=-p\log_2p$. Its second derivative is

$$ \varphi''(p)=-\frac{1}{p\ln 2}<0\qquad\text{for all }p>0 $$ {#eq-concave}

A function with a negative second derivative is **concave**: it bends downwards like a hill, so a straight line drawn between two points of its graph lies **below** the graph. A sum of concave functions is concave, so $H$ is concave.

**Step 3: Jensen's inequality.** For a concave function, "the function of an average" is at least "the average of the function". With weights $\lambda$ and $1-\lambda$:

$$ f\big(\lambda x+(1-\lambda)y\big)\;\ge\;\lambda f(x)+(1-\lambda)f(y) $$ {#eq-jensen}

Apply it to entropy with $\lambda=n_L/n$ and the mixture [[eq-mixture]]:

$$ H(P_N)\;\ge\;\frac{n_L}{n}H(P_L)+\frac{n_R}{n}H(P_R) $$ {#eq-jensen-h}

**Step 4: read off the conclusion.** Move the right side across. The left side minus the right side is exactly $\text{IG}$ from [[eq-gain]]:

$$ \text{IG}\ge0 $$ {#eq-ig-pos}

:::box step | How we got here
- [[eq-mixture]] says the parent is the size-weighted average of its children.
- [[eq-concave]] says entropy is concave, so [[eq-jensen]] applies to it.
- Moving terms in [[eq-jensen-h]] gives [[eq-ig-pos]].
- Because $\varphi$ is **strictly** concave, equality holds only when $P_L=P_R$, that is, when both children have the same class mixture as each other (and so as the parent). A split gains nothing exactly when it tells you nothing.
:::

:::box warn | A subtle point
Information gain is never negative, but it can be **zero** even for a very useful feature, if the usefulness appears only after a second question. Section 9 shows this with the XOR example. That is a limitation of greedy search, not of entropy.
:::

## 6. Another measure: Gini impurity

**The formula.** Gini impurity of a node with class proportions $P$ is

$$ \text{Gini}(P)=\sum_{k=1}^{K}p_k\,(1-p_k) $$ {#eq-gini}

Since $\sum_kp_k=1$, this simplifies:

$$ \text{Gini}(P)=\sum_kp_k-\sum_kp_k^2=1-\sum_{k=1}^{K}p_k^2 $$ {#eq-gini-alt}

:::box plain | What is Gini saying in plain words?
Pick a point at random from the node, then **guess its class by drawing a random label with the node's own proportions**. Gini impurity is the **probability that the guess is wrong**.
- Class $k$ is drawn with probability $p_k$.
- A wrong label is guessed with probability $1-p_k$.
- Summing over classes gives [[eq-gini]].
:::

**Gini and variance.** For two classes with labels $0$ and $1$, the label is a Bernoulli variable with variance $p(1-p)$. Gini is $p(1-p)+(1-p)p=2p(1-p)$, which is exactly **twice that variance**. So Gini measures "how unpredictable is the label in this node", as variance does for numbers.

**Same example, with Gini.** Parent: $1-0.6^2-0.4^2=0.48$. Left child: $0$. Right child: $1-0.2^2-0.8^2=0.32$. The weighted average after the split is $\tfrac5{10}\cdot0.32=0.16$, so the Gini decrease is $0.48-0.16=0.32$.

| | Entropy | Gini |
|---|---|---|
| Zero when | node is pure | node is pure |
| Maximum when | all classes equally likely, $\log_2K$ | all classes equally likely, $1-\tfrac1K$ |
| Two-class maximum | $1$ | $0.5$ |
| Needs a logarithm | yes | no, so slightly faster |
| In practice | picks almost the same splits as Gini | picks almost the same splits as entropy |

The two measures are both zero for a pure node and largest for a perfectly mixed one, and in practice they give very similar trees. The choice rarely matters more than depth or pruning settings.

## 7. Growing a tree

The algorithm is a short recursion:

1. **Start** with all training points in the root.
2. **For every feature** and every candidate threshold, compute the impurity decrease (information gain or Gini decrease).
3. **Choose** the feature and threshold with the largest decrease and split the node.
4. **Repeat** steps 2 and 3 separately inside each child.
5. **Stop** at a node when a stopping rule triggers (it is pure, it has too few points, the best gain is too small, or the maximum depth is reached). That node becomes a leaf.

**Candidate thresholds.** For a numeric feature, sort the values. Only thresholds **between two consecutive different values** can change the split, so there are at most $n-1$ candidates. Sorting once and sweeping from left to right while keeping running class counts lets you score all of them in a single pass.

**Cost.** For $d$ features, one node costs about $O(d\,n\log n)$ operations, dominated by sorting.

## 8. Regression trees

The same tree can predict numbers. Only two things change.

**Step 1: impurity becomes the variance of the targets** in the node:

$$ \text{Imp}(N)=\frac1n\sum_{i\in N}\big(y_i-\bar y_N\big)^2,\qquad\bar y_N=\frac1n\sum_{i\in N}y_i $$ {#eq-reg-imp}

**Step 2: the gain is the drop in weighted variance**, with the same structure as [[eq-gain]]:

$$ \text{Gain}=\text{Imp}(N)-\frac{n_L}{n}\text{Imp}(N_L)-\frac{n_R}{n}\text{Imp}(N_R) $$ {#eq-reg-gain}

**The leaf prediction is the mean.** Why the mean and not, say, the median? A leaf has to give one number $c$ for all its points, and we pay squared error. Treat the $m$ targets in the leaf as a vector $\mathbf{y}$ and the constant prediction as a multiple $c\,\mathbf{1}$ of the all-ones vector. Minimizing $\lVert\mathbf{y}-c\mathbf{1}\rVert^2$ is a projection onto the line spanned by $\mathbf{1}$. By [Chapter 1's formula](chapter.html?c=01-orthogonal-projections#eq-lambda),

$$ c=\frac{\mathbf{y}\cdot\mathbf{1}}{\mathbf{1}\cdot\mathbf{1}}=\frac{\sum_iy_i}{m}=\bar y $$

**The link to classification:** with labels $0$ and $1$, the variance [[eq-reg-imp]] equals $p(1-p)$, so a regression tree with $0/1$ targets is a Gini tree up to a factor of $2$.

## 9. Why greedy splitting is not global optimization

**What the algorithm actually does.** At each node it picks the single split with the highest gain **right now**, then does the same in each child. It never revisits a decision. This is **greedy** search.

**Why not search for the best whole tree?**
- The number of possible trees grows exponentially with the number of features and thresholds.
- Finding the *best* tree under common criteria is known to be **NP-hard**: no efficient algorithm is known that is guaranteed to find it.
- So practical software always uses greedy splitting.

**The loophole: no look-ahead.** A greedy algorithm judges one split at a time, so it cannot see that two weak questions together might be excellent.

**Example: XOR.** Four points, two binary features, label $=1$ exactly when the two features differ:

| $x_1$ | $x_2$ | label |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

1. Split on $x_1$. Each child is a $50/50$ mixture, exactly like the parent, so $\text{IG}=1-1=0$. (By the equality case after [[eq-ig-pos]].)
2. Split on $x_2$: the same, $\text{IG}=0$.
3. So **neither useful feature looks better than nothing** on its own, yet splitting on $x_1$ and then on $x_2$ classifies all four points perfectly.
4. If a noisy third feature happens to have a tiny positive gain on a finite sample, the greedy tree will choose it first and miss the real structure.

:::figure F7.4 | Greedy trap | The XOR data set with a toggle for adding a random noise feature. A panel shows the gain of every possible first split, with the useful ones at zero, and the tree the greedy algorithm builds.

**The way out.** Instead of one perfect tree, build **many different greedy trees** on different samples of the data or features and combine them. Ensembles of trees often beat any single tree by a wide margin, and the rest of this part of the book is about how.

## 10. Why trees have high variance

**What happens when we grow a tree with no limit.**
1. Keep splitting until every leaf is pure.
2. Unless two identical inputs have different labels, **training error is exactly zero**.
3. So the tree is flexible enough to follow the training labels perfectly: **bias is close to zero**.

**Why that costs us variance.** In the notation of [Chapter 4's decomposition](chapter.html?c=04-bias-variance#eq-bv), test error is $\text{Bias}^2+\text{Variance}+\sigma^2$.

1. **The top split depends on small details.** Change a few training points and a different feature can win the first split.
2. **A different first split changes everything below it**, because the two children now contain different points and are grown from scratch.
3. **Deep leaves hold very few points**, so each leaf prediction is based on one or two noisy examples.
4. The result: **two training sets from the same source give very different trees.** That is high variance.

| Tree size | Bias | Variance | Behaviour |
|---|---|---|---|
| Stump (one split) | high | low | underfits |
| Medium depth | medium | medium | often a good compromise |
| Fully grown | near zero | high | overfits |

:::figure F7.3 | Tree depth vs decision boundary | A two-class data set with a depth slider. At depth 1 the boundary is a single cut; at full depth it wraps around individual points. Training error and test error are plotted beside it, with the test error rising again at large depth.

:::figure F7.5 | Resample the data | A button that redraws the training set from the same source and refits the tree. The top split and the branches below it change from draw to draw, while a logistic regression line on the same draws barely moves.

## 11. Taming the variance

There are three levers, from mild to strong.

**Lever 1: stop growing early.** Limit the maximum depth, require a minimum number of points per leaf, or refuse splits with a small gain. This raises bias a little and lowers variance.

**Lever 2: grow big, then prune.** Cost-complexity pruning keeps the tree small by charging for every leaf. For a tree $T$ with leaves $t$:

$$ R_\alpha(T)=\sum_{t\in\text{leaves}(T)}\ \sum_{i\in t}\ell\big(y_i,\hat y_t\big)+\alpha\,\lvert T\rvert $$ {#eq-prune}

- $\ell$ is the loss (misclassification or squared error) and $\lvert T\rvert$ is the number of leaves.
- $\alpha\ge0$ is a price per leaf, exactly like the penalty $\lambda$ in [Ridge](chapter.html?c=03-ridge-lasso#eq-ridge). Larger $\alpha$ gives a smaller tree.
- Choose $\alpha$ by cross-validation, as in [Chapter 4](chapter.html?c=04-bias-variance).

**Lever 3: combine many trees (ensembles).** This is the strongest remedy. There are two main recipes, with different goals:

| | Bagging | Boosting |
|---|---|---|
| Idea | Train many trees **independently** on resampled data and average them | Train trees **one after another**, each fixing the mistakes of those before |
| Trees are | deep, low-bias, high-variance | shallow, high-bias "weak learners" |
| Main effect | **reduces variance** | **reduces bias** |
| Parallelizable | yes | no, each step needs the previous one |
| Well-known examples | Random Forest | AdaBoost, Gradient Boosting, XGBoost |
| Chapters | [Chapter 8](chapter.html?c=08-random-forest) | [Chapter 9](chapter.html?c=09-gradient-boosting), [10](chapter.html?c=10-adaboost), [11](chapter.html?c=11-xgboost) |

:::box plain | What is the difference in plain words?
- **Bagging** asks a crowd of independent, individually unreliable experts and takes a vote. Disagreement between them cancels out, which cuts variance.
- **Boosting** builds a team of simple experts in a line. Each new expert studies where the team so far is wrong and specializes in that. Together they can represent something far richer than any one of them, which cuts bias.
:::

## 12. Bridge cheat-sheet

| Question | Answer | Where |
|---|---|---|
| How mixed is a node? | Entropy or Gini | [[eq-entropy]], [[eq-gini]] |
| How good is a split? | Impurity before minus weighted impurity after | [[eq-gain]] |
| Can a split hurt? | Not on average: $\text{IG}\ge0$ | [[eq-ig-pos]] |
| Equal to zero when? | Both children look like the parent | after [[eq-ig-pos]] |
| What does Gini mean? | Chance of a wrong random guess | [[eq-gini-alt]] |
| What about numeric targets? | Use variance, predict the mean | [[eq-reg-imp]], [[eq-reg-gain]] |
| Why greedy? | Searching all trees is NP-hard | Section 9 |
| Why high variance? | Small data changes alter early splits | Section 10 |
| How to control it? | Stop early, prune, or ensemble | [[eq-prune]], Section 11 |

## 13. Where this unlocks deeper understanding

| Domain | Topics and applications | How this chapter helps |
|---|---|---|
| Machine Learning | [Random Forest](chapter.html?c=08-random-forest) | Its whole design (deep trees, bootstrap samples, random features) answers the variance problem of Section 10. |
| Machine Learning | [Gradient Boosting](chapter.html?c=09-gradient-boosting), [XGBoost](chapter.html?c=11-xgboost) | They fit small regression trees using [[eq-reg-imp]] and [[eq-reg-gain]] as the building block. |
| Machine Learning | [AdaBoost](chapter.html?c=10-adaboost) | Its weak learner is a one-split tree, the stump of Section 10. |
| Probability & Statistics | Information theory: KL divergence, mutual information | Information gain is the **mutual information** between the feature test and the class. |
| Optimization | NP-hardness, greedy and approximate algorithms | Section 9 is a clean everyday example of why we settle for greedy search. |
| Interpretability | Feature importance, SHAP values for trees | Importance scores are built from the total impurity decrease each feature produces. |
| Applications | Credit scoring, medical triage, fraud detection, tabular data competitions | A single tree is easy to explain, and tree ensembles are among the strongest models for table-shaped data. |
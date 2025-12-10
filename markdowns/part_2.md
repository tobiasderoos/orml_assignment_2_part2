# RL Hyper-Heuristic for Greedy–ILP Threshold Selection

This module applies **Q-learning** to determine how many items the greedy heuristic should select before switching to the ILP solver in the **Quadratic Knapsack Problem (QKP)**.  
The goal is to learn an **adaptive stopping criterion** that improves solution quality while keeping ILP computation time low.

The agent makes only **one** decision per episode:  
select the greedy threshold \(k\) at the start, based on the observed instance characteristics.  
After this, the greedy phase and ILP phase run automatically.

---

## Greedy Heuristic

The greedy heuristic selects, at each step, the item with the highest **adjusted profit-to-weight ratio**:

\[
\text{best\_item} = \arg\max_{i \in \text{candidates}}
\frac{p_i + m_i}{w_i}
\]

where  
- \(p_i\) = base profit of item \(i\)  
- \(w_i\) = weight of item \(i\)  
- \(m_i\) = marginal profit of item \(i\)

### Marginal Profit

\[
m_i = \sum_{j \in \text{selected}} q_{ij}
\]

where \(q_{ij}\) represents the quadratic (pairwise) profit contribution between items \(i\) and \(j\).

---

## States

The state encodes instance characteristics that influence the optimal stopping threshold.

### 1. Theoretical Capacity

1. Compute average item weight:  
\[
\bar{w} = \frac{1}{n}\sum_{i=1}^n w_i
\]

2. Compute theoretical capacity:  
\[
T = \frac{W}{\bar{w}}
\]

3. Discretize \(T\) into 3 buckets (33rd and 66th percentile).

A larger \(T\) implies more items fit, suggesting a larger stopping threshold.

### 2. Fraction of High Profit/Weight Items

Compute  
\[
r_i = \frac{p_i}{w_i}
\]

Let \(\bar{r}\) be the mean ratio and discretize into 3 buckets.

The higher \(\bar{r}\), the better greedy tends to perform.

### Combined State

\[
\text{state} = \text{ratio\_bin} \times 3 + \text{capacity\_bin}
\]

→ 9 total states.

---

## Actions

Actions specify how many greedy selections before switching to ILP:

\[
k \in \{5, 10, 20, 30, 50\}
\]

### Rationale

- Coarse thresholds reduce computation time.
- ILP corrects remaining choices, so exact threshold values are not required.
- Choosing \(k = 15\) when the optimum is \(17\) yields the same ILP solution, with small overhead.

---




## Rewards

Rewards evaluate the quality of the combined greedy + ILP solution.

### Reward scaling
Rewards are normalized using full ILP solution and Greedy-only soluiton:
- **Full ILP solution** → reward = 1  
- **Greedy-only solution** → reward = 0  
- **Reduced ILP solution (RILP)** → reward in (0, 1): RILP / (full_ILP - Greedy)

### Negative reward
Negative rewards are assigned when:
- ILP finds **no feasible solution** --> negative reward 2
- if ILP ecxeeds the time limit of 15 seconds --> negative reward of 0.5 

### Timing reward / penalty

Because a faster ILP solve is preferable, reward incorporates a small penalty for runtime:

The closer to 15 secondds, the better. Greedy heuristic is baseline, 15 seconds is upper limit. 

This should ensure that higher thresholds are preferred over lower thresholds, even if they give the same optimal value. 

### Goal
Reward solutions that are:
- close to the full ILP optimum,
- found within the time limit,
- produced using an appropriate greedy–ILP split.

---

## Q-Learning Parameters

- **α (alpha)** – learning rate  set to 0.25 . High learning rate as there is no ...
- **ε (epsilon)** – exploration probability 
- **ε-decay** – gradually reduces exploration  
- No discount factor γ is needed, as each episode consists of a **single decision**.

### Effective update rule
Since the problem is a contextual bandit, the Q-update simplifies to:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big( r - Q(s,a) \big)
\]

This update moves Q(s, a) toward the observed reward, learning the expected performance of each threshold under different instance characteristics.


# Exc 2 

decides which item to insert next into the knapsack. Unlike Exercise 1, the agent here directly selects one
item (from a set) to add, rather than choosing a heuristic move

## States

- A feasibile mask, which item is available?
- Remaining capacity 
- Item features (normalized)
  - weight
  - profit
  - quadratic profit interaction when added
- knapsack features:
  - weight
  - selected items


  ## Actions

- Items possible to select

If already selected, then what?

Q[i] = -∞  if available[i] = 0

- Reward

Q[i] = -∞  if available[i] = 0

- Infeasibility penalty 

# Epsiode ends

- no more items to add
- capacity constraint met
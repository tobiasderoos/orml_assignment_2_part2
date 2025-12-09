# RL Hyper-Heuristic for Greedy–ILP Threshold Selection

This module applies Q-learning to determine how many items the greedy heuristic should select before switching to the ILP solver in the Quadratic Knapsack Problem (QKP).  
The goal is to learn an adaptive stopping criterion that improves solution quality while keeping ILP computation time low.

The agent makes only one decision per episode:  
select the greedy threshold \(k\) at the start, based on the observed instance characteristics.  
After this decision, the greedy phase and ILP phase run automatically.

---

## States

The state represents coarse instance characteristics that influence the optimal stopping threshold.

### Current implementation

### Theoretical capacity
- Compute the average item weight.
- Compute the theoretical capacity:
  \[
  T = \frac{W}{\bar{w}}
  \]
- Discretize \(T\) into 5 buckets to ensure balanced visitation during training.

A larger theoretical capacity implies that more items can fit in the knapsack, so a larger greedy threshold is often beneficial.

# Fraction high state

Fraction items with a high profit/weight ratio. The higher the fracitons, the better greedy works. 


---

## Actions

Actions specify the number of greedy selections before switching to ILP: {5, 10, 15 ... 100}

May not produce the best optimal threshold, but: 
If the optimum is for example 17, the model should pick 15 as this would also give the optimal solution, only take a bit longer to compute as you have two more items for the ILP to solve.

### Rationale
- Using increments of 5 significantly reduces computational cost, and accelerates convergence.
- These thresholds still cover the full practical range of early vs. late ILP intervention.

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

- **α (alpha)** – learning rate  
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
# RL Hyper-Heuristic for Greedy–ILP Threshold Selection

This module applies **Q-learning** to automatically determine how many items should be selected by the greedy heuristic before switching to the ILP solver in the Quadratic Knapsack Problem (QKP).  
The goal is to learn an adaptive stopping criterion that increases solution quality while reducing ILP computation time.

---

## States

The state represents coarse instance characteristics that correlate with the optimal stopping threshold.

### Current implementation
- Compute average item weight.
- Compute theoretical capacity:
  \[
  T = \frac{W}{\bar{w}}
  \]
- Discretize \(T\) into 5 buckets  
  (ensures balanced training across states)

**Interpretation:**  
A larger knapsack → more items fit → a higher greedy threshold is typically more effective.

### Possible improvements
- Include **profit-weight ratio statistics** (mean, variance).
- Add interaction variability . more variability between interactions -> ILP should intervene sooner
## Actions

Actions correspond to the number of greedy selections before switching to ILP:

### Rationale
- Reduces computational complexity.
- Covers the practical range for threshold values.

### Optional: Interpolation
To avoid evaluating unlisted thresholds (e.g., 6–9),  
Q-values can be **linearly interpolated** between adjacent actions.

---

## Rewards

Rewards measure the quality of the solution produced by the combination of greedy + ILP.

### Current reward logic
- High reward when ILP returns a high objective value.
- Negative reward when:
  - ILP finds no feasible solution.
  - ILP reaches the time limit.

### Recommended improvements
#### **1. Normalize reward using the full ILP optimum**
Instead of dividing by a constant (e.g., 10,000), use:

\[
r = \frac{\text{reduced ILP solution}}{\text{full ILP optimal solution}}
\]

Reward ∈ [0, 1], stable and comparable across instances.

#### **2. Time-penalized reward**
Include computational cost:

\[
r = \alpha \cdot \text{solution quality} - \beta \cdot \text{time}
\]

(where α and β are tunable)

#### **3. Reward shaping**
To avoid flat reward landscapes:
- small intermediate reward for greedy phase quality  
- penalty if greedy fills knapsack too early (no room left for ILP improvement)

---

## Q-Learning Parameters

- **α (alpha)** – learning rate  
- **ε (epsilon)** – exploration rate  
- **ε-decay** – decays exploration over training  
- Effective update rule:  
    \[
    Q(s,a) \leftarrow Q(s,a) + \alpha (r - Q(s,a))
    \]

---

## Additional Improvement Ideas

### **1. Use a value baseline**
Reward = ILP solution − greedy-only solution  
This isolates the improvement caused by ILP.



# Exc 2 

decides which item to insert next into the knapsack. Unlike Exercise 1, the agent here directly selects one
item (from a set) to add, rather than choosing a heuristic move

## States

- A high reward if the optimum value is high.
- Negative reward if no solution is found
- negative reward if time limit is reached. 

## Actions

items to pick 

## Rewards


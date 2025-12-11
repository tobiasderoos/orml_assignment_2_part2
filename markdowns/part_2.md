# RL Hyper-Heuristic for Greedy–ILP Threshold Selection

This module applies **Q-learning** to determine how many items the greedy heuristic should select before switching to the ILP solver in the **Quadratic Knapsack Problem (QKP)**.  
The goal is to learn an **adaptive stopping criterion** that improves solution quality while keeping ILP computation time low.

The agent makes only **one** decision per episode:  
select the greedy threshold \(k\) at the start, based on the observed instance characteristics.  
After this, the greedy phase and ILP phase run automatically.

---

# Greedy Heuristic

The greedy heuristic selects items using the marginal linear profit-to-weight ratio:

score(i) = (p_i + Σ_{j∈selected} q_{ij}) / w_i

It selects the item with the highest score and continues until:
- capacity is full, or  
- the agent-specified stopping threshold *k* is reached.

---

# Actions — Stopping Thresholds

The agent chooses:

k ∈ {10, 15, 20, …, 100}

This gives 21 discrete actions each representing “stop greedy after k items.”

Small k:
- ILP has more flexibility  
- ILP takes more time, can take longer than 15 seconds. Hence, this may result in a suboptimal solution. 

Large k:
- greedy dominates and ILP has less correction ability. Also results in a suboptimal solution, closer to greedy solution than optimal ILP solution.

The aim is to learn the threshold that works best per instance. This could improve the ILP solution with 15 seconds time limit in theory. 

---

# State Representation (21 Features)

The model uses a **21-dimensional feature vector** summarizing instance characteristics:

### Weight statistics
- mean weight  
- median weight  
- standard deviation  
- coefficient of variation  
- min/max ratio  

### Capacity structure
- capacity / mean weight  
- capacity / median weight  
- fraction of items lighter than mean  

### Profit/weight structure
- mean (p/w)  
- median (p/w)  
- std (p/w)  

### Quadratic structure
- mean q_ij  
- std q_ij  
- 90th percentile of q_ij  

### One-hot categorical bins
- 3-bin profit/weight ratio category  
- 3-bin capacity category  

### Size feature
- log(n)

The feature vector is fed into a neural network.

---

# Neural Q-Network

Architecture:

Input(21)  
→ Dense(64, relu)  
→ Dense(32, relu)  
→ Dense(21, linear)  (one Q-value per action)

The update rule is contextual-bandit style:

Q(s, a) ← reward

because each episode consists of a single decision; no discount factor is required.

---

# Reward Function

Let:
- G = greedy-only profit  
- R = reduced ILP profit  

Base reward:

reward = (R / G − 1) × 100

Examples:
- ILP improves greedy by 5% → reward = +5  
- ILP equals greedy → reward = 0  

### Penalties
- ILP infeasible → −2.0  
- ILP hits time limit → −0.5  
- Greedy profit ≤ 0 → −2.0  

This encourages:
- ILP improvements  
- stability  
- feasible ILP solves  
- reasonable runtimes  

---

# ILP Caching

Reduced ILP calls are memoized:

cache[(filepath, threshold, fixed_items)] → (objective, solution, status)

This speeds up training significantly by avoiding repeated ILP solves.

---

# Exploration

ε-greedy strategy:
- Start ε = 1.0  
- Decay per episode: ε ← ε × 0.9975  
- Minimum ε = 0.05  

The agent gradually shifts from exploration to exploitation.

---

# Convergence Metrics

Logged to TensorBoard:

- action entropy (distribution concentration)  
- policy stability (changes in argmax(Q))  
- exploitation rate  
- top-3 action reward vs others  
- reward improvement over rolling windows  
- Q-value spread (range of Q-values across actions)  

These help diagnose when the policy stabilizes.

---

# Diagnostic Plots

The plotting script provides:
- reward and optimality curves  
- epsilon decay  
- Q-value statistics  
- heatmap of action distributions  
- policy stability curves  
- reward-by-action bar charts  
- TD-error magnitudes  

---

# Training Workflow

Example:

agent = NeuralBanditQLearning(instance_files, reset_params=True, model_name="exc_1_model/qkeras_model")  
agent.train(n_episodes=1000)  
print_convergence_summary(agent)  
plot_qlearning_diagnostics(agent)

The system saves:
- the neural model (.keras)  
- training parameters (.yaml)  
- plots and TensorBoard logs  

---

# Summary

This module implements a **neural contextual bandit hyper-heuristic** that learns when to stop the greedy heuristic and hand off to ILP in QKP. It uses:
- rich statistical features,  
- a neural Q-function,  
- ILP caching,  
- structured reward shaping,  
- detailed convergence tracking.  

The result is an adaptive, data-driven thresholding strategy that improves greedy–ILP hybrid performance across instances.


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
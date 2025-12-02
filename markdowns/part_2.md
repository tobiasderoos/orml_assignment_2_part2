# Exc1

States is based on how big the knapsack is. The bigger, the more items it can carry thus it should increase the threshold.

These are determined in such a way that training is equally distributed. 

Actions

The different actions are based on increments of 5 in order to reduce computational complexity. Hence, 5, 10, ... 50. Should cover most

Possible: interpolate between actions? 

## Rewards

- A high reward if the optimum value is high.
- Negative reward if no solution is found
- negative reward if time limit is reached. 

## Q learning parameters

- alpha learning rate
- gamma
- epsilon
- decay

# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP

## Aim

To implement iterative policy evaluation using the Gymnasium FrozenLake-v1 environment and estimate the state-value function \(V^\pi(s)\) for a fixed random policy.

---

## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from the start state and attempts to reach the goal while avoiding holes.

### Environment Details

| Component | Description |
|-----------|-------------|
| Observation Space | 16 discrete states (4 × 4 grid) |
| Action Space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal States | Goal state and hole states |
| Transition Model | Stochastic (when `is_slippery=True`) |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy where each action is selected with equal probability.

\[
\pi(a|s)=\frac{1}{4}
\]

The transition probabilities are obtained from the environment using:

```python
env.P[state][action]
```

The objective is to estimate the state-value function:

\[
V^\pi(s)
\]

using iterative policy evaluation until convergence.

---

## Theory

The state-value function under a policy \(\pi\) is the expected cumulative discounted reward obtained by starting from state \(s\) and following the policy thereafter.

The Bellman Expectation Equation is

```math
V^\pi(s)=
\sum_a \pi(a|s)
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')+\gamma V^\pi(s')
\right]
```

where

| Symbol | Meaning |
|---------|---------|
| \(s\) | Current state |
| \(a\) | Action |
| \(s'\) | Next state |
| \(\pi(a \mid s)\) | Policy probability of selecting action \(a\) |
| \(P(s' \mid s,a)\) | Transition probability |
| \(R(s,a,s')\) | Immediate reward |
| \(\gamma\) | Discount factor |
| \(V^\pi(s)\) | State-value function |

The algorithm repeatedly applies the Bellman expectation update until the maximum change in state values becomes smaller than a predefined threshold.

---

## Algorithm

1. Import Gymnasium and NumPy.
2. Create the FrozenLake-v1 environment.
3. Access the transition model using `env.P`.
4. Initialize the state-value function \(V(s)=0\) for all states.
5. Define a random policy where every action has probability 0.25.
6. For every state:
   - For each possible action:
     - Obtain the transition probability, next state, reward, and terminal flag.
     - Update the expected value using the Bellman expectation equation.
7. Repeat the above updates until convergence.
8. Print the total number of iterations.
9. Display the final state-value function as a 4 × 4 grid.

---

# Program

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Access the unwrapped environment to use the transition model
env = env.unwrapped
print(env)

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Parameters
gamma = 0.99
theta = 1e-8

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions

# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------
def policy_evaluation(policy, env, gamma=0.99, theta=1e-8):
    V = np.zeros(n_states)
    iterations = 0
    delta_history = []  # To store delta at each iteration

    while True:
        delta = 0

        for state in range(n_states):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * prob * (reward + gamma * V[next_state] * (not done))

            delta = max(delta, abs(v - V[state]))
            V[state] = v

        iterations += 1
        delta_history.append(delta)

        if delta < theta:
            break

    return V, iterations, delta_history

# Run policy evaluation
V, iterations, delta_history = policy_evaluation(policy, env, gamma, theta)

print("Name: Vikamuhan Reddy")
print("Register Number: 212223240181")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name: Vikamuhan Reddy")
print("Register Number: 212223240181")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```
---

# Output

<img width="514" height="257" alt="image" src="https://github.com/user-attachments/assets/3bc25ad1-184d-443e-9add-cfb3a8f7cdc5" />



# Result

The iterative policy evaluation algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The state-value function for the fixed random policy was computed by repeatedly applying the Bellman expectation equation until convergence.

---

# Inference
• The random policy converged after approximately 131 iterations using iterative policy evaluation.

• States closer to the goal obtained higher state values, while hole and terminal states had a value of 0.

• The computed state-value function represents the expected discounted return for each state under the fixed random policy.

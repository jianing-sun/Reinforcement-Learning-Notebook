# Some times, solving the linear system for policy evaluation may be
# too time consuming (e.g., for large state spaces).
import numpy as np
import random

states = [0, 1]
actions = [0, 1, 2]
discount_factor = 0.95

# Transition Probabilities: P[s',s,a]  Rewards: r[s,a]
P = np.zeros([len(states), len(states), len(actions)])
R = np.zeros([len(states), len(actions)])

P[0, 0, 0] = 0.5
P[1, 0, 0] = 0.5
P[0, 0, 1] = 0
P[1, 0, 1] = 1
P[1, 0, 2] = 0
P[1, 1, 2] = 1

R[0, 0] = 5
R[0, 1] = 10
R[1, 2] = -1


def policy_iteration():
    U = dict([(s, 0) for s in states])
    pi = dict([(s, random.choice(actions)) for s in states])
    while True:
        U = modified_evaluation(pi, P, R, U)
        policy_stable = True
        for s in range(len(states)):
            action_array = np.full(len(actions), fill_value=-np.inf)  # np.zeros(len(actions))
            for a in range(len(actions)):

                if (s == 0 and a == 2) or (s == 1 and a == 0) or (s == 1 and a == 1):
                    continue

                action_array[a] = sum(
                    P[next_s, s, a] * (R[s, a] + discount_factor * U[next_s]) for next_s in
                    range(len(states)))

            new_a = np.argmax(action_array)

            if new_a != pi[s]:
                pi[s] = new_a
                policy_stable = False

        if policy_stable:
            return pi


def modified_evaluation(pi, P, R, U, k=20):
    for i in range(k):
        for s in range(len(states)):
            U[s] = R[s, pi[s]] + discount_factor * sum([P[next_s, s, pi[s]] * U[s] for next_s in range(len(states))])
    return U


policy = policy_iteration()
print(policy)

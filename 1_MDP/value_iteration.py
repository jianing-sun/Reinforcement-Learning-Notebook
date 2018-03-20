import numpy as np


def value_iteration(P, R, theta=0.0001, discount_factor=0.95):
    def one_sweep(state, V):
        A = np.zeros(len(actions))
        # A = np.full(len(actions), fill_value=-np.inf)
        for a in range(len(actions)):
            A[a] = sum([P[next_s, state, a] * (R[state, a] + discount_factor * V[next_s]) for next_s in
                        range(len(states))])
            if(state == 0 and a == 2) or (state == 1 and a == 0) or (state == 1 and a == 1):
                A[a] = -np.inf
        return A

    V = np.zeros(len(states))
    counter = 0

    while True:
        counter += 1
        delta = 0
        for s in range(len(states)):
            A = one_sweep(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([len(states), len(actions)])
    for s in range(len(states)):
        A = one_sweep(s, V)
        best_action = np.argmax(A)

        policy[s, best_action] = 1.0
    return policy, V, counter


if __name__ == '__main__':
    states = [0, 1]
    actions = [0, 1, 2]
    discount_factor = 1

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

    policy, v, counter = value_iteration(P, R)

    print(counter)
    print("Policy Probability Distribution: ")
    print(policy)
    print("")

    print("Value Function:")
    print(v)
    print("")

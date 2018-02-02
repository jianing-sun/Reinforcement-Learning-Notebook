import numpy as np


def policy_eval(policy, P, R, gamma, theta=0.0001):
    V = np.zeros(len(states))
    # repeat until delta < theta, then break
    while True:
        delta = 0
        # for each state, full backup
        for s in range(len(states)):
            v = V[s]
            # calculate values for every state (no improvement yet)
            V[s] = sum([P[next_s, s, policy[s]] * (R[s, policy[s]] + gamma * V[next_s]) for next_s in
                        range(len(states))])
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return np.array(V)


def policy_improve(policy_eval_fn=policy_eval, gamma=0.95):
    # Initialization policy and value arbitrarily for all states
    policy = [0 for s in range(len(states))]

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, P, R, discount_factor)
        policy_stable = True

        for s in range(len(states)):
            # old action
            old_a = policy[s]

            # Find the best action
            action_array = np.zeros(len(actions))
            for a in range(len(actions)):
                action_array[a] = sum(
                    P[next_s, s, a] * (R[s, a] + gamma * V[next_s]) for next_s in range(len(states)))
            # for a in range(len(actions)):
            #     policy[s] = np.argmax(
            #         sum(P[next_s, s, a] * (R[s, a, next_s] + gamma * V[next_s]) for next_s in range(len(states))))
            new_a = np.argmax(action_array)
            if new_a != old_a:
                policy_stable = False
            policy[s] = new_a

        if policy_stable:
            return policy, V


if __name__ == '__main__':
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



    policy, V = policy_improve()

    print("final policy: ", policy)
    print("value: ", V)

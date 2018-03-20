import numpy as np


def policy_eval(policy, P, R, gamma, theta=0.0001):

    V = np.zeros(len(states))
    # repeat until delta < theta, then break
    counter = 0
    while True:
        delta = 0
        # for each state, full backup
        for s in range(len(states)):

            v = 0

            # V[s] = 0
            # calculate values for every state (no improvement yet)
            # for a in range(len(actions)):
            #
            #     if (s == 0 and a == 2) or (s == 1 and a == 0) or (s == 1 and a == 1):
            #         continue
            #
            #     V[s] = sum([P[next_s, s, a] * (R[s, a] + gamma * V[next_s]) for next_s in
            #                 range(len(states))])

            for a, a_prob in enumerate(policy[s]):
                print('a: {}, a_prob: {}'.format(a, a_prob))
                if (s == 0 and a == 2) or (s == 1 and a == 0) or (s == 1 and a == 1):
                    continue

                v += sum([a_prob * P[next_s, s, a] * (R[s, a] + gamma * V[next_s]) for next_s in range(len(states))])

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        counter += 1
        print('state value: {}'.format(V))
        if delta < theta:
            print(counter)
            break
    return np.array(V)


def policy_improve(policy_eval_fn=policy_eval, gamma=0.95):
    # Initialization policy and value arbitrarily for all states
    # policy = [0 for s in range(len(states))]
    policy = np.zeros([len(states), len(actions)], dtype=int)
    policy[0][0] = 1
    policy[1][2] = 1
    # print(policy)

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, P, R, gamma=0.95)
        policy_stable = True

        for s in range(len(states)):
            # old action
            old_a = np.argmax(policy[s])

            # Find the best action
            action_array = np.full(len(actions), fill_value=-np.inf)  # np.zeros(len(actions))
            for a in range(len(actions)):

                if (s == 0 and a == 2) or (s == 1 and a == 0) or (s == 1 and a == 1):
                    continue

                action_array[a] = sum(
                    P[next_s, s, a] * (R[s, a] + gamma * V[next_s]) for next_s in
                    range(len(states)))
            new_a = np.argmax(action_array)

            if new_a != old_a:
                policy_stable = False

            policy[s] = np.eye(len(actions))[new_a]

            print("policy", policy[s])

        if policy_stable:
            return policy, V


if __name__ == '__main__':
    states = [0, 1]
    # actions = [[0, 1], [2]]
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

    print("final policy")
    print(policy)
    print("value")
    print(V)

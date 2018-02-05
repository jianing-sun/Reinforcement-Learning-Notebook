import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def gridWorld():

    shape = [4, 4]
    # define the boundary
    MAX_X = shape[0]
    MAX_Y = shape[1]

    # define the grid
    nActions = 4
    nStates = np.prod(shape)

    grid = np.arange(nStates).reshape([nActions, nActions])
    P = {}

    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.nditer.html
    it = np.nditer(grid, flags=['multi_index'])

    while not it.finished:
        s = it.iterindex
        y, x = it.multi_index  # y -> number of row, x -> number of column

        # initialize prob distribution
        P[s] = {a: [] for a in range(nActions)}

        # check if arrive at terminal states (position 0 and 15)
        check_terminal = lambda s: s == 0 or s == (nStates - 1)
        reward = 0.0 if check_terminal(s) else -1.0

        if check_terminal(s):
            # got terminal, next state should be itself
            P[s][UP] = [(1.0, s, reward)]
            P[s][DOWN] = [(1.0, s, reward)]
            P[s][LEFT] = [(1.0, s, reward)]
            P[s][RIGHT] = [(1.0, s, reward)]

        else:

            ns_up = s if y == 0 else s - MAX_X
            ns_right = s if x == (MAX_X - 1) else s + 1
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s - 1

            P[s][UP] = [(1.0, ns_up, reward)]
            P[s][RIGHT] = [(1.0, ns_right, reward)]
            P[s][DOWN] = [(1.0, ns_down, reward)]
            P[s][LEFT] = [(1.0, ns_left, reward)]

        it.iternext()

    # Initial state distribution is uniform
    isd = np.ones(nStates) / nStates

    return nStates, nActions, P, isd


def value_iteration(P, theta=0.0001, discount_factor=0.95):
    def one_sweep(state, V):
        action_val = np.zeros(nActions)
        # A = np.full(len(actions), fill_value=-np.inf)
        for a in range(nActions):
            for prob, next_s, reward in P[s][a]:
                action_val[a] = prob * (reward + discount_factor * V[next_s])
            if (state == 0 and a == 2) or (state == 1 and a == 0) or (state == 1 and a == 1):
                action_val[a] = -np.inf
        return action_val

    V = np.zeros(nStates)
    counter = 0

    while True:
        counter += 1
        delta = 0
        for s in range(nStates):
            action_val = one_sweep(s, V)
            best_action_value = np.max(action_val)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([nStates, nActions])
    for s in range(nStates):
        action_val = one_sweep(s, V)
        best_action = np.argmax(action_val)

        policy[s, best_action] = 1.0
    return policy, V


if __name__ == '__main__':

    nStates, nActions, P, isd = gridWorld()

    policy, V = value_iteration(P)

    print("policy")
    print(policy)
    print("")

    print("value")
    print(V)
    print("")

    print(V.reshape([4, 4]))

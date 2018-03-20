import numpy as np
from pycolab import ascii_art
from pycolab.prefab_parts import sprites as prefab_sprites
import matplotlib.pyplot as plt


def make_BoyanChain(art):
    return ascii_art.ascii_art_to_game(art, what_lies_beneath=' ', sprites={'P': PlayerSprite_BoyanChain})


def make_FiveStates(art):
    return ascii_art.ascii_art_to_game(art, what_lies_beneath=' ', sprites={'P': PlayerSprite_FiveStates})


class PlayerSprite_FiveStates(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        super(PlayerSprite_FiveStates, self).__init__(corner, position, character, impassable='#')

        self.transition_matrix = np.array([
            [0.42, 0.13, 0.14, 0.03, 0.28],
            [0.25, 0.08, 0.16, 0.35, 0.15],
            [0.08, 0.20, 0.33, 0.17, 0.22],
            [0.36, 0.05, 0.00, 0.51, 0.07],
            [0.17, 0.24, 0.19, 0.18, 0.22]
        ])

        self.reward_matrix = np.array([
            [104.66, 29.69, 82.36, 37.49, 68.82],
            [75.86, 29.24, 100.37, 0.31, 35.99],
            [57.68, 65.66, 56.95, 100.44, 47.63],
            [96.23, 14.01, 0.88, 89.77, 66.77],
            [70.35, 23.69, 73.41, 70.70, 85.41]
        ])

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things  # Unused in this application.
        _, position = self.position

        if actions == 0:  # Fake action.. just to avoid making a move during its_showtime() call

            # New position is sampled from a multinouilli distribution parametrized by the environment's transition matrix
            new_position = np.argmax(np.random.multinomial(n=1, pvals=self.transition_matrix[position, :]))

            # Receives a reward according to the reward matrix (deterministic reward associated with each transition)
            the_plot.add_reward(self.reward_matrix[position, new_position])

            # Move the agent to the new position
            self._teleport((0, new_position))


class PlayerSprite_BoyanChain(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Inform superclass that the '#' delimits the walls."""
        super(PlayerSprite_BoyanChain, self).__init__(corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things  # Unused in this application.
        _, position = self.position

        if actions == 0:  # Just to avoid making a move during its_showtime() call
            # print(layers)
            # From the last state before the Goal
            if layers["G"][0, position + 1]:
                self._east(board, the_plot)  # single jump east
                the_plot.add_reward(-2.0)

            # From any other state
            else:
                # Each one of the 2 possible transitions have a probability of 0.5
                if np.random.rand() > 0.5:  # single jump east
                    self._east(board, the_plot)
                else:  # double jump east
                    self._east(board, the_plot)
                    self._east(board, the_plot)

                # Any transition from those states give a reward of -3
                the_plot.add_reward(-3.0)

        # Check if our agent is on the goal position
        if layers["G"][self.position]:
            the_plot.terminate_episode()


class lstd_params:
    ''' hyperparameters for LSTD algorithms
    d: length of feature vector for states, in boyan chain env, d = 4
    gamma: discount factor
    epsilon: initial A = epsilon * I
    n: number of trajectory
    '''
    def __init__(self):
        self.d = 4

        self.gamma = 1.0

        self.epsilon = 0.01

        self.n = 3000

        self.la = 1


def lstd_recursive(lstdparams, state_feature, V_pi):
    A_re = np.eye(lstdparams.d) / lstdparams.epsilon
    b = np.zeros(lstdparams.d)
    terminal_state = 12
    rms = np.zeros(lstdparams.n)
    for n in range(0, lstdparams.n):
        s = 0
        x = state_feature[s]
        z = x
        game = make_BoyanChain(BOYAN_CHAIN)
        game.its_showtime()
        while s is not terminal_state:
            obs, reward, gamma = game.play(0)
            state_array = np.array(obs.layers['P'], dtype=int)
            _s = list(state_array[0]).index(1)
            _x = state_feature[_s]
            v = np.dot(A_re.T, (np.array(x) - lstdparams.gamma * np.array(_x))).reshape(lstdparams.d, 1)
            A_re -= A_re @ np.array(z).reshape(lstdparams.d, 1) @ v.T / (1 + v.T @ z)
            b += np.array(z) * reward
            theta = np.dot(A_re, b)
            z = lstdparams.la * np.array(z) + _x
            s = _s
            x = _x
        rms[n] = RMS(theta, V_pi)
    return theta, rms


def lstd_recursive_five_states(lstdparams, state_feature, threshold=0.01):
    A_re = np.eye(lstdparams.d) / lstdparams.epsilon
    b = np.zeros(lstdparams.d)
    diff = 1.0
    last_theta = 0
    t = 0
    s = 0
    x = state_feature[s]
    z = x
    game = make_FiveStates(FIVE_STATES)
    game.its_showtime()
    while diff >= threshold:
        t += 1
        obs, reward, gamma = game.play(0)
        state_array = np.array(obs.layers['P'], dtype=int)
        _s = list(state_array[0]).index(1)
        _x = state_feature[_s]
        v = np.dot(A_re.T, (np.array(x) - lstdparams.gamma * np.array(_x))).reshape(lstdparams.d, 1)
        A_re -= A_re @ np.array(z).reshape(lstdparams.d, 1) @ v.T / (1 + v.T @ z)
        b += np.array(z) * reward
        theta = np.dot(A_re, b)
        diff = np.sqrt(np.sum((theta - last_theta) * (theta - last_theta)))
        last_theta = theta
        z = lstdparams.la * np.array(z) + _x
        s = _s
        x = _x

    theta = np.dot(A_re, b)
    return theta, t


def lstd_offline(lstdparams, state_feature, V_pi):
    A = np.eye(lstdparams.d)
    b = np.zeros(lstdparams.d)
    terminal_state = 12
    rms = np.zeros(lstdparams.n)
    for n in range(0, lstdparams.n):
        x = 0
        z = state_feature[0]
        game = make_BoyanChain(BOYAN_CHAIN)
        game.its_showtime()
        while x is not terminal_state:
            obs, reward, gamma = game.play(0)
            state_array = np.array(obs.layers['P'], dtype=int)
            _x = list(state_array[0]).index(1)
            # lstd update
            A += np.outer(z, (np.array(state_feature[x])
                                - lstdparams.gamma * np.array(state_feature[_x])).T)
            b += np.array(z) * reward
            _z = lstdparams.la * np.array(z) + state_feature[_x]
            x = _x
            z = _z
        theta = np.dot(np.linalg.pinv(A), b)
        rms[n] = RMS(theta, V_pi)
    print(np.linalg.pinv(A))
    print(b)
        # print(theta)
    return theta, rms


def lstd_offline_five_states(lstdparams, state_feature, threshold=0.01):
    A = np.eye(lstdparams.d)
    b = np.zeros(lstdparams.d)
    diff = 1.0
    last_theta = 0
    t = 0
    x = 0
    z = state_feature[0]
    game = make_FiveStates(FIVE_STATES)
    game.its_showtime()
    while diff >= threshold:
        t += 1
        obs, reward, gamma = game.play(0)
        state_array = np.array(obs.layers['P'], dtype=int)
        _x = list(state_array[0]).index(1)
        # lstd update
        A += np.outer(z, (np.array(state_feature[x])
                            - lstdparams.gamma * np.array(state_feature[_x])).T)
        b += np.array(z) * reward
        _z = lstdparams.la * np.array(z) + state_feature[_x]
        x = _x
        z = _z
        theta = np.dot(np.linalg.pinv(A), b)
        diff = np.sqrt(np.sum((theta - last_theta)*(theta - last_theta)))
        last_theta = theta
    theta = np.dot(np.linalg.pinv(A), b)
    return theta, t


def RMS(l1, l2):
    l = np.array(l1) - np.array(l2)
    rms = np.sqrt(np.sum(l*l) / len(l))
    return rms


def implementBoyanChain():

    # initialize environment as 13-states boyan chain
    env = BOYAN_CHAIN
    env_width = len(BOYAN_CHAIN[0])
    game = make_BoyanChain(BOYAN_CHAIN)

    # initialize A, b, z
    lstdparams = lstd_params()
    lstdparams.d = 4

    optimal = [-24, -16, -8, 0]
    state_feature = [[1,    0,    0,    0   ],
                     [0.75, 0.25, 0,    0   ],
                     [0.5,  0.5,  0,    0   ],
                     [0.25, 0.75, 0,    0   ],
                     [0,    1,    0,    0   ],
                     [0,    0.75, 0.25, 0   ],
                     [0,    0.5,  0.5,  0   ],
                     [0,    0.25, 0.75, 0   ],
                     [0,    0,    1,    0   ],
                     [0,    0,    0.75, 0.25],
                     [0,    0,    0.5,  0.5 ],
                     [0,    0,    0.25, 0.75],
                     [0,    0,    0,    0   ]]

    theta, rms = lstd_offline(lstdparams, state_feature, optimal)
    # print(theta)
    # print('rms:{}.'.format(rms))
    # print(len(rms))
    # plt.figure(1)
    # plt.plot(range(0, lstdparams.n), rms[:])
    # plt.xlabel('trajectory number')
    # plt.ylabel('RMS error')

    # theta, rms = lstd_recursive(lstdparams, state_feature, optimal)
    print(theta)
    # print('rms:{}.'.format(rms))
    # print(len(rms))
    # plt.figure(2)
    # plt.plot(range(0, lstdparams.n), rms[:])
    # plt.xlabel('trajectory number')
    # plt.ylabel('RMS error')


def implementFiveStates():
    # initialize A, b, z
    lstdparams = lstd_params()
    lstdparams.d = 5
    state_feature = [[74.29, 34.61, 73.48, 53.29, 7.79 ],
                     [61.60, 48.07, 34.68, 36.19, 82.02],
                     [97.00, 4.88,  8.51,  87.89, 5.17 ],
                     [41.10, 40.13, 64.63, 92.67, 31.09],
                     [7.76,  79.82, 43.78, 8.56,  61.11]]

    theta, timestep = lstd_offline_five_states(lstdparams, state_feature)
    print(theta, timestep)
    theta, timestep = lstd_recursive_five_states(lstdparams, state_feature)
    print(theta, timestep)


def equivalence(lstdparams, state_feature):
    A = np.zeros((lstdparams.d, lstdparams.d))
    b = np.zeros(lstdparams.d)
    terminal_state = 12
    x = 0
    z = state_feature[0]
    game = make_BoyanChain(BOYAN_CHAIN)
    game.its_showtime()
    LR_feature = list()
    LR_reward = list()
    while x is not terminal_state:
        obs, reward, gamma = game.play(0)
        state_array = np.array(obs.layers['P'], dtype=int)
        _x = list(state_array[0]).index(1)
        LR_feature.append(state_feature[x])
        LR_reward.append(reward)
        A += np.outer(z, (np.array(state_feature[x])
                          - lstdparams.gamma * np.array(state_feature[_x])).T)
        b += np.array(z) * reward
        _z = lstdparams.la * np.array(z) + state_feature[_x]
        x = _x
        z = _z
    return LR_feature, LR_reward, A, b


def equivalence_recursive(lstdparams, state_feature):
    A_re = np.eye(lstdparams.d)
    A = np.zeros((lstdparams.d, lstdparams.d))
    b = np.zeros(lstdparams.d)
    terminal_state = 12
    LR_feature = list()
    LR_reward = list()

    s = 0
    x = state_feature[s]
    z = x
    game = make_BoyanChain(BOYAN_CHAIN)
    game.its_showtime()
    while s is not terminal_state:
        obs, reward, gamma = game.play(0)
        state_array = np.array(obs.layers['P'], dtype=int)
        _s = list(state_array[0]).index(1)
        _x = state_feature[_s]
        LR_feature.append(state_feature[s])
        LR_reward.append(reward)
        v = np.dot(A_re.T, (np.array(x) - lstdparams.gamma * np.array(_x))).reshape(lstdparams.d, 1)
        A_re -= A_re @ np.array(z).reshape(lstdparams.d, 1) @ v.T / (1 + v.T @ z)
        A += np.outer(z, (np.array(state_feature[s])
                          - lstdparams.gamma * np.array(state_feature[_s])).T)
        b += np.array(z) * reward
        z = lstdparams.la * np.array(z) + _x
        s = _s
        x = _x
    return LR_feature, LR_reward, A, b


if __name__ == '__main__':
    BOYAN_CHAIN = ['P           G']
    FIVE_STATES = ['P    ']

    # implementBoyanChain()
    # implementFiveStates()
    # plt.show()

    state_feature = [[1, 0, 0, 0],
                     [0.75, 0.25, 0, 0],
                     [0.5, 0.5, 0, 0],
                     [0.25, 0.75, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0.75, 0.25, 0],
                     [0, 0.5, 0.5, 0],
                     [0, 0.25, 0.75, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0.75, 0.25],
                     [0, 0, 0.5, 0.5],
                     [0, 0, 0.25, 0.75],
                     [0, 0, 0, 0]]

    lstdparams = lstd_params()
    lstdparams.d = 4
    lstdparams.la = 1
    # LR_feature, LR_reward, A, b = equivalence(lstdparams, state_feature)
    # LR_A = np.zeros((lstdparams.d, lstdparams.d))
    # LR_y = np.zeros(len(LR_feature))
    # LR_b = np.zeros(lstdparams.d)
    # print("LR_feature: {}.".format(LR_feature))
    # print("LR_reward: {}.".format(LR_reward))
    # print("")
    #
    # for i in range(0, len(LR_feature)):
    #     LR_A += LR_feature[i] * np.array(LR_feature[i]).reshape(lstdparams.d, 1)
    #     LR_y[i] = np.sum(LR_reward[i:])
    #     LR_b += np.array(LR_feature[i], dtype=float) * LR_y[i]
    #
    # print("Result from linear regression:")
    # print("linear regression y: {}.".format(LR_y))
    # print("linear regression matrix A: {}.".format(LR_A))
    # print("linear regression matrix b: {}.".format(LR_b))
    # print("")
    # print("Result from off-line LSTD(lambda=1):")
    # print("LSTD matrix A: {}.".format(A))
    # print("LSTD matrix b: {}.".format(b))

    LR_feature, LR_reward, A, b = equivalence_recursive(lstdparams, state_feature)
    LR_A = np.zeros((lstdparams.d, lstdparams.d))
    LR_y = np.zeros(len(LR_feature))
    LR_b = np.zeros(lstdparams.d)
    print("LR_feature: {}.".format(LR_feature))
    print("LR_reward: {}.".format(LR_reward))
    print("")

    for i in range(0, len(LR_feature)):
        LR_A += LR_feature[i] * np.array(LR_feature[i]).reshape(lstdparams.d, 1)
        LR_y[i] = np.sum(LR_reward[i:])
        LR_b += np.array(LR_feature[i], dtype=float) * LR_y[i]

    print("Result from linear regression:")
    print("linear regression y: {}.".format(LR_y))
    print("linear regression matrix A: {}.".format(LR_A))
    print("linear regression matrix b: {}.".format(LR_b))
    print("")
    print("Result from off-line LSTD(lambda=1):")
    print("LSTD matrix A: {}.".format(A))
    print("LSTD matrix b: {}.".format(b))





import my_maze as env
import numpy as np


maze = ['###########',
        '#         #',
        '#         #',
        '#         #',
        '######### #',
        '#         #',
        '#   P     #',
        '###########']

game = env.make_game(maze)
game.its_showtime()
# obv, reward, gamma = game.play(2)
obv, reward, gamma = game.play(2)
# obv, reward, gamma = game.play(3)
# if game.game_over:
#     print("game over")
# state = np.array(obv.layers['P'], dtype=np.float)
# print(state)
# print(reward)
#
# game = env.make_game(maze)
# obv, reward, gamma = game.its_showtime()
# state = np.array(obv.layers['P'], dtype=np.float)
#
# print(state)
if game.game_over:
    print("Game Over")
    game = env.make_game(maze)
    obv, reward, gamma = game.its_showtime()

print(game.game_over)



import numpy as np

import machikoro_actions as a
import machikoro_cards as c
import machikoro_state_space as sp
import machikoro_turn_states as ts

from random import choice

class RandomPolicyAgent():

    def __init__(self, n_players=4) -> None:
        self.n_players = n_players

    def predict(self, obs):
        if len(obs.shape) == 1:
            obs = np.reshape(obs, (self.n_players, -1))
        this_agent = np.nonzero(obs[:, 0])[0][0]
        turnstate = obs[this_agent, sp.TURN]
        match turnstate:

            case ts.ROLL_DICE:
                # print("Simulate roll dice for player", this_agent)
                return choice(
                    [a.ROLL_1, a.ROLL_2]
                )

            case ts.MAY_REROLL:
                # print("Simulate reroll for player", this_agent)
                return choice(
                    [a.REROLL, a.PASS]
                )

            case ts.MAY_BUY:
                # print("Simulate buying for player", this_agent)
                return choice(
                    [a.PASS] + list(range(a.STATION, a.MARKET + 1))
                )        

            case ts.MAY_CHOOSE_CARD:
                # print("Simulate may choose card for player", this_agent)
                return choice(
                    list(range(a.STATION, a.MARKET + 1))
                )
            
            case ts.MAY_CHOOSE_PLAYER_FOR_CARD:
                # print("Simulate may choose player for player", this_agent)
                return choice(
                    [a.CHOOSE_PLAYER_0, a.CHOOSE_PLAYER_1, a.CHOOSE_PLAYER_2, a.CHOOSE_PLAYER_3]
                )

            case ts.MAY_CHOOSE_PLAYER_FOR_COINS:
                # print("Simulate may choose player for player", this_agent)
                return choice(
                    [a.CHOOSE_PLAYER_0, a.CHOOSE_PLAYER_1, a.CHOOSE_PLAYER_2, a.CHOOSE_PLAYER_3]
                )
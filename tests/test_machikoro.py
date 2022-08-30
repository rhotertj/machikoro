
import pytest
import numpy as np

from machikoro import MachiKoroEnv
import machikoro_actions as a
import machikoro_state_space as sp
import machikoro_turn_states as ts


@pytest.fixture(scope="function")
def env():
    env= MachiKoroEnv(4, test_mode=True)
    env.reset()
    return env

def test_roll1(env):
    env.step(a.ROLL_1)
    assert 0 in env.current_throw  

def test_negative_funds(env):
    env.step(a.ROLL_1) # roll once
    pre_buy_state = env.state.copy()
    env.step(a.STADIUM) # too expensive
    assert (env.state == pre_buy_state).all()


def test_empty_inventory(env):
    """Player X should be stuck buying a card out of stock"""

    for _ in range(6):
        env.step(a.ROLL_1, (1,0))
        _, reward, _, _ = env.step(a.WHEAT_FIELD)
        env.render()
    assert reward == -3, "Should be stuck buying out of stock"


def test_win_game(env):
    env.state[0, sp.MALL:sp.RADIO_TOWER + 1] = 1
    env.step(a.ROLL_1, (1,0))
    env.step(a.PASS) # skip reroll
    state, _, done, info = env.step(a.STATION)
    assert done == True, state
    

def test_buy_monument_twice(env):
    for _ in range(5):
        env.step(a.ROLL_1, (1,0))
        if env.current_player == 0:
            _ , reward, _, _ = env.step(a.STATION)
        else:
            env.step(a.PASS)
    assert reward == -3, "Player cannot buy 2 stations"


def test_buy_card(env):
    env.step(a.ROLL_1, (6,0))
    state1, *_ = env.step(a.RANCH)
    
    env.step(a.ROLL_1, (6,0))
    state2, *_ = env.step(a.RANCH)
    print(state1, env.test_mode)
    assert state1[0, sp.RANCH] == 1
    assert state1[0, sp.COINS] == 2

    assert state2[1, sp.RANCH] == 1
    assert state2[1, sp.COINS] == 2
    assert state2[0, sp.RANCH] == 1
    assert state2[0, sp.COINS] == 2

    env.step(a.ROLL_1)
    env.step(a.PASS)
    assert env.current_turn_state == ts.ROLL_DICE, "If player refuses to buy, next player may roll"

def test_roll_2dice(env):
    _, reward, *_ = env.step(a.ROLL_2)
    assert reward == -3, "You are not allowed to roll with 2 dice yet"
    env.state[0, sp.STATION] = 1
    env.step(a.ROLL_2)
    assert env.current_turn_state == ts.MAY_BUY, "After rolling 2 dice, you should may buy something"

def test_reroll(env):
    env.state[0:2, sp.RADIO_TOWER] = 1 # give reroll ability to first 2 players
    env.state[0:2, sp.STATION] = 1 # give ability to use 2 dice
    print(env.state)
    env.step(a.ROLL_1)
    assert env.current_turn_state == ts.MAY_REROLL, "Play with radio tower may reroll"
    env.step(a.PASS) # pass on reroll
    assert env.current_turn_state == ts.MAY_BUY, "Player does not want to reroll"
    env.step(a.PASS) # pass on buying, next player

    assert env.current_player == 1
    env.step(a.ROLL_2, (1,2))
    env.step(a.REROLL)
    assert 0 not in env.current_throw, f"Should have rerolled 2 dice {env.current_throw=}"
    assert env.current_turn_state == ts.MAY_BUY
    assert not np.array_equal(env.current_throw, [1,2]), f"{env.current_throw} should have been rerolled and not equal 1,2. This might fail due to bad luck."

def test_mall_bonus(env):
    env.state[0, sp.MALL] = 1
    env.state[0, sp.KOMBINI] = 2
    env.step(a.ROLL_1, (4, 0))
    assert env.funds == 3 + 2 * (3 + 1), "Kombini should earn 2 * (3 + 1)"

def test_second_turn(env):
    env.state[0:2, sp.STATION] = 1
    env.state[0:2, sp.AMUSEMENT_PARK] = 1
    env.step(a.ROLL_2, (5,5))
    assert env.current_turn_state == ts.MAY_BUY
    env.step(a.WHEAT_FIELD)
    env.render()
    assert env.current_player == 0
    # check rolling with 1 dice allowed
    env.step(a.ROLL_1)
    env.step(a.WHEAT_FIELD)
    assert env.current_player == 1
    # check no third turn allowed
    env.step(a.ROLL_2, (5,5))
    env.step(a.RANCH)
    env.step(a.ROLL_2, (4,4))
    env.step(a.RANCH)
    assert env.current_player == 2


def test_second_turn_with_reroll(env):
    env.state[0:2, sp.STATION] = 1
    env.state[0:2, sp.RADIO_TOWER] = 1
    env.state[0:2, sp.AMUSEMENT_PARK] = 1
    # player may reroll, but passes, has a second turn with another just 1 die
    env.step(a.ROLL_2, (5,5))
    env.step(a.PASS)
    env.step(a.WHEAT_FIELD)
    assert env.current_player == 0
    env.step(a.ROLL_1, (5,0))
    env.step(a.PASS) # pass on reroll again
    env.step(a.PASS) # pass on buying
    print(env.current_turn_state)
    assert env.current_player == 1
    # player may reroll, takes it, has a second turn, may reroll, rerolls again
    env.step(a.ROLL_2, (5,3))
    env.step(a.REROLL, (5,5))
    env.step(a.WHEAT_FIELD)
    assert env.current_player == 1
    env.step(a.ROLL_2, (5,1))
    env.step(a.REROLL)
    env.step(a.PASS) # buy nothing
    assert env.current_player == 2


def test_steal_card(env):
    # Test implementation of business center
    env.state[0, sp.BUSINESS_CENTER] = 1
    env.step(a.ROLL_1, (6,0))
    assert env.current_turn_state == ts.MAY_CHOOSE_PLAYER_FOR_CARD
    env.step(a.CHOOSE_PLAYER_1)
    assert env.current_turn_state == ts.MAY_CHOOSE_CARD
    env.step(a.BAKERY) 
    assert env.state[0, sp.BAKERY] == 2
    assert env.state[1, sp.BAKERY] == 0


def test_steal_money(env):
    # Test implementation of tv station
    env.state[0, sp.TV_STATION] = 1
    env.state[0, sp.BUSINESS_CENTER] = 1
    assert env._owns(sp.TV_STATION)
    env.step(a.ROLL_1, (6,0))
    assert env.current_turn_state == ts.MAY_CHOOSE_PLAYER_FOR_COINS
    _, reward, _, _ = env.step(a.CHOOSE_PLAYER_0)
    assert reward == -3, "You may not steal from yourself"
    env.step(a.CHOOSE_PLAYER_1)
    assert env.state[0, sp.COINS] > env.state[1, sp.COINS], "Stealing appears to not work"
    assert env.current_turn_state == ts.MAY_CHOOSE_PLAYER_FOR_CARD
    

def test_earn_money_from_bank(env):
    env.step(a.ROLL_1, (1,0))
    assert (env.state[:, sp.COINS] == 4).all()
    

def test_pay_money_to_players(env):
    # Tests cafe (and somewhat family restaurant)
    env.state[3, sp.CAFE] = 1 # gets 1
    env.state[2, sp.CAFE] = 2 # gets 2
    env.state[1, sp.CAFE] = 2 # gets 2
    print(env.state)
    env.funds = 4
    env.step(a.ROLL_1, (3,0))
    assert env.funds == 1 # we pay all, then earn 1 from bakery
    assert env.state[3, sp.COINS] == 3 + 1
    assert env.state[2, sp.COINS] == 3 + 2
    assert env.state[1, sp.COINS] == 3 + 1 # player ran out of coins

def test_earn_money_from_players(env):
    # Tests stadium card
    env.state[0, sp.STADIUM] = 1
    env.step(a.ROLL_1, (6,0))
    env.funds == 3 + (env.n_players - 1) * 2

def test_simulate_step(env):
    # call simulate step directly

    pass

def test_step():
    # create env without test mode
    # test step with simulations
    env = MachiKoroEnv(4)
    pass

def test_player_not_first():
    env = MachiKoroEnv(n_players=4, player_index=3)
    # player 0 starts
    assert env.current_player == 0
    # step simulates all players, then we roll
    env.step(a.ROLL_1) 
    assert env.current_turn_state == ts.MAY_BUY
    assert env.current_player == 3

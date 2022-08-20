import numpy as np
import gym

from machikoro_cards import INVENTORY
import machikoro_cards as c
import machikoro_actions as a
import machikoro_state_space as sp
import machikoro_turn_states as ts


def a2c(action):
    """Turns an action into a card

    Args:
        action (int): The action space representation of a card.

    Returns:
        int: The nth card.
    """    
    
    return action - 3

def a2sp(action):
    """Turns an action into a state index

    Args:
        action (int): The action space representation of a card.

    Returns:
        int: The state space representation of a card.
    """    
    
    return action - 1

class MachiKoroEnv(gym.Env):


    def __init__(self, n_players=4, test_mode=False) -> None:
        """A gym environment of the game 'MachiKoro'.

        Args:
            n_players (int, optional): Number of players (2-4). Defaults to 4.
            test_mode (bool, optional): Whether you want to set the dice manually. Defaults to False.
        """        
        super().__init__()
        self.n_players = n_players
        self.reset()

        self.action_space = gym.spaces.Discrete(a.CHOOSE_PLAYER_3 + 1)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=self.state.shape)

        self.test_mode = test_mode
    
    ###############
    # PROPERTY GETTERS / SETTERS
    ###############

    @property
    def current_turn_state(self):
        return self.state[0, sp.TURN]
    
    @current_turn_state.setter
    def current_turn_state(self, new):
        self.state[0,0] = new

    @property
    def current_throw(self):
        return self.state[self.current_player, -2:]

    @current_throw.setter
    def current_throw(self, dice):
        """Change the current throw.

        Args:
            dice (iterable): 2 values, one for each die.
        """        
        
        d1, d2 = dice
        if d1 not in range(1,7):
            d1 = 0
        if d2 not in range(1,7):
            d2 = 0
        self.state[self.current_player,-2:] = [d1, d2]
        dice_sum = int(d1 + d2)
        self.cards_activated = c.get_cards_by_throw(dice_sum).copy()

    @property
    def funds(self):
        """The amount of money the current player has.

        Returns:
            int: The amount of money the current player has.
        """        
        return self.state[self.current_player, sp.COINS]

    @funds.setter
    def funds(self, new):
        """Changes the budget of the current player. 

        Args:
            new (int): New amount of money.

        Raises:
            ValueError: Must be a positive amount, there is no debt in this game.
        """        
        if new < 0:
            raise ValueError
        else:
            self.state[self.current_player, sp.COINS] = new

    @property
    def ILLEGAL_MOVE(self):
        """Crafts a return tuple for an illegally played action.

        Returns:
            tuple: (current state, reward (minus three), end turn (false), end game (false))
        """        
        return self.state, -3, {"end_turn" : False}, False

    def _owns(self, card):
        """Whether the current player already owns the card

        Args:
            card (int): The state space representation of the card.

        Returns:
            bool: Whether it is owned or not.
        """        
        
        return self.state[self.current_player, card] >= 1

    def _reward(self):
        """ Calculates the reward.
        It is computed by adding the current amount of coins and
        the amount that was spent already.

        Returns:
            int: The reward.
        """
        coins = self.state[self.current_player, 1]
        bought = 0
        for card, n in enumerate(self.state[self.current_player, sp.STATION:sp.MARKET + 1]):
            bought += n * c.get_price(card)
        return coins + bought

    #
    # HELPERS THAT RUN PARTS OF STEP OR BUY 
    #

    def _init_state(self,):
        """ Initialize game state.

        Each player starts with 3 coins, a wheat field and a bakery.
        The state is represented by an array of dim (num players)x(23 variables).
        23 state variables divide into the following:
            1 "Turn state" - ROLL, REROLL or BUY
            1 Coins - Current funds
            4 Monuments
            15 Companies
            2 Dice
        """
        self.inventory = INVENTORY.copy()
        state = np.zeros((self.n_players, sp.DIE2 + 1))
        # starting coins
        state[:, sp.COINS] = 3
        # starting cards, offset monuments
        state[:, sp.WHEAT_FIELD] = 1 
        state[:, sp.BAKERY] = 1 # BAKERY
        self.state = state
        self.current_turn_state = ts.ROLL_DICE
        self.current_player = 0
        self.second_turn = False
        self.steal_card_target_action = None
        

    
    def _step_roll(self, action):
        """Performs dice rolling based on given action

        Args:
            action (int): Should be either ROLL_1, ROLL_2 or REROLL

        """        
        
        if action == a.ROLL_1:
            d1 = np.random.randint(1,6)
            self.current_throw = (d1, 0)

        elif action == a.ROLL_2:
            d1, d2 = np.random.randint(1,6, size=2)
            self.current_throw = (d1, d2)

        elif action == a.REROLL:
            reroll_action = len(np.nonzero(self.current_throw)[0]) - 1

            return self._step_roll(reroll_action)
        

    def _step_buy(self, action):
        """Performs buying based on action.

        Returns:
            bool: Whether the action was valid or not.
        """
        if action == a.PASS:
            return True

        # no such card left
        if self.inventory[a2c(action)] == 0:
            return False

        # insufficient funds
        price = c.get_price(a2c(action))
        if self.funds < price:
            return False
        
        # these cards can only be bought once
        if action in [
            a.STATION,
            a.MALL,
            a.AMUSEMENT_PARK,
            a.RADIO_TOWER,
            a.STADIUM,
            a.TV_STATION,
            a.BUSINESS_CENTER
            ] and self._owns(a2sp(action)):
            return False

        self.inventory[a2c(action)] -= 1
        self.state[self.current_player, a2sp(action)] += 1
        self.funds -= price

        return True

    # NOTE: The following functions are always from the perspective of the current player.

    def _pay_all_per_index(self, card, money):
        """Pay all players holding the card at state_index the appropiate amount of money.

        Args:
            card (int): The state space representation of the card.
            money (int): The amount of money everyone is paid by this card.
        """        
        self.state[:, sp.COINS] += money * self.state[:, card]

    def _pay_self_per_index(self, card, money):
        """Pay only yourself the appropiate amount of money, if you hold the card.

        Args:
            card (int): The state space representation of the card.
            money (int): The amount of money you are paid by this card.
        """        
        self.state[self.current_player, sp.COINS] += money * self.state[self.current_player, card]

    def _pay_self_multiplied_per_index(self, cards, multiplier_card, money):
        """Pay yourself money per card at state_indices, multiplied by mulitplier index

        Example: You have 2 cheese factories and 3 ranches. Each ranch (card) gives you 3 (money) coins per cheese factory (multiplier card). 
        You should get 2 * (3 * 3)

        Args:
            cards (int): State space representations of cards you earn money from.
            multiplier_card (int): State space representation of the card that multiplies gained money from cards.
            money (int): The amount you get per card.
        """        

        for card in cards:
            self.state[self.current_player, sp.COINS] += money * self.state[self.current_player, card] * self.state[self.current_player, multiplier_card]


    def _pay_others_from_index(self, card, money):
        """Pay other players owning the card in reverse order.

        Args:
            card (int): State space representations of card other earn money from.
            money (int): The amount others get per card.
        """        
        others = np.delete(np.arange(self.n_players), self.current_player)
        others = others[::-1] # pay in reverse playing order
        for i in others:
            money_to_pay = money * self.state[i, card]
            if money_to_pay > self.funds:
                money_to_pay = self.funds
            self.funds = self.funds - money_to_pay
            self.state[i, sp.COINS] += money_to_pay

    def _pay_self_from_others_per_index(self, card, money):
        """Earn / steal money from other players.

        Args:
            card (int):  State space representations of card you earn money from.
            money (int): The amount you get from others per card.
        """        
        others = np.delete(np.arange(self.n_players), self.current_player)
        for i in others:
            money_to_pay = money * self.state[self.current_player, card]
            if money > self.state[i, sp.COINS]:
                money_to_pay = self.state[i, sp.COINS]
            self.funds += money_to_pay
            self.state[i, sp.COINS] = self.state[i, sp.COINS] - money_to_pay

    def _steal_coins_from_player(self, player_action, money):
        """You may steal coins from one other player of choice.

        Args:
            player_action (int): An action describing a player.
            money (int): The amount of money you steal.
        """        
        other_player = self.current_player
        match player_action:
            case a.CHOOSE_PLAYER_0:
                other_player = 0
            case a.CHOOSE_PLAYER_1:
                other_player = 1
            case a.CHOOSE_PLAYER_2:
                other_player = 2
            case a.CHOOSE_PLAYER_3:
                other_player = 3
        money_to_pay = money
        if money > self.state[other_player, sp.COINS]:
                money_to_pay = self.state[other_player, sp.COINS]
        
        self.state[other_player, sp.COINS] -= money_to_pay
        self.state[self.current_player, sp.COINS] += money_to_pay

    def _steal_card_from_player(self, action):
        """Steals a card from another player.

        Args:
            action (int): The action representation of the card you want to steal.

        Returns:
            bool: Whether the action was valid or not.
        """        
        other_player = self.current_player
        match self.steal_card_target_action:
            case a.CHOOSE_PLAYER_0:
                other_player = 0
            case a.CHOOSE_PLAYER_1:
                other_player = 1
            case a.CHOOSE_PLAYER_2:
                other_player = 2
            case a.CHOOSE_PLAYER_3:
                other_player = 3

        card = a2sp(action)

        # may only be owned once
        if sp.BUSINESS_CENTER >= card and card >= sp.STADIUM:
            if self._owns(card):
                return False
        
        # other player needs to own the card
        if self.state[other_player, card] == 0:
            return False

        self.state[other_player, card] -= 1
        self.state[self.current_player, card] += 1

        return True

    def _economy(self):
        """Hands out money to eligible players.
        First, debt is paid from current funds.
        Then, money is given out of the bank.
        """
        cards_activated = self.cards_activated.copy()

        BONUS = 1 if self._owns(sp.MALL) else 0

        # pay debt first
        if c.CAFE in self.cards_activated:
            self._pay_others_from_index(sp.CAFE, 1 + BONUS)
            self.cards_activated.remove(c.CAFE)
        
        if c.FAMILY_RESTAURANT in self.cards_activated:
            self._pay_others_from_index(sp.FAMILY_RESTAURANT, 2 + BONUS)
            self.cards_activated.remove(c.FAMILY_RESTAURANT)
            
        i = -1
        for card in cards_activated:
            i +=1
            match card:
                case c.WHEAT_FIELD:
                    self._pay_all_per_index(sp.WHEAT_FIELD, 1)
                    self.cards_activated.remove(c.WHEAT_FIELD)
                case c.RANCH:
                    self._pay_all_per_index(sp.RANCH, 1)
                    self.cards_activated.remove(c.RANCH)
                case c.BAKERY:
                    self._pay_self_per_index(sp.BAKERY, 1 + BONUS)
                    self.cards_activated.remove(c.BAKERY)
                case c.KOMBINI:
                    self._pay_self_per_index(sp.KOMBINI, 3 + BONUS)
                    self.cards_activated.remove(c.KOMBINI)
                case c.FOREST:
                    self._pay_all_per_index(sp.FOREST, 1)
                    self.cards_activated.remove(c.FOREST)
                case c.STADIUM:
                    self._pay_self_from_others_per_index(sp.STADIUM, 2)
                    self.cards_activated.remove(c.STADIUM)
                case c.TV_STATION:
                    # tv station lets us steal 5 coins from a player of choice
                    # if the card is still activated, we set the turn state to may choose player for coins
                    # otherwise, we call the appropriate function from the state and can easily ignore it here
                    if self._owns(sp.TV_STATION):
                        self.current_turn_state = ts.MAY_CHOOSE_PLAYER_FOR_COINS
                        self.cards_activated.remove(c.TV_STATION)
                        # if we dont return here, business center might override next turn state
                        # business center will stay active and gets acted on after TV Station
                        return
                case c.BUSINESS_CENTER:
                    # business center lets us steal a card from an opponent
                    # we set the turn state here, let the step funtion handle the rest and just remove the card
                    # to avoid endless card stealings
                    if self._owns(sp.BUSINESS_CENTER):
                        self.current_turn_state = ts.MAY_CHOOSE_PLAYER_FOR_CARD
                        self.cards_activated.remove(c.BUSINESS_CENTER)
                        return

                case c.CHEESE_FACTORY:
                    self._pay_self_multiplied_per_index([sp.RANCH], sp.CHEESE_FACTORY, 3)
                    self.cards_activated.remove(c.CHEESE_FACTORY)
                case c.FURNITURE_FACTORY:
                    self._pay_self_multiplied_per_index([sp.FOREST, sp.MINE], sp.FURNITURE_FACTORY, 3)
                    self.cards_activated.remove(c.FURNITURE_FACTORY)
                case c.MINE:
                    self._pay_all_per_index(sp.MINE, 5)
                    self.cards_activated.remove(c.MINE)
                case c.APPLE_ORCHARD:
                    self._pay_all_per_index(sp.APPLE_ORCHARD, 3)
                    self.cards_activated.remove(c.APPLE_ORCHARD)
                case c.MARKET:
                    self._pay_self_multiplied_per_index([sp.WHEAT_FIELD, sp.APPLE_ORCHARD], sp.MARKET, 2)
                    self.cards_activated.remove(c.MARKET)


    def step(self, action, dice=None):
        """Performs one step in the game.
        Note that in MachiKoro, one turn consists of many steps.
        The return info dict contains the "end_turn" information.

        Args:
            action (int): An action to play. Should fit the current turn state.
            dice (_type_, optional): If you want to test the code, you may want to activate test mode
                when initializing the environment and set dice throws manually here. Defaults to None.

        Returns:
            tuple[np.ndarray, int, dict, bool]: state, reward, info, end of game
        """        
        
        # result = (state, reward, end turn, end game)
        result = None
        cts = self.current_turn_state

        # check monument ownership before turn
        HAS_REROLL = self._owns(sp.RADIO_TOWER)
        HAS_SECOND_TURN = self._owns(sp.AMUSEMENT_PARK)
        HAS_2_DICE = self._owns(sp.STATION)

        if cts == ts.ROLL_DICE:

            if action not in (a.ROLL_1, a.ROLL_2):
                return self.ILLEGAL_MOVE

            if action == a.ROLL_2 and not HAS_2_DICE:
                return self.ILLEGAL_MOVE

            if self.test_mode and not dice is None:
                self.current_throw = dice
            else:
                self._step_roll(action)

            if HAS_REROLL:
                # next input should be whether player wants to reroll or accept dice
                self.current_turn_state = ts.MAY_REROLL
            else:
                # set may buy before economy so it can be overriden by special cards like tv station and business center
                self.current_turn_state = ts.MAY_BUY
                self._economy()

            return self.state, self._reward(), {"end_turn" : False}, False

        elif cts == ts.MAY_REROLL:

            if action not in (a.REROLL, a.PASS):
                return self.ILLEGAL_MOVE
            
            if action == a.REROLL:
                if self.test_mode and not dice is None:
                    self.current_throw = dice
                else:
                    self._step_roll(action)
                
            # set may buy before economy so it can be overriden by special cards like tv station and business center
            self.current_turn_state = ts.MAY_BUY  
            self._economy()
                

            return self.state, self._reward(), {"end_turn" : False}, False

        if cts == ts.MAY_CHOOSE_PLAYER_FOR_COINS:
            # only allow stealing from other players
            valid_players = [a.CHOOSE_PLAYER_0, a.CHOOSE_PLAYER_1, a.CHOOSE_PLAYER_2, a.CHOOSE_PLAYER_3]
            valid_players.pop(self.current_player)
            valid_players = valid_players[:self.n_players - 1]

            if action not in valid_players:
                return self.ILLEGAL_MOVE

            self._steal_coins_from_player(action, 5)
            # set may buy before economy so it can be overriden by special cards like tv station and business center
            self.current_turn_state = ts.MAY_BUY
            self._economy()
            
            return self.state, self._reward(), {"end_turn" : False}, False

        if cts == ts.MAY_CHOOSE_PLAYER_FOR_CARD:
            # only allow stealing from other players
            valid_players = [a.CHOOSE_PLAYER_0, a.CHOOSE_PLAYER_1, a.CHOOSE_PLAYER_2, a.CHOOSE_PLAYER_3]
            valid_players.pop(self.current_player)
            valid_players = valid_players[:self.n_players - 1]

            if action not in valid_players:
                return self.ILLEGAL_MOVE

            self.steal_card_target_action = action 
            self.current_turn_state = ts.MAY_CHOOSE_CARD
            return self.state, self._reward(), {"end_turn" : False}, False

        if cts == ts.MAY_CHOOSE_CARD:

            if action < a.WHEAT_FIELD or action > a.MARKET:
                return self.ILLEGAL_MOVE

            if not self._steal_card_from_player(action):
                return self.ILLEGAL_MOVE

            self.current_turn_state = ts.MAY_BUY
            # this should not do anything as the business center is the only card activated by throw=6,
            # but who knows what kind of add-ons will come :)
            self._economy()
            return self.state, self._reward(), {"end_turn" : False}, False

        if cts == ts.MAY_BUY:

            if (action < a.STATION or action > a.MARKET) and not action == a.PASS:
                return self.ILLEGAL_MOVE


            # buying action not allowed
            if not self._step_buy(action):
                return self.ILLEGAL_MOVE

            # end game, player has all monuments, excluding index
            if sum(self.state[self.current_player, sp.STATION:sp.RADIO_TOWER + 1]) == 4:
                return self.state, 1000, {"end_turn" : False}, True

            d1, d2 = self.current_throw
            if (d1 == d2 and HAS_SECOND_TURN) and not self.second_turn:
                # only allow one second turn
                self.current_turn_state = ts.ROLL_DICE
                self.current_throw = (0,0)
                self.second_turn = True
                self.steal_card_target_action = None   
                return self.state, self._reward(), {"end_turn" : False}, False
            else:
                # end turn
                self.current_player  = (self.current_player + 1) % self.n_players
                self.current_turn_state = ts.ROLL_DICE
                self.current_throw = (0,0)
                self.second_turn = False 
                self.steal_card_target_action = None               
                return self.state, self._reward(), {"end_turn" : True}, False
                
        return result

    def render(self):
        """Prints out some information about the current state.
        """        
        print("-"*5, "Begin Render")
        print("Current player:", self.current_player)
        print("Current dice:", self.current_throw)
        print("Current reward:", self._reward())
        print(self.state)
        print("-"*5, "End Render")

    def close(self):
        """Closes the environment. We do not open any files or save anything so this does nothing.
        """        
        pass

    def reset(self):
        self._init_state()


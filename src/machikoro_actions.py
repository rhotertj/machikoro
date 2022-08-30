# Action space:
ROLL_1 = 0
ROLL_2 = 1
REROLL = 2
# Monuments
STATION = 3
MALL = 4
AMUSEMENT_PARK = 5
RADIO_TOWER = 6
# Companies
WHEAT_FIELD = 7
RANCH = 8
BAKERY = 9
CAFE = 10
KOMBINI = 11
FOREST = 12
STADIUM = 13
TV_STATION = 14
BUSINESS_CENTER = 15
CHEESE_FACTORY = 16
FURNITURE_FACTORY = 17
MINE = 18
FAMILY_RESTAURANT = 19
APPLE_ORCHARD = 20
MARKET = 21
# Special for reroll or not buying anything
PASS = 22
# Choosing players for stealing
CHOOSE_PLAYER_0 = 23
CHOOSE_PLAYER_1 = 24
CHOOSE_PLAYER_2 = 25
CHOOSE_PLAYER_3 = 26

def to_card(action):
    """Turns an action into a card

    Args:
        action (int): The action space representation of a card.

    Returns:
        int: The nth card.
    """    
    
    return action - 3

def to_state_index(action):
    """Turns an action into a state index

    Args:
        action (int): The action space representation of a card.

    Returns:
        int: The state space representation of a card.
    """    
    
    return action + 1

from collections import namedtuple, deque

import numpy as np
from typing import List

import rewards

from agent_code.q_learning_task_1.feature_vector import FeatureVector
from agent_code.q_learning_task_1.game_state import GameState
from agent_code.q_learning_task_1.feature_extractor import extract_features, convert_to_state_object

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.qs = np.zeros((FeatureVector, len(ACTIONS)))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    old_state = convert_to_state_object(last_game_state)
    new_state = convert_to_state_object(new_game_state)

    custom_events = extract_events_from_state(old_state, new_state)

    total_events = custom_events + events

    reward = reward_from_events(self, total_events)

    # TODO Update self.qs
    pass


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # TODO save training model (self.qs)
    pass


def extract_events_from_state(old_state: GameState, new_state: GameState) -> List:
    old_features = extract_features(old_state)
    new_features = extract_features(new_state)

    coin_events = []
    if old_features.coin_distance.minimum() < new_features.coin_distance.minimum():
        coin_events.append(rewards.APPROACH_COIN)
    elif old_features.coin_distance.minimum() > new_features.coin_distance.minimum():
        coin_events.append(rewards.MOVED_AWAY_FROM_COIN)

    return coin_events


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in rewards.rewards:
            reward_sum += rewards.rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

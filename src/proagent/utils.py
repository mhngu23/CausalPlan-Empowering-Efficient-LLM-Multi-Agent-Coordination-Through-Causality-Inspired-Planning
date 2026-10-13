import os
import random
import time
import torch
import numpy as np
import re


from overcooked_ai_py.mdp.actions import Action, Direction
import numpy as np 


def convert_messages_to_prompt(messages):
    """
    Converts a list of messages(for chat) to a prompt (for completion) for OpenAI's API.

    :param messages:
    :return: prompt
    """
    prompt = ""
    for message in messages:
        prompt += f"{message['content']}\n"

    return prompt


def gpt_state_list(mdp, state):
    """List representation of the current state, modified for GPT."""
    players_dict = {player.position: player for player in state.players}
    grid_list = []

    for y, terrain_row in enumerate(mdp.terrain_mtx):
        grid_row = []
        for x, element in enumerate(terrain_row):
            if (x, y) in players_dict.keys():
                player = players_dict[(x, y)]
                orientation = player.orientation
                player_object = player.held_object
                assert orientation in Direction.ALL_DIRECTIONS
                player_idx_lst = [
                    i
                    for i, p in enumerate(state.players)
                    if p.position == player.position
                ]
                assert len(player_idx_lst) == 1

                if player_object:
                    if player_object.name[0] == "s":
                        # this is a soup
                        grid_row.append("{}-{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation], str(player_object)))
                    else:
                        grid_row.append("{}-{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation], player_object.name[:1]))
                else:
                    grid_row.append("{}-{}".format(player_idx_lst[0], Action.ACTION_TO_CHAR[orientation]))
            else:
                if element == "X" and state.has_object((x, y)):
                    state_obj = state.get_object((x, y))
                    if state_obj.name[0] == "s":
                        grid_row.append(str(state_obj))
                    else:
                        grid_row.append(state_obj.name[:1])
                elif element == "P" and state.has_object((x, y)):
                    soup = state.get_object((x, y))
                    grid_row.append(element+str(soup))
                else:
                    grid_row.append(element)

        grid_list.append(grid_row)

    if state.bonus_orders:
        bonus_orders = ["Bonus orders: {}".format(state.bonus_orders)]
        grid_list.append(bonus_orders)

    return grid_list


def softmax(x):
    # Subtracting the max value for numerical stability (prevents overflow)
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)

ACTION_PATTERN = r'pickup\(onion\)|put_onion_in_pot\(\)|pickup\(dish\)|fill_dish_with_soup\(\)|deliver_soup\(\)|place_obj_on_counter\(\)|wait\(1\)'


def normalize_action(response):
    """Map an LLM proposal to its canonical semantic action (lowercasing + regex matching
    against the predefined action patterns). Returns None if no action is recognised."""
    match = re.search(ACTION_PATTERN, response.lower())
    return match.group() if match else None


def merge_equivalent_actions(actions, probs):
    """Sum the (already normalised) probabilities of semantically equivalent proposals."""
    merged = {}
    for action, prob in zip(actions, probs):
        merged[action] = merged.get(action, 0.0) + float(prob)
    return list(merged.keys()), list(merged.values())


def generate_action_index(response, index, layout):
    # if layout == "cramped_room":
    if "pickup(onion)" in response:
        return 0
    elif "put_onion_in_pot" in response:
        return 1
    elif "pickup(dish)" in response:
        return 2
    elif "fill_dish_with_soup" in response:
        return 3
    elif "deliver_soup" in response:
        return 4
    elif "place_obj_on_counter" in response:
        if index[1] == 1:
            return 5
        elif index[2] == 1:
            return 6
        else:
            print("fail reponse", response)
            return None
    return None


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA (if using GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoids non-deterministic algorithms

def parse_response_analysis(response):
	response = response.replace("Plan", "").strip()
	response = response.replace("Please give a plan!", "").strip()
	return response
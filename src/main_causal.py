import os
import time

from tqdm import tqdm
import argparse
from argparse import ArgumentParser
import numpy as np
from rich import print as rprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
if hasattr(tf, "config") and hasattr(tf.config, "set_visible_devices"):
    tf.config.set_visible_devices([], 'GPU')  # RL partners run on CPU; keep the GPUs for the LLM

import importlib_metadata
try:
    VERSION = importlib_metadata.version("overcooked_ai")
except importlib_metadata.PackageNotFoundError:
    VERSION = '0.0.1'  # version of the bundled lib/overcooked_ai
print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from causal_graph.SCMOp_Predict_Next_Action_Only import SCMOp_Predict_Next_Action_Only
from causal_graph.SCMOp_Predict_Next_Action_Only_2pot import SCMOp_Predict_Next_Action_Only_2pot


from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent, set_seed, save_result
from utils_causal import DATA_DIR, convert_state_to_input, convert_action_to_input, write_buffer_to_folder, load_buffer, save_causal_graph, load_causal_graph_DAG


# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


import torch
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)


OvercookedState = ['empty_hand', # empty hand 0
                'hold_onion', # holding oninon 1
                'hold_dish', # holding empty dish 2
                'dish_with_soup', # holding dish with soup 3
                'pot_0', # pot with 0 onion 4
                'pot_1', # pot with 1 onion 5
                'pot_2', # pot with 2 onions 6
                'pot_3', # pot with 3 oninons 7
                'pot_finished', # pot with soup cooked 8
                'goal_delivered', # delivered_goal 9
                'pickup(onion)', # 10
                'put_onion_in_pot()', # 11
                'pickup(dish)', # 12
                'fill_dish_with_soup()', # 13
                'deliver_soup()', # 14
                'place_onion_on_counter()', # 15
                'place_dish_on_counter()',] # 16

OvercookedState_2_pot = ['empty_hand', # empty hand 0
                'hold_onion', # holding oninon 1
                'hold_dish', # holding empty dish 2
                'dish_with_soup', # holding dish with soup 3
                'pot_0', # pot with 0 onion 4
                'pot_1', # pot with 1 onion 5
                'pot_2', # pot with 2 onions 6
                'pot_3', # pot with 3 oninons 7
                'pot_finished', # pot with soup cooked 8
                'pot_1_0', # pot with 0 onion 9
                'pot_1_1', # pot with 1 onion 10
                'pot_1_2', # pot with 2 onions 11
                'pot_1_3', # pot with 3 oninons 12
                'pot_1_finished', # pot with soup cooked 13
                'goal_delivered', # delivered_goal 14
                'pickup(onion)', # 15
                'put_onion_in_pot()', # 16
                'pickup(dish)', # 17
                'fill_dish_with_soup()', # 18
                'deliver_soup()', # 19
                'place_onion_on_counter()', # 20
                'place_dish_on_counter()',] # 21

interact_action_list = ['pickup_onion', # 15
                'put_onion_in_pot', # 16
                'pickup_dish', # 17
                'fill_dish_with_soup', # 18
                'deliver_soup', # 19
                'place_onion_on_counter', # 20
                'place_dish_on_counter',] # 21

LAYOUT_ABBR = {
    "cramped_room": "cr",
    "asymmetric_advantages": "aa",
    "coordination_ring": "cor",
    "forced_coordination": "fc",
    "counter_circuit": "cc",
}


def train_causality(data_buffer, path, train_step, layout, sparsity_lambda=1e-7):
    """
    Phase 1 of CausalPlan: learn the Structural Causal Action (SCA) model from the
    buffer B and save the causal action matrix M (sigmoid of the structural parameters).
    """
    if layout == "cramped_room":
        causal_model = SCMOp_Predict_Next_Action_Only(data_buffer, sparsity_lambda=sparsity_lambda)
    else:
        causal_model = SCMOp_Predict_Next_Action_Only_2pot(data_buffer, sparsity_lambda=sparsity_lambda)

    for it in tqdm(range(train_step)):
        # Alternate between the functional parameters (delta) and structural parameters (eta)
        for _ in range(10):
            causal_model.train_f()
        for _ in range(10):
            causal_model.train_s()

        if it % 100 == 0:
            print(torch.sigmoid(causal_model.s_params.edge_params))
            save_causal_graph(torch.sigmoid(causal_model.best_s_params), path)

    causal_graph = torch.sigmoid(causal_model.best_s_params)
    save_causal_graph(causal_graph, path)
    causal_model.finish()
    return causal_graph


def main(variant):

    set_seed(variant['seed'])

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']

    mode = variant['mode']
    
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    data_buffer = []

    # Phase 1: learn the SCA model / causal action matrix from a previously collected buffer.
    if variant['train_SCM']:
        data_buffer = load_buffer(path=variant['buffer_path'])
        if data_buffer is None:
            raise FileNotFoundError(f"Buffer {variant['buffer_path']} not found in {DATA_DIR}")
        print(f"The length of the training buffer is: {len(data_buffer)}")
        train_causality(data_buffer, variant["causal_graph_path"], variant['train_SCM_step'],
                        layout, sparsity_lambda=variant['sparsity_lambda'])
        return

    # Phase 2: load the learned causal action matrix for causal-aware planning.
    causal_graph = None
    if "ProAgent" in (p0_algo, p1_algo) and (variant['use_causal_graph'] or variant['use_failure_handled']):
        causal_graph = load_causal_graph_DAG(variant["causal_graph_path"], layout)

    for i in range(episode):  
        agents_list = []
        for alg in [p0_algo, p1_algo]:
            if alg == "ProAgent":

                assert variant['gpt_planner_model']!=None, print(f'you should choose a gpt model')
                assert variant['gpt_explainer_model']!=None, print(f'you should choose a gpt model')

                print(f"\n----Use Planner: {variant['gpt_planner_model']}----\n")
                print(f"\n----Use Explainer: {variant['gpt_explainer_model']}----\n")
                print(f"\n----Use Analysis: {variant['gpt_analysis_model']}----\n")

                agent = make_agent(alg, mdp, layout, planner_model=variant['gpt_planner_model'], explainer_model=variant['gpt_explainer_model'],
                                   analysis_model=variant['gpt_analysis_model'],   
                                   prompt_level=variant['prompt_level'], 
                                   belief_revision=variant['belief_revision'],
                                   use_causal_graph=variant['use_causal_graph'],
                                   use_failure_handled=variant['use_failure_handled'],
                                   causal_graph=causal_graph,
                                   retrival_method=variant['retrival_method'], 
                                   K=variant['K'],
                                   gamma=variant['gamma'])
                
            elif alg == "BC":
                agent = make_agent(alg, mdp, layout, seed_id=i)

            else:
                agent = make_agent(alg, mdp, layout)

            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0
        old_state = None
        old_op_state = None

        if variant["record_action_only"]:
            interact_act_p0 = None
            interact_act_p1 = None

        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                if variant["layout"] == "cramped_room":
                    state = convert_state_to_input(OvercookedState, s_t.objects, s_t.players[0].held_object, variant["layout"])
                    state_op = convert_state_to_input(OvercookedState, s_t.objects, s_t.players[1].held_object, variant["layout"])
                else:
                    state = convert_state_to_input(OvercookedState_2_pot, s_t.objects, s_t.players[0].held_object, variant["layout"])
                    state_op = convert_state_to_input(OvercookedState_2_pot, s_t.objects, s_t.players[1].held_object, variant["layout"])

                a_t = team.joint_action(s_t) 
                print(f"\n-----------Controller-----------\n")  
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}") 
                obs, reward, done, env_info = env.step(a_t)

                ml_actions = obs.ml_actions
                # Append the completed medium-level action of each player to its state vector
                if not variant["record_action_only"]:
                    state = convert_action_to_input(state, ml_actions[0])
                    state_op = convert_action_to_input(state_op, ml_actions[1])
                else:

                    if ml_actions[0] in interact_action_list:
                        interact_act_p0 = ml_actions[0]
                    if ml_actions[1] in interact_action_list:
                        interact_act_p1 = ml_actions[1] 
                    state = convert_action_to_input(state, interact_act_p0, variant["layout"])
                    state_op = convert_action_to_input(state_op, interact_act_p1, variant["layout"])

                if reward == 20:
                    if variant["layout"] == "cramped_room":
                        state[9] = 1
                        state_op[9] = 1
                    else:
                        state[14] = 1
                        state_op[14] = 1

                if variant['collect_data'] == 'p0':
                    timestep = [old_state, state, old_op_state, reward]
                else:
                    timestep = [old_op_state, state_op, old_state, reward]

                if old_state is not None and variant['save_buffer']:
                    if not variant['record_action_only']:
                        data_buffer.append(timestep)
                    else:
                        if variant["layout"] == "cramped_room":
                            if timestep[1] and any(timestep[1][i] == 1 for i in range(10, 17)):
                                data_buffer.append(timestep)
                        else:
                            if timestep[1] and any(timestep[1][i] == 1 for i in range(15, 22)):
                                data_buffer.append(timestep)

                    if t % 10 == 0:
                        write_buffer_to_folder(data_buffer, variant['buffer_path'])

                old_state = state
                old_op_state = state_op

                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')

            if variant['save_buffer']:
                write_buffer_to_folder(data_buffer, variant['buffer_path'])

            ## finish one episode
            if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
                print(f"\n================\n")
                try: # ProAgent id = 0
                    print(f"P1's real behavior: {team.agents[0].teammate_ml_actions_dict}")
                    print(f"The infered P1's intention: {team.agents[0].teammate_intentions_dict}")
                except: # ProAgent id = 1
                    print(f"P0's real behavior: {team.agents[1].teammate_ml_actions_dict}")
                    print(f"The infered P0's intention: {team.agents[1].teammate_intentions_dict}")
                print(f"\n================\n")
 
        elif mode == 'demo':
            pass
         
        print(f"Episode {i+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)
       
    # Number of LLM plans rejected by the verifier (causally invalid actions), last episode
    fail_count = 0
    if p0_algo == "ProAgent":
        fail_count = agents_list[0].fail_count
    elif p1_algo == "ProAgent":
        fail_count = agents_list[1].fail_count

    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")

    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
        "fail_count": int(fail_count)
    }
    for (k,v) in result_dict.items():
        print(f'{k}: {v}')

    if variant['save']:
        save_result(variant, result_dict, p0_algo, p1_algo, episode, layout, horizon)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':

    parser = ArgumentParser(description='CausalPlan on Overcooked-AI')

    # environment / partners
    parser.add_argument('--layout', '-l', type=str, default='cramped_room', choices=list(LAYOUT_ABBR.keys()))
    parser.add_argument('--p0', type=str, default='COLE', choices=['ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay'], help='Algorithm for Player 0')
    parser.add_argument('--p1', type=str, default='ProAgent', choices=['ProAgent', 'Greedy', 'COLE', 'FCP', 'MEP', 'PBT', 'SP', 'BC', 'Random', 'Stay'], help='Algorithm for Player 1')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon (timesteps) of one episode')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # LLM agent (only used when one of the players is ProAgent / CausalPlan)
    HF_MODELS = ["google/gemma-1.1-7b-it", "meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-14B-Instruct-1M", "meta-llama/Llama-3.3-70B-Instruct"]
    COHERE_MODELS = ["command-r-08-2024", "command-r-plus-08-2024"]
    parser.add_argument('--gpt_planner_model', type=str, default="meta-llama/Llama-3.3-70B-Instruct", choices=HF_MODELS,
                        help='Backbone LLM used as the planner (local Hugging Face model)')
    parser.add_argument('--gpt_analysis_model', type=str, default='command-r-08-2024', choices=COHERE_MODELS + HF_MODELS,
                        help='LLM producing the scene analysis in the dual-prompt pipeline')
    parser.add_argument('--gpt_explainer_model', type=str, default='command-r-08-2024', choices=COHERE_MODELS + HF_MODELS,
                        help='LLM used by the explainer (belief correction) module')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l2-ap', choices=['l1-p', 'l2-ap', 'l3-aip'], help="'l1-p': plan directly; 'l2-ap': analysis then plan; 'l3-aip': analysis, intention and plan.")
    parser.add_argument('--belief_revision', '-br', type=str2bool, default=False, help='Use belief revision (only with l3-aip)')
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k'], help='Dialogue history retrieval method')
    parser.add_argument('--K', type=int, default=1, help="Number of history dialogues to retrieve")

    # CausalPlan
    parser.add_argument('--use_causal_graph', '-cg', type=str2bool, default=True, help='Causal-Aware Planning: re-weight LLM candidate actions with the causal action matrix')
    parser.add_argument('--use_failure_handled', '-fh', type=str2bool, default=True, help='Causal Backup Plan: fall back to the causal action matrix when no valid candidate is generated')
    parser.add_argument('--gamma', type=float, default=0.5, help='Weight of the LLM probability p_a in the score fusion (Eq. 6)')
    parser.add_argument('--causal_graph_path', '-cgp', type=str, default=None, help='Causal action matrix file in data/ (auto-selected from layout/player if omitted)')
    parser.add_argument('--transfer_graph_source', '-tgs', type=str, default=None, choices=[None, 'aa', 'cor', 'fc', 'cc'],
                        help='Use the causal graph learned on another layout (cross-layout transfer). Only valid among the two-pot layouts (aa/cor/fc/cc).')
    parser.add_argument('--ablation', type=str2bool, default=False, help='Use the Cramped Room causal graph learned from Llama-8B-collected data instead of MEP')

    # SCA model training / data collection
    parser.add_argument('--train_SCM', '-scm', type=str2bool, default=False, help='Train the SCA model on --buffer_path and save it to --causal_graph_path')
    parser.add_argument('--train_SCM_step', type=int, default=50000, help='Number of SCA training iterations')
    parser.add_argument('--sparsity_lambda', type=float, default=1e-7, help='Sparsity coefficient lambda (Eq. 4)')
    parser.add_argument('--save_buffer', type=str2bool, default=False, help='Collect trajectories into --buffer_path')
    parser.add_argument('--buffer_path', '-bp', type=str, default=None, help='Buffer file in data/')
    parser.add_argument('--record_action_only', '-ra', type=str2bool, default=False, help='Only record transitions that complete an interact (medium-level) action')
    parser.add_argument('--collect_data', type=str, default='p1', choices=['p1', 'p0'], help='Which player the buffer is collected for')

    # logging
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'], help='exp mode runs step-by-step')
    parser.add_argument('--save', type=str2bool, default=True, help='Save the results')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory to save results')

    args = parser.parse_args()
    variant = vars(args)

    if (variant['train_SCM'] or variant['save_buffer']) and variant['buffer_path'] is None:
        parser.error('--buffer_path is required with --train_SCM True or --save_buffer True')
    if variant['train_SCM'] and variant['causal_graph_path'] is None:
        parser.error('--causal_graph_path is required with --train_SCM True')

    # Select the pretrained causal action matrix for the CausalPlan player
    if variant['causal_graph_path'] is None and "ProAgent" in (variant["p0"], variant["p1"]):
        player = "p0" if variant["p0"] == "ProAgent" else "p1"
        if variant["ablation"]:
            graph_source, suffix = "cr", "_llama"
        else:
            graph_source, suffix = variant["transfer_graph_source"] or LAYOUT_ABBR[variant["layout"]], ""
        variant["causal_graph_path"] = f"edge_params_with_action_after_200k_train_50k_test_w_op_final_{graph_source}_act_only_{player}{suffix}.pt"

    print(variant)
    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finished all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")

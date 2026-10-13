import time
import sys 
import itertools, os, json, re

from collections import defaultdict
import numpy as np
import pkg_resources
import copy 

from .modules import Module
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path 
from overcooked_ai_py.planning.search import get_intersect_counter 
from overcooked_ai_py.planning.search import query_counter_states 

from .utils import set_seed, parse_response_analysis, normalize_action

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

NAME_TO_ACTION = {
	"NORTH": Direction.NORTH,
	"SOUTH": Direction.SOUTH,
	"EAST": Direction.EAST,
	"WEST": Direction.WEST,
	"INTERACT": Action.INTERACT,
	"STAY": Action.STAY
}


set_seed(1)

OvercookedState = [ 'empty_hand', # empty hand 0
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

class ProAgent(object):
	"""
	Base class for LLM-based Overcooked agents.
	"""
	def __init__(self, planner_model="meta-llama/Llama-3.3-70B-Instruct", explainer_model="command-r-08-2024",
				 analysis_model="command-r-08-2024"):
		self.agent_index = None
		self.planner_model = planner_model
		self.explainer_model = explainer_model
		self.analysis_model = analysis_model


	def set_agent_index(self, agent_index):
		raise NotImplementedError

	def action(self, state):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError


class ProMediumLevelAgent(ProAgent):
	"""
	ProAgent-style medium-level LLM agent extended with CausalPlan
	(Causal-Aware Planning and Causal Backup Plan).
	"""
	def __init__(
			self,
			mlam,
			layout,
			planner_model='meta-llama/Llama-3.3-70B-Instruct',
			explainer_model='command-r-08-2024',
			analysis_model='command-r-08-2024',
			prompt_level='l2-ap', # ['l1-p', 'l2-ap', 'l3-aip']
			belief_revision=False,
			use_causal_graph=False,
			causal_graph=None,
			use_failure_handled=False,
			gamma=0.5,
			retrival_method="recent_k",
			K=1, 
			auto_unstuck=False,
			controller_mode='new', # the default overcooked-ai Greedy controller
			debug_mode='N', 
			agent_index=None,
			outdir = None,
			debug = False):
		super().__init__(planner_model=planner_model, explainer_model=explainer_model, analysis_model=analysis_model)

		self.trace = True
		self.debug_mode = 'Y'
		self.controller_mode = controller_mode
		self.mlam = mlam
		self.layout = layout
		self.mdp = self.mlam.mdp
		
		self.out_dir = outdir 
		self.agent_index = agent_index

		self.prompt_level = prompt_level
		self.belief_revision = belief_revision
	
		self.retrival_method = retrival_method
		self.K = K
		
		self.prev_state = None
		self.auto_unstuck = auto_unstuck

		self.current_ml_action = None
		self.past_ml_actions = None

		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.pot_id_to_pos = []
		self.plan_action = {}
		self.plan_path = {}
		self.plan_path_with_goal = {}

		self.layout_prompt = self.generate_layout_prompt()
		
		if self.layout == 'cramped_room':
			self.index = [0] * len(OvercookedState)
			self.op_index = [0] * len(OvercookedState)
		else:
			self.index = [0] * len(OvercookedState_2_pot)
			self.op_index = [0] * len(OvercookedState_2_pot)
		
		self.previous_action = None

		self.use_causal_graph = use_causal_graph
		self.causal_graph = causal_graph
		self.use_failure_handled = use_failure_handled
		self.gamma = gamma
		self.debug = debug
		self.fail_count = 0

	def set_mdp(self, mdp):
		self.mdp = mdp

	def create_gptmodule(self, module_name, full_model_name, file_type='txt', retrival_method='recent_k', K=10, backend=None):
		print(f"\n--->Initializing GPT {module_name}<---\n")    

		# prompt_file = os.path.join(PROMPT_DIR, self.model, module_name, self.layout+f'_{self.agent_index}.'+file_type)

		if "gpt" in full_model_name or "text-davinci" in full_model_name:
			model_name = "gpt"
		elif "meta-llama" in full_model_name or "google" in full_model_name or "Qwen" in full_model_name or "command" in full_model_name:
			# All open-source / Cohere backbones share the ProAgent prompt library in prompts/gpt
			model_name = "gpt"

		# print("Prompt_dir", PROMPT_DIR)

		if module_name == "planner":
			if self.use_causal_graph:
				prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level, f'{self.layout}_{self.agent_index}.{file_type}')
			else:
				prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level, f'{self.layout}_{self.agent_index}.{file_type}')
		elif module_name == "explainer":
			prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'player{self.agent_index}.{file_type}')
		else:
			raise Exception(f"Module {module_name} not supported.")

		# print("Prompt_file", prompt_file)
		with open(prompt_file, "r") as f:
			if file_type == 'json':
				messages = json.load(f)
			elif file_type == 'txt':
				messages = [{"role": "system", "content": f.read()}]
			else:
				print("Unsupported file format.")
		
		return Module(messages, full_model_name, retrival_method, K, causal_graph=self.causal_graph, backend=backend)

	def reset(self):
		self.planner.reset()
		self.explainer.reset()
		self.analyzer.dialog_history_list = self.planner.dialog_history_list
		self.prev_state = None
		self.current_ml_action = None
		self.current_ml_action_steps = 0
		self.time_to_wait = 0
		self.possible_motion_goals = None
		self.current_timestep = 0
		self.teammate_ml_actions_dict = {}
		self.teammate_intentions_dict = {}

	def set_agent_index(self, agent_index):
		self.agent_index = agent_index
		self.planner = self.create_gptmodule("planner", full_model_name=self.planner_model, retrival_method=self.retrival_method, K=self.K)
		# Scene analysis (first prompt of the dual-prompt pipeline); shares the planner's prompt and dialogue history
		self.analyzer = self.create_gptmodule("planner", full_model_name=self.analysis_model, retrival_method=self.retrival_method, K=self.K, backend=self.planner)
		self.analyzer.dialog_history_list = self.planner.dialog_history_list
		backend = self.planner if self.explainer_model == self.planner_model else self.analyzer
		self.explainer = self.create_gptmodule("explainer", full_model_name=self.explainer_model, retrival_method='recent_k', K=self.K, backend=backend)

		print(self.planner.instruction_head_list[0]['content'])

	def generate_layout_prompt(self):
		layout_prompt_dict = {
			"onion_dispenser": " <Onion Dispenser {id}>",
			"dish_dispenser": " <Dish Dispenser {id}>",
			"tomato_dispenser": " <Tomato Dispenser {id}>",
			"serving": " <Serving Loc {id}>",
			"pot": " <Pot {id}>",
		}
		layout_prompt = "Here's the layout of the kitchen:"
		for obj_type, prompt_template in layout_prompt_dict.items():
			locations = getattr(self.mdp, f"get_{obj_type}_locations")()
			for obj_id, obj_pos in enumerate(locations):
				layout_prompt += prompt_template.format(id=obj_id) + ","
				if obj_type == "pot":
					self.pot_id_to_pos.append(obj_pos)
		layout_prompt = layout_prompt[:-1] + ".\n"
		return layout_prompt

	def generate_state_index(self, state):
		if self.layout == 'cramped_room':
			ego = state.players[self.agent_index]
			ego_object = ego.held_object.name if ego.held_object else "nothing"

			if ego_object == "nothing":
				self.index[0] = 1
			if ego_object == "onion":  
				self.index[1] = 1
			if ego_object == "dish":  
				self.index[2] = 1
			if "soup" in str(ego_object):
				self.index[3] = 1
			pot_states_dict = self.mdp.get_pot_states(state)
			for key in pot_states_dict.keys():
				if key == "empty":
					if pot_states_dict[key] != []:
						self.index[4] = 1
				else:
					for soup_key in pot_states_dict[key].keys():
						# soup_key: ready, cooking, partially_full
						for pos in pot_states_dict[key][soup_key]:
							soup_object = state.get_object(pos)
							soup_type, num_items, cook_time = soup_object.state
							if soup_key == "1_items":
								self.index[5] = 1
							elif soup_key == "2_items":
								self.index[6] = 1
							elif soup_key == "cooking":
								self.index[7] = 1
							elif soup_key == "ready":
								self.index[8] = 1
							else:
								pass
		else:
			ego = state.players[self.agent_index]
			ego_object = ego.held_object.name if ego.held_object else "nothing"

			if ego_object == "nothing":
				self.index[0] = 1
			if ego_object == "onion":  
				self.index[1] = 1
			if ego_object == "dish":  
				self.index[2] = 1
			if "soup" in str(ego_object):
				self.index[3] = 1
			pot_states_dict = self.mdp.get_pot_states(state)
			for key in pot_states_dict.keys():
				if key == "empty":
					if len(pot_states_dict[key]) == 2:
						self.index[4] = 1
						self.index[9] = 1
					if len(pot_states_dict[key]) == 1 and pot_states_dict[key][0] in [(4, 2), (3, 0)]:
						self.index[4] = 1
					if len(pot_states_dict[key]) == 1 and pot_states_dict[key][0] in [(4, 3), (4, 1), (4, 0)]:
						self.index[9] = 1

				else:
					for soup_key in pot_states_dict[key].keys():
						# soup_key: ready, cooking, partially_full
						for pos in pot_states_dict[key][soup_key]:
							soup_object = state.get_object(pos)
							soup_type, num_items, cook_time = soup_object.state
							if soup_key == "1_items" and pos in [(4, 2), (3, 0)]:
								self.index[5] = 1
							if soup_key == "2_items" and pos in [(4, 2), (3, 0)]:
								self.index[6] = 1
							if soup_key == "cooking" and pos in [(4, 2), (3, 0)]:
								self.index[7] = 1
							if soup_key == "ready" and pos in [(4, 2), (3, 0)]:
								self.index[8] = 1
							if soup_key == "1_items" and pos in [(4, 3), (4, 1), (4, 0)]:
								self.index[10] = 1
							if soup_key == "2_items" and pos in [(4, 3), (4, 1), (4, 0)]:
								self.index[11] = 1
							if soup_key == "cooking" and pos in [(4, 3), (4, 1), (4, 0)]:
								self.index[12] = 1
							if soup_key == "ready" and pos in [(4, 3), (4, 1), (4, 0)]:
								self.index[13] = 1

	def generate_state_index_w_op(self, state):
		teammate = state.players[1 - self.agent_index]
		ego_object = teammate.held_object.name if teammate.held_object else "nothing"

		if ego_object == "nothing":
			self.op_index[0] = 1
		if ego_object == "onion":  
			self.op_index[1] = 1
		if ego_object == "dish":  
			self.op_index[2] = 1
		if "soup" in str(ego_object):
			self.op_index[3] = 1

	def generate_action_index(self, state):
		if self.layout == 'cramped_room':
			ego = state.players[self.agent_index]
			ego_object = ego.held_object.name if ego.held_object else "nothing"

			if "pickup_onion" in self.previous_action:
				self.index[10] = 1
			elif "put_onion_in_pot" in self.previous_action:
				self.index[11] = 1
			elif "pickup_dish" in self.previous_action:
				self.index[12] = 1
			elif "fill_dish_with_soup" in self.previous_action:
				self.index[13] = 1
			elif "deliver_soup" in self.previous_action:
				self.index[14] = 1
			elif "place_obj_on_counter" in self.previous_action:
				if ego_object == "onion":  
					self.index[15] = 1
				if ego_object == "onion":  
					self.index[16] = 1
			else:
				pass
		else:
			ego = state.players[self.agent_index]
			ego_object = ego.held_object.name if ego.held_object else "nothing"

			if "pickup_onion" in self.previous_action:
				self.index[15] = 1
			elif "put_onion_in_pot" in self.previous_action:
				self.index[16] = 1
			elif "pickup_dish" in self.previous_action:
				self.index[17] = 1
			elif "fill_dish_with_soup" in self.previous_action:
				self.index[18] = 1
			elif "deliver_soup" in self.previous_action:
				self.index[19] = 1
			elif "place_obj_on_counter" in self.previous_action:
				if ego_object == "onion":  
					self.index[20] = 1
				if ego_object == "onion":  
					self.index[21] = 1
			else:
				pass
							
	def generate_state_prompt(self, state):
		# self.generate_state_index(state)

		ego = state.players[self.agent_index]
		teammate = state.players[1 - self.agent_index]

		time_prompt = f"Scene {state.timestep}: "
		ego_object = ego.held_object.name if ego.held_object else "nothing"
		teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
		ego_state_prompt = f"<Player {self.agent_index}> holds "
		if ego_object == 'soup':
			ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
		elif ego_object == 'nothing':
			ego_state_prompt += f"{ego_object}. "
		else:
			ego_state_prompt += f"one {ego_object}. "
		
		teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
		if teammate_object == 'soup':
			teammate_state_prompt += f"a dish with {teammate_object}. "
		elif teammate_object == "nothing":
			teammate_state_prompt += f"{teammate_object}. "
		else:
			teammate_state_prompt += f"one {teammate_object}. "

		
		kitchen_state_prompt = "Kitchen states: "
		prompt_dict = {
			"empty": "<Pot {id}> is empty. ",
			"cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps. ",
			"ready": "<Pot {id}> has already cooked the soup. ",
			"1_items": "<Pot {id}> has 1 onion. ",
			"2_items": "<Pot {id}> has 2 onions. ",
			"3_items": "<Pot {id}> has 3 onions and is full. "
		}

		pot_states_dict = self.mdp.get_pot_states(state)   

		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			for key in pot_states_dict.keys():
				if key == "cooking":
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						soup_object = state.get_object(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
				else:
					for pos in pot_states_dict[key]:
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
		
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			for key in pot_states_dict.keys():
				if key == "empty":
					for pos in pot_states_dict[key]: 
						pot_id = self.pot_id_to_pos.index(pos)
						kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
				else: # key = 'onion' or 'tomota'
					for soup_key in pot_states_dict[key].keys():
						# soup_key: ready, cooking, partially_full
						for pos in pot_states_dict[key][soup_key]:
							pot_id = self.pot_id_to_pos.index(pos)
							soup_object = state.get_object(pos)
							soup_type, num_items, cook_time = soup_object.state
							if soup_key == "cooking":
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
							elif soup_key == "partially_full":
								pass
							else:
								kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)
 

		intersect_counters = get_intersect_counter(
								state.players_pos_and_or[self.agent_index], 
								state.players_pos_and_or[1 - self.agent_index], 
								self.mdp, 
								self.mlam
							)
		counter_states = query_counter_states(self.mdp, state)  

		if self.layout == 'forced_coordination': 
			kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
			count_states = {}  
			for i in intersect_counters:  
				obj_i = 'nothing' 
				if counter_states[i] != ' ': 
					obj_i = counter_states[i]                
				if obj_i in count_states:  
					count_states[obj_i] += 1
				else: 
					count_states[obj_i]  = 1 
			total_obj = ['onion', 'dish']
			for i in count_states:   
				if i == 'nothing': 
					continue 
				kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
			for i in total_obj: 
				if i not in count_states:        
					kitchen_state_prompt += f'No counters have {i}. ' 

		if self.layout == 'forced_coordination': 
			teammate_state_prompt = ""		
		
		self.generate_state_index_w_op(state)

		return (self.layout_prompt + time_prompt + ego_state_prompt +
				teammate_state_prompt + kitchen_state_prompt)

	def count_action(self):
		if self.current_ml_action not in self.plan_action:
			self.plan_action[self.current_ml_action] = 1
		else:
			self.plan_action[self.current_ml_action] += 1
	
	def count_consequence_action(self, state):
		ego = state.players[self.agent_index]
		ego_object = ego.held_object.name if ego.held_object else "nothing"

		if ego_object == "nothing":
			control_agent = "nothing"
		if ego_object == "onion":  
			control_agent = "onion"
		if ego_object == "dish":  
			control_agent = "dish"
		if "soup" in str(ego_object):
			control_agent = "soup"
		
		ego = state.players[1-self.agent_index]
		ego_object = ego.held_object.name if ego.held_object else "nothing"

		if ego_object == "nothing":
			other_agent = "nothing"
		if ego_object == "onion":  
			other_agent = "onion"
		if ego_object == "dish":  
			other_agent = "dish"
		if "soup" in str(ego_object):
			other_agent = "soup"

		pot_states_dict = self.mdp.get_pot_states(state)
		# time.sleep(3)

		pot_1 = "Not_available"
		pot_states_dict = self.mdp.get_pot_states(state)
		for key in pot_states_dict.keys():
			if key == "empty":
				if pot_states_dict[key] != []:
					pot_1 = "pot_1_empty"
			else:
				for soup_key in pot_states_dict[key].keys():
					# soup_key: ready, cooking, partially_full
					for pos in pot_states_dict[key][soup_key]:
						soup_object = state.get_object(pos)
						soup_type, num_items, cook_time = soup_object.state
						if soup_key == "1_items":
							pot_1 = "pot_1_1_items"
						elif soup_key == "2_items":
							pot_1 = "pot_1_2_items"
						elif soup_key == "cooking":
							pot_1 = "pot_1_cooking"
						elif soup_key == "ready":
							pot_1 = "pot_1_ready"
						else:
							pass

		if self.past_ml_actions is not None:
			path = control_agent + ", " + other_agent + ", " + pot_1 + ", " + self.past_ml_actions + "---->" + self.current_ml_action
			if path not in self.plan_path:
				self.plan_path[path] = 1
			else:
				self.plan_path[path] += 1
			

	def generate_belief_prompt(self):
		ego_id = self.agent_index
		intention_prompt = f"All <Player {ego_id}> infered intentions about <Player {1-ego_id}>: {self.teammate_intentions_dict}.\n"
		real_behavior_prompt = f"<Player {1-ego_id}> real behaviors: {self.teammate_ml_actions_dict}.\n"
		belief_prompt = intention_prompt + real_behavior_prompt
		return belief_prompt
	
	##################
	'''
	The followings are the Planner part
	'''
	##################

	def action(self, state):
		
		start_pos_and_or = state.players_pos_and_or[self.agent_index]

		# only use to record the teammate ml_action, 
		# if teammate finish ml_action in t-1, it will record in s_t, 
		# otherwise, s_t will just record None,
		# and we here check this information and store it into proagent
		self.current_timestep = state.timestep
		if state.ml_actions[1-self.agent_index] != None:
			self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]

		# if current ml action does not exist, generate a new one
		if self.current_ml_action is None:
			self.current_ml_action = self.generate_ml_action(state)
			self.count_action()
			self.count_consequence_action(state)

		# if the current ml action is in process, Player{self.agent_index} done, else generate a new one
		if self.current_ml_action_steps > 0:
			current_ml_action_done = self.check_current_ml_action_done(state)
			if current_ml_action_done:
				# generate a new ml action
				self.generate_success_feedback(state)
				self.current_ml_action = self.generate_ml_action(state)
				self.count_action()
				self.count_consequence_action(state)


		count = 0
		while not self.validate_current_ml_action(state):
			self.fail_count += 1
			print("Fail count", self.fail_count)
			# if validation fails, generate a new ml action. 
			# validate_current_ml_action True --> pass

			self.trace = False
			self.generate_failure_feedback(state)
			self.current_ml_action = self.generate_ml_action(state, failure=True, count=count)
			self.count_action()
			self.count_consequence_action(state)
			
			count += 1
			if count > 3:
				self.current_ml_action = "wait(1)"
				self.time_to_wait = 1

		
		self.trace = True 
		if "wait" in self.current_ml_action:
			# If wait in action 
			self.current_ml_action_steps += 1
			self.time_to_wait -= 1
			lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
			chosen_action = lis_actions[np.random.randint(0, len(lis_actions))]
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				self.prev_state = state
				return chosen_action, {}
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				self.prev_state = state
				return chosen_action
		else:
			possible_motion_goals = self.find_motion_goals(state)    
			current_motion_goal, chosen_action = self.choose_motion_goal(
				start_pos_and_or, 
				possible_motion_goals, 
				state
			)
		# if "wait" in self.current_ml_action: 
		# 	print(f'current motion goal for P{self.agent_index} is wait') 
		# else: 
		# 	if current_motion_goal is None: 
		# 		current_motion_goal = 'None' 
		# 	print(f'current motion goal for P{self.agent_index} is {current_motion_goal}') 


		if self.auto_unstuck and chosen_action != Action.INTERACT:
			if (
					self.prev_state is not None
					and state.players
					== self.prev_state.players
			):
				if self.agent_index == 0:
					joint_actions = list(
						itertools.product(Action.ALL_ACTIONS, [Action.STAY])
					)
				elif self.agent_index == 1:
					joint_actions = list(
						itertools.product([Action.STAY], Action.ALL_ACTIONS)
					)
				else:
					raise ValueError("Player index not recognized")

				unblocking_joint_actions = []
				for j_a in joint_actions:
					if j_a != [Action.INTERACT,Action.STAY] and  j_a != [Action.STAY,Action.INTERACT]:
						if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
							new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
						elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
							new_state, _, _ = self.mlam.mdp.get_state_transition(state, j_a)		
						if (
								new_state.players_pos_and_or
								!= self.prev_state.players_pos_and_or
							):
							unblocking_joint_actions.append(j_a)
				unblocking_joint_actions.append([Action.STAY, Action.STAY])
				chosen_action = unblocking_joint_actions[
					np.random.choice(len(unblocking_joint_actions))
				][self.agent_index]

		self.prev_state = state
		if chosen_action is None:
			self.current_ml_action = "wait(1)"
			self.time_to_wait = 1
			chosen_action = Action.STAY
		self.current_ml_action_steps += 1

		# print(f'ml_action = {self.current_ml_action}') 
		# print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
		print(self.plan_action)
		print(self.plan_path)
		self.past_ml_actions = copy.deepcopy(self.current_ml_action)

		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			return chosen_action, {}
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			return chosen_action

	def parse_ml_action(self, response, agent_index): 
		"""
		Convert the response from the LLM's query to a medium level action string.
		This includes wait action.
		"""
		if agent_index == 0: 
			pattern = r'layer\s*0: (.+)'
		elif agent_index == 1: 
			pattern = r'layer\s*1: (.+)'
		else:
			raise ValueError("Unsupported agent index.")

		match = re.search(pattern, response)
		if match:
			action_string = match.group(1)
		else:
			# raise Exception("please check the query")
			action_string = response
			# print("please check the query")

		# Parse the response to get the medium level action string
		try: 
			ml_action = action_string.split()[0]
		except: 
			print('failed on 528') 
			action_string = 'wait(1)'
			ml_action = action_string

		if "place" in action_string:
			ml_action = "place_obj_on_counter"
		elif "pick" in action_string:
			if "onion" in action_string:
				ml_action = "pickup_onion"
			elif "tomato" in action_string:
				ml_action = "pickup_tomato"
			elif "dish" in action_string:
				ml_action = "pickup_dish"
		elif "put" in action_string:
			if "onion" in action_string:
				ml_action = "put_onion_in_pot"
			elif "tomato" in action_string:
				ml_action = "put_tomato_in_pot"
		elif "fill" in action_string:   
			ml_action = "fill_dish_with_soup"
		elif "deliver" in action_string:
			ml_action = "deliver_soup"
		elif "wait" not in action_string:
			ml_action='wait(1)'  
			action_string = ml_action
		if "wait" in action_string:
			
			def parse_wait_string(s):
				# Check if it's just "wait"
				if s == "wait":
					return 1

				# Remove 'wait' and other characters from the string
				s = s.replace('wait', '').replace('(', '').replace(')', '').replace('"', '').replace('.', '') 

				# If it's a number, return it as an integer
				if s.isdigit():
					return int(s)

				# If it's not a number, return a default value or raise an exception
				return 1
			if self.layout == 'forced_coordination': 
				self.time_to_wait = max(3, parse_wait_string(action_string))
			else: 
				self.time_to_wait = parse_wait_string(action_string)    
			# print(ml_action) 
			# print(self.time_to_wait) 
			
			ml_action = f"wait({self.time_to_wait})"

		else:
			pass
		
		# aviod to generate two skill, eg, Plan for Player 0: "deliver_soup(), pickup(onion)".
		if "," in ml_action:
			ml_action = ml_action.split(',')[0].strip()

		return ml_action    
	
	def generate_ml_action(self, state, failure=False, count=0):
		"""
		Selects a medium level action for the current state.
		Motion goals can be thought of instructions of the form:
			[do X] at location [Y]

		In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
		a simple set of  heuristics based on the current state.

		Effectively, will return a list of all possible locations Y in which the selected
		medium level action X can be performed.
		"""
		# If belief revision is enabled, generate a belief prompt
		if self.prompt_level == "l3-aip" and self.belief_revision:
			belief_prompt = self.generate_belief_prompt()
		else:
			belief_prompt = ''

		state_prompt = belief_prompt + self.generate_state_prompt(state) 
			
		print(f"\n\n### Observation module to GPT\n")   
		print(f"State Prompt: {state_prompt}\n")	

		state_prompt = state_prompt	+ "Please analyze: "

		if self.use_causal_graph or self.use_failure_handled:
			# Active features of the causal action matrix: ego state, partner state and previous action
			self.generate_state_index(state)
			self.generate_state_index_w_op(state)
			if self.previous_action:
				self.generate_action_index(state)


		state_message = {"role": "user", "content": state_prompt}
				
		self.analyzer.current_user_message = state_message

		if failure:		# Query analyse when fail
			response_analysis = self.analyzer.query(trace = self.trace)
		else:			# Initial query analyse
			response_analysis = self.analyzer.query(stop='Plan', trace = self.trace)

		response_analysis = parse_response_analysis(response_analysis)

		state_prompt = belief_prompt + self.generate_state_prompt(state) + response_analysis + "Please give a plan: "

		state_message = {"role": "user", "content": state_prompt}
				
		self.planner.current_user_message = state_message
		if self.use_causal_graph:
			if failure :
				response = self.planner.query(stop='Scene', trace = self.trace, count=count)
			else:
				response = self.planner.query_w_cg(stop='Scene', trace = self.trace, agent_index = self.agent_index,
											index=self.index, op_index=self.op_index, failure_handled=self.use_failure_handled,
											gamma=self.gamma, layout=self.layout,
											verifier=lambda action: self.is_valid_ml_action(state, action))
		elif self.use_failure_handled and not failure:
			# Causal Backup Plan only: keep the LLM plan if it is valid, otherwise fall back to M
			response = self.planner.query(stop='Scene', trace = self.trace)
			action = normalize_action(response)
			if action is None or not self.is_valid_ml_action(state, action):
				backup_action = self.planner.get_action_based_on_causal_graph(self.index, self.op_index, self.layout)
				if backup_action is not None:
					response = f"Plan for Player {self.agent_index}: {backup_action}."
		else:
			response = self.planner.query(stop='Scene', trace = self.trace)

		# response = response_analysis.replace("Scene", "").strip()

		# print(f"Response:\n{response}\n")

		if 'wait' not in response:
			self.planner.add_msg_to_dialog_history(state_message) 
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
		
		print(f"\n\n\n### GPT Planner module\n")   
		print("====== GPT Query ======")
		print("\n===== Parser =====\n")
		## specific for prompt need intention
		if self.prompt_level == "l3-aip":
			# Generated the intention for the teammate: 1 - self.agent_index -> 2 agents --> will give the other one index
			generated_intention = self.parse_ml_action(response, 1-self.agent_index)
			# Add the new predicted intention to the teammate_intentions_dict
			self.teammate_intentions_dict[str(self.current_timestep)] = generated_intention
			print(f"Intention for Player {1 - self.agent_index}: {generated_intention}")  

		ml_action = self.parse_ml_action(response, self.agent_index)
		self.previous_action = ml_action

		if "wait" not in ml_action:
			self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
		
		print(f"Player {self.agent_index}: {ml_action}")
		self.current_ml_action_steps = 0

		if self.layout == 'cramped_room':
			self.index = [0] * len(OvercookedState)
			self.op_index = [0] * len(OvercookedState)
		else:
			self.index = [0] * len(OvercookedState_2_pot)
			self.op_index = [0] * len(OvercookedState_2_pot)

		return ml_action


	##################
	'''
	The followings are the Verificator part
	'''
	##################

	def check_current_ml_action_done(self,state):
		"""
		checks if the current ml action is done
		:return: True or False
		"""
		player = state.players[self.agent_index]
		# pot_states_dict = self.mlam.mdp.get_pot_states(state)
		if "pickup" in self.current_ml_action:
			pattern = r"pickup(?:[(]|_)(\w+)(?:[)]|)" # fit both pickup(onion) and pickup_onion
			obj_str = re.search(pattern, self.current_ml_action).group(1)
			# Get obj str if player is holding it --> True else False
			return player.has_object() and player.get_object().name == obj_str
		
		elif "fill" in self.current_ml_action:
			# If filled but held object not soup --> false
			return player.held_object.name == 'soup'
		
		elif "put" in self.current_ml_action or "place" in self.current_ml_action:
			# If put in action but not holding object --> false
			return not player.has_object()
		
		elif "deliver" in self.current_ml_action:
			# If put in action but not holding object --> false
			return not player.has_object()
		
		elif "wait" in self.current_ml_action:
			return self.time_to_wait == 0

	def is_valid_ml_action(self, state, ml_action):
		"""Verifier for a candidate action: check that it satisfies the physical constraints in `state`."""
		current_ml_action = self.current_ml_action
		self.current_ml_action = ml_action
		try:
			return bool(self.validate_current_ml_action(state))
		finally:
			self.current_ml_action = current_ml_action

	def validate_current_ml_action(self, state):
		"""
		make sure the current_ml_action exists and is valid
		"""
		if self.current_ml_action is None:
			return False

		pot_states_dict = self.mdp.get_pot_states(state)
		player = state.players[self.agent_index]
		if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
			soup_cooking = len(pot_states_dict['cooking']) > 0
			soup_ready = len(pot_states_dict['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)
			cookable_pots = self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
		elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
			soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
			soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
			pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full'] + pot_states_dict["tomato"]['partially_full']
			cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)] # pot has max onions/tomotos

		
		has_onion = False
		has_tomato = False
		has_dish = False
		has_soup = False
		has_object = player.has_object()
		if has_object:
			has_onion = player.get_object().name == 'onion'
			has_tomato = player.get_object().name == 'tomato'
			has_dish = player.get_object().name == 'dish'
			has_soup = player.get_object().name == 'soup'
		empty_counter = self.mdp.get_empty_counter_locations(state)


		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:   

			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False 
			return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
		if self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			return not has_object and len(self.mdp.get_tomato_dispenser_locations()) > 0
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			flag2 = len(self.find_motion_goals(state)) == 0 
			if flag2: 
				return False 
			return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
		elif "put_onion_in_pot" in self.current_ml_action:
			return has_onion and len(pot_not_full) > 0
		elif "put_tomato_in_pot" in self.current_ml_action:
			return has_tomato and len(pot_not_full) > 0
		elif "place_obj_on_counter" in self.current_ml_action:
			return has_object and len(empty_counter) > 0
		elif "fill_dish_with_soup" in self.current_ml_action:
			return has_dish and (soup_ready or soup_cooking)
		elif "deliver_soup" in self.current_ml_action:
			return has_soup
		elif "wait" in self.current_ml_action:
			return 0 < int(self.current_ml_action.split('(')[1][:-1]) <= 20


	def generate_success_feedback(self, state):
		success_feedback = f"### Controller Validation\nPlayer {self.agent_index} succeeded at {self.current_ml_action}. \n"
		# print(success_feedback)  
		if 'wait' not in success_feedback:
			self.planner.add_msg_to_dialog_history({"role": "user", "content": f'Player {self.agent_index} succeeded at {self.current_ml_action}.'})
		
	def generate_failure_feedback(self, state):
		failure_feedback = self.generate_state_prompt(state)
		failure_feedback += f" Player {self.agent_index} failed at {self.current_ml_action}."
		failure_feedback += f" Why did Player {self.agent_index} fail ?"     
		print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")  
		failure_message = {"role": "user", "content": failure_feedback}
		self.explainer.current_user_message = failure_message
		failure_explanation = self.explainer.query()
		if "wait" not in failure_explanation:
			self.explainer.add_msg_to_dialog_history({"role": "user", "content": failure_feedback})
			self.explainer.add_msg_to_dialog_history({"role": "assistant", "content": failure_explanation})
		self.planner.add_msg_to_dialog_history({"role": "user", "content": failure_explanation})

	##################
	'''
	The followings are the Controller part almost inherited from GreedyHumanModel class
	'''
	##################	
		
	def find_shared_counters(self, state, mlam):  
		counter_dicts = query_counter_states(self.mdp, state) 

		counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
						state.players_pos_and_or[1 - self.agent_index], 
						self.mdp, 
						self.mlam
					)    

		print('counter_list = {}'.format(counter_list))  
		lis = [] 
		for i in counter_list:  
			if counter_dicts[i] == ' ':  
				lis.append(i)       
		available_plans = mlam._get_ml_actions_for_positions(lis)
		return available_plans          

	def find_motion_goals(self, state):
		"""
		Generates the motion goals for the given medium level action.
		:param state:
		:return:
		"""
		am = self.mlam
		motion_goals = []
		player = state.players[self.agent_index]
		pot_states_dict = self.mdp.get_pot_states(state)
		counter_objects = self.mdp.get_counter_objects_dict(
			state, list(self.mdp.terrain_pos_dict["X"])
		)
		if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
			motion_goals = am.pickup_onion_actions_new(state, counter_objects, state.players_pos_and_or, self.agent_index) 


		elif self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
			motion_goals = am.pickup_tomato_actions(state, counter_objects)
		elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
			motion_goals = am.pickup_dish_actions_new(state, counter_objects , state.players_pos_and_or, self.agent_index)
		elif "put_onion_in_pot" in self.current_ml_action:
			motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
		elif "put_tomato_in_pot" in self.current_ml_action:
			motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
		elif "place_obj_on_counter" in self.current_ml_action:  
			motion_goals = self.find_shared_counters(state, self.mlam)     
			if len(motion_goals) == 0: 
				motion_goals = am.place_obj_on_counter_actions(state)

		elif "start_cooking" in self.current_ml_action:
			if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
				next_order = list(state.all_orders)[0]
				soups_ready_to_cook_key = "{}_items".format(len(next_order.ingredients))
				soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
			elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
				soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]
			only_pot_states_ready_to_cook = defaultdict(list)
			only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
			motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
		elif "fill_dish_with_soup" in self.current_ml_action:
			motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
		elif "deliver_soup" in self.current_ml_action:
			motion_goals = am.deliver_soup_actions()
		elif "wait" in self.current_ml_action:
			motion_goals = am.wait_actions(player)
		else:
			raise ValueError("Invalid action: {}".format(self.current_ml_action))

		motion_goals = [
			mg
			for mg in motion_goals
			if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
				player.pos_and_or, mg
			)
		]

		return motion_goals

	def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
		"""
		For each motion goal, consider the optimal motion plan that reaches the desired location.
		Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
		or rationally), and returns the plan and the corresponding first action on that plan.
		"""

		if self.controller_mode == 'new':
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal_new(
				start_pos_and_or, motion_goals, state
			)
		else: 
			(
				chosen_goal,
				chosen_goal_action,
			) = self.get_lowest_cost_action_and_goal(
				start_pos_and_or, motion_goals
			)
		return chosen_goal, chosen_goal_action
	
	def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""
		min_cost = np.inf
		best_action, best_goal = None, None
		for goal in motion_goals:
			action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
				start_pos_and_or, goal
			)
			if plan_cost < min_cost:
				best_action = action_plan[0]
				min_cost = plan_cost
				best_goal = goal
		return best_goal, best_action

	
	def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
		"""
		Chooses motion goal that has the lowest cost action plan.
		Returns the motion goal itself and the first action on the plan.
		"""   
		min_cost = np.inf
		best_action, best_goal = None, None
		for goal in motion_goals:   
			action_plan, plan_cost = self.real_time_planner(
				start_pos_and_or, goal, state
			)     
			if plan_cost < min_cost:
				best_action = action_plan
				min_cost = plan_cost
				best_goal = goal     
		if best_action is None: 
			# print('\n\n\nBlocking Happend, executing default path\n\n\n')
			# print('current position = {}'.format(start_pos_and_or)) 
			# print('goal position = {}'.format(motion_goals))        
			if np.random.rand() < 0.5:  
				return None, Action.STAY
			else: 
				return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
		return best_goal, best_action

	def real_time_planner(self, start_pos_and_or, goal, state):   
		terrain_matrix = {
			'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
			'height': len(self.mlam.mdp.terrain_mtx), 
			'width' : len(self.mlam.mdp.terrain_mtx[0]) 
		}
		other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
		action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, terrain_matrix) 

		return action_plan, plan_cost

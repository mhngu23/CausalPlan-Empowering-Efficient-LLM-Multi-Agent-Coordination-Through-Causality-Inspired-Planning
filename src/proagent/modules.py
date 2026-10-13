import os
import re
import random
import httpx

import torch
import time
from rich import print as rprint

import cohere
from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from transformers import StoppingCriteria, StoppingCriteriaList

from .utils import normalize_action, merge_equivalent_actions, softmax, generate_action_index, set_seed

set_seed(1)

# Cohere API key (only needed for the command-r / command-r-plus backends).
# Set COHERE_API_KEY in the environment or in src/proagent/secrets.env.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.env"))
cohere_api_key = os.getenv("COHERE_API_KEY") or os.getenv("API_KEY")

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

# Number of candidate plans sampled from the LLM at each decision step
NUM_CANDIDATES = 20


# Define a custom stopping criterion
class StopOnWordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stopword, input_length):
        self.tokenizer = tokenizer
        self.stopword_id = tokenizer.convert_tokens_to_ids(stopword)
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the stopword token exists in the latest generated tokens
        for seq in input_ids:
            if self.stopword_id in seq[self.input_length:]:
                return True
        return False



class Module(object):
    """
    This module is responsible for communicating with the LLMs.
    """
    def __init__(self, 
                 role_messages, 
                 model="meta-llama/Meta-Llama-3-8B-Instruct",
                 retrival_method="recent_k",
                 K=3, 
                 causal_graph=None,
                 backend=None):
        '''
        args:
        retrival_method / K: retrieve the K most recent dialogues from the history
        causal_graph: causal action matrix M used for causal-aware planning
        backend: another Module using the same model whose loaded LLM / client is reused
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None
        if backend is not None and backend.model == self.model:
            for attr in ("co", "tokenizer", "llm_model"):
                if hasattr(backend, attr):
                    setattr(self, attr, getattr(backend, attr))
        elif "command" in self.model:
            self.co = cohere.ClientV2(cohere_api_key)
        elif "meta-llama" in self.model or "google" in self.model or "Qwen" in self.model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                )
        
        else:
            raise ValueError(f"Model {self.model} not supported.")

        self.causal_graph = causal_graph

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    def query(self, stop=None, temperature=0.0, trace=True, count=0):
        """Plain LLM query (greedy decoding), used for the analysis prompt, re-planning and the explainer."""
        rec = self.K  
        if trace == True: 
            self.K = 0 
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 
        response = "" 
        get_response = False
        retry_count = 0

        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query LLM failed for over 3 times!")
                return {}

            elif "meta-llama" in self.model or "google" in self.model or "Qwen" in self.model:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                if  messages[0]['role'] == "system":
                    if "meta-llama" in self.model or "google" in self.model:
                        messages[1]['content'] = "<system> " + messages[0]['content'] + "</system>\n" + messages[1]['content'] 
                    else:
                        messages[1]['content'] = messages[0]['content'] + messages[1]['content']
                    messages = messages[1]['content']
                else:
                    messages = messages[1]['content']

                input_ids = self.tokenizer(messages, return_tensors="pt").to(self.llm_model.device)

                if stop:
                    stopword = stop
                    stopping_criteria = StoppingCriteriaList([StopOnWordCriteria(self.tokenizer, stopword, input_length=input_ids["input_ids"].shape[1])])
                    outputs = self.llm_model.generate(**input_ids, 
                                        max_new_tokens=256, 
                                        do_sample=False,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        stopping_criteria=stopping_criteria)
                else:
                    outputs = self.llm_model.generate(**input_ids, 
                                        max_new_tokens=256, 
                                        do_sample=False,
                                        return_dict_in_generate=True,
                                        output_scores=True)

                input_length = input_ids["input_ids"].shape[1]

                generated_tokens = outputs.sequences[:, input_length:]

                response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                return response
            
            elif 'command' in self.model:
                for attempt in range(5):
                    try:
                        response = self.co.chat(
                            model=self.model, 
                            messages=messages,
                            max_tokens=256,
                            temperature=temperature,
                            stop_sequences=[stop] if stop else None
                        )
                        break  # If successful, exit the loop
                    except httpx.ReadTimeout:
                        print(f"Timeout occurred. Retrying {attempt + 1}/5...")
                        time.sleep(5)
                else:
                    raise Exception("API request failed after multiple retries.")

            else:
                raise Exception(f"Model {self.model} not supported.")
            
            get_response = True

        return self.parse_response(response)


    
    def get_active_features(self, index, op_index, layout):
        """Column indices J = Active(s_t, a_{t-1}) of the causal action matrix: active ego
        state / previous-action features followed by the active partner-state features."""
        offset = 17 if layout == "cramped_room" else 22
        return [i for i, x in enumerate(index) if x == 1] + \
               [i + offset for i, x in enumerate(op_index) if x == 1]

    def get_causal_score(self, action_index, index, op_index, layout):
        """Causal score p_c(a) = sum_{j in J} M[a, j] (Sect. 3.1). Actions without a row in M
        (e.g. wait) receive a causal score of 0."""
        if action_index is None:
            return 0.0
        return float(sum(self.causal_graph[action_index][j] for j in self.get_active_features(index, op_index, layout)))

    def get_action_based_on_causal_graph(self, index, op_index, layout):
        """Causal Backup Plan (Sect. 3.2.2): a_t = argmax_{a in A} p_c(a) over the whole action space."""
        action_names = OvercookedState[10:] if layout == "cramped_room" else OvercookedState_2_pot[15:]
        if not self.get_active_features(index, op_index, layout):
            return None
        scores = [self.get_causal_score(a, index, op_index, layout) for a in range(len(action_names))]
        best = max(range(len(scores)), key=lambda a: scores[a])
        print(f"Causal backup plan: {action_names[best]} (p_c={scores[best]:.3f})")
        return action_names[best]

    def query_w_cg(self, stop=None, temperature=0.0, trace=True, agent_index=None, index=None,
                   op_index=None, failure_handled=False, gamma=0.5, layout=None, verifier=None):
        """
        Causal-aware planning (Sect. 3.2): sample candidate plans A' from the LLM, keep those
        accepted by the verifier, fuse the LLM probability p_a with the causal score p_c
        (Eq. 6), normalise with a softmax, merge semantically equivalent proposals by summation
        and sample the final action. If A' is empty, fall back to the Causal Backup Plan.
        """
        rec = self.K
        if trace == True:
            self.K = 0
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False:
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again."
        self.K = rec

        if not ("meta-llama" in self.model or "google" in self.model or "Qwen" in self.model):
            raise ValueError(f"Causal-aware planning requires a local Hugging Face backbone, got {self.model}.")

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if messages[0]['role'] == "system":
            if "meta-llama" in self.model or "google" in self.model:
                messages[1]['content'] = "<system> " + messages[0]['content'] + "</system>\n" + messages[1]['content']
            else:
                messages[1]['content'] = messages[0]['content'] + messages[1]['content']
        prompt = messages[1]['content']

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        input_length = input_ids["input_ids"].shape[1]

        # Sample candidate plans from the LLM
        batched_input_ids = {key: value.repeat(NUM_CANDIDATES, 1) for key, value in input_ids.items()}
        generate_kwargs = dict(max_new_tokens=256,
                               do_sample=True,
                               temperature=1.0,
                               top_k=50,
                               top_p=0.9,
                               return_dict_in_generate=True,
                               output_scores=True)
        if stop:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnWordCriteria(self.tokenizer, stop, input_length=input_length)])
        outputs = self.llm_model.generate(**batched_input_ids, **generate_kwargs)

        generated_tokens = outputs.sequences[:, input_length:]
        transition_scores = self.llm_model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        transition_scores = torch.where(transition_scores == float('-inf'), 0.0, transition_scores)
        responses = [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
        # Sequence-level probability p_a of each proposal
        llm_probs = [float(torch.exp(score.sum(axis=0)).cpu()) for score in transition_scores]

        # Candidate set A': proposals that map to a known action and pass the verifier
        candidates = []
        for response, p_a in zip(responses, llm_probs):
            action = normalize_action(response)
            if action is None:
                continue
            if verifier is not None and not verifier(action):
                continue
            candidates.append((action, p_a))

        if not candidates:
            print("No valid candidate action:", responses)
            if not failure_handled:
                return ""
            backup_action = self.get_action_based_on_causal_graph(index, op_index, layout)
            if backup_action is None:
                return ""
            best_resp = f"Plan for Player {agent_index}: {backup_action}."
            print(f"Best response: {best_resp}")
            return best_resp

        # Eq. 6: fuse LLM and causal scores, then normalise over the candidates
        fused = [gamma * p_a + (1 - gamma) * self.get_causal_score(generate_action_index(action, index, layout), index, op_index, layout)
                 for action, p_a in candidates]
        candidate_probs = softmax(fused)

        # Merge semantically equivalent proposals by summing their probabilities
        final_actions, final_probs = merge_equivalent_actions([action for action, _ in candidates], candidate_probs)
        for action, prob in zip(final_actions, final_probs):
            print(f"Response: Plan for Player {agent_index}: {action} | Probability: {prob:.4f}")

        # Sample the final action from the categorical distribution over unique actions
        best_resp = random.choices(final_actions, weights=final_probs, k=1)[0]
        best_resp = f"Plan for Player {agent_index}: {best_resp}."
        print(f"Best response: {best_resp}")
        return best_resp

    def parse_response(self, response):
        if "meta-llama" in self.model or "google" in self.model or "Qwen" in self.model:
            return response[0]['generated_text']
        elif "command" in self.model: 
            return response.message.content[0].text

    def reset(self):
        self.dialog_history_list = []


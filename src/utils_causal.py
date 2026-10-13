import os
import torch

DATA_DIR = os.environ.get(
    "CAUSALPLAN_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

def convert_state_to_input(OvercookedState, env_obj, held_obj, layout="cramped_room"):
    """
    Convert a state to the input format of the model
    """
    if layout == "cramped_room":
        updated_state = [0] * len(OvercookedState)
        if not held_obj:
            updated_state[0] = 1
        if held_obj and "onion" in held_obj.name:  
            updated_state[1] = 1
        if held_obj and "dish" in held_obj.name:  
            updated_state[2] = 1
        if held_obj and "soup" in str(held_obj.name) and held_obj.state[0] == "onion" and held_obj.state[1] == 3:
            updated_state[3] = 1
        if not env_obj:
            updated_state[4] = 1

        for value in env_obj.values():  
            if env_obj and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 1:
                updated_state[5] = 1 
            if env_obj and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 2:
                updated_state[6] = 1
            if env_obj and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 3 and value.state[2] < 20:
                updated_state[7] = 1
            if env_obj and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 3 and value.state[2] >= 20:
                updated_state[8] = 1
    else:
        updated_state = [0] * len(OvercookedState)
        if not held_obj:
            updated_state[0] = 1
        if held_obj and "onion" in held_obj.name:  
            updated_state[1] = 1
        if held_obj and "dish" in held_obj.name:  
            updated_state[2] = 1
        if held_obj and "soup" in str(held_obj.name) and held_obj.state[0] == "onion" and held_obj.state[1] == 3:
            updated_state[3] = 1
        if not env_obj:
            updated_state[4] = 1
            updated_state[9] = 1

        for value in env_obj.values():  

            if env_obj and (value.position in [(4, 2), (3, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 1:
                updated_state[5] = 1 
            if env_obj and (value.position in [(4, 2), (3, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 2:
                updated_state[6] = 1
            if env_obj and (value.position in [(4, 2), (3, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 3 and value.state[2] < 20:
                updated_state[7] = 1
            if env_obj and (value.position in [(4, 2), (3, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 3 and value.state[2] >= 20:
                updated_state[8] = 1
            if env_obj and (value.position in [(4, 3), (4, 1), (4, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 1:
                updated_state[10] = 1 
            if env_obj and (value.position in [(4, 3), (4, 1), (4, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 2:
                updated_state[11] = 1
            if env_obj and (value.position in [(4, 3), (4, 1), (4, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 3 and value.state[2] < 20:
                updated_state[12] = 1
            if env_obj and (value.position in [(4, 3), (4, 1), (4, 0)]) and "soup" in str(value.name) and value.state[0] == "onion" and value.state[1] == 3 and value.state[2] >= 20:
                updated_state[13] = 1
    return updated_state

def convert_action_to_input(updated_state, action, layout="cramped_room"):
    if layout == "cramped_room":
        if action == "pickup_onion":
            updated_state[10] = 1
        elif action == "put_onion_in_pot":
            updated_state[11] = 1
        elif action == "pickup_dish":
            updated_state[12] = 1
        elif action == "fill_dish_with_soup":
            updated_state[13] = 1
        elif action == "deliver_soup":
            updated_state[14] = 1
        elif action == "place_onion_on_counter":
            updated_state[15] = 1
        elif action == "place_dish_on_counter":
            updated_state[16] = 1
    else:
        if action == "pickup_onion":
            updated_state[15] = 1
        elif action == "put_onion_in_pot":
            updated_state[16] = 1
        elif action == "pickup_dish":
            updated_state[17] = 1
        elif action == "fill_dish_with_soup":
            updated_state[18] = 1
        elif action == "deliver_soup":
            updated_state[19] = 1
        elif action == "place_onion_on_counter":
            updated_state[20] = 1
        elif action == "place_dish_on_counter":
            updated_state[21] = 1

    return updated_state

def write_buffer_to_folder(buffer, path):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, path)

    # # Convert the buffer to a PyTorch tensor
    # tensor_data = torch.tensor(buffer)
    tensor_data = buffer

    # Save the tensor data to the .pt file
    torch.save(tensor_data, file_path)

    print(f"Tensor has been written to {file_path}")

def load_buffer(path):
    try:
        # Define the path to the .pt file
        file_path = os.path.join(DATA_DIR, path)
        # Load the tensor data from the .pt file
        buffer = torch.load(file_path)
        return buffer

    except FileNotFoundError:
        print(f"Buffer file not found: {file_path}")
        


def save_causal_graph(causal_graph, path):
    try:
        # Define the path to save the .pt file
        file_path = os.path.join(DATA_DIR, path)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the causal graph as a .pt file
        torch.save(causal_graph, file_path)
        print(f"Causal graph successfully saved to {file_path}")

    except Exception as e:
        print(f"An error occurred while saving the causal graph: {e}")

def load_causal_graph(path):
    try:
        # Define the path to the .pt file
        file_path = os.path.join(DATA_DIR, path)

        # Load the tensor data from the .pt file
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        causal_graph = torch.load(file_path, map_location=device)
        causal_graph = torch.where(causal_graph < 0.1, 0.0, causal_graph)

        # print(device)
        if torch.cuda.is_available():
            causal_graph = causal_graph.detach().to('cpu').numpy()

        return causal_graph
    except FileNotFoundError:
        raise FileNotFoundError(f"Causal graph file not found: {file_path}")

def load_causal_graph_DAG(path, layout):
    try:
        # Define the path to the .pt file
        file_path = os.path.join(DATA_DIR, path)

        # Load the tensor data from the .pt file
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        causal_graph = torch.load(file_path, map_location=device)
        # causal_graph = torch.where(causal_graph < 0.1, 0.0, causal_graph)

        # print(device)
        if torch.cuda.is_available():
            causal_graph = causal_graph.detach().to('cpu').numpy()

        n, m = causal_graph.shape
        min_dim = min(n, m)
        
        
        if layout == "cramped_room":
            offset = 10  # Adjustment for cramped room layout
        else:
            offset = 15  # Adjustment for other layout
        # Remove trivial pairwise cycles between action nodes: keep only the stronger direction
        for i in range(min_dim):
            for j in range(min_dim):
                if j + offset != i+ offset:  # Exclude diagonal elements  
                    if causal_graph[i, j + offset] > 0 and causal_graph[j, i + offset] > 0:
                        if causal_graph[i, j + offset] > causal_graph[j, i + offset]:
                            causal_graph[j, i + offset] = 0  # Remove the weaker edge
                        else:
                            causal_graph[i, j + offset] = 0  # Remove the weaker edge
        return causal_graph

    except FileNotFoundError:
        raise FileNotFoundError(f"Causal graph file not found: {file_path}")

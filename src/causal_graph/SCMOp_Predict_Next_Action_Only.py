import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except ImportError:  # wandb is optional and only used for logging training curves
    wandb = None

# Set print options for better visibility of tensor values
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

# Including Action Prob in Prediction
class StructureParamsOp(nn.Module):
    def __init__(self, state_num, action_num, op_state_num):
        """
        Parameters that define the structure (causal graph) of the model.
        Args:
            state_num: Number of state variables.
            action_num: Number of action variables.
        """
        super(StructureParamsOp, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.op_state_num = op_state_num
        
        # Edge parameters defining connections between states and actions
        self.edge_params = torch.nn.Parameter(
            torch.empty((self.action_num, self.action_num + self.state_num + self.op_state_num)).uniform_(-0.1, 0.1)
        )
        

class FunctionalNetOp(nn.Module):
    def __init__(self, state_num, action_num, op_state_num):
        """
        Neural network that learns the functional relationships in the causal graph.
        Args:
            state_num: Number of state variables.
            action_num: Number of action variables.
        """
        super(FunctionalNetOp, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.op_state_num = op_state_num

        # Hidden dimensions for the MLP
        self.h_dim_1 = 64
        self.h_dim_2 = 256
        self.out_dim = 1  # Output dimension

        # Create a list of networks, one for each variable
        self.fs = nn.ModuleList()
        for i in range(self.action_num):
            self.fs.append(
                nn.Sequential(
                    nn.Linear(self.state_num + self.action_num + self.op_state_num, 64),  # Input layer
                    nn.ReLU(),                                       # Activation
                    nn.Linear(self.h_dim_1, self.h_dim_2),           # Hidden layer 1
                    nn.ReLU(),                                       # Activation
                    nn.Linear(self.h_dim_2, self.h_dim_2),           # Hidden layer 2
                    nn.ReLU(),                                       # Activation
                    nn.Linear(self.h_dim_2, self.h_dim_1),           # Hidden layer 2
                    nn.ReLU(),                                       # Activation
                    nn.Linear(self.h_dim_1, self.out_dim),           # Output layer
                    nn.Sigmoid()                                     # Activation for binary output
                )
            )
        
    def forward(self, index, x, mask):
        """
        Forward pass for the FunctionalNet.
        Args:
            index: Index of the variable being predicted.
            x: Input tensor.
            mask: Mask tensor to zero out certain inputs.
        """
        mask = mask.float()  # Convert mask to float for multiplication
        return self.fs[index](x * mask)  # Apply mask and forward pass through the respective network


class SCMOp_Predict_Next_Action_Only:
    def __init__(self, buffer=None, sparsity_lambda=1e-7, use_wandb=True):
        """
        Structural Causal Model (SCM) class for training functional and structural parameters.
        Args:
            buffer: Data buffer containing input-output samples.
            sparsity_lambda: Sparsity coefficient lambda of the regularisation loss (Eq. 4).
            use_wandb: Log training losses to Weights & Biases if it is installed.
        """
        if buffer is None:
            raise ValueError("Buffer cannot be None. Please provide a valid buffer.")

        self.buffer = buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.batch_size = 65  # Batch size for training
        self.state_num = 10  # Number of state variables
        self.action_num = 7  # Number of action variables
        self.op_state_num = 4 # Number of opponent states variables

        # Initialize FunctionalNet and StructureParams
        self.f_model = FunctionalNetOp(self.state_num, self.action_num, self.op_state_num).to(self.device)
        self.s_params = StructureParamsOp(self.state_num, self.action_num, self.op_state_num).to(self.device)

        # Loss function and optimizers
        self.criterion = nn.BCELoss()  # Binary cross-entropy loss
        self.s_optimizer = optim.Adam(self.s_params.parameters(), lr=3e-4) 
        self.f_optimizer = optim.Adam(self.f_model.parameters(), lr=3e-4) 

        self.sparsity_lambda = sparsity_lambda

        # Unique run name for logging with wandb
        self.wandb = None
        if use_wandb and wandb is not None:
            run_name = f"train_f_and_s_{int(time.time())}_rand_{random.randint(1, 1000)}"
            self.wandb = wandb.init(project='CausalPlan_SCA', name=run_name)
        self.best_loss = float('inf')  # Initialize best loss to infinity
        self.best_s_params = None  # Best structural parameters
    
    def sample_batch(self, buffer):
        """
        Sample a batch of data from the buffer.
        Args:
            buffer: Data buffer.
        """
        random_indices = random.sample(range(len(buffer)), self.batch_size)
        batch = [buffer[i] for i in random_indices]  # Extract random samples
        return batch

    def create_input(self, batch):
        """
        Create input tensor by combining specific parts of consecutive elements.
        Args:
            batch: List of input-output pairs.
        """
        states = torch.stack([torch.tensor(b[0][:self.state_num]) for b in batch[1:]])
        # Previous action a_{t-1} is taken from the same record as the state s_t and target a_t
        actions = torch.stack([torch.tensor(b[0][self.state_num:self.state_num + self.action_num]) for b in batch[1:]])
        op_states = torch.stack([torch.tensor(b[2][:self.op_state_num]) for b in batch[1:]])
        combined_tensors = torch.cat([states, actions, op_states], dim=1)

        return combined_tensors if len(combined_tensors) > 0 else torch.empty(0)
    
    def create_output(self, batch):
        """
        Create output tensor from a batch of data.
        Args:
            batch: List of input-output pairs.
        """
        actions = torch.stack([torch.tensor(b[1][self.state_num:self.state_num + self.action_num]) for b in batch[1:]])
        
        return actions if len(actions) > 0 else torch.empty(0)

    
    def sample_configuration(self):
        """
        Sample a binary parent configuration e_ji ~ Bernoulli(sigmoid(eta_ji)).
        """
        configuration = torch.bernoulli(torch.sigmoid(self.s_params.edge_params))

        # Same mask as in train_s: the previous instance of the predicted action is not used as its own parent
        for i in range(min(configuration.shape)):
            configuration[i, i + self.state_num] = 0
        return configuration.to(self.device)

    def train_f(self):
        """
        Train the FunctionalNet (f_model) using sampled configurations.
        """
        for param in self.f_model.parameters():
            param.requires_grad = True  # Enable gradient computation for f_model
        
        batch = self.sample_batch(self.buffer)
        input_tensor = self.create_input(batch)
        output_tensor = self.create_output(batch)

        # Move tensors to device
        input_tensor = input_tensor.to(torch.float).to(self.device)
        output_tensor = output_tensor.to(torch.float).to(self.device)
        
        total_loss = 0
        for i in range(len(self.f_model.fs)):
            # Sample a random configuration
            configuration = self.sample_configuration().detach()
            # print(configuration.shape)
            mask = configuration[i]  # Only the sampled parents of action i are revealed to f_i
            extended_mask = mask.repeat(self.batch_size-1, 1)  # Extend mask to batch size
            self.f_optimizer.zero_grad()  # Zero gradients for optimizer

            # Make predictions
            predict = self.f_model(i, extended_mask, input_tensor)
            output = output_tensor[:, i].unsqueeze(1)  # Target output for the current variable
            loss = self.criterion(predict, output)  # Compute loss
            loss.backward()  # Backpropagation
            self.f_optimizer.step()  # Update model weights
            
            total_loss += loss.item()

        # Log training loss for f_model
        if self.wandb is not None:
            self.wandb.log({'total_train_f_loss': total_loss})

    def train_s(self):
        """
        Train the StructureParams (s_params) using sampled configurations.
        """
        for param in self.f_model.parameters():
            param.requires_grad = False  # Disable gradient computation for f_model
        
        batch = self.sample_batch(self.buffer)
        input_tensor = self.create_input(batch)
        output_tensor = self.create_output(batch)
        
        # Move tensors to device
        input_tensor = input_tensor.to(torch.float).to(self.device)
        output_tensor = output_tensor.to(torch.float).to(self.device)
        
        total_loss = 0
        for i in range(len(self.f_model.fs)):
            # Mask for structural parameters (to exclude self-loops)
            mask = torch.ones_like(self.s_params.edge_params[i])
            mask[i+self.state_num] = 0  
            mask_param = self.s_params.edge_params[i] * mask
            extended_mask = torch.sigmoid(mask_param.repeat(self.batch_size-1, 1))  # Sigmoid activation
            
            self.s_optimizer.zero_grad()  # Zero gradients for optimizer

            # Make predictions
            predict = self.f_model(i, extended_mask, input_tensor)
            output = output_tensor[:, i].unsqueeze(1)  # Target output for the current variable
            loss = self.criterion(predict, output)  # Compute loss
            total_loss += loss
        
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_s_params = self.s_params.edge_params.detach().clone()

        # Add sparsity loss to encourage simpler structures
        siggamma = self.s_params.edge_params.sigmoid()
        # Sparsity regulariser L_reg (Eq. 4)
        Lsparse = siggamma.sum().mul(self.sparsity_lambda)
        total_loss = total_loss + Lsparse  # Combine losses
        total_loss.backward()  # Backpropagation
        
        # Log training loss for s_params
        if self.wandb is not None:
            self.wandb.log({'total_train_s_loss': total_loss.item()})
        self.s_optimizer.step()  # Update model weights

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()

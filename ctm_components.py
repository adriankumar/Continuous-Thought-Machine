import torch
import torch.nn as nn 
import math


#removes specified dimension from tensor, pytorch does not have an in-built nn.Squeeze(), hence this custom class is used for nn.<Module> chains rather than torch.tensors
class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)
    
#nlm class
class NeuronLevelModel(nn.Module):
    def __init__(self, memory_length, num_neurons, hidden_dim=None, use_layernorm=False, dropout=0.0, temperature=1.0):
        super().__init__()
        
        self.memory_length = memory_length #length of pre-activation history used for post activation; also the input dim of this component
        self.num_neurons = num_neurons #number of independent neurons the model conceptually uses
        self.is_deep = hidden_dim is not None #whether to use 2-layer or 1-layer mlp per neuron
        
        #dropout and normalisation setup
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layernorm = nn.LayerNorm(memory_length, elementwise_affine=True) if use_layernorm else nn.Identity()
        
        #learnable temperature scaling parameter
        self.register_parameter('temperature', nn.Parameter(torch.tensor(temperature)))
        
        #choose network to build, both initialised using Xavier/Glorot initialisation (+- 1/sqrt(fan_in + fan_out))
        if self.is_deep:
            self._build_deep_nlm(hidden_dim) #2 layer mlp
        else:
            self._build_shallow_nlm() #in -> out
    
    #constructs 2-layer mlp per neuron with glu activations
    def _build_deep_nlm(self, hidden_dim):
        #first layer: memory_length -> 2*hidden_dim (for glu)
        self.register_parameter('w1', nn.Parameter(
            torch.empty(self.memory_length, 2 * hidden_dim, self.num_neurons).uniform_(
                -1/math.sqrt(self.memory_length + 2 * hidden_dim), #Xavier/Glorot initialisation
                1/math.sqrt(self.memory_length + 2 * hidden_dim)
            )
        ))

        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2 * hidden_dim)))
        
        #second layer: hidden_dim -> 2 (for glu then squeeze to 1)
        self.register_parameter('w2', nn.Parameter(
            torch.empty(hidden_dim, 2, self.num_neurons).uniform_(
                -1/math.sqrt(hidden_dim + 2),
                1/math.sqrt(hidden_dim + 2)
            )
        ))

        self.register_parameter('b2', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))
    
    #constructs 1-layer mlp per neuron with glu activation
    def _build_shallow_nlm(self):
        #single layer: memory_length -> 2 (for glu then squeeze to 1)
        self.register_parameter('w1', nn.Parameter(
            torch.empty(self.memory_length, 2, self.num_neurons).uniform_(
                -1/math.sqrt(self.memory_length + 2),
                1/math.sqrt(self.memory_length + 2)
            )
        ))

        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))
    
    #applies parallel mlps to neuron histories using einsum operations
    def forward(self, neuron_histories):
        #input shape: (batch, num_neurons, memory_length)
        x = self.dropout(neuron_histories)
        x = self.layernorm(x)  #normalise across memory dimension per neuron
        
        if self.is_deep:
            return self._forward_deep(x)
        else:
            return self._forward_shallow(x)
    
    #processes input through 2-layer mlp per neuron
    def _forward_deep(self, x):
        #first layer with glu activation
        out = torch.einsum('bnm,mhn->bnh', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #splits last dim and applies gating
        
        #second layer with glu activation and squeeze
        out = torch.einsum('bnh,hrn->bnr', out, self.w2) + self.b2
        out = nn.functional.glu(out, dim=-1)  #results in single output per neuron
        
        return out.squeeze(-1) / self.temperature
    
    #1-layer mlp per neuron  
    def _forward_shallow(self, x):
        #single layer with glu activation and squeeze
        out = torch.einsum('bnm,mrn->bnr', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #single output per neuron
        
        return out.squeeze(-1) / self.temperature


#manages temporal memory buffers and synchronisation state for standard ctm
class MemoryManager(nn.Module):
    def __init__(self, num_neurons, memory_length, sync_size_out, sync_size_action):
        super().__init__()
        
        self.num_neurons = num_neurons #number of neurons in model
        self.memory_length = memory_length #length of pre-activation history
        self.sync_size_out = sync_size_out #size of output synchronisation representation
        self.sync_size_action = sync_size_action #size of action synchronisation representation
        
        #learnable decay parameters for synchronisation temporal weighting
        self.register_parameter('decay_params_out', nn.Parameter(torch.zeros(sync_size_out)))
        self.register_parameter('decay_params_action', nn.Parameter(torch.zeros(sync_size_action)))
        
        #initial pre-activation history buffer for all neurons
        self.register_parameter('initial_pre_history', nn.Parameter(
            torch.zeros(num_neurons, memory_length).uniform_(
                -math.sqrt(1/(num_neurons + memory_length)), 
                math.sqrt(1/(num_neurons + memory_length))
            )
        ))
    
    #creates initial memory states and synchronisation accumulators for batch processing
    def initialise_memory_state(self, batch_size, device):
        #pre-activation history buffer for nlm input
        pre_history = self.initial_pre_history.unsqueeze(0).expand(batch_size, -1, -1).to(device).clone()
        
        #synchronisation state accumulators initialised to none for first update
        sync_state = {
            'out': {'decay_alpha': None, 'decay_beta': None},
            'action': {'decay_alpha': None, 'decay_beta': None}
        }
        
        return pre_history, sync_state
    
    #updates memory buffer using fifo strategy
    def update_memory_buffer(self, buffer, new_activations):
        #drop oldest timestep and append newest
        return torch.cat((buffer[:, :, 1:], new_activations.unsqueeze(-1)), dim=-1)
    
    #formats pre-activation history for neuron level model processing
    def get_histories_for_nlm(self, pre_activation_buffer):
        return pre_activation_buffer  #shape: (batch, num_neurons, memory_length)
    
    #computes exponential decay weights from learnable parameters
    def get_decay_weights(self, batch_size, sync_type='out'):
        decay_params = self.decay_params_out if sync_type == 'out' else self.decay_params_action
        clamped_params = torch.clamp(decay_params, 0, 15)  #clamp for numerical stability
        decay_weights = torch.exp(-clamped_params)
        return decay_weights.unsqueeze(0).expand(batch_size, -1)
    
    #updates synchronisation accumulators using temporal recurrence
    def update_synchronisation_state(self, sync_state, pairwise_products, decay_weights, sync_type='out'):
        state = sync_state[sync_type]
        
        if state['decay_alpha'] is None:
            #first update initialises accumulators
            state['decay_alpha'] = pairwise_products.clone()
            state['decay_beta'] = torch.ones_like(pairwise_products)
        else:
            #recurrent update with exponential temporal decay
            state['decay_alpha'] = decay_weights * state['decay_alpha'] + pairwise_products
            state['decay_beta'] = decay_weights * state['decay_beta'] + 1
        
        #compute normalised synchronisation representation
        synchronisation = state['decay_alpha'] / torch.sqrt(state['decay_beta'])
        return synchronisation


#vector-based unet for inter-neuron synaptic communication using skip connections and multi-scale processing
class SynapseUNet(nn.Module):
    def __init__(self, output_neurons, network_depth, minimum_width=16, dropout=0.0):
        super().__init__()
        
        self.output_neurons = output_neurons #number of neurons to output (matches num_neurons)
        self.network_depth = network_depth #depth of unet architecture
        self.minimum_width = minimum_width #smallest bottleneck width
        
        #compute width schedule from output down to minimum and back
        self.layer_widths = self._compute_layer_widths()
        
        #build unet architecture components
        self.initial_projection = self._build_initial_projection()
        self.down_blocks, self.up_blocks, self.skip_normalisers = self._build_unet_blocks(dropout)
    
    #calculates width progression for unet layers using linear interpolation
    def _compute_layer_widths(self):
        import numpy as np
        widths = np.linspace(self.output_neurons, self.minimum_width, self.network_depth)
        return [int(w) for w in widths]
    
    #creates initial projection layer that maps input to first unet width
    def _build_initial_projection(self):
        return nn.Sequential(
            nn.LazyLinear(self.layer_widths[0]),
            nn.LayerNorm(self.layer_widths[0]),
            nn.SiLU()
        )
    
    #constructs downward encoding, upward decoding and skip connection layers
    def _build_unet_blocks(self, dropout):
        down_blocks = nn.ModuleList()
        up_blocks = nn.ModuleList()
        skip_normalisers = nn.ModuleList()
        
        num_blocks = len(self.layer_widths) - 1 #number of down/up block pairs
        
        for block_idx in range(num_blocks):
            #downward compression block
            down_block = self._create_projection_block(
                self.layer_widths[block_idx], 
                self.layer_widths[block_idx + 1], 
                dropout
            )
            down_blocks.append(down_block)
            
            #upward expansion block
            up_block = self._create_projection_block(
                self.layer_widths[block_idx + 1], 
                self.layer_widths[block_idx], 
                dropout
            )
            up_blocks.append(up_block)
            
            #skip connection normaliser for corresponding level
            skip_normalisers.append(nn.LayerNorm(self.layer_widths[block_idx]))
        
        return down_blocks, up_blocks, skip_normalisers
    
    #creates standardised projection block with dropout, linear transform, normalisation and activation
    def _create_projection_block(self, input_width, output_width, dropout):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_width, output_width),
            nn.LayerNorm(output_width),
            nn.SiLU()
        )
    
    #processes input through unet with downward encoding, bottleneck and upward decoding with skip connections
    def forward(self, concatenated_input):
        #initial projection to unet width
        current_activation = self.initial_projection(concatenated_input)
        
        #downward encoding path with skip connection storage
        skip_activations = self._encode_downward(current_activation)
        
        #upward decoding path with skip connections
        final_output = self._decode_upward(skip_activations)
        
        return final_output
    
    #encodes input through downward path storing intermediate activations for skip connections
    def _encode_downward(self, initial_activation):
        skip_activations = [initial_activation] #store initial for skip connections
        
        current_activation = initial_activation
        for down_block in self.down_blocks:
            current_activation = down_block(current_activation)
            skip_activations.append(current_activation)
        
        return skip_activations
    
    #decodes through upward path applying skip connections and normalisation
    def _decode_upward(self, skip_activations):
        current_activation = skip_activations[-1] #start from bottleneck
        num_up_blocks = len(self.up_blocks)
        
        for block_idx in range(num_up_blocks):
            #process in reverse order for upward path
            up_block_idx = num_up_blocks - 1 - block_idx
            skip_connection_idx = up_block_idx
            
            #apply upward projection
            current_activation = self.up_blocks[up_block_idx](current_activation)
            
            #add skip connection and normalise
            skip_connection = skip_activations[skip_connection_idx]
            current_activation = self.skip_normalisers[skip_connection_idx](
                current_activation + skip_connection
            )
        
        return current_activation


#manages neuron pairing strategies and computes synchronisation pairwise products
class SynchPairing(nn.Module):
    def __init__(self, pairing_strategy, num_neurons, sync_size_out, sync_size_action, self_pairing_count=0):
        super().__init__()
        
        self.pairing_strategy = pairing_strategy #'random-pairing', 'first-last', or 'random'
        self.num_neurons = num_neurons #total number of neurons available for pairing
        self.sync_sizes = {'out': sync_size_out, 'action': sync_size_action} #synchronisation sizes per type
        self.self_pairing_count = self_pairing_count #number of self-connections for random-pairing
        
        #validate configuration and compute representation sizes
        self._validate_configuration()
        self.representation_sizes = self._calculate_representation_sizes()
        
        #initialise neuron pairing indices as buffers for device handling
        self._initialise_pairing_indices()
    
    #validates pairing configuration for mathematical consistency
    def _validate_configuration(self):
        if self.pairing_strategy == 'first-last':
            required_neurons = self.sync_sizes['out'] + self.sync_sizes['action']
            assert self.num_neurons >= required_neurons, f"need {required_neurons} neurons for first-last pairing"
        
        if self.pairing_strategy == 'random-pairing':
            for sync_type, sync_size in self.sync_sizes.items():
                assert self.self_pairing_count <= sync_size, f"self pairing count exceeds {sync_type} sync size"
    
    #computes synchronisation representation dimensions for each sync type
    def _calculate_representation_sizes(self):
        sizes = {}
        for sync_type, sync_size in self.sync_sizes.items():
            if self.pairing_strategy == 'random-pairing':
                sizes[sync_type] = sync_size #direct dimensional control
            else:
                sizes[sync_type] = (sync_size * (sync_size + 1)) // 2 #upper triangular matrix
        return sizes
    
    #creates and registers neuron pairing indices for both sync types
    def _initialise_pairing_indices(self):
        for sync_type, sync_size in self.sync_sizes.items():
            left_indices, right_indices = self._generate_pairing_indices(sync_type, sync_size)
            self.register_buffer(f'{sync_type}_left_indices', left_indices)
            self.register_buffer(f'{sync_type}_right_indices', right_indices)
    
    #generates neuron pairing indices based on strategy and sync type
    def _generate_pairing_indices(self, sync_type, sync_size):
        if self.pairing_strategy == 'first-last':
            return self._create_first_last_indices(sync_type, sync_size)
        elif self.pairing_strategy == 'random':
            return self._create_random_dense_indices(sync_size)
        elif self.pairing_strategy == 'random-pairing':
            return self._create_random_pairing_indices(sync_size)
        else:
            raise ValueError(f"unsupported pairing strategy: {self.pairing_strategy}")
    
    #creates indices for first-last strategy using fixed neuron positions
    def _create_first_last_indices(self, sync_type, sync_size):
        if sync_type == 'out':
            indices = torch.arange(0, sync_size) #first n neurons
        else: #action
            indices = torch.arange(self.num_neurons - sync_size, self.num_neurons) #last n neurons
        return indices, indices #same indices for both left and right
    
    #creates random neuron subset indices for dense synchronisation computation
    def _create_random_dense_indices(self, sync_size):
        import numpy as np
        left_indices = torch.from_numpy(np.random.choice(self.num_neurons, size=sync_size, replace=False))
        right_indices = torch.from_numpy(np.random.choice(self.num_neurons, size=sync_size, replace=False))
        return left_indices, right_indices
    
    #creates direct neuron-to-neuron mappings for random-pairing strategy
    def _create_random_pairing_indices(self, sync_size):
        import numpy as np
        
        #select left neurons randomly
        left_indices = torch.from_numpy(np.random.choice(self.num_neurons, size=sync_size, replace=False))
        
        #create right indices with self-connections priority
        right_indices = torch.zeros_like(left_indices)
        right_indices[:self.self_pairing_count] = left_indices[:self.self_pairing_count] #self-connections first
        
        #fill remaining with random selections
        if self.self_pairing_count < sync_size:
            remaining_right = torch.from_numpy(
                np.random.choice(self.num_neurons, size=sync_size - self.self_pairing_count, replace=False)
            )
            right_indices[self.self_pairing_count:] = remaining_right
        
        return left_indices, right_indices
    
    #computes pairwise products between selected neurons for synchronisation
    def compute_pairwise_products(self, activations, sync_type):
        #retrieve pairing indices for sync type
        left_indices = getattr(self, f'{sync_type}_left_indices')
        right_indices = getattr(self, f'{sync_type}_right_indices')
        
        #ensure device compatibility
        left_indices = left_indices.to(activations.device)
        right_indices = right_indices.to(activations.device)
        
        if self.pairing_strategy == 'random-pairing':
            return self._compute_sparse_pairwise_products(activations, left_indices, right_indices)
        else:
            return self._compute_dense_pairwise_products(activations, left_indices, right_indices)
    
    #computes element-wise products for random-pairing strategy
    def _compute_sparse_pairwise_products(self, activations, left_indices, right_indices):
        selected_left = activations[:, left_indices] #shape: (batch, sync_size)
        selected_right = activations[:, right_indices] #shape: (batch, sync_size)
        return selected_left * selected_right #element-wise multiplication
    
    #computes outer product and upper triangle for dense strategies
    def _compute_dense_pairwise_products(self, activations, left_indices, right_indices):
        selected_left = activations[:, left_indices] #shape: (batch, sync_size)
        selected_right = activations[:, right_indices] #shape: (batch, sync_size)
        
        #compute outer product between selected neurons
        outer_product = selected_left.unsqueeze(2) * selected_right.unsqueeze(1) #shape: (batch, sync_size, sync_size)
        
        #extract upper triangular elements for symmetric relationships
        sync_size = len(left_indices)
        upper_triangle_i, upper_triangle_j = torch.triu_indices(sync_size, sync_size)
        pairwise_products = outer_product[:, upper_triangle_i, upper_triangle_j] #shape: (batch, triangular_size)
        
        return pairwise_products
    
    #returns synchronisation representation size for tensor allocation
    def get_representation_size(self, sync_type):
        return self.representation_sizes[sync_type]
    
    #resets pairing indices for experimentation with new strategy
    def reset_pairings(self, new_strategy=None):
        if new_strategy is not None:
            self.pairing_strategy = new_strategy
            self._validate_configuration()
            self.representation_sizes = self._calculate_representation_sizes()
        self._initialise_pairing_indices()

#cross-attention module for ctm input processing using synchronisation vectors as queries
class AttentionModule(nn.Module):
    def __init__(self, d_input, num_heads, dropout=0.0):
        super().__init__()
        
        self.d_input = d_input #attention embedding dimension
        self.num_heads = num_heads #number of attention heads
        
        #query projection for action synchronisation vectors
        self.query_projection = nn.LazyLinear(d_input)
        
        #key-value projection for input features with normalisation
        self.kv_projection = nn.Sequential(
            nn.LazyLinear(d_input),
            nn.LayerNorm(d_input)
        )
        
        #core multi-head cross-attention mechanism
        self.attention = nn.MultiheadAttention(
            d_input, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
    
    #processes input features and sync vectors through cross-attention
    def forward(self, input_features, action_sync_vectors):
        #input_features shape: (batch, seq_len, feature_dim)
        #action_sync_vectors shape: (batch, sync_size)
        
        #project sync vectors to queries and add sequence dimension
        queries = self.query_projection(action_sync_vectors).unsqueeze(1) #shape: (batch, 1, d_input)
        
        #project input features to keys and values
        keys_values = self.kv_projection(input_features) #shape: (batch, seq_len, d_input)
        
        #compute cross-attention with sync vectors attending to input features
        attended_output, attention_weights = self.attention(
            queries, keys_values, keys_values,
            need_weights=True  #for potential visualisation
        )
        
        #remove sequence dimension and return attended features
        return attended_output.squeeze(1) #shape: (batch, d_input)
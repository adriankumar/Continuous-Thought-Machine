import torch 
import torch.nn as nn 
import numpy as np


#--------------------------------------------------------------------------------------------------------------
#                                                Continuous Thought Machine
#--------------------------------------------------------------------------------------------------------------
class ContinuousThoughtMachine(nn.Module):
    def __init__(self, num_neurons, memory_length, sync_size_out, sync_size_action, 
                 d_input, num_heads, unet_depth, iterations, output_classes, 
                 self_pairing_count=0, use_deep_nlm=False, use_layernorm=False, 
                 dropout=0.0, temperature=1.0, min_unet_width=16):
        super().__init__()
        
        self.num_neurons = num_neurons #total number of neurons in the model
        self.iterations = iterations #number of internal thinking steps
        self.output_classes = output_classes #final prediction dimension
        
        #build ctm architecture components
        self._build_ctm_components(
            num_neurons, memory_length, sync_size_out, sync_size_action,
            d_input, num_heads, unet_depth, self_pairing_count, use_deep_nlm,
            use_layernorm, dropout, temperature, min_unet_width
        )
        
        #initial post-activation state for neurons - shape: num_neurons
        self.register_parameter('initial_post_activations', 
                              nn.Parameter(torch.zeros(num_neurons).uniform_(
                                  -np.sqrt(1/num_neurons), np.sqrt(1/num_neurons)
                              )))

#----------------------------
# Architecture stuff
#----------------------------
    def _build_ctm_components(self, num_neurons, memory_length, sync_size_out, sync_size_action,
                             d_input, num_heads, unet_depth, self_pairing_count, use_deep_nlm,
                             use_layernorm, dropout, temperature, min_unet_width):
        
        #pre-activation history management
        self.pre_activations_manager = PreActivationsManager(
            num_neurons=num_neurons,
            pre_activation_history_len=memory_length
        )
        
        #synchronisation computation and neuron pairing
        self.synchronisation_manager = SynchronisationManager(
            num_neurons=num_neurons,
            sync_size_out=sync_size_out,
            sync_size_action=sync_size_action,
            self_pairing_count=self_pairing_count
        )
        
        #attention mechanism for input interaction
        self.attention_module = AttentionModule(
            d_input=d_input,
            num_heads=num_heads,
            dropout=dropout
        )
        
        #u-net synapse model for inter-neuron communication
        self.synapse_model = SynapseModelUNet(
            neurons=num_neurons,
            depth=unet_depth,
            min_width=min_unet_width,
            dropout=dropout
        )
        
        #neuron-level models for temporal processing
        self.neuron_level_model = NeuronLevelModel(
            memory_length=memory_length,
            num_neurons=num_neurons,
            is_deep=use_deep_nlm,
            use_layernorm=use_layernorm,
            dropout=dropout,
            temperature=temperature
        )
        
        #final output projection from output synchronisation
        self.output_projection = nn.Linear(sync_size_out, self.output_classes)

#----------------------------
# Forward Processing
#----------------------------
    def forward(self, input_features, track_internals=False):
        #input_features shape: batch x sequence_length x feature_dim
        batch_size = input_features.size(0)
        device = input_features.device
        
        #storage for tracking internal states if requested
        if track_internals:
            internal_states = {
                'post_activations': [],
                'action_sync': [],
                'output_sync': [],
                'attention_weights': []
            }
        
        #initialise ctm states
        pre_history = self.pre_activations_manager.initialise_history(batch_size, device)
        sync_state = self.synchronisation_manager.initialise_synchronisation_state(batch_size, device)
        
        #current post-activations start from learned initial state
        current_post_activations = self.initial_post_activations.unsqueeze(0).expand(batch_size, -1)
        
        #storage for predictions across iterations
        predictions = torch.empty(batch_size, self.output_classes, self.iterations, device=device)
        
        #main thinking loop - iterate over internal time steps
        for iteration in range(self.iterations):
            
            #compute action synchronisation from current neuron states
            action_pairwise_products = self.synchronisation_manager.compute_pairwise_products(
                current_post_activations, 'action'
            )
            action_sync_vector = self.synchronisation_manager.update_synchronisation_state(
                sync_state, action_pairwise_products, 'action', batch_size, device
            )
            
            #use action sync to attend to input features
            attended_features, attention_weights = self.attention_module(
                input_features, action_sync_vector
            )
            
            #concatenate attended features with current post-activations for synapse input
            synapse_input = torch.cat([attended_features, current_post_activations], dim=-1)
            
            #process through u-net synapse model to get new pre-activations
            new_pre_activations = self.synapse_model(synapse_input)
            
            #update pre-activation history with fifo strategy
            pre_history = self.pre_activations_manager.update_history(pre_history, new_pre_activations)
            
            #process neuron histories through nlms to get new post-activations
            current_post_activations = self.neuron_level_model(pre_history)
            
            #compute output synchronisation from updated post-activations
            output_pairwise_products = self.synchronisation_manager.compute_pairwise_products(
                current_post_activations, 'out'
            )
            output_sync_vector = self.synchronisation_manager.update_synchronisation_state(
                sync_state, output_pairwise_products, 'out', batch_size, device
            )
            
            #generate prediction from output synchronisation
            current_prediction = self.output_projection(output_sync_vector)
            predictions[:, :, iteration] = current_prediction
            
            #track internal states if requested
            if track_internals:
                internal_states['post_activations'].append(current_post_activations.detach().cpu())
                internal_states['action_sync'].append(action_sync_vector.detach().cpu())
                internal_states['output_sync'].append(output_sync_vector.detach().cpu())
                internal_states['attention_weights'].append(attention_weights.detach().cpu())
        
        #return predictions and optionally internal states
        if track_internals:
            return predictions, internal_states
        else:
            return predictions

#--------------------------------------------------------------------------------------------------------------
#                                                Attention Module
#--------------------------------------------------------------------------------------------------------------
class AttentionModule(nn.Module):
    def __init__(self, d_input, num_heads, dropout=0.0):
        super().__init__()
        
        self.d_input = d_input #attention embedding dimension
        self.num_heads = num_heads #number of attention heads
        self.dropout = dropout
        
        #build attention components
        self._build_attention_layers()

#----------------------------
# Architecture stuff
#----------------------------
    def _build_attention_layers(self):
        #query projection maps action synchronisation vectors to attention dimension
        self.query_projection = nn.LazyLinear(self.d_input) #sync_size -> d_input
        
        #key-value projection processes input features with normalisation
        self.kv_projection = nn.Sequential(
            nn.LazyLinear(self.d_input), #feature_dim -> d_input
            nn.LayerNorm(self.d_input) #normalise before attention
        )
        
        #core multi-head cross-attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_input,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True #batch dimension comes first
        )

#----------------------------
# Forward Processing
#----------------------------
    def forward(self, input_features, action_sync_vector):
        #input_features shape: batch x sequence_length x feature_dim (e.g., from backbone)
        #action_sync_vector shape: batch x sync_size_action (from synchronisation manager)
        
        #project action sync vector to query dimension and add sequence dimension
        queries = self.query_projection(action_sync_vector).unsqueeze(1) #shape: batch x 1 x d_input
        
        #project input features to keys and values
        keys_values = self.kv_projection(input_features) #shape: batch x sequence_length x d_input
        
        #compute cross-attention where sync vector attends to input features
        attended_output, attention_weights = self.attention(
            query=queries, #what we're asking about (current thoughts)
            key=keys_values, #what we're searching through (input features)
            value=keys_values, #what we extract (same as keys)
            need_weights=True #return attention patterns for analysis
        )
        
        #remove sequence dimension from attended output for further processing
        attended_features = attended_output.squeeze(1) #shape: batch x d_input
        
        return attended_features, attention_weights #attention_weights shape: batch x 1 x sequence_length

#--------------------------------------------------------------------------------------------------------------
#                                                Synchronisation Manager
#--------------------------------------------------------------------------------------------------------------
class SynchronisationManager(nn.Module):
    def __init__(self, num_neurons, sync_size_out, sync_size_action, self_pairing_count=0):
        super().__init__()
        
        self.num_neurons = num_neurons #total number of neurons available for pairing
        self.sync_size_out = sync_size_out #size of output synchronisation representation
        self.sync_size_action = sync_size_action #size of action synchronisation representation  
        self.self_pairing_count = self_pairing_count #number of self-connections in random pairing
        
        #learnable decay parameters for temporal weighting - shape: sync_size for each type
        self.register_parameter('decay_params_out', nn.Parameter(torch.zeros(self.sync_size_out)))
        self.register_parameter('decay_params_action', nn.Parameter(torch.zeros(self.sync_size_action)))
        
        #generate and register neuron pairing indices for both sync types
        self._build_neuron_pairings()

#----------------------------
# Architecture stuff  
#----------------------------
    def _build_neuron_pairings(self):
        #create pairing indices for output synchronisation
        out_left_indices, out_right_indices = self._generate_random_pairing_indices(self.sync_size_out)
        self.register_buffer('out_left_indices', out_left_indices) #shape: sync_size_out
        self.register_buffer('out_right_indices', out_right_indices) #shape: sync_size_out
        
        #create pairing indices for action synchronisation  
        action_left_indices, action_right_indices = self._generate_random_pairing_indices(self.sync_size_action)
        self.register_buffer('action_left_indices', action_left_indices) #shape: sync_size_action
        self.register_buffer('action_right_indices', action_right_indices) #shape: sync_size_action
    
    def _generate_random_pairing_indices(self, sync_size):
        #select left neurons randomly without replacement
        left_indices = torch.from_numpy(np.random.choice(self.num_neurons, size=sync_size, replace=False))
        
        #create right indices with self-connections priority
        right_indices = torch.zeros_like(left_indices)
        right_indices[:self.self_pairing_count] = left_indices[:self.self_pairing_count] #self-connections first
        
        #fill remaining positions with random selections
        if self.self_pairing_count < sync_size:
            remaining_right = torch.from_numpy(
                np.random.choice(self.num_neurons, size=sync_size - self.self_pairing_count, replace=False)
            )
            right_indices[self.self_pairing_count:] = remaining_right
        
        return left_indices, right_indices

#----------------------------
# Forward Processing
#----------------------------
    def initialise_synchronisation_state(self, batch_size, device):
        #synchronisation state accumulators initialised to none for first update
        sync_state = {
            'out': {'decay_alpha': None, 'decay_beta': None},
            'action': {'decay_alpha': None, 'decay_beta': None}
        }
        return sync_state
    
    def compute_pairwise_products(self, post_activations, sync_type):
        #post_activations shape: batch x num_neurons -> select paired neurons and multiply
        if sync_type == 'out':
            left_indices = self.out_left_indices.to(post_activations.device)
            right_indices = self.out_right_indices.to(post_activations.device)
        else: #action
            left_indices = self.action_left_indices.to(post_activations.device) 
            right_indices = self.action_right_indices.to(post_activations.device)
        
        #select neurons based on pairing indices
        selected_left = post_activations[:, left_indices] #shape: batch x sync_size
        selected_right = post_activations[:, right_indices] #shape: batch x sync_size
        
        #compute element-wise multiplication for synchronisation
        pairwise_products = selected_left * selected_right #shape: batch x sync_size
        
        return pairwise_products
    
    def get_decay_weights(self, batch_size, sync_type, device):
        #get appropriate decay parameters and clamp for numerical stability
        if sync_type == 'out':
            decay_params = self.decay_params_out
        else: #action
            decay_params = self.decay_params_action
            
        clamped_params = torch.clamp(decay_params, 0, 15) #prevent extreme values
        decay_weights = torch.exp(-clamped_params) #convert to exponential decay
        
        #expand for batch processing: sync_size -> batch x sync_size
        return decay_weights.unsqueeze(0).expand(batch_size, -1).to(device)
    
    def update_synchronisation_state(self, sync_state, pairwise_products, sync_type, batch_size, device):
        #get decay weights for temporal weighting
        decay_weights = self.get_decay_weights(batch_size, sync_type, device)
        
        state = sync_state[sync_type]
        
        if state['decay_alpha'] is None:
            #first iteration initialises accumulators
            state['decay_alpha'] = pairwise_products.clone()
            state['decay_beta'] = torch.ones_like(pairwise_products)
        else:
            #recurrent update with exponential temporal decay
            state['decay_alpha'] = decay_weights * state['decay_alpha'] + pairwise_products
            state['decay_beta'] = decay_weights * state['decay_beta'] + 1
        
        #compute normalised synchronisation representation
        synchronisation_vector = state['decay_alpha'] / torch.sqrt(state['decay_beta'])
        
        return synchronisation_vector #shape: batch x sync_size

#--------------------------------------------------------------------------------------------------------------
#                                                Neuron Level Model
#--------------------------------------------------------------------------------------------------------------
class NeuronLevelModel(nn.Module):
    def __init__(self, memory_length, num_neurons, is_deep=False, use_layernorm=False, dropout=0.0, temperature=1.0):
        super().__init__()

        self.memory_length = memory_length #is the same as pre activation history length
        self.num_neurons = num_neurons
        self.is_deep = is_deep

        #dropout and layernorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        #elementwise_affine introduces learnable weight and bias to the normalised output and performs a standard perceptron based calculation (weight * normalised + bias)
        self.layernorm = nn.LayerNorm(self.memory_length, elementwise_affine=True) if use_layernorm else nn.Identity()

        #learnable temperature scaling parameter
        self.register_parameter('temperature', nn.Parameter(torch.tensor(temperature)))

        if self.is_deep:
            self._build_deep_nlm() #2 layer nlm
        else:
            self._build_nlm() 

#----------------------------
# Architecture stuff
#----------------------------
    def _build_deep_nlm(self):
        # first layer: memory_length -> 2*hidden_dim (for glu)
        self.register_parameter('w1', nn.Parameter(torch.empty(self.memory_length, 2 * 2, self.num_neurons).uniform_(-1/np.sqrt(self.memory_length + 2 * 2), 1/np.sqrt(self.memory_length + 2 * 2))))
        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2 * 2)))

        #second layer: hidden_dim -> 2 (for glu then squeeze to 1)
        self.register_parameter('w2', nn.Parameter(torch.empty(2, 2, self.num_neurons).uniform_(-1/np.sqrt(2 + 2), 1/np.sqrt(2 + 2))))
        self.register_parameter('b2', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))
    
    def _build_nlm(self):
        #w1 has shape: memory_length x 2 x num_neurons, b1 has shape: 1 x num_neurons x 2
        self.register_parameter('w1', nn.Parameter(torch.empty(self.memory_length, 2, self.num_neurons).uniform_(-1/np.sqrt(self.memory_length + 2), 1/np.sqrt(self.memory_length + 2))))
        self.register_parameter('b1', nn.Parameter(torch.zeros(1, self.num_neurons, 2)))

#----------------------------
# Forward Processing
#----------------------------
    def forward(self, pre_activation_history):
        x = self.dropout(pre_activation_history) #input shape: batch, num_neurons, memory_length
        x = self.layernorm(x) #normalise

        #output shape should be batch x num neurons -> each neuron has one post activation 
        if self.is_deep:
            return self._forward_deep(x)
        else:
            return self._forward_shallow(x)
    
    def _forward_deep(self, x):
        #first layer with glu activation
        out = torch.einsum('bnm,mhn->bnh', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #splits last dim and applies gating
        
        #second layer with glu activation and squeeze
        out = torch.einsum('bnh,hrn->bnr', out, self.w2) + self.b2
        out = nn.functional.glu(out, dim=-1)  #results in single output per neuron

        post_activations = out.squeeze(-1) / self.temperature
        
        return post_activations

    def _forward_shallow(self, x):
        #single layer with glu activation and squeeze
        out = torch.einsum('bnm,mrn->bnr', x, self.w1) + self.b1
        out = nn.functional.glu(out, dim=-1)  #single output per neuron
        
        post_activations = out.squeeze(-1) / self.temperature
        
        return post_activations

#--------------------------------------------------------------------------------------------------------------
#                                                Pre-activation history manager
#--------------------------------------------------------------------------------------------------------------
#to manage the pre-activation histories
class PreActivationsManager(nn.Module):
    def __init__(self, num_neurons, pre_activation_history_len):
        super().__init__()

        self.num_neurons = num_neurons
        self.memory_length = pre_activation_history_len #only pre activations have a stored history, official github implementation uses a decay calculation for post activation history

        #pre-activation history initialised using Xavier/Glorot initialisation (+- 1/sqrt(fan_in + fan_out))
        self.register_parameter('initial_pre_history', 
                                nn.Parameter(
                                    torch.zeros(self.num_neurons, self.memory_length).uniform_(
                                        -np.sqrt(1/(self.num_neurons + self.memory_length)), 
                                        np.sqrt(1/(self.num_neurons + self.memory_length))
                                        )
                                    )
                                ) #shape of num_neurons x memory_length

#reshape from num_neurons x memory_length to -> batch_size x num_neurons x memory_length
#clone creates independent copies so each batch has its own evolving history
    def initialise_history(self, batch_size, device):
        pre_history = self.initial_pre_history.unsqueeze(0).expand(batch_size, -1, -1).to(device).clone() 
        return pre_history


#update the pre history with the new computed one for each neuron, history has shape batch x num_neurons x memory_length
#new pre activations is expected to have shape batch x num_neurons; so unsqueeze(-1) adds time dim batch x num_neurons x memory_length=1 and concatenates these pre activation values along the feature dim 
    def update_history(self, history, new_pre_activations):
        return torch.cat((history[:, :, 1:], new_pre_activations.unsqueeze(-1)), dim=-1)


#--------------------------------------------------------------------------------------------------------------
#                                                U-Net Synapse Model
#--------------------------------------------------------------------------------------------------------------
class SynapseModelUNet(nn.Module):
    def __init__(self, neurons, depth, min_width=16, dropout=0.0, bias=False):
        super().__init__()
        self.neurons = neurons #same number as neurons to be used in CTM
        self.depth = depth
        self.min_width = min_width #smallest bottleneck
        self.dr = dropout
        self.bias = bias

        self.layer_widths = self._interpolate_width(self.neurons, self.depth, self.min_width) #list of neurons in each layer from top->bottom
        self.input_projection = self._input_projection_layer()

        #down, up, and skip connections
        self.down_path, self.up_path, self.skip_norm = self._build_network(self.dr)

        #TODO: Add getters for certain attributes for live display of network behaviour

#----------------------------
# Forward Processing
#----------------------------
    def forward(self, input):
        input_mapping = self.input_projection(input) #map raw input to num of neurons

        skip_activations = self.traverse_down(input_mapping) #down the u-net

        pre_activations = self.traverse_up(skip_activations)

        return pre_activations

    def traverse_up(self, skip_activations):
        current_activation = skip_activations[-1] #start from end/bottleneck layer
        layers = len(self.up_path)

        for layer_id in range(layers):
            reversed_layer_id = layers - 1 - layer_id #layer index backwards

            current_activation = self.up_path[reversed_layer_id](current_activation) #project in upward layer

            #add skip connection and normalise
            current_activation = self.skip_norm[reversed_layer_id](current_activation + skip_activations[reversed_layer_id])

        return current_activation #return the final outputs
    
    def traverse_down(self, input_mapping):
        #initial pass
        current_activation = input_mapping
        skip_activations = [current_activation] #keep a list of all layer activations for skip connection

        for layer in self.down_path:
            current_activation = layer(current_activation) #downsized until it reaches bottleneck
            skip_activations.append(current_activation) #store layer-wise activations for skip connection
        
        return skip_activations


#----------------------------
# Architecture stuff
#----------------------------
    #returns a list of linearly interpolated number of neurons in each layer
    def _interpolate_width(self, num_neurons, depth, min_neurons):
        widths = np.linspace(num_neurons, min_neurons, depth) #start:num_neurons, end: min_neurons, number of elements:depth
        return [int(w) for w in widths]

    #maps input -> neurons to feed the network
    def _input_projection_layer(self):
        return nn.Sequential(
            nn.LazyLinear(self.layer_widths[0], bias=self.bias), #lazy infers the input dim, which will come from out concatennated input with post neuron activation
            nn.LayerNorm(self.layer_widths[0], bias=self.bias), #normalise
            nn.SiLU() #activation function - basically copying the offical ctm code
        )
    
    #reusable layer for each block
    def _create_projection(self, input_size, output_size, dr):
        return nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(input_size, output_size), #single linear projection
            nn.LayerNorm(output_size), #normalise b4 silu activation
            nn.SiLU()
        )    
    
    #builds the actual u-net structures
    def _build_network(self, dropout_rate):
        down_path = nn.ModuleList() #downward layers
        up_path = nn.ModuleList() #upward layers
        skip_norm = nn.ModuleList() #normaliser for each skip connection in the upward layer

        #loop through layers
        for layer in range(len(self.layer_widths) - 1):
            #down
            down_block = self._create_projection(self.layer_widths[layer], self.layer_widths[layer + 1], dropout_rate) #using current layer as input size, and next layer as output size
            down_path.append(down_block) #append to module list

            #up
            up_block = self._create_projection(self.layer_widths[layer + 1], self.layer_widths[layer], dropout_rate) #same as down block but swap input size to be layer + 1 and output to just layer
            up_path.append(up_block)

            #skip connection normalisers
            skip_norm.append(nn.LayerNorm(self.layer_widths[layer])) #normaliser size will be same as output size of the up block 

        return down_path, up_path, skip_norm
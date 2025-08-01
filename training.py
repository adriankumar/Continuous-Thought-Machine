import torch
import numpy as np
from ctm_architecture import ContinuousThoughtMachine
from dataset import (
    prepare_dataset, 
    show_dataset_examples, 
    generate_arithmetic_sample
)
from training_utilities import (
    train_ctm_model, 
    validate_training_data
)

#--------------------------------------------------------------------------------------------------------------
#                                                Configuration
#--------------------------------------------------------------------------------------------------------------

#ctm architecture hyperparameters
CTM_CONFIG = {
    'num_neurons': 64,              #total neurons in ctm latent space
    'memory_length': 10,            #pre-activation history length per neuron
    'sync_size_out': 32,           #output synchronisation representation size
    'sync_size_action': 16,        #action synchronisation representation size
    'attention_size': 32,          #attention embedding dimension
    'num_heads': 4,                #multi-head attention heads
    'unet_depth': 3,               #u-net synapse model depth
    'thinking_steps': 8,           #internal reasoning iterations
    'output_dim': 1,               #single output for arithmetic prediction
    'self_pairing_count': 4,       #self-connections in neuron pairing
    'use_deep_nlm': True,          #use 2-layer neuron level models
    'use_layernorm': False,        #layernorm in nlms (keep false for phase 1)
    'dropout': 0.1,                #dropout rate
    'temperature': 1.0,            #nlm temperature scaling
    'min_unet_width': 16,          #minimum u-net bottleneck width
    'prediction_reshaper': [-1]    #shape for certainty calculation
}

#training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': r"training_metrics\step_size_reasoning\test",
    'save_freq': 2
}

#dataset configuration
DATASET_CONFIG = {
    'train_samples': 2000,         #training set size
    'test_samples': 400            #test set size (for future evaluation)
}

#--------------------------------------------------------------------------------------------------------------
#                                                Model Setup
#--------------------------------------------------------------------------------------------------------------

#create ctm model with specified configuration
def create_ctm_model(config):
    model = ContinuousThoughtMachine(
        num_neurons=config['num_neurons'],
        memory_length=config['memory_length'],
        sync_size_out=config['sync_size_out'],
        sync_size_action=config['sync_size_action'],
        attention_size=config['attention_size'],
        num_heads=config['num_heads'],
        unet_depth=config['unet_depth'],
        thinking_steps=config['thinking_steps'],
        output_dim=config['output_dim'],
        self_pairing_count=config['self_pairing_count'],
        use_deep_nlm=config['use_deep_nlm'],
        use_layernorm=config['use_layernorm'],
        dropout=config['dropout'],
        temperature=config['temperature'],
        min_unet_width=config['min_unet_width'],
        prediction_reshaper=config['prediction_reshaper']
    )
    
    return model

#print model summary and parameter count, cant do this until after training has model using lazylinear layers which get initialised during forward pass
def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print(f"ctm model created successfully")
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print(f"thinking steps: {model.thinking_steps}")
    print(f"neurons: {model.num_neurons}")
    print(f"memory length: {model.pre_activations_manager.memory_length}")

#--------------------------------------------------------------------------------------------------------------
#                                                Training Pipeline
#--------------------------------------------------------------------------------------------------------------

#complete training pipeline with all setup
def run_training_pipeline():
    print("=" * 60)
    print("ctm arithmetic reasoning training - phase 1")
    print("=" * 60)
    
    training_config = TRAINING_CONFIG.copy()
    
    #prepare dataset
    print("\n1. preparing dataset...")
    train_data, train_labels, _, _ = prepare_dataset(DATASET_CONFIG, sequence_length=5)
    validate_training_data(train_data, train_labels)
    show_dataset_examples(train_data, train_labels)
    
    #create model
    print("\n2. creating ctm model...")
    model = create_ctm_model(CTM_CONFIG)
    # print_model_info(model)
    
    #start training
    print(f"\n3. starting training on {training_config['device']}...")
    print(f"configuration: {training_config}")
    
    try:
        #run training
        history = train_ctm_model(
            model=model,
            train_data=train_data,
            train_labels=train_labels,
            config=training_config
        )
        
        #training completed successfully
        print(f"\ntraining completed successfully!")
        print(f"final loss: {history['loss'][-1]:.4f}")
        print(f"final accuracy: {history['accuracy'][-1]:.3f}")
        print(f"final confidence: {history['confidence'][-1]:.3f}")
        
        return history, model
        
    except Exception as e:
        print(f"training failed with error: {e}")
        raise e


#--------------------------------------------------------------------------------------------------------------
#                                                Main Execution
#--------------------------------------------------------------------------------------------------------------

#main training execution
def main():
    #set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:

        history, model = run_training_pipeline()
        
        print("\ntraining pipeline completed successfully!")

        print("\n Printing model parameters for task")
        print_model_info(model)
        
        #optional: run quick evaluation on a few samples
        print("\nrunning quick evaluation...")
        
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                #generate test sample
                test_seq, true_answer = generate_arithmetic_sample(8)
                
                #prepare input
                test_input = torch.FloatTensor(test_seq).unsqueeze(0).unsqueeze(-1)
                
                #predict with ctm
                if torch.cuda.is_available():
                    test_input = test_input.cuda()
                    model = model.cuda()
                
                predictions, certainties = model(test_input)
                predicted_answer = predictions[0, 0, -1].item()
                confidence = certainties[0, 1, -1].item()
                
                print(f"  {test_seq} → predicted: {predicted_answer:.2f}, true: {true_answer}, confidence: {confidence:.3f}")
        
    except KeyboardInterrupt:
        print("\ntraining interrupted by user")
    except Exception as e:
        print(f"\ntraining failed: {e}")
        raise e

if __name__ == "__main__":
    main()
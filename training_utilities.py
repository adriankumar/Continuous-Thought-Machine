import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import datetime
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

#default training hyperparameters for phase 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_FREQ = 10
DEFAULT_TOLERANCE = 0.5  #accuracy tolerance for arithmetic predictions

#--------------------------------------------------------------------------------------------------------------
#                                                Data Handling
#--------------------------------------------------------------------------------------------------------------

#convert numpy arithmetic sequences to torch tensors with proper ctm input format
def numpy_to_torch_dataset(inputs, labels, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):
    #convert to tensors
    input_tensor = torch.FloatTensor(inputs)
    label_tensor = torch.FloatTensor(labels)
    
    #create dataset and dataloader
    dataset = TensorDataset(input_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

#prepare arithmetic sequence batch for ctm input format
def create_ctm_input_batch(sequences):
    #sequences shape: (batch_size, sequence_length)
    # batch_size, seq_len = sequences.shape
    
    #add feature dimension for ctm attention module
    #shape becomes: (batch_size, sequence_length, feature_dim=1)
    input_features = sequences.unsqueeze(-1)
    
    return input_features

#--------------------------------------------------------------------------------------------------------------
#                                                Loss Functions
#--------------------------------------------------------------------------------------------------------------

#simple mse loss on final predictions for phase 1
def compute_mse_loss(predictions, targets):
    #predictions shape: (batch_size, output_dim, thinking_steps)
    #extract final thinking step predictions
    final_predictions = predictions[:, :, -1].squeeze(-1)  #shape: (batch_size,)
    
    #compute mse loss
    loss = F.mse_loss(final_predictions, targets)
    return loss

#calculate batch metrics for monitoring training progress
def compute_batch_metrics(predictions, targets, certainties, tolerance=DEFAULT_TOLERANCE):
    #extract final predictions and certainties
    final_predictions = predictions[:, :, -1].squeeze(-1)  #shape: (batch_size,)
    final_certainties = certainties[:, 1, -1]  #confidence values (1 - entropy)
    
    #compute accuracy within tolerance
    prediction_errors = torch.abs(final_predictions - targets)
    accurate_predictions = (prediction_errors <= tolerance).float()
    accuracy = accurate_predictions.mean().item()
    
    #compute average confidence
    avg_confidence = final_certainties.mean().item()
    
    #compute convergence analysis - how many steps to reach stable prediction
    convergence_steps = compute_convergence_steps(predictions)
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'convergence_steps': convergence_steps,
        'final_predictions': final_predictions.detach().cpu(),
        'prediction_errors': prediction_errors.detach().cpu()
    }

#analyse how quickly predictions converge across thinking steps
def compute_convergence_steps(predictions, stability_threshold=0.1):
    #predictions shape: (batch_size, output_dim, thinking_steps)
    batch_size, _, thinking_steps = predictions.shape
    
    if thinking_steps < 2:
        return 1.0
    
    convergence_steps = []
    
    for b in range(batch_size):
        #track prediction changes across thinking steps
        pred_sequence = predictions[b, 0, :]  #single output dim
        
        #find when prediction becomes stable
        converged_step = thinking_steps  #default to max steps
        
        for step in range(1, thinking_steps):
            change = torch.abs(pred_sequence[step] - pred_sequence[step-1])
            
            if change < stability_threshold:
                converged_step = step + 1
                break
        
        convergence_steps.append(converged_step)
    
    return np.mean(convergence_steps)

#--------------------------------------------------------------------------------------------------------------
#                                                Training Core
#--------------------------------------------------------------------------------------------------------------

#process single batch through ctm with gradient updates
def process_single_batch(model, input_batch, target_batch, optimiser, device):
    #move data to device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    #prepare ctm input format
    ctm_input = create_ctm_input_batch(input_batch)
    
    #zero gradients
    optimiser.zero_grad()
    
    #forward pass through ctm
    predictions, certainties = model(ctm_input)
    
    #compute loss
    loss = compute_mse_loss(predictions, target_batch)
    
    #backward pass
    loss.backward()
    optimiser.step()
    
    #compute batch metrics
    with torch.no_grad():
        batch_metrics = compute_batch_metrics(predictions, target_batch, certainties)
    
    #return batch results
    return {
        'loss': loss.item(),
        'predictions': predictions.detach().cpu(),
        'certainties': certainties.detach().cpu(),
        **batch_metrics
    }

#train model for single epoch
def train_single_epoch(model, dataloader, optimiser, device):
    model.train()
    
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_confidence = 0.0
    epoch_convergence = 0.0
    batch_count = 0
    
    #process all batches in epoch
    for input_batch, target_batch in dataloader:
        batch_results = process_single_batch(model, input_batch, target_batch, optimiser, device)
        
        #accumulate metrics
        epoch_loss += batch_results['loss']
        epoch_accuracy += batch_results['accuracy']
        epoch_confidence += batch_results['avg_confidence']
        epoch_convergence += batch_results['convergence_steps']
        batch_count += 1
    
    #calculate epoch averages
    return {
        'avg_loss': epoch_loss / batch_count,
        'avg_accuracy': epoch_accuracy / batch_count,
        'avg_confidence': epoch_confidence / batch_count,
        'avg_convergence': epoch_convergence / batch_count,
        'batch_count': batch_count
    }

#complete ctm training pipeline
def train_ctm_model(model, train_data, train_labels, config=None):
    #set default configuration if none provided
    if config is None:
        config = {}
    
    #extract configuration with defaults
    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
    epochs = config.get('epochs', DEFAULT_EPOCHS)
    learning_rate = config.get('learning_rate', DEFAULT_LEARNING_RATE)
    device = config.get('device', DEFAULT_DEVICE)
    save_path = config.get('save_path', './checkpoints')
    save_freq = config.get('save_freq', DEFAULT_SAVE_FREQ)
    
    #setup data loader
    train_loader = numpy_to_torch_dataset(train_data, train_labels, batch_size)
    
    #setup optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #move model to device
    model = model.to(device)
    
    #initialise training history
    history = {
        'loss': [],
        'accuracy': [],
        'confidence': [],
        'convergence': []
    }
    
    #training loop with progress bar
    epoch_bar = tqdm(range(epochs), desc="training ctm")
    
    for epoch in epoch_bar:
        #train single epoch
        epoch_metrics = train_single_epoch(model, train_loader, optimiser, device)
        
        #store metrics
        history['loss'].append(epoch_metrics['avg_loss'])
        history['accuracy'].append(epoch_metrics['avg_accuracy'])
        history['confidence'].append(epoch_metrics['avg_confidence'])
        history['convergence'].append(epoch_metrics['avg_convergence'])
        
        #update progress bar
        epoch_bar.set_postfix(
            loss=f"{epoch_metrics['avg_loss']:.4f}",
            acc=f"{epoch_metrics['avg_accuracy']:.3f}",
            conf=f"{epoch_metrics['avg_confidence']:.3f}"
        )
        
        #save checkpoint periodically
        if save_path and (epoch + 1) % save_freq == 0:
            save_checkpoint(model, epoch, epoch_metrics, config, save_path)
    
    #save final training metrics
    save_training_metrics(history, save_path, config)
    
    print("training completed successfully")
    return history

#--------------------------------------------------------------------------------------------------------------
#                                                Checkpointing & Metrics
#--------------------------------------------------------------------------------------------------------------

#save model checkpoint with configuration
def save_checkpoint(model, epoch, metrics, config, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_path = os.path.join(save_path, f"ctm_epoch_{epoch+1}.pt")
    
    #prepare checkpoint data
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'model_config': get_ctm_config(model),
        'training_config': config,
        'final_loss': metrics['avg_loss'],
        'final_accuracy': metrics['avg_accuracy'],
        'final_confidence': metrics['avg_confidence']
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"checkpoint saved: {checkpoint_path}")

#save complete training metrics history
def save_training_metrics(history, save_path, config):
    os.makedirs(save_path, exist_ok=True)
    
    #prepare metadata
    metadata = {
        'training_type': 'ctm_phase1',
        'num_epochs': len(history['loss']),
        'final_loss': history['loss'][-1],
        'final_accuracy': history['accuracy'][-1],
        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': config
    }
    
    #save metadata
    with open(os.path.join(save_path, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    #save metrics history
    np.savez(os.path.join(save_path, 'training_metrics.npz'),
             epoch=np.arange(1, len(history['loss']) + 1),
             loss=np.array(history['loss']),
             accuracy=np.array(history['accuracy']),
             confidence=np.array(history['confidence']),
             convergence=np.array(history['convergence']))
    
    print(f"training metrics saved to {save_path}")

#extract ctm model configuration for reproducibility
def get_ctm_config(model):
    return {
        'num_neurons': model.num_neurons,
        'memory_length': model.pre_activations_manager.memory_length,
        'sync_size_out': model.synchronisation_manager.sync_size_out,
        'sync_size_action': model.synchronisation_manager.sync_size_action,
        'attention_size': model.attention_module.attention_size,
        'num_heads': model.attention_module.num_heads,
        'unet_depth': model.synapse_model.depth,
        'thinking_steps': model.thinking_steps,
        'output_dim': model.output_dim,
        'self_pairing_count': model.synchronisation_manager.self_pairing_count,
        'use_deep_nlm': model.neuron_level_model.is_deep,
        'prediction_reshaper': model.prediction_reshaper
    }

#--------------------------------------------------------------------------------------------------------------
#                                                Validation
#--------------------------------------------------------------------------------------------------------------

#validate training data format for ctm compatibility
def validate_training_data(inputs, labels):
    #check input format
    assert isinstance(inputs, np.ndarray), "inputs must be numpy array"
    assert isinstance(labels, np.ndarray), "labels must be numpy array"
    assert len(inputs.shape) == 2, "inputs must be 2d array (samples, sequence_length)"
    assert len(labels.shape) == 1, "labels must be 1d array"
    assert inputs.shape[0] == labels.shape[0], "inputs and labels must have same number of samples"
    
    print(f"training data validated: {inputs.shape[0]} samples, {inputs.shape[1]} sequence length")
    return True
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import datetime
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

#default training hyperparameters for phase 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_FREQ = 10
DEFAULT_TOLERANCE = 0.5  #accuracy tolerance for arithmetic predictions

#default loss component weights for ctm combined loss
DEFAULT_LOSS_WEIGHTS = {
    'progressive': 1.0,      #main prediction accuracy loss
    'confidence': 0.2,       #confidence regularisation strength
    'convergence': 0.1       #reasoning stability penalty
}

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
    #add feature dimension for ctm attention module
    #shape becomes: (batch_size, sequence_length, feature_dim=1)
    input_features = sequences.unsqueeze(-1)
    
    return input_features

#--------------------------------------------------------------------------------------------------------------
#                                                CTM Loss Functions
#--------------------------------------------------------------------------------------------------------------

#progressive weighted mse loss - weight later thinking steps more heavily
def compute_progressive_weighted_mse(predictions, targets, step_weighting='linear'):
    #predictions shape: (batch_size, output_dim, thinking_steps)
    batch_size, output_dim, thinking_steps = predictions.shape
    
    #create step weights - later steps more important
    if step_weighting == 'linear':
        step_weights = torch.linspace(0.1, 1.0, thinking_steps, device=predictions.device)
    elif step_weighting == 'exponential':
        step_weights = torch.exp(torch.linspace(0, 1, thinking_steps, device=predictions.device))
        step_weights = step_weights / step_weights.sum() * thinking_steps  #normalise
    else:
        step_weights = torch.ones(thinking_steps, device=predictions.device)
    
    #compute mse loss for each thinking step
    step_losses = []
    for step in range(thinking_steps):
        step_pred = predictions[:, :, step].squeeze(-1)  #shape: (batch_size,)
        step_loss = F.mse_loss(step_pred, targets, reduction='mean')
        step_losses.append(step_loss * step_weights[step])
    
    #combine weighted losses
    progressive_loss = torch.stack(step_losses).sum()
    return progressive_loss

#confidence regularisation loss - penalise overconfidence on wrong predictions
def compute_confidence_regularisation_loss(predictions, certainties, targets, overall_confidence):
    #extract final predictions and compute prediction quality
    final_predictions = predictions[:, :, -1].squeeze(-1)  #shape: (batch_size,)
    prediction_errors = torch.abs(final_predictions - targets)
    
    #convert errors to quality measure (lower error = higher quality)
    prediction_quality = 1.0 / (1.0 + prediction_errors)
    
    #extract final step confidence and overall reasoning confidence
    final_step_confidence = certainties[:, 1, -1]  #shape: (batch_size,)
    reasoning_confidence = overall_confidence[:, 1]  #shape: (batch_size,)
    
    #penalise mismatch between confidence and actual prediction quality
    step_confidence_penalty = F.mse_loss(final_step_confidence, prediction_quality)
    overall_confidence_penalty = F.mse_loss(reasoning_confidence, prediction_quality)
    
    #combine confidence regularisation losses
    confidence_reg_loss = step_confidence_penalty + overall_confidence_penalty
    return confidence_reg_loss

#convergence loss - reward stable reasoning that converges quickly
def compute_convergence_loss(predictions):
    #predictions shape: (batch_size, output_dim, thinking_steps)
    batch_size, _, thinking_steps = predictions.shape
    
    if thinking_steps < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    #compute prediction variance across thinking steps for each sample
    prediction_variance = torch.var(predictions[:, 0, :], dim=-1)  #shape: (batch_size,)
    
    #penalise high variance (unstable reasoning)
    convergence_loss = prediction_variance.mean()
    return convergence_loss

#combined ctm loss function integrating all loss components
def compute_ctm_combined_loss(predictions, targets, certainties, overall_confidence, loss_weights=None):
    if loss_weights is None:
        loss_weights = DEFAULT_LOSS_WEIGHTS
    
    #compute individual loss components
    progressive_loss = compute_progressive_weighted_mse(predictions, targets)
    confidence_loss = compute_confidence_regularisation_loss(predictions, certainties, targets, overall_confidence)
    convergence_loss = compute_convergence_loss(predictions)
    
    #combine losses with configurable weights
    total_loss = (loss_weights['progressive'] * progressive_loss + 
                  loss_weights['confidence'] * confidence_loss + 
                  loss_weights['convergence'] * convergence_loss)
    
    return total_loss, {
        'progressive': progressive_loss.item(),
        'confidence': confidence_loss.item(), 
        'convergence': convergence_loss.item(),
        'total': total_loss.item()
    }

#--------------------------------------------------------------------------------------------------------------
#                                               Metrics
#--------------------------------------------------------------------------------------------------------------

#compute batch metrics using per-step data and overall confidence
def compute_batch_metrics(predictions, targets, certainties, overall_confidence, tolerance=DEFAULT_TOLERANCE):
    #extract final predictions and certainties
    final_predictions = predictions[:, :, -1].squeeze(-1)  #shape: (batch_size,)
    final_certainties = certainties[:, 1, -1]  #confidence values (1 - entropy)
    overall_reasoning_conf = overall_confidence[:, 1]  #overall reasoning confidence
    
    #compute final step accuracy
    prediction_errors = torch.abs(final_predictions - targets)
    accurate_predictions = (prediction_errors <= tolerance).float()
    final_accuracy = accurate_predictions.mean().item()
    
    #compute per-step accuracy evolution
    step_accuracies = []
    thinking_steps = predictions.shape[-1]
    for step in range(thinking_steps):
        step_preds = predictions[:, :, step].squeeze(-1)
        step_errors = torch.abs(step_preds - targets)
        step_acc = (step_errors <= tolerance).float().mean().item()
        step_accuracies.append(step_acc)
    
    #compute confidence evolution metrics
    confidence_trajectory = certainties[:, 1, :].mean(0).detach().cpu().numpy()  #average confidence per step
    
    #compute convergence analysis
    convergence_steps = compute_convergence_steps(predictions)
    reasoning_stability = 1.0 / (1.0 + torch.var(predictions[:, 0, :], dim=-1).mean().item())
    
    return {
        'final_accuracy': final_accuracy,
        'avg_final_confidence': final_certainties.mean().item(),
        'avg_reasoning_confidence': overall_reasoning_conf.mean().item(),
        'step_accuracies': step_accuracies,
        'confidence_trajectory': confidence_trajectory,
        'convergence_steps': convergence_steps,
        'reasoning_stability': reasoning_stability,
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

#process single batch through ctm with gradient updates using loss
def process_single_batch(model, input_batch, target_batch, optimiser, device, loss_config=None):
    #move data to device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    #prepare ctm input format
    ctm_input = create_ctm_input_batch(input_batch)
    
    #zero gradients
    optimiser.zero_grad()
    
    #forward pass through ctm - now returns overall confidence
    predictions, certainties, overall_confidence = model(ctm_input)
    
    #compute ctm combined loss
    loss_weights = loss_config.get('loss_weights', DEFAULT_LOSS_WEIGHTS) if loss_config else DEFAULT_LOSS_WEIGHTS
    total_loss, loss_components = compute_ctm_combined_loss(predictions, target_batch, certainties, overall_confidence, loss_weights)
    
    #backward pass
    total_loss.backward()
    optimiser.step()
    
    #compute batch metrics
    with torch.no_grad():
        batch_metrics = compute_batch_metrics(predictions, target_batch, certainties, overall_confidence)
    
    #return batch results with loss breakdown
    return {
        'loss': total_loss.item(),
        'loss_components': loss_components,
        'predictions': predictions.detach().cpu(),
        'certainties': certainties.detach().cpu(),
        'overall_confidence': overall_confidence.detach().cpu(),
        **batch_metrics
    }

#train model for single epoch with metrics tracking
def train_single_epoch(model, dataloader, optimiser, device, loss_config=None):
    model.train()
    
    #epoch accumulators
    epoch_loss = 0.0
    epoch_loss_components = {'progressive': 0.0, 'confidence': 0.0, 'convergence': 0.0}
    epoch_final_accuracy = 0.0
    epoch_final_confidence = 0.0
    epoch_reasoning_confidence = 0.0
    epoch_convergence = 0.0
    epoch_stability = 0.0
    
    #per-step tracking across epoch
    thinking_steps = None
    epoch_step_accuracies = None
    epoch_confidence_trajectory = None
    
    batch_count = 0
    
    #process all batches in epoch
    for input_batch, target_batch in dataloader:
        batch_results = process_single_batch(model, input_batch, target_batch, optimiser, device, loss_config)
        
        #accumulate main metrics
        epoch_loss += batch_results['loss']
        for component in epoch_loss_components:
            epoch_loss_components[component] += batch_results['loss_components'][component]
        
        epoch_final_accuracy += batch_results['final_accuracy']
        epoch_final_confidence += batch_results['avg_final_confidence']
        epoch_reasoning_confidence += batch_results['avg_reasoning_confidence']
        epoch_convergence += batch_results['convergence_steps']
        epoch_stability += batch_results['reasoning_stability']
        
        #accumulate per-step metrics
        if epoch_step_accuracies is None:
            thinking_steps = len(batch_results['step_accuracies'])
            epoch_step_accuracies = np.zeros(thinking_steps)
            epoch_confidence_trajectory = np.zeros(thinking_steps)
        
        epoch_step_accuracies += np.array(batch_results['step_accuracies'])
        epoch_confidence_trajectory += batch_results['confidence_trajectory']
        
        batch_count += 1
    
    #calculate epoch averages
    return {
        'avg_loss': epoch_loss / batch_count,
        'avg_loss_components': {k: v / batch_count for k, v in epoch_loss_components.items()},
        'avg_final_accuracy': epoch_final_accuracy / batch_count,
        'avg_final_confidence': epoch_final_confidence / batch_count,
        'avg_reasoning_confidence': epoch_reasoning_confidence / batch_count,
        'avg_convergence': epoch_convergence / batch_count,
        'avg_stability': epoch_stability / batch_count,
        'avg_step_accuracies': epoch_step_accuracies / batch_count,
        'avg_confidence_trajectory': epoch_confidence_trajectory / batch_count,
        'batch_count': batch_count
    }

#complete ctm training pipeline with loss and metrics
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
        'loss_components': {'progressive': [], 'confidence': [], 'convergence': []},
        'final_accuracy': [],
        'final_confidence': [],
        'reasoning_confidence': [],
        'convergence': [],
        'stability': [],
        'step_accuracies': [],
        'confidence_trajectory': []
    }
    
    #training loop with progress bar
    epoch_bar = tqdm(range(epochs), desc="training ctm phase 2")
    
    for epoch in epoch_bar:
        #train single epoch with metrics
        epoch_metrics = train_single_epoch(model, train_loader, optimiser, device, config)
        
        #store metrics
        history['loss'].append(epoch_metrics['avg_loss'])
        for component in history['loss_components']:
            history['loss_components'][component].append(epoch_metrics['avg_loss_components'][component])
        
        history['final_accuracy'].append(epoch_metrics['avg_final_accuracy'])
        history['final_confidence'].append(epoch_metrics['avg_final_confidence'])
        history['reasoning_confidence'].append(epoch_metrics['avg_reasoning_confidence'])
        history['convergence'].append(epoch_metrics['avg_convergence'])
        history['stability'].append(epoch_metrics['avg_stability'])
        history['step_accuracies'].append(epoch_metrics['avg_step_accuracies'])
        history['confidence_trajectory'].append(epoch_metrics['avg_confidence_trajectory'])
        
        #update progress bar with key metrics
        epoch_bar.set_postfix(
            loss=f"{epoch_metrics['avg_loss']:.4f}",
            acc=f"{epoch_metrics['avg_final_accuracy']:.3f}",
            conf=f"{epoch_metrics['avg_reasoning_confidence']:.3f}",
            stab=f"{epoch_metrics['avg_stability']:.3f}"
        )
        
        #save checkpoint periodically
        if save_path and (epoch + 1) % save_freq == 0:
            save_checkpoint(model, epoch, epoch_metrics, config, save_path)
    
    #save final training metrics
    save_training_metrics(history, save_path, config)
    
    print("ctm phase 2 training completed successfully")
    return history

#--------------------------------------------------------------------------------------------------------------
#                                               Checkpointing & Metrics
#--------------------------------------------------------------------------------------------------------------

#save model checkpoint with loss component breakdown
def save_checkpoint(model, epoch, metrics, config, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_path = os.path.join(save_path, f"ctm_phase2_epoch_{epoch+1}.pt")
    
    #prepare checkpoint data
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'model_config': get_ctm_config(model),
        'training_config': config,
        'final_loss': metrics['avg_loss'],
        'loss_components': metrics['avg_loss_components'],
        'final_accuracy': metrics['avg_final_accuracy'],
        'reasoning_confidence': metrics['avg_reasoning_confidence'],
        'reasoning_stability': metrics['avg_stability']
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"checkpoint saved: {checkpoint_path}")

#save complete training metrics history
def save_training_metrics(history, save_path, config):
    os.makedirs(save_path, exist_ok=True)
    
    #prepare  metadata
    metadata = {
        'num_epochs': len(history['loss']),
        'final_loss': history['loss'][-1],
        'final_accuracy': history['final_accuracy'][-1],
        'final_reasoning_confidence': history['reasoning_confidence'][-1],
        'final_stability': history['stability'][-1],
        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': config,
        'loss_components': {k: v[-1] for k, v in history['loss_components'].items()}
    }
    
    #save metadata
    with open(os.path.join(save_path, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    #save metrics history
    np.savez(os.path.join(save_path, 'training_metrics.npz'),
             epoch=np.arange(1, len(history['loss']) + 1),
             loss=np.array(history['loss']),
             progressive_loss=np.array(history['loss_components']['progressive']),
             confidence_loss=np.array(history['loss_components']['confidence']),
             convergence_loss=np.array(history['loss_components']['convergence']),
             final_accuracy=np.array(history['final_accuracy']),
             final_confidence=np.array(history['final_confidence']),
             reasoning_confidence=np.array(history['reasoning_confidence']),
             convergence=np.array(history['convergence']),
             stability=np.array(history['stability']),
             step_accuracies=np.array(history['step_accuracies']),
             confidence_trajectory=np.array(history['confidence_trajectory']))
    
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
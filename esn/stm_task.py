"""
Short Term Memory (STM) task implementation for evaluating ESN memory capacity.

This module implements the Short Term Memory task as described in:
Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
GMD Report 148, German National Research Center for Information Technology.

The memory capacity is a quantitative measure of the ability of an ESN to reconstruct
delayed versions of the input signal.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from model import EchoStateNetwork, ESNRegressor
import warnings


def generate_stm_data(
    sequence_length: int,
    num_delays: int,
    num_sequences: int = 1,
    input_type: str = 'uniform',
    random_state: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for Short Term Memory task.
    
    Args:
        sequence_length (int): Length of each sequence
        num_delays (int): Maximum number of delays to test
        num_sequences (int): Number of independent sequences
        input_type (str): Type of input signal ('uniform', 'gaussian', 'binary')
        random_state (Optional[int]): Random seed for reproducibility
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input sequences and target sequences
    """
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    # Generate input sequences
    if input_type == 'uniform':
        inputs = torch.rand(num_sequences, sequence_length, 1) * 2 - 1  # [-1, 1]
    elif input_type == 'gaussian':
        inputs = torch.randn(num_sequences, sequence_length, 1)
    elif input_type == 'binary':
        inputs = (torch.rand(num_sequences, sequence_length, 1) > 0.5).float() * 2 - 1
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # Generate delayed targets
    targets = torch.zeros(num_sequences, sequence_length, num_delays)
    
    for delay in range(1, num_delays + 1):
        # Shift input by delay steps
        delayed_input = torch.zeros_like(inputs[:, :, 0])
        if delay < sequence_length:
            delayed_input[:, delay:] = inputs[:, :-delay, 0]
        targets[:, :, delay - 1] = delayed_input
    
    return inputs, targets


def compute_memory_capacity(
    esn: EchoStateNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float = 0.7,
    washout: int = 100,
    ridge_alpha: float = 1e-6
) -> Tuple[float, List[float], List[int], Dict]:
    """
    Compute memory capacity of an ESN using the STM task.
    
    Args:
        esn (EchoStateNetwork): The ESN model to evaluate
        inputs (torch.Tensor): Input sequences of shape (num_seq, seq_len, 1)
        targets (torch.Tensor): Target sequences of shape (num_seq, seq_len, num_delays)
        train_ratio (float): Ratio of data to use for training (rest for validation)
        washout (int): Number of initial timesteps to ignore
        ridge_alpha (float): Ridge regression regularization parameter
        
    Returns:
        Tuple[float, List[float], List[int], Dict]: Total memory capacity, capacity per delay, 
                                                  delay indices, and detailed results
    """
    num_sequences, sequence_length, num_delays = targets.shape
    
    # Get reservoir states for entire sequence
    states = esn.forward(inputs)
    
    # Apply washout
    if washout > 0:
        states = states[:, washout:, :]
        targets = targets[:, washout:, :]
        effective_length = sequence_length - washout
    else:
        effective_length = sequence_length
    
    # Split into train and validation
    train_length = int(effective_length * train_ratio)
    
    # Training data
    states_train = states[:, :train_length, :]
    targets_train = targets[:, :train_length, :]
    
    # Validation data
    states_val = states[:, train_length:, :]
    targets_val = targets[:, train_length:, :]
    
    # Reshape training data for regression
    batch_size, train_seq_len, reservoir_size = states_train.shape
    X_train = states_train.reshape(-1, reservoir_size)
    
    # Add bias term to training data
    X_train_with_bias = torch.cat([
        X_train,
        torch.ones(X_train.shape[0], 1, device=X_train.device)
    ], dim=1)
    
    # Reshape validation data
    val_seq_len = states_val.shape[1]
    X_val = states_val.reshape(-1, reservoir_size)
    X_val_with_bias = torch.cat([
        X_val,
        torch.ones(X_val.shape[0], 1, device=X_val.device)
    ], dim=1)
    
    memory_capacities = []
    trained_weights = []
    validation_predictions = []
    validation_targets = []
    
    # Compute capacity for each delay
    for delay in range(num_delays):
        # Training targets
        y_train = targets_train[:, :, delay].reshape(-1, 1)
        
        # Validation targets
        y_val = targets_val[:, :, delay].reshape(-1, 1)
        
        # Ridge regression on training data
        A = X_train_with_bias.T @ X_train_with_bias
        A.diagonal().add_(ridge_alpha)
        b = X_train_with_bias.T @ y_train
        
        # Solve linear system
        w_out = torch.linalg.solve(A, b)
        trained_weights.append(w_out.clone())
        
        # Make predictions on validation data
        y_pred = X_val_with_bias @ w_out
        validation_predictions.append(y_pred.clone())
        validation_targets.append(y_val.clone())
        
        # Compute R² on validation data
        ss_res = torch.sum((y_val - y_pred) ** 2)
        ss_tot = torch.sum((y_val - torch.mean(y_val)) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
            capacity = max(0, r_squared.item())  # Ensure non-negative
        else:
            capacity = 0.0
        
        memory_capacities.append(capacity)
    
    # Total memory capacity is sum of individual capacities
    total_capacity = sum(memory_capacities)
    delay_indices = list(range(1, num_delays + 1))
    
    # Detailed results for visualization
    detailed_results = {
        'trained_weights': trained_weights,
        'validation_predictions': validation_predictions,
        'validation_targets': validation_targets,
        'validation_states': states_val,
        'train_length': train_length,
        'val_length': val_seq_len,
        'washout': washout
    }
    
    return total_capacity, memory_capacities, delay_indices, detailed_results


def plot_delay_accuracy(
    detailed_results: Dict,
    delay_indices: List[int] = [1, 5],
    sequence_indices: List[int] = [0],
    max_points: int = 200,
    save_path: Optional[str] = None
):
    """
    Plot truth vs prediction accuracy for specific delays.
    
    Args:
        detailed_results (Dict): Detailed results from compute_memory_capacity
        delay_indices (List[int]): Which delays to visualize (1-indexed)
        sequence_indices (List[int]): Which sequences to plot (0-indexed)
        max_points (int): Maximum number of points to plot for clarity
        save_path (Optional[str]): Path to save the plot
    """
    n_delays = len(delay_indices)
    n_sequences = len(sequence_indices)
    
    fig, axes = plt.subplots(n_sequences, n_delays, figsize=(6 * n_delays, 4 * n_sequences))
    
    if n_sequences == 1 and n_delays == 1:
        axes = [[axes]]
    elif n_sequences == 1:
        axes = [axes]
    elif n_delays == 1:
        axes = [[ax] for ax in axes]
    
    for seq_idx, seq in enumerate(sequence_indices):
        for delay_idx, delay in enumerate(delay_indices):
            ax = axes[seq_idx][delay_idx]
            
            # Get predictions and targets for this delay (convert to 0-indexed)
            delay_0indexed = delay - 1
            y_pred = detailed_results['validation_predictions'][delay_0indexed]
            y_true = detailed_results['validation_targets'][delay_0indexed]
            
            # Extract data for specific sequence
            val_length = detailed_results['val_length']
            start_idx = seq * val_length
            end_idx = (seq + 1) * val_length
            
            pred_seq = y_pred[start_idx:end_idx, 0].cpu().numpy()
            true_seq = y_true[start_idx:end_idx, 0].cpu().numpy()
            
            # Subsample for visualization if too many points
            if len(pred_seq) > max_points:
                indices = np.linspace(0, len(pred_seq) - 1, max_points, dtype=int)
                pred_seq = pred_seq[indices]
                true_seq = true_seq[indices]
                time_steps = indices
            else:
                time_steps = np.arange(len(pred_seq))
            
            # Plot time series comparison
            ax.plot(time_steps, true_seq, 'b-', label='Truth', linewidth=2, alpha=0.8)
            ax.plot(time_steps, pred_seq, 'r--', label='Prediction', linewidth=2, alpha=0.8)
            
            # Calculate R² for this specific delay
            ss_res = np.sum((true_seq - pred_seq) ** 2)
            ss_tot = np.sum((true_seq - np.mean(true_seq)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            ax.set_title(f'Delay k={delay}, Sequence {seq}\nR² = {r2:.4f}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            y_min = min(np.min(true_seq), np.min(pred_seq))
            y_max = max(np.max(true_seq), np.max(pred_seq))
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Delay accuracy plot saved to {save_path}")
    
    plt.show()


def plot_scatter_accuracy(
    detailed_results: Dict,
    delay_indices: List[int] = [1, 5],
    sequence_indices: List[int] = [0],
    max_points: int = 1000,
    save_path: Optional[str] = None
):
    """
    Plot scatter plots of truth vs prediction for specific delays.
    
    Args:
        detailed_results (Dict): Detailed results from compute_memory_capacity
        delay_indices (List[int]): Which delays to visualize (1-indexed)
        sequence_indices (List[int]): Which sequences to plot (0-indexed)
        max_points (int): Maximum number of points to plot for clarity
        save_path (Optional[str]): Path to save the plot
    """
    n_delays = len(delay_indices)
    
    fig, axes = plt.subplots(1, n_delays, figsize=(6 * n_delays, 5))
    
    if n_delays == 1:
        axes = [axes]
    
    for delay_idx, delay in enumerate(delay_indices):
        ax = axes[delay_idx]
        
        # Get predictions and targets for this delay (convert to 0-indexed)
        delay_0indexed = delay - 1
        y_pred = detailed_results['validation_predictions'][delay_0indexed]
        y_true = detailed_results['validation_targets'][delay_0indexed]
        
        # Combine data from all specified sequences
        all_pred = []
        all_true = []
        
        for seq in sequence_indices:
            val_length = detailed_results['val_length']
            start_idx = seq * val_length
            end_idx = (seq + 1) * val_length
            
            pred_seq = y_pred[start_idx:end_idx, 0].cpu().numpy()
            true_seq = y_true[start_idx:end_idx, 0].cpu().numpy()
            
            all_pred.extend(pred_seq)
            all_true.extend(true_seq)
        
        all_pred = np.array(all_pred)
        all_true = np.array(all_true)
        
        # Subsample if too many points
        if len(all_pred) > max_points:
            indices = np.random.choice(len(all_pred), max_points, replace=False)
            all_pred = all_pred[indices]
            all_true = all_true[indices]
        
        # Create scatter plot
        ax.scatter(all_true, all_pred, alpha=0.6, s=8)
        
        # Add diagonal line (perfect prediction)
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # Calculate R²
        ss_res = np.sum((all_true - all_pred) ** 2)
        ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        ax.set_title(f'Truth vs Prediction (Delay k={delay})\nR² = {r2:.4f}')
        ax.set_xlabel('Truth')
        ax.set_ylabel('Prediction')
        ax.grid(True, alpha=0.3)
        
        # Make axes equal
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter accuracy plot saved to {save_path}")
    
    plt.show()


def evaluate_memory_capacity(
    reservoir_size: int,
    spectral_radius: float = 0.95,
    input_scaling: float = 1.0,
    connectivity: float = 0.1,
    leaking_rate: float = 1.0,
    sequence_length: int = 2000,
    max_delay: int = 100,
    num_sequences: int = 1,
    train_ratio: float = 0.7,
    washout: int = 100,
    ridge_alpha: float = 1e-6,
    input_type: str = 'uniform',
    random_state: Optional[int] = None,
    device: str = 'cpu',
    plot_delays: Optional[List[int]] = None
) -> Dict:
    """
    Evaluate memory capacity of an ESN with given parameters.
    
    Args:
        reservoir_size (int): Number of reservoir units
        spectral_radius (float): Spectral radius of reservoir
        input_scaling (float): Input scaling factor
        connectivity (float): Reservoir connectivity
        leaking_rate (float): Leaking rate
        sequence_length (int): Length of test sequences
        max_delay (int): Maximum delay to test
        num_sequences (int): Number of test sequences
        train_ratio (float): Ratio of data for training (rest for validation)
        washout (int): Washout period
        ridge_alpha (float): Ridge regression regularization
        input_type (str): Type of input signal
        random_state (Optional[int]): Random seed
        device (str): Computing device
        plot_delays (Optional[List[int]]): Delays to plot for visualization
        
    Returns:
        Dict: Results containing memory capacity and other metrics
    """
    print(f"Evaluating ESN with {reservoir_size} reservoir units...")
    print(f"Parameters: SR={spectral_radius}, IS={input_scaling}, "
          f"Conn={connectivity}, LR={leaking_rate}")
    print(f"Train ratio: {train_ratio:.2f}, Validation ratio: {1-train_ratio:.2f}")
    
    # Generate data
    inputs, targets = generate_stm_data(
        sequence_length=sequence_length,
        num_delays=max_delay,
        num_sequences=num_sequences,
        input_type=input_type,
        random_state=random_state
    )
    
    # Move to device
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Create ESN
    esn = EchoStateNetwork(
        input_size=1,
        reservoir_size=reservoir_size,
        output_size=max_delay,
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        connectivity=connectivity,
        leaking_rate=leaking_rate,
        random_state=random_state,
        device=device
    )
    
    # Compute memory capacity
    total_capacity, capacities_per_delay, delay_indices, detailed_results = compute_memory_capacity(
        esn=esn,
        inputs=inputs,
        targets=targets,
        train_ratio=train_ratio,
        washout=washout,
        ridge_alpha=ridge_alpha
    )
    
    print(f"Total Memory Capacity: {total_capacity:.4f}")
    print(f"Theoretical Maximum: {reservoir_size}")
    print(f"Capacity Ratio: {total_capacity/reservoir_size:.4f}")
    
    # Plot delay accuracy if requested
    if plot_delays is not None:
        print(f"Plotting accuracy for delays: {plot_delays}")
        plot_delay_accuracy(detailed_results, delay_indices=plot_delays)
        plot_scatter_accuracy(detailed_results, delay_indices=plot_delays)
    
    return {
        'total_capacity': total_capacity,
        'capacities_per_delay': capacities_per_delay,
        'delay_indices': delay_indices,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'input_scaling': input_scaling,
        'connectivity': connectivity,
        'leaking_rate': leaking_rate,
        'capacity_ratio': total_capacity / reservoir_size,
        'train_ratio': train_ratio,
        'esn_info': esn.get_info(),
        'detailed_results': detailed_results
    }


def plot_memory_capacity(results: Dict, save_path: Optional[str] = None):
    """
    Plot memory capacity results.
    
    Args:
        results (Dict): Results from evaluate_memory_capacity
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Capacity per delay
    plt.subplot(1, 2, 1)
    plt.plot(results['delay_indices'], results['capacities_per_delay'], 'b-', linewidth=2)
    plt.xlabel('Delay (k)')
    plt.ylabel('Memory Capacity MC(k)')
    plt.title(f'Memory Capacity per Delay\nReservoir Size: {results["reservoir_size"]}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative capacity
    cumulative_capacity = np.cumsum(results['capacities_per_delay'])
    plt.subplot(1, 2, 2)
    plt.plot(results['delay_indices'], cumulative_capacity, 'r-', linewidth=2)
    plt.axhline(y=results['total_capacity'], color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Delay (k)')
    plt.ylabel('Cumulative Memory Capacity')
    plt.title(f'Cumulative Memory Capacity\nTotal: {results["total_capacity"]:.2f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def compare_esn_parameters(
    parameter_ranges: Dict,
    reservoir_size: int = 100,
    sequence_length: int = 1000,
    max_delay: int = 50,
    train_ratio: float = 0.7,
    random_state: Optional[int] = None,
    device: str = 'cpu'
) -> Dict:
    """
    Compare memory capacity across different ESN parameter settings.
    
    Args:
        parameter_ranges (Dict): Dictionary with parameter names and their ranges
        reservoir_size (int): Base reservoir size
        sequence_length (int): Length of test sequences
        max_delay (int): Maximum delay to test
        train_ratio (float): Ratio of data for training
        random_state (Optional[int]): Random seed
        device (str): Computing device
        
    Returns:
        Dict: Comparison results
    """
    results = {}
    
    for param_name, param_values in parameter_ranges.items():
        print(f"\nTesting {param_name}...")
        results[param_name] = {
            'values': param_values,
            'capacities': [],
            'capacity_ratios': []
        }
        
        for value in param_values:
            # Set up parameters
            kwargs = {
                'reservoir_size': reservoir_size,
                'sequence_length': sequence_length,
                'max_delay': max_delay,
                'train_ratio': train_ratio,
                'random_state': random_state,
                'device': device
            }
            kwargs[param_name] = value
            
            # Evaluate
            result = evaluate_memory_capacity(**kwargs)
            results[param_name]['capacities'].append(result['total_capacity'])
            results[param_name]['capacity_ratios'].append(result['capacity_ratio'])
    
    return results


def plot_parameter_comparison(comparison_results: Dict, save_path: Optional[str] = None):
    """
    Plot parameter comparison results.
    
    Args:
        comparison_results (Dict): Results from compare_esn_parameters
        save_path (Optional[str]): Path to save the plot
    """
    n_params = len(comparison_results)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    
    if n_params == 1:
        axes = [axes]
    
    for i, (param_name, results) in enumerate(comparison_results.items()):
        axes[i].plot(results['values'], results['capacities'], 'bo-', linewidth=2, markersize=6)
        axes[i].set_xlabel(param_name.replace('_', ' ').title())
        axes[i].set_ylabel('Total Memory Capacity')
        axes[i].set_title(f'Memory Capacity vs {param_name.replace("_", " ").title()}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to demonstrate STM task and memory capacity evaluation."""
    print("Echo State Network Short Term Memory Task")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random_state = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Evaluate single ESN configuration with visualization
    print("\n1. Evaluating single ESN configuration...")
    results = evaluate_memory_capacity(
        reservoir_size=100,
        spectral_radius=0.95,
        input_scaling=1.0,
        connectivity=0.1,
        sequence_length=1000,  # Reduced from 2000 for better visualization
        max_delay=100,         # Reduced from 150
        train_ratio=0.7,
        random_state=random_state,
        device=device,
        plot_delays=[1, 5]  # Visualize delays 1 and 5
    )
    
    # Plot results
    plot_memory_capacity(results)
    
    # Parameter comparison
    print("\n2. Comparing different spectral radius values...")
    parameter_ranges = {
        'spectral_radius': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.1]
    }
    
    comparison_results = compare_esn_parameters(
        parameter_ranges=parameter_ranges,
        reservoir_size=100,
        sequence_length=800,   # Reduced from 1000
        max_delay=60,          # Reduced from 80
        train_ratio=0.7,
        random_state=random_state,
        device=device
    )
    
    # Plot comparison
    plot_parameter_comparison(comparison_results)
    
    print("\nSTM task evaluation completed!")


if __name__ == "__main__":
    main() 
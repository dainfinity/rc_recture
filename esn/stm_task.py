"""
Short-term Memory (STM) task implementation for ESN evaluation.

This module implements the short-term memory capacity benchmark for echo state networks.
The task evaluates how well the reservoir can remember past inputs at various delays.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
from model import EchoStateNetwork, ESNRegressor
import warnings
import os
from datetime import datetime

# Set up matplotlib for better visualization
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def generate_stm_data(
    sequence_length: int,
    num_delays: int,
    num_sequences: int = 1,
    input_type: str = 'uniform',
    random_state: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate input sequences and delayed target sequences for memory capacity test.
    
    Args:
        sequence_length (int): Length of each sequence
        num_delays (int): Maximum number of delays to test
        num_sequences (int): Number of independent sequences
        input_type (str): Type of input signal ('uniform', 'gaussian')
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
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # Create delayed target sequences
    targets = torch.zeros(num_sequences, sequence_length, num_delays)
    
    for delay in range(num_delays):
        # Target at time t is input at time t-(delay+1)
        delay_steps = delay + 1
        targets[:, delay_steps:, delay] = inputs[:, :-delay_steps, 0]
    
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
    Compute short-term memory capacity using linear regression on reservoir states.
    
    Args:
        esn (EchoStateNetwork): The ESN model
        inputs (torch.Tensor): Input sequences
        targets (torch.Tensor): Target sequences (delayed inputs)
        train_ratio (float): Ratio of data to use for training
        washout (int): Number of initial timesteps to ignore
        ridge_alpha (float): Ridge regression regularization parameter
        
    Returns:
        Tuple: (total_capacity, individual_capacities, delay_indices, detailed_results)
    """
    num_sequences, sequence_length, num_delays = targets.shape
    
    # Apply washout
    if washout > 0:
        inputs_clean = inputs[:, washout:, :]
        targets_clean = targets[:, washout:, :]
        effective_length = sequence_length - washout
    else:
        inputs_clean = inputs
        targets_clean = targets
        effective_length = sequence_length
    
    # Get reservoir states
    states = esn.get_reservoir_states(inputs_clean, reset_state=True)
    reservoir_size = states.shape[2]
    
    # Split into training and validation
    train_length = int(effective_length * train_ratio)
    
    states_train = states[:, :train_length, :]
    targets_train = targets_clean[:, :train_length, :]
    states_val = states[:, train_length:, :]
    targets_val = targets_clean[:, train_length:, :]
    
    # Reshape training data
    X_train = states_train.reshape(-1, reservoir_size)
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
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(6 * n_delays, 4 * n_sequences))
    gs = fig.add_gridspec(n_sequences, n_delays, hspace=0.35, wspace=0.25,
                         left=0.08, right=0.95, top=0.90, bottom=0.12)
    
    for seq_idx, seq in enumerate(sequence_indices):
        for delay_idx, delay in enumerate(delay_indices):
            ax = fig.add_subplot(gs[seq_idx, delay_idx])
            
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
            
            ax.set_title(f'Delay k={delay}, Sequence {seq}\nR² = {r2:.4f}', fontsize=11, pad=8)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            y_min = min(np.min(true_seq), np.min(pred_seq))
            y_max = max(np.max(true_seq), np.max(pred_seq))
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    fig.suptitle('STM Task: Delay Accuracy Analysis', fontsize=14, y=0.96)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Delay accuracy plot saved to: {save_path}")
    
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
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(6 * n_delays, 5))
    gs = fig.add_gridspec(1, n_delays, hspace=0.3, wspace=0.25,
                         left=0.08, right=0.95, top=0.88, bottom=0.15)
    
    for delay_idx, delay in enumerate(delay_indices):
        ax = fig.add_subplot(gs[0, delay_idx])
        
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
        
        # Subsample for visualization if too many points
        if len(all_pred) > max_points:
            indices = np.random.choice(len(all_pred), max_points, replace=False)
            all_pred = all_pred[indices]
            all_true = all_true[indices]
        
        # Create scatter plot
        ax.scatter(all_true, all_pred, alpha=0.6, s=20)
        
        # Add diagonal line
        min_val = min(np.min(all_true), np.min(all_pred))
        max_val = max(np.max(all_true), np.max(all_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate R²
        ss_res = np.sum((all_true - all_pred) ** 2)
        ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        ax.set_title(f'Delay k={delay}\nR² = {r2:.4f}', fontsize=12, pad=10)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle('STM Task: Truth vs Prediction Scatter Plots', fontsize=14, y=0.94)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Scatter accuracy plot saved to: {save_path}")
    
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
    plot_delays: Optional[List[int]] = None,
    save_results: bool = True
) -> Dict:
    """
    Evaluate memory capacity of an ESN with given parameters.
    
    Args:
        reservoir_size (int): Number of reservoir units
        spectral_radius (float): Spectral radius of reservoir matrix
        input_scaling (float): Input scaling factor
        connectivity (float): Reservoir connectivity
        leaking_rate (float): Leaking rate
        sequence_length (int): Length of sequences
        max_delay (int): Maximum delay to test
        num_sequences (int): Number of sequences to generate
        train_ratio (float): Training data ratio
        washout (int): Washout period
        ridge_alpha (float): Ridge regression parameter
        input_type (str): Type of input signal
        random_state (Optional[int]): Random seed
        device (str): Device to use
        plot_delays (Optional[List[int]]): Specific delays to plot
        save_results (bool): Whether to save results to file
        
    Returns:
        Dict: Memory capacity evaluation results
    """
    print(f"Evaluating memory capacity (max delay: {max_delay})")
    print("=" * 50)
    
    # Generate data
    print("Generating STM data...")
    inputs, targets = generate_stm_data(
        sequence_length=sequence_length,
        num_delays=max_delay,
        num_sequences=num_sequences,
        input_type=input_type,
        random_state=random_state
    )
    
    # Create ESN
    print("Initializing ESN...")
    esn = ESNRegressor(
        input_size=1,
        reservoir_size=reservoir_size,
        output_size=1,  # Will be set during capacity computation
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        connectivity=connectivity,
        leaking_rate=leaking_rate,
        random_state=random_state,
        device=device
    )
    
    # Compute memory capacity
    print("Computing memory capacity...")
    total_capacity, memory_capacities, delay_indices, detailed_results = compute_memory_capacity(
        esn=esn,
        inputs=inputs,
        targets=targets,
        train_ratio=train_ratio,
        washout=washout,
        ridge_alpha=ridge_alpha
    )
    
    print(f"Total memory capacity: {total_capacity:.4f}")
    print(f"Theoretical maximum: {max_delay}")
    print(f"Efficiency: {total_capacity/max_delay:.2%}")
    
    # Prepare results
    results = {
        'total_capacity': total_capacity,
        'memory_capacities': memory_capacities,
        'delay_indices': delay_indices,
        'detailed_results': detailed_results,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'input_scaling': input_scaling,
        'connectivity': connectivity,
        'leaking_rate': leaking_rate,
        'max_delay': max_delay,
        'sequence_length': sequence_length
    }
    
    # Plot results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"STM/results/STM_capacity_{timestamp}"
        
        plot_memory_capacity(results, save_path=f"{base_path}_capacity.png")
        
        if plot_delays is None:
            plot_delays = [1, 5, 10, min(20, max_delay)]
        
        plot_delays_available = [d for d in plot_delays if d <= max_delay]
        if plot_delays_available:
            plot_delay_accuracy(
                detailed_results, 
                delay_indices=plot_delays_available,
                save_path=f"{base_path}_delays.png"
            )
            plot_scatter_accuracy(
                detailed_results,
                delay_indices=plot_delays_available[:2],
                save_path=f"{base_path}_scatter.png"
            )
    else:
        plot_memory_capacity(results)
        
        if plot_delays is None:
            plot_delays = [1, 5, 10, min(20, max_delay)]
        
        plot_delays_available = [d for d in plot_delays if d <= max_delay]
        if plot_delays_available:
            plot_delay_accuracy(detailed_results, delay_indices=plot_delays_available)
            plot_scatter_accuracy(detailed_results, delay_indices=plot_delays_available[:2])
    
    return results


def plot_memory_capacity(results: Dict, save_path: Optional[str] = None):
    """
    Plot memory capacity vs delay.
    
    Args:
        results (Dict): Results from evaluate_memory_capacity
        save_path (Optional[str]): Path to save the plot
    """
    # Create figure with improved layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25,
                         left=0.08, right=0.95, top=0.90, bottom=0.12)
    
    memory_capacities = results['memory_capacities']
    delay_indices = results['delay_indices']
    total_capacity = results['total_capacity']
    max_delay = results['max_delay']
    
    # Main capacity plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(delay_indices, memory_capacities, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title(f'Memory Capacity vs Delay (Total: {total_capacity:.2f})', fontsize=14, pad=10)
    ax1.set_xlabel('Delay k')
    ax1.set_ylabel('Memory Capacity')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, len(delay_indices) + 0.5)
    
    # Cumulative capacity
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_capacity = np.cumsum(memory_capacities)
    ax2.plot(delay_indices, cumulative_capacity, 'o-', linewidth=2, markersize=4)
    ax2.set_title('Cumulative Memory Capacity', fontsize=12, pad=10)
    ax2.set_xlabel('Delay k')
    ax2.set_ylabel('Cumulative Capacity')
    ax2.grid(True, alpha=0.3)
    
    # Capacity decay
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(delay_indices, memory_capacities, 'o-', linewidth=2, markersize=4)
    ax3.set_title('Memory Capacity (Log Scale)', fontsize=12, pad=10)
    ax3.set_xlabel('Delay k')
    ax3.set_ylabel('Memory Capacity (log)')
    ax3.grid(True, alpha=0.3)
    
    # Add summary statistics
    fig.text(0.02, 0.02, 
            f'Reservoir Size: {results["reservoir_size"]}, '
            f'Spectral Radius: {results["spectral_radius"]:.2f}, '
            f'Total Capacity: {total_capacity:.2f}/{max_delay} ({total_capacity/max_delay:.1%})',
            fontsize=10, ha='left')
    
    fig.suptitle('Short-Term Memory Capacity Analysis', fontsize=16, y=0.96)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Memory capacity plot saved to: {save_path}")
    
    plt.show()


def compare_esn_parameters(
    parameter_ranges: Dict,
    reservoir_size: int = 100,
    sequence_length: int = 1000,
    max_delay: int = 50,
    train_ratio: float = 0.7,
    random_state: Optional[int] = None,
    device: str = 'cpu',
    save_results: bool = True
) -> Dict:
    """
    Compare ESN performance across different parameter values.
    
    Args:
        parameter_ranges (Dict): Dictionary of parameter names and their ranges
        reservoir_size (int): Base reservoir size
        sequence_length (int): Length of sequences
        max_delay (int): Maximum delay to test
        train_ratio (float): Training data ratio
        random_state (Optional[int]): Random seed
        device (str): Device to use
        save_results (bool): Whether to save results to file
        
    Returns:
        Dict: Comparison results
    """
    print("Comparing ESN parameters")
    print("=" * 50)
    
    results = {}
    
    for param_name, param_values in parameter_ranges.items():
        print(f"\nTesting {param_name}: {param_values}")
        param_results = []
        
        for value in param_values:
            kwargs = {
                'reservoir_size': reservoir_size,
                'sequence_length': sequence_length,
                'max_delay': max_delay,
                'train_ratio': train_ratio,
                'random_state': random_state,
                'device': device,
                'save_results': False
            }
            kwargs[param_name] = value
            
            result = evaluate_memory_capacity(**kwargs)
            param_results.append({
                'value': value,
                'total_capacity': result['total_capacity'],
                'memory_capacities': result['memory_capacities']
            })
        
        results[param_name] = param_results
    
    # Plot comparison
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"STM/results/STM_parameter_comparison_{timestamp}.png"
        plot_parameter_comparison(results, save_path=save_path)
    else:
        plot_parameter_comparison(results)
    
    return results


def plot_parameter_comparison(comparison_results: Dict, save_path: Optional[str] = None):
    """
    Plot comparison of different parameter values.
    
    Args:
        comparison_results (Dict): Results from compare_esn_parameters
        save_path (Optional[str]): Path to save the plot
    """
    n_params = len(comparison_results)
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(6 * n_params, 5))
    gs = fig.add_gridspec(1, n_params, hspace=0.3, wspace=0.25,
                         left=0.08, right=0.95, top=0.85, bottom=0.15)
    
    for i, (param_name, param_data) in enumerate(comparison_results.items()):
        ax = fig.add_subplot(gs[0, i])
        
        values = [data['value'] for data in param_data]
        capacities = [data['total_capacity'] for data in param_data]
        
        ax.plot(values, capacities, 'o-', linewidth=2, markersize=8)
        ax.set_title(f'{param_name.replace("_", " ").title()}', fontsize=12, pad=10)
        ax.set_xlabel(param_name.replace("_", " ").title())
        ax.set_ylabel('Total Memory Capacity')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Parameter Comparison: Memory Capacity', fontsize=14, y=0.92)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Parameter comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run STM task examples."""
    print("Short-Term Memory Task Demonstration")
    print("=" * 50)
    
    # Basic memory capacity evaluation
    print("\n1. Basic memory capacity evaluation")
    results = evaluate_memory_capacity(
        reservoir_size=100,
        spectral_radius=0.95,
        max_delay=50,
        sequence_length=1500,
        random_state=42
    )
    
    # Parameter comparison
    print("\n2. Parameter comparison")
    parameter_ranges = {
        'spectral_radius': [0.5, 0.8, 0.95, 1.2],
        'reservoir_size': [50, 100, 200],
    }
    
    comparison_results = compare_esn_parameters(
        parameter_ranges=parameter_ranges,
        max_delay=30,
        sequence_length=1000,
        random_state=42
    )


if __name__ == "__main__":
    main() 
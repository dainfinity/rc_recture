"""
NARMA (Non-linear Auto-Regressive Moving Average) task implementation for ESN evaluation.

This module implements the NARMA task which is a standard benchmark for testing
the computational capabilities of reservoir computing systems.

The NARMA-m equation is:
y(n+1) = 0.3*y(n) + 0.05*y(n)*sum(y(n-i), i=0 to m-1) + 1.5*u(n-m)*u(n) + 0.1

Where:
- y(n) is the output at time n
- u(n) is the input at time n  
- m is the order of the NARMA system
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


def generate_narma_data(
    sequence_length: int,
    order: int = 10,
    num_sequences: int = 1,
    input_type: str = 'uniform',
    random_state: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate NARMA time series data.
    
    Args:
        sequence_length (int): Length of each sequence
        order (int): Order of the NARMA system (default: 10 for NARMA-10)
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
        inputs = torch.rand(num_sequences, sequence_length, 1) * 0.5  # [0, 0.5]
    elif input_type == 'gaussian':
        inputs = torch.abs(torch.randn(num_sequences, sequence_length, 1)) * 0.2
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # Initialize output
    outputs = torch.zeros(num_sequences, sequence_length, 1)
    
    for seq in range(num_sequences):
        u = inputs[seq, :, 0]  # Input sequence
        y = torch.zeros(sequence_length)  # Output sequence
        
        # Initialize first few values
        for n in range(order):
            y[n] = 0.1  # Small initial value
        
        # Generate NARMA sequence
        for n in range(order, sequence_length - 1):
            # NARMA equation:
            # y(n+1) = 0.3*y(n) + 0.05*y(n)*sum(y(n-i), i=0 to order-1) + 1.5*u(n-order)*u(n) + 0.1
            
            sum_term = torch.sum(y[n-order+1:n+1])  # sum of y(n-i) for i=0 to order-1
            
            # Clip extreme values to prevent instability
            y_term = torch.clamp(y[n], -10, 10)
            sum_term = torch.clamp(sum_term, -50, 50)
            u_term = torch.clamp(u[n-order] * u[n], -10, 10)
            
            y[n+1] = (0.3 * y_term + 
                     0.05 * y_term * sum_term + 
                     1.5 * u_term + 
                     0.1)
            
            # Clamp output to prevent explosion
            y[n+1] = torch.clamp(y[n+1], -20, 20)
        
        outputs[seq, :, 0] = y
    
    return inputs, outputs


def evaluate_narma_task(
    esn: EchoStateNetwork,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float = 0.7,
    washout: int = 100,
    ridge_alpha: float = 1e-6
) -> Dict:
    """
    Evaluate ESN performance on NARMA task.
    
    Args:
        esn (EchoStateNetwork): The ESN model to evaluate
        inputs (torch.Tensor): Input sequences of shape (num_seq, seq_len, 1)
        targets (torch.Tensor): Target sequences of shape (num_seq, seq_len, 1)
        train_ratio (float): Ratio of data to use for training
        washout (int): Number of initial timesteps to ignore
        ridge_alpha (float): Ridge regression regularization parameter
        
    Returns:
        Dict: Evaluation results including RMSE, NMSE, R2, and predictions
    """
    num_sequences, sequence_length, _ = targets.shape
    
    # Apply washout
    if washout > 0:
        inputs_clean = inputs[:, washout:, :]
        targets_clean = targets[:, washout:, :]
        effective_length = sequence_length - washout
    else:
        inputs_clean = inputs
        targets_clean = targets
        effective_length = sequence_length
    
    # Split into train and test
    train_length = int(effective_length * train_ratio)
    
    # Training data
    inputs_train = inputs_clean[:, :train_length, :]
    targets_train = targets_clean[:, :train_length, :]
    
    # Test data
    inputs_test = inputs_clean[:, train_length:, :]
    targets_test = targets_clean[:, train_length:, :]
    
    # Train ESN
    esn.fit(inputs_train, targets_train, washout=0, ridge_alpha=ridge_alpha)
    
    # Predict on test data
    predictions = esn.predict(inputs_test, reset_state=True)
    
    # Flatten for evaluation
    y_true = targets_test.flatten()
    y_pred = predictions.flatten()
    
    # Check for NaN or inf values
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        print("Warning: NaN or inf values detected in predictions. Using fallback values.")
        rmse = float('inf')
        nmse = float('inf')
        r2 = 0.0
        mse = float('inf')
    else:
        # Calculate evaluation metrics
        mse = torch.mean((y_true - y_pred) ** 2).item()
        rmse = np.sqrt(mse)
        
        # NMSE (Normalized Mean Squared Error)
        var_true = torch.var(y_true).item()
        if var_true > 0:
            nmse = mse / var_true
        else:
            nmse = float('inf')
        
        # R² (Coefficient of determination)
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0
    
    results = {
        'rmse': rmse,
        'nmse': nmse,
        'r2': r2,
        'mse': mse,
        'predictions': predictions,
        'targets_test': targets_test,
        'inputs_test': inputs_test,
        'train_length': train_length,
        'test_length': inputs_test.shape[1],
        'washout': washout
    }
    
    return results


def plot_narma_results(
    results: Dict,
    sequence_idx: int = 0,
    max_points: int = 500,
    save_path: Optional[str] = None,
    order: int = 10
):
    """
    Plot NARMA task results with multiple visualizations.
    
    Args:
        results (Dict): Results from evaluate_narma_task
        sequence_idx (int): Which sequence to plot (for multiple sequences)
        max_points (int): Maximum number of points to plot for clarity
        save_path (Optional[str]): Path to save the plot
        order (int): NARMA order for title
    """
    predictions = results['predictions'][sequence_idx, :, 0]
    targets = results['targets_test'][sequence_idx, :, 0]
    
    # Limit points for visualization
    if len(predictions) > max_points:
        step = len(predictions) // max_points
        predictions = predictions[::step]
        targets = targets[::step]
    
    # Create figure with improved layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35, 
                         left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # Time series comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(targets.numpy(), 'b-', label='True values', linewidth=2, alpha=0.8)
    ax1.plot(predictions.detach().numpy(), 'r--', label='Predictions', linewidth=2, alpha=0.8)
    ax1.set_title(f'NARMA-{order} Time Series Prediction Results', fontsize=14, pad=10)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(targets.numpy(), predictions.detach().numpy(), alpha=0.6, s=20)
    min_val = min(targets.min().item(), predictions.min().item())
    max_val = max(targets.max().item(), predictions.max().item())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_title('True vs Predicted', fontsize=12, pad=10)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predicted Values')
    ax2.grid(True, alpha=0.3)
    # Make scatter plot square
    ax2.set_aspect('equal', adjustable='box')
    
    # Error over time
    ax3 = fig.add_subplot(gs[1, :2])
    error = (targets - predictions).abs()
    ax3.plot(error.numpy(), 'g-', linewidth=1.5)
    ax3.set_title('Absolute Error Over Time', fontsize=12, pad=10)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Absolute Error')
    ax3.grid(True, alpha=0.3)
    
    # Metrics display
    ax4 = fig.add_subplot(gs[1, 2])
    metrics_text = f"""Evaluation Metrics

RMSE: {results['rmse']:.6f}
NMSE: {results['nmse']:.6f}
R²: {results['r2']:.6f}
MSE: {results['mse']:.6f}

Data Information:
Train length: {results['train_length']}
Test length: {results['test_length']}
Washout: {results['washout']}"""
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Main title
    fig.suptitle(f'NARMA-{order} Task Results', fontsize=16, y=0.97)
    
    # Save figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def run_narma_experiment(
    order: int = 10,
    reservoir_size: int = 100,
    spectral_radius: float = 0.95,
    input_scaling: float = 1.0,
    connectivity: float = 0.1,
    leaking_rate: float = 1.0,
    sequence_length: int = 2000,
    num_sequences: int = 1,
    train_ratio: float = 0.7,
    washout: int = 100,
    ridge_alpha: float = 1e-6,
    input_type: str = 'uniform',
    random_state: Optional[int] = None,
    device: str = 'cpu',
    plot_results: bool = True,
    save_results: bool = True
) -> Dict:
    """
    Run complete NARMA experiment with specified parameters.
    
    Args:
        order (int): NARMA order (e.g., 10 for NARMA-10)
        reservoir_size (int): Number of reservoir units
        spectral_radius (float): Spectral radius of reservoir
        input_scaling (float): Input scaling factor
        connectivity (float): Reservoir connectivity
        leaking_rate (float): Leaking rate
        sequence_length (int): Length of generated sequences
        num_sequences (int): Number of sequences to generate
        train_ratio (float): Training data ratio
        washout (int): Washout period
        ridge_alpha (float): Ridge regression parameter
        input_type (str): Type of input signal
        random_state (Optional[int]): Random seed
        device (str): Device to use
        plot_results (bool): Whether to plot results
        save_results (bool): Whether to save results to file
        
    Returns:
        Dict: Complete experiment results
    """
    print(f"Running NARMA-{order} task")
    print("=" * 50)
    
    # Generate NARMA data
    print("Generating data...")
    inputs, targets = generate_narma_data(
        sequence_length=sequence_length,
        order=order,
        num_sequences=num_sequences,
        input_type=input_type,
        random_state=random_state
    )
    
    # Create ESN
    print("Initializing ESN...")
    esn = ESNRegressor(
        input_size=1,
        reservoir_size=reservoir_size,
        output_size=1,
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        connectivity=connectivity,
        leaking_rate=leaking_rate,
        random_state=random_state,
        device=device
    )
    
    # Evaluate
    print("Training and evaluating ESN...")
    results = evaluate_narma_task(
        esn=esn,
        inputs=inputs,
        targets=targets,
        train_ratio=train_ratio,
        washout=washout,
        ridge_alpha=ridge_alpha
    )
    
    # Print results
    print(f"\nResults:")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"NMSE: {results['nmse']:.6f}")
    print(f"R²: {results['r2']:.6f}")
    print(f"MSE: {results['mse']:.6f}")
    
    # Generate save path if saving results
    save_path = None
    if save_results and not np.isinf(results['rmse']):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"NARMA/results/NARMA_{order}_order_{timestamp}.png"
    
    # Plot results
    if plot_results and not np.isinf(results['rmse']):
        plot_narma_results(results, save_path=save_path, order=order)
    
    # Add experiment parameters to results
    results.update({
        'order': order,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'input_scaling': input_scaling,
        'connectivity': connectivity,
        'leaking_rate': leaking_rate,
        'sequence_length': sequence_length,
        'input_type': input_type
    })
    
    return results


def compare_narma_orders(
    orders: List[int] = [5, 10, 15, 20],
    reservoir_size: int = 100,
    sequence_length: int = 2000,
    random_state: Optional[int] = None,
    device: str = 'cpu',
    save_results: bool = True
) -> Dict:
    """
    Compare ESN performance across different NARMA orders.
    
    Args:
        orders (List[int]): List of NARMA orders to test
        reservoir_size (int): Number of reservoir units
        sequence_length (int): Length of sequences
        random_state (Optional[int]): Random seed
        device (str): Device to use
        save_results (bool): Whether to save results to file
        
    Returns:
        Dict: Comparison results
    """
    print("Comparing different NARMA orders")
    print("=" * 50)
    
    results = {}
    
    for order in orders:
        print(f"\nRunning NARMA-{order}...")
        result = run_narma_experiment(
            order=order,
            reservoir_size=reservoir_size,
            sequence_length=sequence_length,
            random_state=random_state,
            device=device,
            plot_results=False,
            save_results=False
        )
        results[f'NARMA-{order}'] = result
    
    # Create improved comparison plot
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3,
                         left=0.08, right=0.95, top=0.88, bottom=0.15)
    
    orders_list = [results[f'NARMA-{order}']['order'] for order in orders]
    rmse_list = [results[f'NARMA-{order}']['rmse'] for order in orders]
    nmse_list = [results[f'NARMA-{order}']['nmse'] for order in orders]
    r2_list = [results[f'NARMA-{order}']['r2'] for order in orders]
    
    # Filter out inf values for plotting
    finite_orders = []
    finite_rmse = []
    finite_nmse = []
    finite_r2 = []
    
    for i, (order, rmse, nmse, r2) in enumerate(zip(orders_list, rmse_list, nmse_list, r2_list)):
        if np.isfinite(rmse) and np.isfinite(nmse) and np.isfinite(r2):
            finite_orders.append(order)
            finite_rmse.append(rmse)
            finite_nmse.append(nmse)
            finite_r2.append(r2)
    
    if finite_orders:
        # RMSE plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(finite_orders, finite_rmse, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('RMSE vs NARMA Order', fontsize=12, pad=10)
        ax1.set_xlabel('NARMA Order')
        ax1.set_ylabel('RMSE')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(finite_orders)
        
        # NMSE plot  
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(finite_orders, finite_nmse, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('NMSE vs NARMA Order', fontsize=12, pad=10)
        ax2.set_xlabel('NARMA Order')
        ax2.set_ylabel('NMSE')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(finite_orders)
        
        # R² plot
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(finite_orders, finite_r2, 'go-', linewidth=2, markersize=8)
        ax3.set_title('R² vs NARMA Order', fontsize=12, pad=10)
        ax3.set_xlabel('NARMA Order')
        ax3.set_ylabel('R²')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(finite_orders)
    else:
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle('NARMA Task Performance Comparison', fontsize=14)
    
    # Save comparison plot
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"NARMA/results/NARMA_comparison_{timestamp}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comparison figure saved to: {save_path}")
    
    plt.show()
    
    return results


def main():
    """Main function to run NARMA task examples."""
    print("NARMA Task Demonstration")
    print("=" * 50)
    
    # Basic NARMA-10 experiment
    print("\n1. Basic NARMA-10 experiment")
    results_10 = run_narma_experiment(
        order=10,
        reservoir_size=100,
        sequence_length=2000,
        random_state=42
    )
    
    # Compare different NARMA orders (limited to lower orders for stability)
    print("\n2. Comparing different NARMA orders")
    comparison_results = compare_narma_orders(
        orders=[5, 10],  # Limited to orders that are more stable
        reservoir_size=100,
        sequence_length=1500,
        random_state=42
    )
    
    # NARMA-5 experiment (more stable)
    print("\n3. NARMA-5 experiment (easier task)")
    results_5 = run_narma_experiment(
        order=5,
        reservoir_size=100,
        sequence_length=2000,
        random_state=42
    )


if __name__ == "__main__":
    main() 
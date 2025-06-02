"""
Example scripts demonstrating various ESN applications and evaluations.

This module provides practical examples of how to use the ESN implementation
for different tasks and parameter optimization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from model import EchoStateNetwork, ESNRegressor
from stm_task import (
    evaluate_memory_capacity, 
    compare_esn_parameters, 
    plot_parameter_comparison,
    plot_delay_accuracy,
    plot_scatter_accuracy
)


def example_simple_prediction():
    """Simple example: Predict delayed sine wave."""
    print("Example 1: Simple Prediction Task")
    print("-" * 40)
    
    # Generate sine wave data
    t = torch.linspace(0, 10 * np.pi, 1000).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t)
    y = torch.sin(t + 0.5)  # Delayed sine wave
    
    # Split into train/test
    train_len = 800
    x_train, x_test = x[:, :train_len, :], x[:, train_len:, :]
    y_train, y_test = y[:, :train_len, :], y[:, train_len:, :]
    
    # Create and train ESN
    esn = ESNRegressor(
        input_size=1,
        reservoir_size=100,
        output_size=1,
        spectral_radius=0.95,
        random_state=42
    )
    
    esn.fit(x_train, y_train, washout=50)
    
    # Predict
    y_pred = esn.predict(x_test)
    
    # Calculate error
    mse = esn.score(x_test, y_test, metric='mse')
    print(f"Test MSE: {mse:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t[0, :train_len, 0], x_train[0, :, 0], 'b-', label='Input')
    plt.plot(t[0, :train_len, 0], y_train[0, :, 0], 'r-', label='Target')
    plt.title('Training Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(t[0, train_len:, 0], y_test[0, :, 0], 'r-', label='True', linewidth=2)
    plt.plot(t[0, train_len:, 0], y_pred[0, :, 0], 'g--', label='Predicted', linewidth=2)
    plt.title(f'Test Prediction (MSE: {mse:.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return esn


def example_mackey_glass():
    """Example: Mackey-Glass time series prediction."""
    print("\nExample 2: Mackey-Glass Time Series")
    print("-" * 40)
    
    def mackey_glass(length=2000, tau=17, n=10, beta=0.2, gamma=0.1, 
                     initial=1.2, dt=1.0, discard=500):
        """Generate Mackey-Glass time series."""
        # History initialization
        history_length = int(tau / dt)
        history = [initial] * history_length
        
        # Generate series
        series = []
        for _ in range(length + discard):
            delayed_value = history[-history_length] if len(history) >= history_length else initial
            new_value = history[-1] + dt * (
                beta * delayed_value / (1 + delayed_value ** n) - gamma * history[-1]
            )
            history.append(new_value)
            if len(history) > discard:
                series.append(new_value)
        
        return np.array(series[:length])
    
    # Generate Mackey-Glass data
    data = mackey_glass(length=2000)
    data = (data - np.mean(data)) / np.std(data)  # Normalize
    
    # Create input-output pairs
    delay = 1
    x = torch.tensor(data[:-delay], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    y = torch.tensor(data[delay:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    # Split data
    train_len = 1500
    x_train, x_test = x[:, :train_len, :], x[:, train_len:, :]
    y_train, y_test = y[:, :train_len, :], y[:, train_len:, :]
    
    # Create and train ESN
    esn = ESNRegressor(
        input_size=1,
        reservoir_size=200,
        output_size=1,
        spectral_radius=0.99,
        connectivity=0.1,
        random_state=42
    )
    
    esn.fit(x_train, y_train, washout=100)
    
    # Multi-step prediction
    prediction_steps = 200
    x_pred = x_test[:, :1, :]  # Start with first test input
    predictions = []
    
    for _ in range(prediction_steps):
        y_pred = esn.predict(x_pred, reset_state=False)
        predictions.append(y_pred[:, -1:, :])
        x_pred = y_pred[:, -1:, :]  # Use prediction as next input
    
    predictions = torch.cat(predictions, dim=1)
    
    # Calculate errors
    one_step_pred = esn.predict(x_test)
    one_step_mse = esn.score(x_test, y_test, metric='mse')
    multi_step_mse = torch.mean((y_test[:, :prediction_steps, :] - predictions) ** 2).item()
    
    print(f"One-step MSE: {one_step_mse:.6f}")
    print(f"Multi-step MSE ({prediction_steps} steps): {multi_step_mse:.6f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(data[:500])
    plt.title('Mackey-Glass Time Series (Sample)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    test_range = range(len(y_test[0, :100, 0]))
    plt.plot(test_range, y_test[0, :100, 0], 'b-', label='True', linewidth=2)
    plt.plot(test_range, one_step_pred[0, :100, 0], 'r--', label='One-step pred', linewidth=2)
    plt.title(f'One-step Prediction (MSE: {one_step_mse:.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    pred_range = range(prediction_steps)
    plt.plot(pred_range, y_test[0, :prediction_steps, 0], 'b-', label='True', linewidth=2)
    plt.plot(pred_range, predictions[0, :, 0], 'g--', label='Multi-step pred', linewidth=2)
    plt.title(f'Multi-step Prediction (MSE: {multi_step_mse:.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return esn


def example_memory_capacity_analysis():
    """Example: Comprehensive memory capacity analysis."""
    print("\nExample 3: Memory Capacity Analysis")
    print("-" * 40)
    
    # Single ESN evaluation with detailed visualization
    print("Evaluating single ESN configuration...")
    results = evaluate_memory_capacity(
        reservoir_size=100,
        spectral_radius=0.95,
        sequence_length=1000,      # Reduced from 2000
        max_delay=80,              # Reduced from 120
        train_ratio=0.7,
        random_state=42,
        plot_delays=[1, 5, 10]     # Reduced from [1, 5, 10, 20]
    )
    
    # Multi-parameter comparison
    print("\nComparing multiple parameters...")
    parameter_ranges = {
        'spectral_radius': [0.1, 0.5, 0.9, 0.95, 0.99, 1.05],
        'connectivity': [0.05, 0.1, 0.2, 0.3, 0.5],
        'input_scaling': [0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    comparison_results = compare_esn_parameters(
        parameter_ranges=parameter_ranges,
        reservoir_size=80,
        sequence_length=800,       # Reduced from 1500
        max_delay=40,              # Reduced from 60
        train_ratio=0.7,
        random_state=42
    )
    
    # Plot comparison
    plot_parameter_comparison(comparison_results)
    
    return results, comparison_results


def example_detailed_delay_analysis():
    """Example: Detailed analysis of different delays."""
    print("\nExample 4: Detailed Delay Analysis")
    print("-" * 40)
    
    # Evaluate with multiple ESN configurations
    configs = [
        {'spectral_radius': 0.9, 'connectivity': 0.1, 'name': 'Conservative'},
        {'spectral_radius': 0.95, 'connectivity': 0.1, 'name': 'Balanced'},
        {'spectral_radius': 0.99, 'connectivity': 0.2, 'name': 'Aggressive'}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\nEvaluating {config['name']} configuration...")
        results = evaluate_memory_capacity(
            reservoir_size=100,
            spectral_radius=config['spectral_radius'],
            connectivity=config['connectivity'],
            sequence_length=800,       # Reduced from 1500
            max_delay=30,              # Reduced from 50
            train_ratio=0.7,
            random_state=42
        )
        
        all_results.append({**results, 'config_name': config['name']})
        
        # Detailed visualization for specific delays
        print(f"Visualizing delays for {config['name']} configuration...")
        plot_delay_accuracy(
            results['detailed_results'], 
            delay_indices=[1, 5, 10], 
            max_points=150             # Reduced from 300
        )
    
    # Compare configurations
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for result in all_results:
        plt.plot(result['delay_indices'], result['capacities_per_delay'], 
                'o-', label=result['config_name'], linewidth=2, markersize=4)
    plt.xlabel('Delay (k)')
    plt.ylabel('Memory Capacity MC(k)')
    plt.title('Memory Capacity per Delay (Different Configurations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    config_names = [r['config_name'] for r in all_results]
    total_capacities = [r['total_capacity'] for r in all_results]
    capacity_ratios = [r['capacity_ratio'] for r in all_results]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    plt.bar(x - width/2, total_capacities, width, label='Total Capacity', alpha=0.8)
    plt.bar(x + width/2, [r * 100 for r in capacity_ratios], width, 
            label='Capacity Ratio (×100)', alpha=0.8)
    
    plt.xlabel('Configuration')
    plt.ylabel('Capacity')
    plt.title('Total Memory Capacity Comparison')
    plt.xticks(x, config_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return all_results


def example_train_validation_split_analysis():
    """Example: Analyze the effect of train/validation split ratio."""
    print("\nExample 5: Train/Validation Split Analysis")
    print("-" * 40)
    
    train_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    capacities = []
    capacity_ratios = []
    
    for ratio in train_ratios:
        print(f"Testing train ratio: {ratio:.1f}")
        result = evaluate_memory_capacity(
            reservoir_size=100,
            spectral_radius=0.95,
            sequence_length=800,       # Reduced from 1500
            max_delay=50,              # Reduced from 80
            train_ratio=ratio,
            random_state=42
        )
        capacities.append(result['total_capacity'])
        capacity_ratios.append(result['capacity_ratio'])
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_ratios, capacities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Train Ratio')
    plt.ylabel('Total Memory Capacity')
    plt.title('Memory Capacity vs Train/Validation Split')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_ratios, capacity_ratios, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Train Ratio')
    plt.ylabel('Capacity Ratio (MC/N)')
    plt.title('Capacity Efficiency vs Train/Validation Split')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return train_ratios, capacities, capacity_ratios


def example_reservoir_size_scaling():
    """Example: How memory capacity scales with reservoir size."""
    print("\nExample 6: Reservoir Size Scaling")
    print("-" * 40)
    
    reservoir_sizes = [20, 50, 100, 150, 200, 300]
    capacities = []
    capacity_ratios = []
    
    for size in reservoir_sizes:
        print(f"Testing reservoir size: {size}")
        result = evaluate_memory_capacity(
            reservoir_size=size,
            spectral_radius=0.95,
            sequence_length=800,               # Reduced from 1500
            max_delay=min(size//2 + 10, 80),   # Adjusted scaling
            train_ratio=0.7,
            random_state=42
        )
        capacities.append(result['total_capacity'])
        capacity_ratios.append(result['capacity_ratio'])
    
    # Plot scaling results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(reservoir_sizes, capacities, 'bo-', linewidth=2, markersize=8)
    plt.plot(reservoir_sizes, reservoir_sizes, 'r--', alpha=0.7, label='Theoretical maximum')
    plt.xlabel('Reservoir Size')
    plt.ylabel('Total Memory Capacity')
    plt.title('Memory Capacity vs Reservoir Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(reservoir_sizes, capacity_ratios, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Reservoir Size')
    plt.ylabel('Capacity Ratio (MC/N)')
    plt.title('Capacity Efficiency vs Reservoir Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return reservoir_sizes, capacities, capacity_ratios


def benchmark_performance():
    """Benchmark ESN performance."""
    print("\nBenchmark: Performance Analysis")
    print("-" * 40)
    
    reservoir_sizes = [50, 100, 200, 500]
    sequence_lengths = [500, 1000, 2000]
    
    results = {}
    
    for res_size in reservoir_sizes:
        results[res_size] = {}
        for seq_len in sequence_lengths:
            print(f"Benchmarking: Reservoir={res_size}, Sequence={seq_len}")
            
            # Generate data
            x = torch.randn(1, seq_len, 1)
            y = torch.randn(1, seq_len, 1)
            
            # Create ESN
            esn = ESNRegressor(
                input_size=1,
                reservoir_size=res_size,
                output_size=1,
                random_state=42
            )
            
            # Time training
            start_time = time.time()
            esn.fit(x, y, washout=50)
            train_time = time.time() - start_time
            
            # Time prediction
            start_time = time.time()
            _ = esn.predict(x)
            pred_time = time.time() - start_time
            
            results[res_size][seq_len] = {
                'train_time': train_time,
                'pred_time': pred_time
            }
            
            print(f"  Train time: {train_time:.4f}s, Pred time: {pred_time:.4f}s")
    
    # Plot benchmark results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for seq_len in sequence_lengths:
        train_times = [results[res_size][seq_len]['train_time'] for res_size in reservoir_sizes]
        plt.plot(reservoir_sizes, train_times, 'o-', label=f'Seq len: {seq_len}', linewidth=2)
    plt.xlabel('Reservoir Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Reservoir Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for seq_len in sequence_lengths:
        pred_times = [results[res_size][seq_len]['pred_time'] for res_size in reservoir_sizes]
        plt.plot(reservoir_sizes, pred_times, 's-', label=f'Seq len: {seq_len}', linewidth=2)
    plt.xlabel('Reservoir Size')
    plt.ylabel('Prediction Time (s)')
    plt.title('Prediction Time vs Reservoir Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Run all examples."""
    print("Echo State Network Examples")
    print("=" * 50)
    
    # Run examples
    example_simple_prediction()
    example_mackey_glass()
    example_memory_capacity_analysis()
    example_detailed_delay_analysis()
    example_train_validation_split_analysis()
    example_reservoir_size_scaling()
    benchmark_performance()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main() 
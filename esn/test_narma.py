"""
Test script for NARMA task implementation.
This script demonstrates the NARMA task with visualization.
"""

from narma_task import run_narma_experiment, compare_narma_orders
import matplotlib.pyplot as plt

def test_single_narma():
    """Test a single NARMA-10 experiment with visualization."""
    print("Testing single NARMA-10 experiment...")
    results = run_narma_experiment(
        order=10,
        reservoir_size=100,
        spectral_radius=0.95,
        sequence_length=1500,
        train_ratio=0.7,
        washout=50,
        random_state=42,
        plot_results=True
    )
    return results

def test_narma_comparison():
    """Test comparison of different NARMA orders."""
    print("Testing NARMA order comparison...")
    results = compare_narma_orders(
        orders=[3, 5, 8, 10],
        reservoir_size=100,
        sequence_length=1200,
        random_state=42
    )
    return results

def test_easy_narma():
    """Test an easier NARMA-5 experiment."""
    print("Testing NARMA-5 experiment...")
    results = run_narma_experiment(
        order=5,
        reservoir_size=80,
        spectral_radius=0.9,
        sequence_length=1500,
        train_ratio=0.8,
        washout=50,
        random_state=42,
        plot_results=True
    )
    return results

if __name__ == "__main__":
    print("NARMA Task Testing")
    print("=" * 50)
    
    # Test single NARMA experiment
    print("\n1. Single NARMA-10 Test:")
    results_10 = test_single_narma()
    
    # Test NARMA comparison
    print("\n2. NARMA Order Comparison Test:")
    comparison_results = test_narma_comparison()
    
    # Test easier NARMA
    print("\n3. NARMA-5 Test:")
    results_5 = test_easy_narma()
    
    print("\nAll tests completed successfully!")
    print(f"NARMA-10 RMSE: {results_10['rmse']:.6f}, R²: {results_10['r2']:.6f}")
    print(f"NARMA-5 RMSE: {results_5['rmse']:.6f}, R²: {results_5['r2']:.6f}") 
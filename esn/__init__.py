"""
Echo State Network (ESN) Package

A comprehensive implementation of Echo State Networks using PyTorch,
including various tasks for evaluation and analysis.

Main Components:
- EchoStateNetwork: Core ESN implementation
- ESNRegressor: ESN for regression tasks
- ESNClassifier: ESN for classification tasks
- Short Term Memory task for memory capacity evaluation

Example usage:
    from esn import EchoStateNetwork, evaluate_memory_capacity
    
    # Create an ESN
    esn = EchoStateNetwork(
        input_size=1,
        reservoir_size=100,
        output_size=1,
        spectral_radius=0.95
    )
    
    # Evaluate memory capacity
    results = evaluate_memory_capacity(reservoir_size=100)
"""

__version__ = "1.0.0"
__author__ = "ESN Implementation Team"

# Import main classes and functions
from .model import (
    EchoStateNetwork,
    ESNRegressor,
    ESNClassifier
)

from .stm_task import (
    generate_stm_data,
    compute_memory_capacity,
    evaluate_memory_capacity,
    plot_memory_capacity,
    compare_esn_parameters,
    plot_parameter_comparison,
    plot_delay_accuracy,
    plot_scatter_accuracy
)

__all__ = [
    # Core ESN classes
    'EchoStateNetwork',
    'ESNRegressor', 
    'ESNClassifier',
    
    # STM task functions
    'generate_stm_data',
    'compute_memory_capacity',
    'evaluate_memory_capacity',
    'plot_memory_capacity',
    'compare_esn_parameters',
    'plot_parameter_comparison',
    'plot_delay_accuracy',
    'plot_scatter_accuracy'
] 
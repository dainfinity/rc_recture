"""
Echo State Network implementation using PyTorch.

This module provides a flexible and extensible implementation of Echo State Networks
with customizable reservoir properties and training methods.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import warnings


class EchoStateNetwork(nn.Module):
    """
    Echo State Network (ESN) implementation.
    
    An ESN consists of a sparsely connected reservoir of recurrent units
    and a linear readout layer that is trained using linear regression.
    
    Args:
        input_size (int): Number of input features
        reservoir_size (int): Number of reservoir units
        output_size (int): Number of output features
        spectral_radius (float): Spectral radius of the reservoir weight matrix
        input_scaling (float): Scaling factor for input weights
        connectivity (float): Connectivity rate of the reservoir (0-1)
        leaking_rate (float): Leaking rate for reservoir update (0-1)
        bias_scaling (float): Scaling factor for bias weights
        noise_level (float): Noise level added to reservoir states
        random_state (Optional[int]): Random seed for reproducibility
        device (str): Device to run the model on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        spectral_radius: float = 0.95,
        input_scaling: float = 1.0,
        connectivity: float = 0.1,
        leaking_rate: float = 1.0,
        bias_scaling: float = 0.0,
        noise_level: float = 0.0,
        random_state: Optional[int] = None,
        device: str = 'cpu'
    ):
        super(EchoStateNetwork, self).__init__()
        
        # Store hyperparameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.connectivity = connectivity
        self.leaking_rate = leaking_rate
        self.bias_scaling = bias_scaling
        self.noise_level = noise_level
        self.device = device
        
        # Set random state for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Initialize weights
        self._initialize_weights()
        
        # Reservoir state
        self.reservoir_state = None
        
        # Readout weights (to be trained)
        self.W_out = None
        
        # Move to device
        self.to(device)
    
    def _initialize_weights(self):
        """Initialize reservoir and input weights."""
        # Input weights (W_in)
        self.W_in = torch.empty(self.reservoir_size, self.input_size + 1, device=self.device)
        nn.init.uniform_(self.W_in, -self.input_scaling, self.input_scaling)
        
        # Bias weights
        if self.bias_scaling > 0:
            self.W_in[:, -1] = torch.empty(self.reservoir_size, device=self.device).uniform_(
                -self.bias_scaling, self.bias_scaling
            )
        else:
            self.W_in[:, -1] = 0
        
        # Reservoir weights (W_res)
        self.W_res = self._create_reservoir_matrix()
    
    def _create_reservoir_matrix(self) -> torch.Tensor:
        """Create sparsely connected reservoir weight matrix with desired spectral radius."""
        # Create sparse random matrix
        W_res = torch.zeros(self.reservoir_size, self.reservoir_size, device=self.device)
        
        # Number of connections
        n_connections = int(self.connectivity * self.reservoir_size ** 2)
        
        # Random connections
        for _ in range(n_connections):
            i = np.random.randint(0, self.reservoir_size)
            j = np.random.randint(0, self.reservoir_size)
            W_res[i, j] = np.random.uniform(-1, 1)
        
        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W_res)
        current_spectral_radius = torch.max(torch.abs(eigenvalues)).real
        
        if current_spectral_radius > 0:
            W_res = W_res * (self.spectral_radius / current_spectral_radius)
        
        return W_res
    
    def reset_state(self, batch_size: int = 1):
        """Reset reservoir state to zero."""
        self.reservoir_state = torch.zeros(
            batch_size, self.reservoir_size, device=self.device
        )
    
    def forward(self, input_data: torch.Tensor, reset_state: bool = True) -> torch.Tensor:
        """
        Forward pass through the ESN.
        
        Args:
            input_data (torch.Tensor): Input data of shape (batch_size, seq_len, input_size)
            reset_state (bool): Whether to reset reservoir state before processing
            
        Returns:
            torch.Tensor: Reservoir states of shape (batch_size, seq_len, reservoir_size)
        """
        batch_size, seq_len, _ = input_data.shape
        
        if reset_state or self.reservoir_state is None:
            self.reset_state(batch_size)
        
        # Store all reservoir states
        all_states = torch.zeros(
            batch_size, seq_len, self.reservoir_size, device=self.device
        )
        
        for t in range(seq_len):
            # Add bias term
            input_with_bias = torch.cat([
                input_data[:, t, :],
                torch.ones(batch_size, 1, device=self.device)
            ], dim=1)
            
            # Update reservoir state
            input_activation = torch.matmul(input_with_bias, self.W_in.T)
            reservoir_activation = torch.matmul(self.reservoir_state, self.W_res.T)
            
            # Add noise if specified
            if self.noise_level > 0:
                noise = torch.normal(
                    0, self.noise_level,
                    size=self.reservoir_state.shape,
                    device=self.device
                )
                reservoir_activation += noise
            
            # Apply leaking rate and activation function
            new_state = torch.tanh(input_activation + reservoir_activation)
            self.reservoir_state = (
                (1 - self.leaking_rate) * self.reservoir_state +
                self.leaking_rate * new_state
            )
            
            all_states[:, t, :] = self.reservoir_state
        
        return all_states
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        washout: int = 0,
        ridge_alpha: float = 1e-6
    ) -> 'EchoStateNetwork':
        """
        Train the ESN readout layer using ridge regression.
        
        Args:
            X (torch.Tensor): Input data of shape (batch_size, seq_len, input_size)
            y (torch.Tensor): Target data of shape (batch_size, seq_len, output_size)
            washout (int): Number of initial timesteps to discard
            ridge_alpha (float): Ridge regression regularization parameter
            
        Returns:
            EchoStateNetwork: Self for method chaining
        """
        # Get reservoir states
        states = self.forward(X)
        
        # Apply washout
        if washout > 0:
            states = states[:, washout:, :]
            y_train = y[:, washout:, :]
        else:
            y_train = y
        
        # Reshape for linear regression
        batch_size, seq_len, _ = states.shape
        X_train = states.reshape(-1, self.reservoir_size)
        y_train = y_train.reshape(-1, self.output_size)
        
        # Add bias term
        X_train_with_bias = torch.cat([
            X_train,
            torch.ones(X_train.shape[0], 1, device=self.device)
        ], dim=1)
        
        # Ridge regression
        A = X_train_with_bias.T @ X_train_with_bias
        A.diagonal().add_(ridge_alpha)
        b = X_train_with_bias.T @ y_train
        
        # Solve linear system
        self.W_out = torch.linalg.solve(A, b).T
        
        return self
    
    def predict(
        self,
        X: torch.Tensor,
        reset_state: bool = True
    ) -> torch.Tensor:
        """
        Make predictions using the trained ESN.
        
        Args:
            X (torch.Tensor): Input data of shape (batch_size, seq_len, input_size)
            reset_state (bool): Whether to reset reservoir state before prediction
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, seq_len, output_size)
        """
        if self.W_out is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        # Get reservoir states
        states = self.forward(X, reset_state=reset_state)
        
        # Add bias term
        batch_size, seq_len, _ = states.shape
        states_with_bias = torch.cat([
            states,
            torch.ones(batch_size, seq_len, 1, device=self.device)
        ], dim=-1)
        
        # Compute output
        predictions = torch.matmul(states_with_bias, self.W_out.T)
        
        return predictions
    
    def get_reservoir_states(
        self,
        X: torch.Tensor,
        reset_state: bool = True
    ) -> torch.Tensor:
        """Get reservoir states for given input."""
        return self.forward(X, reset_state=reset_state)
    
    def get_spectral_radius(self) -> float:
        """Get actual spectral radius of the reservoir matrix."""
        eigenvalues = torch.linalg.eigvals(self.W_res)
        return torch.max(torch.abs(eigenvalues)).real.item()
    
    def get_info(self) -> dict:
        """Get information about the ESN configuration."""
        return {
            'input_size': self.input_size,
            'reservoir_size': self.reservoir_size,
            'output_size': self.output_size,
            'spectral_radius': self.spectral_radius,
            'actual_spectral_radius': self.get_spectral_radius(),
            'input_scaling': self.input_scaling,
            'connectivity': self.connectivity,
            'leaking_rate': self.leaking_rate,
            'bias_scaling': self.bias_scaling,
            'noise_level': self.noise_level,
            'device': self.device,
            'trained': self.W_out is not None
        }


class ESNRegressor(EchoStateNetwork):
    """ESN for regression tasks with additional utilities."""
    
    def score(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        metric: str = 'mse'
    ) -> float:
        """
        Calculate performance score.
        
        Args:
            X (torch.Tensor): Input data
            y (torch.Tensor): True targets
            metric (str): Metric to use ('mse', 'mae', 'r2')
            
        Returns:
            float: Performance score
        """
        y_pred = self.predict(X)
        
        if metric == 'mse':
            return torch.mean((y - y_pred) ** 2).item()
        elif metric == 'mae':
            return torch.mean(torch.abs(y - y_pred)).item()
        elif metric == 'r2':
            ss_res = torch.sum((y - y_pred) ** 2)
            ss_tot = torch.sum((y - torch.mean(y)) ** 2)
            return (1 - ss_res / ss_tot).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")


class ESNClassifier(EchoStateNetwork):
    """ESN for classification tasks."""
    
    def predict_proba(
        self,
        X: torch.Tensor,
        reset_state: bool = True
    ) -> torch.Tensor:
        """Predict class probabilities using softmax."""
        logits = self.predict(X, reset_state=reset_state)
        return torch.softmax(logits, dim=-1)
    
    def predict_classes(
        self,
        X: torch.Tensor,
        reset_state: bool = True
    ) -> torch.Tensor:
        """Predict class labels."""
        probs = self.predict_proba(X, reset_state=reset_state)
        return torch.argmax(probs, dim=-1) 
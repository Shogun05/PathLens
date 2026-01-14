#!/usr/bin/env python3
"""
PNMLR Neural Utility Model.

A Multi-Layer Perceptron that learns to predict accessibility utility
given node features and user profile context.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PNMLRModel:
    """
    Personalized Neuro Mixed Logit Regression Model.
    
    A neural network that predicts accessibility utility scores
    given node features, conditioned on user preference profiles.
    
    Architecture:
        Input: [node_features (F) | profile_weights (A)]
        Hidden: 2 layers with ReLU activation
        Output: utility score (scalar)
    
    Where F = number of node features, A = number of amenity types
    """
    
    def __init__(
        self,
        n_node_features: int,
        n_amenity_types: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        learning_rate: float = 0.001,
        seed: int = 42,
    ):
        self.n_node_features = n_node_features
        self.n_amenity_types = n_amenity_types
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.seed = seed
        
        # Total input dimension: node features + profile weights
        self.input_dim = n_node_features + n_amenity_types
        
        # Initialize weights
        np.random.seed(seed)
        self._init_weights()
        
        self._trained = False
        self.training_history: List[Dict] = []
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        self.weights = []
        self.biases = []
        
        dims = [self.input_dim] + list(self.hidden_dims) + [1]
        
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative for backprop."""
        return (x > 0).astype(float)
    
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass returning activations and pre-activations.
        
        Args:
            X: (N, input_dim) input matrix
        
        Returns:
            Tuple of (activations, pre_activations) for backprop
        """
        activations = [X]
        pre_activations = []
        
        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            pre_activations.append(z)
            
            # ReLU for hidden layers, linear for output
            if i < len(self.weights) - 1:
                current = self._relu(z)
            else:
                current = z
            activations.append(current)
        
        return activations, pre_activations
    
    def predict(
        self,
        node_features: np.ndarray,
        profile_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Predict utility scores for nodes given a user profile.
        
        Args:
            node_features: (N, F) normalized node features
            profile_weights: (A,) or (N, A) user profile weights
        
        Returns:
            (N,) utility scores
        """
        N = node_features.shape[0]
        
        # Broadcast profile to all nodes if single profile
        if profile_weights.ndim == 1:
            profile_weights = np.tile(profile_weights, (N, 1))
        
        # Concatenate node features with profile
        X = np.concatenate([node_features, profile_weights], axis=1)
        
        # Forward pass
        activations, _ = self._forward(X)
        
        return activations[-1].flatten()
    
    def predict_multi_profile(
        self,
        node_features: np.ndarray,
        profiles: np.ndarray,
    ) -> np.ndarray:
        """
        Predict utilities for all nodes across all profiles.
        
        Args:
            node_features: (N, F) normalized node features
            profiles: (P, A) array of P user profiles
        
        Returns:
            (N, P) matrix of utility scores
        """
        N = node_features.shape[0]
        P = profiles.shape[0]
        
        utilities = np.zeros((N, P))
        for p in range(P):
            utilities[:, p] = self.predict(node_features, profiles[p])
        
        return utilities
    
    def predict_average_utility(
        self,
        node_features: np.ndarray,
        profiles: np.ndarray,
    ) -> np.ndarray:
        """
        Predict average utility across all profiles.
        
        This is used during optimization to evaluate placements
        that benefit diverse population groups.
        
        Args:
            node_features: (N, F) normalized node features
            profiles: (P, A) array of P user profiles
        
        Returns:
            (N,) average utility scores
        """
        utilities = self.predict_multi_profile(node_features, profiles)
        return utilities.mean(axis=1)
    
    def fit(
        self,
        node_features: np.ndarray,
        profile_weights: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> 'PNMLRModel':
        """
        Train the model using gradient descent.
        
        Args:
            node_features: (N, F) normalized node features
            profile_weights: (N, A) profile weights for each sample
            targets: (N,) target utility values
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Whether to log progress
        
        Returns:
            self for chaining
        """
        N = node_features.shape[0]
        X = np.concatenate([node_features, profile_weights], axis=1)
        y = targets.reshape(-1, 1)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations, pre_activations = self._forward(X_batch)
                
                # Compute loss (MSE)
                predictions = activations[-1]
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # Backpropagation
                batch_size_actual = X_batch.shape[0]
                
                # Output layer gradient
                delta = 2 * (predictions - y_batch) / batch_size_actual
                
                # Backprop through layers
                gradients_w = []
                gradients_b = []
                
                for i in range(len(self.weights) - 1, -1, -1):
                    # Gradient for weights and biases
                    grad_w = activations[i].T @ delta
                    grad_b = np.sum(delta, axis=0)
                    gradients_w.insert(0, grad_w)
                    gradients_b.insert(0, grad_b)
                    
                    if i > 0:
                        # Backprop through ReLU
                        delta = (delta @ self.weights[i].T) * self._relu_derivative(pre_activations[i-1])
                
                # Update weights
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * gradients_w[i]
                    self.biases[i] -= self.learning_rate * gradients_b[i]
            
            epoch_loss /= n_batches
            self.training_history.append({'epoch': epoch, 'loss': epoch_loss})
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        self._trained = True
        return self
    
    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        state = {
            'n_node_features': self.n_node_features,
            'n_amenity_types': self.n_amenity_types,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'weights': self.weights,
            'biases': self.biases,
            'trained': self._trained,
            'training_history': self.training_history,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved PNMLR model to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'PNMLRModel':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(
            n_node_features=state['n_node_features'],
            n_amenity_types=state['n_amenity_types'],
            hidden_dims=state['hidden_dims'],
            learning_rate=state['learning_rate'],
            seed=state['seed'],
        )
        model.weights = state['weights']
        model.biases = state['biases']
        model._trained = state['trained']
        model.training_history = state.get('training_history', [])
        
        logger.info(f"Loaded PNMLR model from {path}")
        return model


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Fake data
    N = 1000
    F = 10  # node features
    A = 7   # amenity types
    
    node_features = np.random.randn(N, F)
    profiles = np.random.dirichlet(np.ones(A), size=N)
    targets = np.random.rand(N)  # Random utilities for testing
    
    model = PNMLRModel(n_node_features=F, n_amenity_types=A)
    model.fit(node_features, profiles, targets, epochs=20, verbose=True)
    
    # Test prediction
    test_profile = np.random.dirichlet(np.ones(A))
    utilities = model.predict(node_features[:10], test_profile)
    print(f"Sample predictions: {utilities}")

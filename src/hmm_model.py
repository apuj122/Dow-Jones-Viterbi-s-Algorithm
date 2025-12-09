"""
Hidden Markov Model implementation for regime detection.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional

class GaussianHMM:
    """Simple Gaussian Hidden Markov Model."""
    
    def __init__(self, n_states: int = 3, n_iter: int = 100, 
                 random_state: Optional[int] = None):
        """
        Initialize Gaussian HMM.
        
        Args:
            n_states: Number of hidden states
            n_iter: Number of Baum-Welch iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Model parameters
        self.pi = None  # Initial state probabilities
        self.A = None   # Transition matrix
        self.mu = None  # Emission means
        self.sigma = None  # Emission standard deviations
        
    def initialize_parameters(self, observations: np.ndarray):
        """
        Initialize model parameters.
        
        Args:
            observations: Observation sequence
        """
        n = len(observations)
        
        # Uniform initial state probabilities
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Random transition matrix
        self.A = np.random.dirichlet(np.ones(self.n_states), self.n_states)
        
        # Initialize means by dividing data into regions
        sorted_obs = np.sort(observations)
        self.mu = np.array([
            sorted_obs[int(i * n / self.n_states)] 
            for i in range(self.n_states)
        ])
        
        # Initialize standard deviations
        self.sigma = np.ones(self.n_states) * np.std(observations) / 2
        self.sigma = np.maximum(self.sigma, 0.01)  # Avoid zero variance
        
        print(f"Initialized HMM with {self.n_states} states")
        print(f"Initial means: {self.mu}")
        print(f"Initial sigmas: {self.sigma}")
        
    def emission_prob(self, obs: float, state: int) -> float:
        """
        Calculate emission probability P(obs | state).
        
        Args:
            obs: Observation value
            state: Hidden state index
            
        Returns:
            Probability density
        """
        return norm.pdf(obs, self.mu[state], self.sigma[state])
    
    def forward_algorithm(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm for computing forward probabilities.
        
        Args:
            observations: Observation sequence
            
        Returns:
            Tuple of (forward_probabilities, scaling_factors)
        """
        n = len(observations)
        alpha = np.zeros((n, self.n_states))
        scale = np.zeros(n)
        
        # Initialize
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self.emission_prob(observations[0], i)
        
        scale[0] = np.sum(alpha[0, :])
        alpha[0, :] /= scale[0]
        
        # Forward pass
        for t in range(1, n):
            for j in range(self.n_states):
                alpha[t, j] = self.emission_prob(observations[t], j) * \
                              np.sum(alpha[t-1, :] * self.A[:, j])
            
            scale[t] = np.sum(alpha[t, :])
            if scale[t] > 0:
                alpha[t, :] /= scale[t]
            else:
                scale[t] = 1e-10
        
        return alpha, scale
    
    def backward_algorithm(self, observations: np.ndarray, 
                          scale: np.ndarray) -> np.ndarray:
        """
        Backward algorithm for computing backward probabilities.
        
        Args:
            observations: Observation sequence
            scale: Scaling factors from forward algorithm
            
        Returns:
            Backward probabilities
        """
        n = len(observations)
        beta = np.zeros((n, self.n_states))
        
        # Initialize
        beta[n-1, :] = 1.0 / scale[n-1]
        
        # Backward pass
        for t in range(n-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i, :] * self.emission_prob(observations[t+1], np.arange(self.n_states)) * beta[t+1, :]
                ) / scale[t]
        
        return beta
    
    def baum_welch(self, observations: np.ndarray):
        """
        Baum-Welch algorithm for training HMM parameters.
        
        Args:
            observations: Observation sequence
        """
        self.initialize_parameters(observations)
        n = len(observations)
        
        for iteration in range(self.n_iter):
            # E-step: Forward-backward algorithm
            alpha, scale = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations, scale)
            
            # Compute posteriors
            gamma = alpha * beta
            gamma_sum = np.sum(gamma, axis=1, keepdims=True)
            gamma_sum = np.where(gamma_sum == 0, 1e-10, gamma_sum)
            gamma /= gamma_sum
            
            # M-step: Update parameters
            # Update initial state probabilities
            self.pi = np.clip(gamma[0, :], 1e-10, 1.0)
            self.pi /= np.sum(self.pi)
            
            # Compute xi (state transition posteriors)
            xi = np.zeros((n-1, self.n_states, self.n_states))
            for t in range(n-1):
                denom = scale[t+1] if scale[t+1] > 0 else 1e-10
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = alpha[t, i] * self.A[i, j] * \
                                     self.emission_prob(observations[t+1], j) * beta[t+1, j] / denom
            
            # Update transition matrix
            for i in range(self.n_states):
                xi_sum = np.sum(xi[:, i, :], axis=0)
                gamma_sum_state = np.sum(gamma[:-1, i])
                if gamma_sum_state > 0:
                    self.A[i, :] = np.clip(xi_sum / gamma_sum_state, 1e-10, 1.0)
                else:
                    self.A[i, :] = 1.0 / self.n_states
                self.A[i, :] /= np.sum(self.A[i, :])
            
            # Update emission parameters
            for j in range(self.n_states):
                gamma_sum_j = np.sum(gamma[:, j])
                if gamma_sum_j > 1e-10:
                    self.mu[j] = np.sum(gamma[:, j] * observations) / gamma_sum_j
                    variance = np.sum(gamma[:, j] * (observations - self.mu[j])**2) / gamma_sum_j
                    self.sigma[j] = np.sqrt(np.maximum(variance, 0.001))
                else:
                    self.sigma[j] = np.std(observations) / 2
            
            if (iteration + 1) % 10 == 0:
                print(f"Baum-Welch iteration {iteration + 1}/{self.n_iter}")
        
        print("Training complete!")
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict hidden states using Viterbi algorithm.
        
        Args:
            observations: Observation sequence
            
        Returns:
            Most likely state sequence
        """
        n = len(observations)
        viterbi = np.zeros((n, self.n_states))
        backpointer = np.zeros((n, self.n_states), dtype=int)
        
        # Initialize
        for i in range(self.n_states):
            viterbi[0, i] = np.log(self.pi[i]) + np.log(self.emission_prob(observations[0], i) + 1e-10)
        
        # Forward pass
        for t in range(1, n):
            for j in range(self.n_states):
                temp = viterbi[t-1, :] + np.log(self.A[:, j] + 1e-10) + \
                       np.log(self.emission_prob(observations[t], j) + 1e-10)
                viterbi[t, j] = np.max(temp)
                backpointer[t, j] = np.argmax(temp)
        
        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(viterbi[-1, :])
        
        for t in range(n-2, -1, -1):
            states[t] = backpointer[t+1, states[t+1]]
        
        return states
    
    def get_parameters(self) -> dict:
        """Get model parameters."""
        return {
            'pi': self.pi.copy(),
            'A': self.A.copy(),
            'mu': self.mu.copy(),
            'sigma': self.sigma.copy()
        }

"""
Visualization module for HMM results and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class HMMVisualizer:
    """Visualization utilities for HMM analysis."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_price_and_states(self, dates: pd.DatetimeIndex, 
                             prices: np.ndarray,
                             states: np.ndarray,
                             state_names: dict = None,
                             title: str = "Dow Jones Price with HMM States"):
        """
        Plot price time series with hidden states highlighted.
        
        Args:
            dates: Date index
            prices: Price values
            states: Hidden state sequence
            state_names: Dictionary mapping states to names
            title: Plot title
        """
        if state_names is None:
            state_names = {0: "Bull", 1: "Normal", 2: "Bear"}
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot price
        ax1.plot(dates, prices, linewidth=1.5, color='black', label='Adj Close')
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot states
        colors = ['green', 'blue', 'red']
        for i, (state, color) in enumerate(zip(sorted(np.unique(states)), colors)):
            mask = states == state
            ax2.scatter(dates[mask], states[mask], alpha=0.5, s=10, 
                       color=color, label=f"State {state}: {state_names.get(state, 'Unknown')}")
        
        ax2.set_ylabel('Hidden State', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_title('Decoded Hidden States (Viterbi)', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.set_yticks(sorted(np.unique(states)))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "price_and_states.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
    
    def plot_returns_and_states(self, dates: pd.DatetimeIndex,
                               returns: np.ndarray,
                               states: np.ndarray,
                               state_names: dict = None):
        """
        Plot returns with state coloring.
        
        Args:
            dates: Date index
            returns: Daily returns
            states: Hidden states
            state_names: Dictionary mapping states to names
        """
        if state_names is None:
            state_names = {0: "Bull", 1: "Normal", 2: "Bear"}
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['green', 'blue', 'red']
        for state, color in zip(sorted(np.unique(states)), colors):
            mask = states == state
            ax.scatter(dates[mask], returns[mask], alpha=0.5, s=20,
                      color=color, label=f"State {state}: {state_names.get(state, 'Unknown')}")
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Daily Log Returns', fontsize=11)
        ax.set_title('Daily Returns Colored by Hidden State', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "returns_and_states.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
    
    def plot_transition_matrix(self, A: np.ndarray, 
                              state_names: dict = None):
        """
        Plot state transition matrix as heatmap.
        
        Args:
            A: Transition matrix
            state_names: Dictionary mapping states to names
        """
        if state_names is None:
            state_names = {i: f"State {i}" for i in range(len(A))}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        state_labels = [state_names.get(i, f"State {i}") for i in range(len(A))]
        sns.heatmap(A, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=state_labels, yticklabels=state_labels,
                   cbar_kws={'label': 'Probability'}, ax=ax)
        
        ax.set_title('HMM State Transition Matrix', fontsize=13, fontweight='bold')
        ax.set_xlabel('To State', fontsize=11)
        ax.set_ylabel('From State', fontsize=11)
        
        plt.tight_layout()
        filepath = self.output_dir / "transition_matrix.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
    
    def plot_emission_distributions(self, mu: np.ndarray, sigma: np.ndarray,
                                   state_names: dict = None):
        """
        Plot emission distributions for each state.
        
        Args:
            mu: Mean for each state
            sigma: Standard deviation for each state
            state_names: Dictionary mapping states to names
        """
        from scipy.stats import norm
        
        if state_names is None:
            state_names = {i: f"State {i}" for i in range(len(mu))}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.linspace(min(mu) - 4*max(sigma), max(mu) + 4*max(sigma), 1000)
        colors = ['green', 'blue', 'red']
        
        for i, (mean, std) in enumerate(zip(mu, sigma)):
            state_name = state_names.get(i, f"State {i}")
            ax.plot(x, norm.pdf(x, mean, std), linewidth=2, 
                   color=colors[i % len(colors)], label=f"{state_name} (μ={mean:.3f}, σ={std:.3f})")
        
        ax.set_xlabel('Return Value', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title('Emission Distributions by Hidden State', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "emission_distributions.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
    
    def plot_state_durations(self, state_durations: dict, 
                            state_names: dict = None):
        """
        Plot state duration statistics.
        
        Args:
            state_durations: Dictionary of duration statistics per state
            state_names: Dictionary mapping states to names
        """
        if state_names is None:
            state_names = {i: f"State {i}" for i in state_durations.keys()}
        
        states = list(state_durations.keys())
        mean_durations = [state_durations[s]['mean_duration'] for s in states]
        max_durations = [state_durations[s]['max_duration'] for s in states]
        labels = [state_names.get(s, f"State {s}") for s in states]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(states))
        width = 0.35
        
        ax.bar(x - width/2, mean_durations, width, label='Mean Duration', color='skyblue')
        ax.bar(x + width/2, max_durations, width, label='Max Duration', color='coral')
        
        ax.set_ylabel('Days', fontsize=11)
        ax.set_title('State Duration Statistics', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.output_dir / "state_durations.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
    
    def create_summary_report(self, dates: pd.DatetimeIndex,
                             prices: np.ndarray,
                             returns: np.ndarray,
                             states: np.ndarray,
                             hmm_params: dict,
                             viterbi_analysis: dict,
                             state_names: dict = None):
        """
        Create a comprehensive summary visualization.
        
        Args:
            dates: Date index
            prices: Price values
            returns: Daily returns
            states: Hidden states
            hmm_params: HMM parameters
            viterbi_analysis: Viterbi analysis results
            state_names: Dictionary mapping states to names
        """
        if state_names is None:
            state_names = {0: "Bull", 1: "Normal", 2: "Bear"}
        
        self.plot_price_and_states(dates, prices, states, state_names)
        self.plot_returns_and_states(dates, returns, states, state_names)
        self.plot_transition_matrix(hmm_params['A'], state_names)
        self.plot_emission_distributions(hmm_params['mu'], hmm_params['sigma'], state_names)
        
        state_durations = viterbi_analysis['state_durations']
        self.plot_state_durations(state_durations, state_names)
        
        print("\nAll visualizations saved to:", self.output_dir)

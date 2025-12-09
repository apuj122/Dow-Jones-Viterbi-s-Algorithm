"""
Viterbi algorithm implementation for HMM state decoding.
"""

import numpy as np
from typing import List, Tuple

class ViterbiDecoder:
    """Viterbi algorithm for decoding hidden states from observations."""
    
    @staticmethod
    def viterbi(observations: np.ndarray, pi: np.ndarray, A: np.ndarray,
                emission_func) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm for finding the most likely state sequence.
        
        Args:
            observations: Observation sequence
            pi: Initial state probabilities
            A: Transition matrix
            emission_func: Function to compute emission probability P(obs|state)
            
        Returns:
            Tuple of (state_sequence, log_probability)
        """
        n = len(observations)
        n_states = len(pi)
        
        # Initialize viterbi matrix and backpointer
        viterbi = np.zeros((n, n_states))
        backpointer = np.zeros((n, n_states), dtype=int)
        
        # Step 1: Initialize (t=0)
        for s in range(n_states):
            viterbi[0, s] = np.log(pi[s] + 1e-10) + np.log(emission_func(observations[0], s) + 1e-10)
        
        # Step 2: Recursion (t=1...T-1)
        for t in range(1, n):
            for s in range(n_states):
                # Compute max probability for each previous state
                temp = viterbi[t-1, :] + np.log(A[:, s] + 1e-10)
                backpointer[t, s] = np.argmax(temp)
                viterbi[t, s] = np.max(temp) + np.log(emission_func(observations[t], s) + 1e-10)
        
        # Step 3: Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(viterbi[-1, :])
        
        for t in range(n-2, -1, -1):
            states[t] = backpointer[t+1, states[t+1]]
        
        # Compute final log probability
        log_prob = viterbi[-1, states[-1]]
        
        return states, log_prob
    
    @staticmethod
    def viterbi_log_prob_path(observations: np.ndarray, state_sequence: np.ndarray,
                              pi: np.ndarray, A: np.ndarray,
                              emission_func) -> float:
        """
        Compute log probability of a given state sequence.
        
        Args:
            observations: Observation sequence
            state_sequence: Given state sequence
            pi: Initial state probabilities
            A: Transition matrix
            emission_func: Emission probability function
            
        Returns:
            Log probability of the sequence
        """
        n = len(observations)
        log_prob = np.log(pi[state_sequence[0]] + 1e-10)
        log_prob += np.log(emission_func(observations[0], state_sequence[0]) + 1e-10)
        
        for t in range(1, n):
            log_prob += np.log(A[state_sequence[t-1], state_sequence[t]] + 1e-10)
            log_prob += np.log(emission_func(observations[t], state_sequence[t]) + 1e-10)
        
        return log_prob
    
    @staticmethod
    def get_state_transitions(state_sequence: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Extract transitions from state sequence.
        
        Args:
            state_sequence: Array of state indices
            
        Returns:
            List of (from_state, to_state, count) tuples
        """
        transitions = {}
        
        for t in range(len(state_sequence) - 1):
            key = (state_sequence[t], state_sequence[t+1])
            transitions[key] = transitions.get(key, 0) + 1
        
        return sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def get_state_duration(state_sequence: np.ndarray, state: int) -> dict:
        """
        Analyze duration of specific state in the sequence.
        
        Args:
            state_sequence: Array of state indices
            state: State to analyze
            
        Returns:
            Dictionary with duration statistics
        """
        durations = []
        current_duration = 0
        
        for s in state_sequence:
            if s == state:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        if not durations:
            return {
                'count': 0,
                'total_days': 0,
                'mean_duration': 0,
                'max_duration': 0,
                'min_duration': 0
            }
        
        return {
            'count': len(durations),
            'total_days': sum(durations),
            'mean_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'min_duration': np.min(durations)
        }
    
    @staticmethod
    def analyze_state_sequence(state_sequence: np.ndarray, 
                               n_states: int) -> dict:
        """
        Comprehensive analysis of the decoded state sequence.
        
        Args:
            state_sequence: Array of state indices
            n_states: Number of hidden states
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_observations': len(state_sequence),
            'state_durations': {},
            'state_frequencies': {},
            'major_transitions': ViterbiDecoder.get_state_transitions(state_sequence)
        }
        
        for state in range(n_states):
            duration_stats = ViterbiDecoder.get_state_duration(state_sequence, state)
            analysis['state_durations'][state] = duration_stats
            
            freq = np.sum(state_sequence == state)
            analysis['state_frequencies'][state] = {
                'count': freq,
                'percentage': 100 * freq / len(state_sequence)
            }
        
        return analysis

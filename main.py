"""
Main execution script for Dow Jones HMM analysis.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_fetcher import DowJonesDataFetcher
from src.feature_engineering import FeatureEngineer
from src.hmm_model import GaussianHMM
from src.viterbi_algorithm import ViterbiDecoder
from src.visualization import HMMVisualizer

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("Dow Jones Hidden Markov Model Analysis")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n[Step 1] Fetching Dow Jones data...")
    fetcher = DowJonesDataFetcher(output_dir="data")
    dji_data = fetcher.get_or_fetch(start_date="1995-01-01")
    
    print(f"Data shape: {dji_data.shape}")
    print(f"Date range: {dji_data.index[0].date()} to {dji_data.index[-1].date()}")
    print(f"Price range: ${dji_data['Adj Close'].min():.2f} to ${dji_data['Adj Close'].max():.2f}")
    
    # Step 2: Feature engineering
    print("\n[Step 2] Engineering features...")
    observations, features_df = FeatureEngineer.prepare_hmm_features(dji_data)
    
    print(f"Observations shape: {observations.shape}")
    print(f"Features:")
    print(features_df.describe())
    
    # Step 3: Train HMM
    print("\n[Step 3] Training Hidden Markov Model...")
    n_states = 3  # Bull, Normal, Bear
    hmm = GaussianHMM(n_states=n_states, n_iter=50, random_state=42)
    hmm.baum_welch(observations)
    
    hmm_params = hmm.get_parameters()
    print("\nTrained HMM Parameters:")
    print(f"Initial state probabilities (π): {hmm_params['pi']}")
    print(f"\nTransition matrix (A):")
    print(hmm_params['A'])
    print(f"\nEmission means (μ): {hmm_params['mu']}")
    print(f"Emission std devs (σ): {hmm_params['sigma']}")
    
    # Step 4: Viterbi algorithm
    print("\n[Step 4] Running Viterbi algorithm...")
    states = hmm.predict(observations)
    
    # Ensure states matches features_df length (features_df may be shorter due to NaN removal)
    states = states[:len(features_df)]
    
    print(f"Decoded state sequence shape: {states.shape}")
    print(f"Unique states: {np.unique(states)}")
    print(f"State counts:")
    for state in np.unique(states):
        count = np.sum(states == state)
        pct = 100 * count / len(states)
        print(f"  State {state}: {count} days ({pct:.1f}%)")
    
    # Step 5: Analyze results
    print("\n[Step 5] Analyzing Viterbi results...")
    viterbi_analysis = ViterbiDecoder.analyze_state_sequence(states, n_states)
    
    print("\nState frequencies:")
    for state, freq_info in viterbi_analysis['state_frequencies'].items():
        print(f"  State {state}: {freq_info['count']} observations ({freq_info['percentage']:.1f}%)")
    
    print("\nState durations (days):")
    for state, duration_info in viterbi_analysis['state_durations'].items():
        print(f"  State {state}:")
        print(f"    Mean: {duration_info['mean_duration']:.1f} days")
        print(f"    Max: {duration_info['max_duration']} days")
        print(f"    Min: {duration_info['min_duration']} days")
        print(f"    Total occurrences: {duration_info['count']}")
    
    print("\nMajor state transitions:")
    for (from_state, to_state), count in viterbi_analysis['major_transitions'][:5]:
        print(f"  {from_state} → {to_state}: {count} times")
    
    # Step 6: Visualize results
    print("\n[Step 6] Creating visualizations...")
    visualizer = HMMVisualizer(output_dir="data")
    
    # Prepare data for visualization
    # Align all arrays to the same length as features_df
    dates = features_df.index
    prices = dji_data.loc[dates, 'Adj Close'].values
    returns = features_df['Returns'].values
    
    state_names = {
        0: "Bull Market",
        1: "Normal Market",
        2: "Bear Market"
    }
    
    visualizer.create_summary_report(
        dates=dates,
        prices=prices,
        returns=returns,
        states=states,
        hmm_params=hmm_params,
        viterbi_analysis=viterbi_analysis,
        state_names=state_names
    )
    
    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    results_df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': returns,
        'State': states
    })
    results_df.set_index('Date', inplace=True)
    results_df.to_csv("data/hmm_results.csv")
    print("Results saved to data/hmm_results.csv")
    
    # Save parameters
    params_df = pd.DataFrame({
        'State': range(n_states),
        'Initial_Prob': hmm_params['pi'],
        'Mean': hmm_params['mu'],
        'Std_Dev': hmm_params['sigma']
    })
    params_df.to_csv("data/hmm_parameters.csv", index=False)
    print("Parameters saved to data/hmm_parameters.csv")
    
    # Save transition matrix
    trans_df = pd.DataFrame(
        hmm_params['A'],
        index=[f"From_State_{i}" for i in range(n_states)],
        columns=[f"To_State_{i}" for i in range(n_states)]
    )
    trans_df.to_csv("data/transition_matrix.csv")
    print("Transition matrix saved to data/transition_matrix.csv")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return {
        'hmm': hmm,
        'states': states,
        'data': results_df,
        'parameters': hmm_params,
        'analysis': viterbi_analysis
    }

if __name__ == "__main__":
    results = main()

# Dow Jones Hidden Markov Model Analysis

A comprehensive project for analyzing Dow Jones financial time series data using Hidden Markov Models (HMM) and the Viterbi algorithm.

## Overview

This project:
1. Fetches historical Dow Jones (^DJI) data from 1995 to present
2. Preprocesses and engineers financial features (returns, volatility, momentum)
3. Builds a Hidden Markov Model with multiple hidden states
4. Applies the Viterbi algorithm to decode the most likely hidden state sequence
5. Visualizes results and analyzes market regimes

## Project Structure

```
├── src/
│   ├── data_fetcher.py          # Download Dow Jones data
│   ├── feature_engineering.py   # Create financial features
│   ├── hmm_model.py             # Hidden Markov Model implementation
│   ├── viterbi_algorithm.py     # Viterbi algorithm for state decoding
│   └── visualization.py         # Plotting and analysis
├── data/                        # Downloaded data and results
├── main.py                      # Main execution script
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Installation

1. Clone or download this project
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main analysis:
```
python main.py
```

This will:
- Download Dow Jones historical data
- Train the HMM model
- Run Viterbi algorithm
- Generate visualizations
- Save results to the `data/` folder

## Key Components

### Data Fetcher
Downloads historical Dow Jones data using `yfinance` library.

### Feature Engineering
Calculates returns, volatility, momentum, and other technical indicators.

### Hidden Markov Model
Implements HMM with:
- Multiple hidden states (representing market regimes: Bull, Normal, Bear)
- Gaussian emission probabilities
- Estimated transition and emission matrices

### Viterbi Algorithm
Decodes the most likely sequence of hidden states given the observed returns data.

## Results

The model identifies different market regimes and their transitions, helping to understand:
- Bull market states (high returns, low volatility)
- Normal market states (moderate returns)
- Bear market states (negative returns, high volatility)

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

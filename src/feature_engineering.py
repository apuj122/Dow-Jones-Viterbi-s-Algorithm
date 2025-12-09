"""
Feature engineering module for financial data preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Tuple

class FeatureEngineer:
    """Handles feature engineering for HMM training."""
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, column: str = "Adj Close") -> pd.Series:
        """
        Calculate daily log returns.
        
        Args:
            data: DataFrame with price data
            column: Column name to calculate returns from
            
        Returns:
            Series of log returns
        """
        return np.log(data[column] / data[column].shift(1))
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation).
        
        Args:
            returns: Series of returns
            window: Window size for rolling calculation
            
        Returns:
            Series of volatility values
        """
        return returns.rolling(window=window).std()
    
    @staticmethod
    def calculate_momentum(data: pd.DataFrame, window: int = 20, 
                          column: str = "Adj Close") -> pd.Series:
        """
        Calculate price momentum.
        
        Args:
            data: DataFrame with price data
            window: Window size for momentum calculation
            column: Column name to calculate momentum from
            
        Returns:
            Series of momentum values
        """
        return data[column].pct_change(window)
    
    @staticmethod
    def prepare_hmm_features(data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare features for HMM training.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (observation_sequence, feature_dataframe)
        """
        fe = FeatureEngineer()
        
        # Calculate returns
        returns = fe.calculate_returns(data)
        
        # Calculate volatility
        volatility = fe.calculate_volatility(returns, window=20)
        
        # Calculate momentum
        momentum = fe.calculate_momentum(data, window=20)
        
        # Create feature dataframe
        features_df = pd.DataFrame({
            'Returns': returns,
            'Volatility': volatility,
            'Momentum': momentum
        })
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        # Normalize features for HMM
        normalized_returns = (returns.dropna() - returns.mean()) / returns.std()
        
        # Stack returns as observation sequence (1D array)
        # We'll use returns as the primary observation
        observation_sequence = normalized_returns.values
        
        print(f"Prepared {len(observation_sequence)} observations for HMM")
        print(f"Return statistics:")
        print(f"  Mean: {normalized_returns.mean():.6f}")
        print(f"  Std Dev: {normalized_returns.std():.6f}")
        print(f"  Min: {normalized_returns.min():.6f}")
        print(f"  Max: {normalized_returns.max():.6f}")
        
        return observation_sequence, features_df
    
    @staticmethod
    def create_multivariate_features(data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare multivariate features for HMM (returns + volatility).
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (observation_sequence, feature_dataframe)
        """
        fe = FeatureEngineer()
        
        # Calculate returns
        returns = fe.calculate_returns(data)
        
        # Calculate volatility
        volatility = fe.calculate_volatility(returns, window=20)
        
        # Create feature dataframe
        features_df = pd.DataFrame({
            'Returns': returns,
            'Volatility': volatility
        }).dropna()
        
        # Normalize features
        returns_norm = (features_df['Returns'] - features_df['Returns'].mean()) / features_df['Returns'].std()
        volatility_norm = (features_df['Volatility'] - features_df['Volatility'].mean()) / features_df['Volatility'].std()
        
        # Stack as 2D observation sequence
        observation_sequence = np.column_stack([returns_norm.values, volatility_norm.values])
        
        print(f"Prepared {len(observation_sequence)} multivariate observations for HMM")
        
        return observation_sequence, features_df

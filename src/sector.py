import pandas as pd
import numpy as np

def construct_peer_sector_index(target_ticker, all_tickers_dict, train_start='2018-01-01', train_end='2018-12-31', k=15):
    """
    target_ticker: The stock we are predicting (e.g., 'AAPL') [cite: 22]
    all_tickers_dict: Dictionary of DataFrames {ticker: df} [cite: 20]
    k: Number of peers to include (10-20) [cite: 52]
    """
    
    # 1. Collect returns for all available tickers during the TRAIN period [cite: 52, 111]
    returns_df = pd.DataFrame()
    
    for ticker, df in all_tickers_dict.items():
        # Ensure we use Adj Close for returns [cite: 21, 121]
        train_slice = df.loc[train_start:train_end]
        if not train_slice.empty:
            returns_df[ticker] = train_slice['Adj Close'].pct_change()

    # 2. Calculate correlation with the target stock [cite: 52]
    correlations = returns_df.corr()[target_ticker].drop(target_ticker)
    
    # 3. Pick top K peers [cite: 52]
    top_peers = correlations.sort_values(ascending=False).head(k).index.tolist()
    print(f"Selected Sector Peers for {target_ticker}: {top_peers}")

    # 4. Form the sector index (r_sec) for the ENTIRE dataset timeline [cite: 53]
    # Use the fixed peer list selected from Train to prevent leakage 
    all_returns = pd.DataFrame()
    for ticker in top_peers:
        all_returns[ticker] = all_tickers_dict[ticker]['Adj Close'].pct_change()
    
    # Sector return is the simple average of peer returns [cite: 53]
    sector_index_returns = all_returns.mean(axis=1)
    
    return sector_index_returns, top_peers
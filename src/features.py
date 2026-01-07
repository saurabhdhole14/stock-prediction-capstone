import pandas as pd
import numpy as np
import statsmodels.api as sm

def create_leakage_safe_features(df_stock, df_mkt, df_sec):
    """
    Creates features for day t to predict return at t+1.
    """
    # Safety Check: Find the correct price column name
    # This prevents the KeyError if your CSV uses 'Close' instead of 'Adj Close'
    price_col = 'Adj Close' if 'Adj Close' in df_stock.columns else 'Close'
    
    # 1. Target Variable: Next-day return (r_{i, t+1})
    df_stock['target_r'] = df_stock[price_col].pct_change().shift(-1)
    
    # 2. Own Stock Features
    df_stock['r_t'] = df_stock[price_col].pct_change()
    for lag in range(1, 6):
        df_stock[f'r_lag_{lag}'] = df_stock['r_t'].shift(lag)
    
    # Rolling Volatility & Mean (5, 20, 63 days)
    for w in [5, 20, 63]:
        df_stock[f'vol_{w}'] = df_stock['r_t'].rolling(window=w).std()
        df_stock[f'rolling_mean_{w}'] = df_stock['r_t'].rolling(window=w).mean()
        # SMA Gaps
        df_stock[f'sma_{w}_gap'] = df_stock[price_col] / df_stock[price_col].rolling(window=w).mean() - 1

    # High-Low Range Z-Score
    # Note: Using 'Close' here as a denominator is standard
    hl_range = (df_stock['High'] - df_stock['Low']) / df_stock[price_col]
    df_stock['hl_zscore_20'] = (hl_range - hl_range.rolling(20).mean()) / hl_range.rolling(20).std()

    # Volume Z-score and Deltas
    log_vol = np.log(df_stock['Volume'].replace(0, np.nan))
    df_stock['volu_z_21'] = (log_vol - log_vol.rolling(21).mean()) / log_vol.rolling(21).std()
    df_stock['volu_delta_5'] = log_vol.diff(5)

    # 3. Market & Sector Context
    mkt_price = 'Adj Close' if 'Adj Close' in df_mkt.columns else 'Close'
    df_stock['r_mkt_t'] = df_mkt[mkt_price].pct_change()

    # Check if df_sec is a Series (already returns) or a DataFrame (needs pct_change)
    if isinstance(df_sec, pd.Series):
        # It's already the sector return series from construct_peer_sector_index
        df_stock['r_sec_t'] = df_sec
    else:
        # Fallback if a DataFrame is passed
        sec_price = 'Adj Close' if 'Adj Close' in df_sec.columns else 'Close'
        df_stock['r_sec_t'] = df_sec[sec_price].pct_change()
    
    # 4. Factor Form: Expanding OLS
    df_stock['beta_mkt'] = np.nan
    df_stock['beta_sec'] = np.nan
    df_stock['idio_resid_t'] = np.nan

    for i in range(63, len(df_stock)):
        past_data = df_stock.iloc[:i+1].dropna(subset=['r_t', 'r_mkt_t', 'r_sec_t'])
        if len(past_data) < 30: continue
        
        Y = past_data['r_t']
        X = sm.add_constant(past_data[['r_mkt_t', 'r_sec_t']])
        model = sm.OLS(Y, X).fit()
        
        df_stock.iloc[i, df_stock.columns.get_loc('beta_mkt')] = model.params['r_mkt_t']
        df_stock.iloc[i, df_stock.columns.get_loc('beta_sec')] = model.params['r_sec_t']
        df_stock.iloc[i, df_stock.columns.get_loc('idio_resid_t')] = model.resid.iloc[-1]

    # 5. Calendar Dummies
    df_stock['day_of_week'] = df_stock.index.dayofweek
    df_stock['month'] = df_stock.index.month
    df_stock = pd.get_dummies(df_stock, columns=['day_of_week', 'month'], drop_first=True)

    return df_stock.dropna()
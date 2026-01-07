import argparse
import pandas as pd
import json

def get_decision(r_hat, theta_buy=0.005, theta_sell=0.005):
    """
    Maps predicted return to Buy/Hold/Sell based on tuned thresholds[cite: 70, 71].
    Keep a wide neutral zone for educational purposes[cite: 72].
    """
    if r_hat >= theta_buy:
        return "Buy"
    elif r_hat <= -theta_sell:
        return "Sell"
    else:
        return "Hold"

def run_query(date_str, ticker, model_ensemble, feature_data):
    """
    Inputs: date (YYYY-MM-DD), ticker, the trained ensemble, and the processed features[cite: 84].
    """
    # 1. Locate the data for the requested date 't' to predict 't+1'
    try:
        row = feature_data.loc[date_str]
    except KeyError:
        return f"Error: No data available for date {date_str}"

    # 2. Get the prediction (r_hat for t+1) [cite: 16]
    # In a real scenario, you'd pass 'row' through the ensemble's predict method
    r_hat = model_ensemble.predict(row.drop(['target_r']).values.reshape(1, -1))[0]
    
    # 3. Derive next-day price: P_{t+1} = P_t * (1 + r_hat) [cite: 38]
    p_t = row['Adj Close']
    p_hat_next = p_t * (1 + r_hat)
    
    # 4. Determine decision [cite: 91]
    decision = get_decision(r_hat)
    
    # 5. Identify top drivers (simplified example) [cite: 93, 104]
    drivers = {
        "own": {"r_t": row['r_t'], "vol20": row['vol_20']},
        "market": {"r_mkt_t": row['r_mkt_t'], "beta_mkt": row['beta_mkt']},
        "sector": {"r_sec_t": row['r_sec_t'], "beta_sec": row['beta_sec']}
    }
    
    # 6. Format Output as JSON [cite: 86-104]
    output = {
        "date": date_str,
        "ticker": ticker,
        "r_hat_next": round(r_hat, 4),
        "p_hat_next": round(p_hat_next, 2),
        "decision": decision,
        "drivers": drivers
    }
    
    return json.dumps(output, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Prediction CLI [cite: 83]")
    parser.add_argument("date", help="Date in YYYY-MM-DD format [cite: 85]")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol [cite: 85]")
    
    args = parser.parse_args()
    
    # Note: In production, you would load your saved models/dataframes here
    # Example call:
    # print(run_query(args.date, args.ticker, trained_stacker, processed_df))
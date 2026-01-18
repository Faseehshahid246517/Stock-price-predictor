import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


load_dotenv()

API_KEY_ID = os.getenv("ALPACA_API_KEY")
SECRET_KEY_ID = os.getenv("ALPACA_SECRET_KEY")

class MarketForecaster:
    def __init__(self, key_id, secret_id):
        self.connector = StockHistoricalDataClient(key_id, secret_id)
        self.estimator = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=101)
        self.pricing_table = None
        self.feature_list = ["close_price", "trade_volume", "volatility_idx", "avg_5d", "avg_10d", "avg_20d"]

    def fetch_market_history(self, ticker, start, end):
        print(f"\n[+] Fetching data for {ticker} from {start} to {end}...")
        req_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        try:
            raw_data = self.connector.get_stock_bars(req_params)
            self.pricing_table = raw_data.df.reset_index()
            self.pricing_table.rename(columns={"close": "close_price", "volume": "trade_volume"}, inplace=True)
            self.pricing_table = self.pricing_table[self.pricing_table["symbol"] == ticker]
            return True
        except Exception as e:
            print(f"[!] Error fetching data: {e}")
            return False

    def generate_indicators(self):
        data = self.pricing_table
        data["price_change"] = data["close_price"].pct_change()
        data["volatility_idx"] = data["price_change"].rolling(5).std()
        data["avg_5d"] = data["close_price"].rolling(5).mean()
        data["avg_10d"] = data["close_price"].rolling(10).mean()
        data["avg_20d"] = data["close_price"].rolling(20).mean()
        
        data["future_target"] = data["close_price"].shift(-1)
        self.pricing_table = data.dropna()

    def train_and_evaluate(self):
        X_data = self.pricing_table[self.feature_list]
        y_data = self.pricing_table["future_target"]

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)
        
        self.estimator.fit(X_train, y_train)
        
        predictions = self.estimator.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        return y_test, predictions, mae, mape

    def predict_target_date(self):
        recent_features = self.pricing_table[self.feature_list].iloc[[-1]]
        predicted_price = self.estimator.predict(recent_features)[0]
        return predicted_price

    def show_results(self, y_test, predictions, ticker):
        plt.figure(figsize=(14, 7))
        plt.plot(y_test.values, label="Actual Market Price", color='blue', alpha=0.7)
        plt.plot(predictions, label="Model Forecast", color='orange', linestyle='--')
        plt.title(f"Market Analysis: {ticker}")
        plt.xlabel("Time Period")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def get_input_or_default(prompt, default_val):
    user_input = input(f"{prompt} [Default: {default_val}]: ")
    return user_input if user_input.strip() != "" else default_val

def main_menu():
    forecaster = MarketForecaster(API_KEY_ID, SECRET_KEY_ID)
    
    while True:
        print("\n" + "="*40)
        print("   STOCK PREDICTION SYSTEM")
        print("="*40)
        print("1. Initialize New Prediction")
        print("2. Exit System")
        
        choice = input("\nEnter selection: ")
        
        if choice == '1':
            ticker = input("Enter Stock Ticker (e.g., NVDA): ").upper()
            
            today = datetime.now()
            default_start = (today - timedelta(days=1095)).strftime('%Y-%m-%d')
            default_end = (today - timedelta(days=10)).strftime('%Y-%m-%d')
            default_target = (today + timedelta(days=1)).strftime('%Y-%m-%d')

            start_date = get_input_or_default("Enter Training Start Date", default_start)
            end_date = get_input_or_default("Enter Training End Date", default_end)
            target_date = get_input_or_default("Enter Target Prediction Date", default_target)
            
            success = forecaster.fetch_market_history(ticker, start_date, end_date)
            
            if success:
                forecaster.generate_indicators()
                actuals, preds, mae, mape = forecaster.train_and_evaluate()
                
                print("\n--- MODEL PERFORMANCE ---")
                print(f"Mean Absolute Error: ${mae:.2f}")
                print(f"Percentile Error (MAPE): {mape:.2f}%")
                
                future_price = forecaster.predict_target_date()
                print(f"\nPrediction for {ticker} on {target_date}:")
                print(f"Estimated Close: ${future_price:.2f}")
                
                print("\nGenerating graph...")
                forecaster.show_results(actuals, preds, ticker)
                
        elif choice == '2':
            break
        else:
            print("Invalid command.")

if __name__ == "__main__":
    main_menu()
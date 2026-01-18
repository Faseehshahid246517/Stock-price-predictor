# Algorithmic Stock Price Forecaster

A machine learning application that predicts future stock prices using historical market data and technical analysis. This tool interfaces with the Alpaca Data API to fetch real-time financial data, processes it through a Random Forest regression model, and visualizes the results.

## 📋 Features

* **Automated Data Retrieval**: Connects directly to Alpaca Markets to fetch high-fidelity historical data (Open, High, Low, Close, Volume).
* **Technical Feature Engineering**: Automatically generates key trading indicators:
    * **Moving Averages**: 5-day, 10-day, and 20-day trends.
    * **Market Volatility**: 5-day rolling standard deviation.
    * **Daily Returns**: Percentage change calculations.
* **Smart CLI Interface**: A text-based menu system that handles user inputs and provides intelligent defaults (e.g., automatically selecting a 3-year training window if no date is provided).
* **Performance Metrics**: Evaluates model accuracy using:
    * **MAE (Mean Absolute Error)**: The average magnitude of errors in dollars.
    * **MAPE (Mean Absolute Percentage Error)**: The accuracy represented as a percentage.
* **Data Visualization**: Generates interactive matplotlib charts comparing the model's predictions against actual historical prices.

## 🧠 The Machine Learning Model

This project utilizes the **Random Forest Regressor** from the Scikit-Learn library, an ensemble learning method ideal for capturing non-linear relationships in financial data.

### Model Architecture
* **Algorithm**: Random Forest Regression (`sklearn.ensemble.RandomForestRegressor`)
* **Estimators**: 200 decision trees.
* **Max Depth**: 12 (optimized to prevent overfitting).
* **Target Variable**: The next trading day's closing price (`shift(-1)`).

### Input Features (X)
The model does not rely solely on price. It is trained on a multi-dimensional feature set:
1.  `close_price`: Raw closing price.
2.  `trade_volume`: Daily trading volume.
3.  `volatility_idx`: Short-term market volatility.
4.  `avg_5d`, `avg_10d`, `avg_20d`: Short, medium, and long-term trend indicators.

## ⚙️ Setup & Installation

### Prerequisites
1.  **Python 3.8** or higher.
2.  An **Alpaca Markets** account (The Free Tier is sufficient).
    * [Sign up here](https://app.alpaca.markets/signup) to generate your API Key and Secret Key.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/stock-forecaster.git](https://github.com/your-username/stock-forecaster.git)
cd stock-forecaster
Step 2: Install Dependencies
Install the required Python libraries using pip:

Bash

pip install pandas numpy matplotlib alpaca-py scikit-learn
Step 3: Configure API Keys
Open the main.py file in your code editor.

Locate the configuration section at the top:

Python

# --- CONFIGURATION ---
API_KEY_ID = "YOUR_API_KEY_HERE"
SECRET_KEY_ID = "YOUR_SECRET_KEY_HERE"
Replace the placeholder text with your actual API Key ID and Secret Key from the Alpaca dashboard.

🚀 How to Run
Execute the script from your terminal:

Bash

python main.py
Usage Instructions
Select Option 1 from the main menu.

Enter Stock Ticker: Input the symbol you wish to analyze (e.g., NVDA, TSLA, AAPL).

Date Configuration:

You will be prompted for a Start Date, End Date, and Target Date.

Press ENTER on any prompt to use the default settings (Default: Train on the last 3 years, predict for tomorrow).

Analyze Results:

The console will display the predicted price and the error margins (MAE/MAPE).

A graph will appear showing how well the model tracked the stock price over the testing period.

⚠️ Disclaimer
This software is provided for educational and research purposes only. It does not constitute financial advice. Algorithmic trading involves significant risk, and this model should not be used as the sole basis for real-money investment decisions.

📄 License
This project is open-source and available under the MIT License.

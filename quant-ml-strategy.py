"""
Quantitative Backtesting Strategy with Machine Learning
Author: Alice Proietti

ML-based backtesting strategy on SPY (2020–today). 
Trains Linear Regression, Random Forest and XGBoost on engineered features, 
selects the best model via time-series CV, and backtests a long-short strategy.
Outputs Sharpe Ratio, Max Drawdown and cumulative performance vs SPY.
"""


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt

# Parameters
ticker = 'SPY'
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
prediction_horizon = 1

# Download Data
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# Feature engineering 
data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
data['Volatility_5'] = data['LogReturn'].rolling(5).std()
data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
data['Volatility_10'] = data['LogReturn'].rolling(10).std()
data['VolumeChange'] = data['Volume'].pct_change()
data = data.dropna()

# Features e target
features = ['LogReturn', 'Momentum_5', 'Volatility_5', 'Momentum_10', 'Volatility_10', 'VolumeChange']
X = data[features]
y = data['LogReturn'].shift(-prediction_horizon)
X = X[:-prediction_horizon]
y = y[:-prediction_horizon]

# TimeSeries Cross Validation
tscv = TimeSeriesSplit(n_splits=5)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Automatic selection of the best model
best_score = float('inf')
best_model = None

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    mean_score = -np.mean(scores)
    print(f"{name} MSE: {mean_score:.6f}")
    if mean_score < best_score:
        best_score = mean_score
        best_model = model
        best_model_name = name

print(f"\nBest model selected: {best_model_name}")

# Best model fit
best_model.fit(X, y)
data['PredictedReturn'] = np.nan
data.iloc[:-prediction_horizon, data.columns.get_loc('PredictedReturn')] = best_model.predict(X)

# Signal and strategy
data['Signal'] = np.where(data['PredictedReturn'] > 0, 1, -1)
data['StrategyReturn'] = data['Signal'].shift(1) * data['LogReturn']

# Metriche performance 
cumulative_strategy = np.exp(data['StrategyReturn'].cumsum())
cumulative_asset = np.exp(data['LogReturn'].cumsum())

sharpe_ratio = (data['StrategyReturn'].mean() / data['StrategyReturn'].std()) * np.sqrt(252)
rolling_max = cumulative_strategy.cummax()
drawdown = (cumulative_strategy - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Comparative graph
plt.figure(figsize=(12,6))
plt.plot(cumulative_strategy, label='Strategy')
plt.plot(cumulative_asset, label=ticker)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title(f'Backtest Strategy vs {ticker}')
plt.legend()
plt.show()

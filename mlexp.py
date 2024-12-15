
import yfinance as yf
import pandas as pd
import numpy as np

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch historical data for a blue-chip stock
ticker = 'pcjeweller.NS'  # Replace with desired blue-chip stock ticker
data = yf.download(ticker, period='1d', interval='5m')

# Calculate RSI
data['RSI'] = calculate_rsi(data, 14)

# Implementing the RSI strategy
def rsi_strategy(data):
    buy_signals = []
    sell_signals = []
    position = False  # Currently not in a position

    for i in range(len(data)):
        if data['RSI'][i] < 30 and not position:
            buy_signals.append(data['Close'][i])
            sell_signals.append(np.nan)
            position = True
        elif data['RSI'][i] > 70 and position:
            buy_signals.append(np.nan)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
    
    data['Buy Signal'] = buy_signals
    data['Sell Signal'] = sell_signals

    return data

# Apply the strategy
data = rsi_strategy(data)

# Display the data with signals
print(data[['Close', 'RSI', 'Buy Signal', 'Sell Signal']])

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.scatter(data.index, data['Buy Signal'], label='Buy Signal', marker='^', color='green')
plt.scatter(data.index, data['Sell Signal'], label='Sell Signal', marker='v', color='red')
plt.title(f'{ticker} Price with RSI Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

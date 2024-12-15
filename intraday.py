import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time

# Define the interval for updating data (in seconds)
UPDATE_INTERVAL = 60  # Update every 60 seconds

def fetch_live_data(ticker, start_date, interval):
    # Fetch live data with the specified interval
    data = yf.download(ticker, start=start_date, interval=interval)
    return data

def calculate_signals(data):
    # Calculate moving averages
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    # Generate signals
    data['Signal'] = 0
    data['Signal'][data['50_MA'] > data['200_MA']] = 1  # Buy signal
    data['Signal'][data['50_MA'] < data['200_MA']] = -1  # Sell signal

    return data

def plot_data(data):
    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['50_MA'], label='50-Period MA')
    plt.plot(data['200_MA'], label='200-Period MA')
    plt.plot(data[data['Signal'] == 1].index, data['50_MA'][data['Signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data[data['Signal'] == -1].index, data['50_MA'][data['Signal'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title('Live Data and Signals')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

def live_trading(ticker, interval='5m'):
    start_date = datetime.now() - timedelta(days=1)  # Start from a day ago to get initial data

    while True:
        # Fetch live data
        data = fetch_live_data(ticker, start_date, interval)

        # Calculate signals
        data = calculate_signals(data)

        # Plot data
        plot_data(data)

        # Update the start_date for the next fetch
        start_date = datetime.now()

        # Wait for the next update
        time.sleep(UPDATE_INTERVAL)

# Example usage
ticker = 'GC=F'  # Gold Futures
live_trading(ticker, interval='5m')

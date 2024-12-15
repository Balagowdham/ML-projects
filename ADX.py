import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate ADX and related indicators
def calculate_adx(data, window=14):
    # Calculate True Range (TR)
    data['TR'] = data['High'] - data['Low']
    data['TR'] = data['TR'].combine_first(abs(data['High'] - data['Close'].shift()))
    data['TR'] = data['TR'].combine_first(abs(data['Low'] - data['Close'].shift()))
    
    # Calculate Directional Movement (DM)
    data['DMplus'] = data['High'] - data['High'].shift()
    data['DMplus'] = data['DMplus'].where(data['DMplus'] > (data['Low'].shift() - data['Low']), 0)
    
    data['DMminus'] = data['Low'].shift() - data['Low']
    data['DMminus'] = data['DMminus'].where(data['DMminus'] > (data['High'] - data['High'].shift()), 0)
    
    # Calculate Smoothed DX
    data['TR14'] = data['TR'].rolling(window=window).mean()
    data['DMplus14'] = data['DMplus'].rolling(window=window).mean()
    data['DMminus14'] = data['DMminus'].rolling(window=window).mean()
    
    data['RSIplus'] = (data['DMplus14'] / data['TR14']) * 100
    data['RSIminus'] = (data['DMminus14'] / data['TR14']) * 100
    
    # Calculate ADX
    data['Diffplus'] = abs(data['RSIplus'] - data['RSIplus'].shift(-1)).rolling(window=window).mean()
    data['Diffminus'] = abs(data['RSIminus'] - data['RSIminus'].shift(-1)).rolling(window=window).mean()
    
    data['DIplus'] = (data['Diffplus'] / data['TR14']) * 100
    data['DIminus'] = (data['Diffminus'] / data['TR14']) * 100
    
    data['DX'] = ((data['DIplus'] - data['DIminus']) / (data['DIplus'] + data['DIminus'])) * 100
    
    # Calculate ADX
    data['ADX'] = data['DX'].rolling(window=window).mean()
    
    return data

# Function to generate buy and sell signals
def generate_signals(data):
    buy_signals = [None]  # Start with None to align with index
    sell_signals = [None]  # Start with None to align with index
    position = False  # Position in market, initially not holding

    for i in range(1, len(data)):
        if data['DIplus'][i] > data['DIminus'][i] and data['ADX'][i] > data['ADX'][i-1]:
            buy_signals.append(data['Close'][i])
            sell_signals.append(None)
            position = True
        elif data['DIplus'][i] < data['DIminus'][i] and data['ADX'][i] > data['ADX'][i-1]:
            buy_signals.append(None)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(None)
            sell_signals.append(None)

    data['Buy Signal'] = buy_signals[:len(data)]  # Ensure lengths match
    data['Sell Signal'] = sell_signals[:len(data)]  # Ensure lengths match

    return data

# Fetch historical data for ITC with 5-minute interval for a week
ticker = 'ITC.NS'
data = yf.download(ticker, start='2024-07-09', end='2024-07-13', interval='5m')

# Drop any rows with NaN values
data.dropna(inplace=True)

# Calculate ADX and related indicators
data = calculate_adx(data)

# Generate buy and sell signals
data = generate_signals(data)

# Plotting ADX, +DI, -DI and signals
plt.figure(figsize=(16, 8))
plt.plot(data['ADX'], label='ADX', color='black')
plt.plot(data['DIplus'], label='+DI', color='blue')
plt.plot(data['DIminus'], label='-DI', color='red')
plt.scatter(data.index, data['Buy Signal'], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(data.index, data['Sell Signal'], marker='v', color='r', label='Sell Signal', alpha=1)

plt.title(f'ADX Strategy with Buy/Sell Signals for {ticker} (9th July to 13th July 2024)')
plt.xlabel('Time')
plt.ylabel('Index Value')
plt.legend()
plt.grid(True)
plt.show()

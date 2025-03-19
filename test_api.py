import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to get the latest data from Yahoo Finance
def get_latest_data(ticker='TSLA', days=70):
    # Get data from Yahoo Finance
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Downloaded {len(data)} days of {ticker} data")
    
    # Calculate technical indicators (same as in your training)
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss = loss + 1e-10  # Avoid division by zero
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = rolling_mean + (rolling_std * 2)
    data['Lower_Band'] = rolling_mean - (rolling_std * 2)
    
    # Calculate OBV
    data['OBV'] = (data['Volume'] * ((data['Close'].diff() > 0) * 2 - 1)).fillna(0).cumsum()
    
    # Log transform volume
    data['Volume'] = np.log1p(data['Volume'])
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Select features
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA7', 'MA20', 'MA30', 
                'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'OBV']
    data = data[features]
    
    return data

# Get the latest data
latest_data = get_latest_data()

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(latest_data)

# Get the last 60 days for prediction
input_data = scaled_data[-60:].tolist()

# Make sure we have exactly 60 time steps
if len(input_data) != 60:
    print(f"Warning: Expected 60 time steps, got {len(input_data)}. Adjusting...")
    if len(input_data) > 60:
        input_data = input_data[-60:]
    else:
        # Pad with the first row if we don't have enough data
        padding = [input_data[0]] * (60 - len(input_data))
        input_data = padding + input_data

# Send the request to the API
url = "http://localhost:8000/predict"
payload = {"features": input_data}

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        predicted_price = result["predicted_price"]
        
        print(f"Predicted Tesla stock price: ${predicted_price:.2f}")
        
        # Get the actual latest price for comparison
        latest_price = latest_data['Close'].iloc[-1]
        print(f"Latest actual price: ${latest_price:.2f}")
        print(f"Difference: ${predicted_price - latest_price:.2f} ({((predicted_price - latest_price) / latest_price) * 100:.2f}%)")
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(latest_data.index[-30:], latest_data['Close'].iloc[-30:], label='Actual Price')
        plt.axhline(y=predicted_price, color='r', linestyle='--', label=f'Predicted Price: ${predicted_price:.2f}')
        plt.title('Tesla Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_result.png')
        plt.show()
        
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error making request: {e}")
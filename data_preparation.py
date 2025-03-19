import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def download_stock_data(ticker='TSLA', start='2015-01-01', end='2023-01-01'):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading {ticker} data from {start} to {end}...")
    ds_stock = yf.download(ticker, start=start, end=end)
    print(f"Downloaded {len(ds_stock)} rows of data")
    return ds_stock

def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # Add small epsilon to avoid division by zero
    loss = loss + 1e-10  
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_obv(close, volume):
    obv = (volume * ((close.diff() > 0) * 2 - 1)).fillna(0).cumsum()
    return obv

def inspect_data(data, name="dataset"):
    """Print statistics about the data to help debug"""
    print(f"\n--- {name} Statistics---")
    print(f"Shape: {data.shape}")
    
    if isinstance(data, np.ndarray):
        print(f"Min: {np.nanmin(data) if not np.all(np.isnan(data)) else 'nan'}")
        print(f"Max: {np.nanmax(data) if not np.all(np.isnan(data)) else 'nan'}")
        print(f"Mean: {np.nanmean(data) if not np.all(np.isnan(data)) else 'nan'}")
        print(f"Median: {np.nanmedian(data) if not np.all(np.isnan(data)) else 'nan'}")
        print(f"Standard Deviation: {np.nanstd(data) if not np.all(np.isnan(data)) else 'nan'}")
        print(f"NaN Count: {np.any(np.isnan(data))}")
        print(f"Inf Count: {np.any(np.isinf(data))}")
        
        if np.any(np.isnan(data)):
            print(f"Position of NaN: {np.where(np.isnan(data))}")
        
        if np.any(np.isinf(data)):
            print(f"Position of Inf: {np.where(np.isinf(data))}")
    else:
        # For pandas DataFrame
        print(f"Min: {data.min().min()}")
        print(f"Max: {data.max().max()}")
        print(f"Mean: {data.mean().mean()}")
        print(f"NaN Count: {data.isna().sum().sum()}")

def prepare_data(ds_stock):
    """Add technical indicators and prepare features with better handling of NaN values"""
    # Make a copy to avoid modifying the original
    df = ds_stock.copy()
    
    # Calculate moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # Calculate additional features
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
    df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(df['Close'])
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    
    # Log transform volume to reduce skewness
    df['Volume'] = np.log1p(df['Volume'])
    
    # Select features
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA7', 'MA20', 'MA30', 
                'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'OBV']
    df = df[features]
    
    # Print NaN counts before dropping
    print("NaN counts before dropping:")
    print(df.isna().sum())
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Print shape after dropping NaNs
    print(f"Shape after dropping NaNs: {df.shape}")
    
    # Check for and handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def create_sequences(data, seq_length):
    """Create sequences for LSTM input with proper shape checking"""
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting 'Close' price (index 0)
    
    x_array = np.array(x)
    y_array = np.array(y)
    
    print(f"Sequence creation - x shape: {x_array.shape}, y shape: {y_array.shape}")
    
    return x_array, y_array

def prepare_train_test_data(ticker='TSLA', start='2015-01-01', end='2023-01-01', seq_length=60, split_rate=0.8):
    """Complete data preparation pipeline with better error handling"""
    # Download and prepare data
    ds_stock = download_stock_data(ticker, start, end)
    
    # Inspect raw data
    inspect_data(ds_stock, "Raw Stock Data")
    
    # Prepare features
    ds_stock = prepare_data(ds_stock)
    
    # Inspect prepared data
    inspect_data(ds_stock, "Prepared Stock Data")
    
    if len(ds_stock) <= seq_length:
        raise ValueError(f"Not enough data points after preprocessing. Got {len(ds_stock)}, need more than {seq_length}.")
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ds_stock)
    
    # Create sequences
    x, y = create_sequences(scaled_data, seq_length)
    
    # Verify shapes
    if len(x) != len(y):
        raise ValueError(f"Sequence creation resulted in mismatched shapes: x: {x.shape}, y: {y.shape}")
    
    # Split data
    split = int(len(x) * split_rate)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training data shape: {x_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, {y_test.shape}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Inspect final training data
    inspect_data(x_train, "x_train")
    inspect_data(y_train, "y_train")
    inspect_data(x_test, "x_test")
    inspect_data(y_test, "y_test")
    
    # Print feature names for reference
    for i, col in enumerate(ds_stock.columns):
        print(f"Feature {i} ({col})")
        print(f" Min: {np.min(x_train[:,:,i])}")
        print(f" Max: {np.max(x_train[:,:,i])}")
        print(f" Mean: {np.mean(x_train[:,:,i])}")
        print(f" Median: {np.median(x_train[:,:,i])}")
    
    return x_train, y_train, x_test, y_test, scaler, ds_stock.columns

if __name__ == "__main__":
    # Test the data preparation
    try:
        x_train, y_train, x_test, y_test, scaler, columns = prepare_train_test_data()
        print("Data preparation successful!")
    except Exception as e:
        print(f"Error in data preparation: {e}")
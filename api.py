from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import tensorflow as tf
import joblib
from pydantic import BaseModel
import os
import sys
import pandas as pd
import yfinance as yf

# Make sure the model module is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model class first before loading the model
from model import ComplexLSTMModel

app = FastAPI(title="Tesla Stock Price Prediction API",
              description="API for predicting Tesla stock prices using LSTM",
              version="1.0.0")

# Define paths
MODEL_PATH = os.path.join('models', 'simple_lstm_model.keras')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Load model and scaler
try:
    # For simple model
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class StockInput(BaseModel):
    features: list[list[float]]  # Expecting a sequence of feature vectors

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post('/predict')
def predict_stock(data: StockInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, 
                          detail="Model or scaler not loaded. Please check server logs.")
    
    try:
        # Validate input shape
        if len(data.features) != 60:
            raise HTTPException(status_code=400, 
                              detail=f"Input must be 60 time steps. Got {len(data.features)} time steps.")
        
        # Get number of features from the first sequence
        expected_features = 14  # This should match your model's expected features
        
        if any(len(features) != expected_features for features in data.features):
            raise HTTPException(status_code=400, 
                              detail=f"Each time step must have {expected_features} features.")
        
        # Convert to numpy array and reshape for model
        input_data = np.array(data.features)
        
        # Make prediction
        prediction = model.predict(np.expand_dims(input_data, axis=0))
        
        # Prepare for inverse scaling (add zeros for other features)
        padded_prediction = np.zeros((prediction.shape[0], expected_features))
        padded_prediction[:, 0] = prediction[:, 0]  # Assuming 'Close' is the first feature
        
        # Inverse transform to get actual price
        prediction_rescaled = scaler.inverse_transform(padded_prediction)[0, 0]
        
        return {
            'predicted_price': float(prediction_rescaled),
            'status': 'success'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-latest")
async def predict_latest():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, 
                          detail="Model or scaler not loaded. Please check server logs.")
    
    try:
        # Get the latest Tesla data
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=100)
        
        # Download data
        data = yf.download("TSLA", start=start_date, end=end_date)
        
        if len(data) < 60:
            raise HTTPException(status_code=500, 
                              detail=f"Not enough data points. Got {len(data)}, need at least 60.")
        
        # Calculate features (same as in training)
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
        
        # Select features
        features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA7', 'MA20', 'MA30', 
                    'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'OBV']
        data = data[features]
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        # Get the latest actual price before scaling
        latest_price = float(data['Close'].iloc[-1])
        
        # Scale the data
        scaled_data = scaler.transform(data)
        
        # Get the last 60 days for prediction
        input_data = scaled_data[-60:]
        
        # Make prediction
        prediction = model.predict(np.expand_dims(input_data, axis=0))
        
        # Prepare for inverse scaling
        padded_prediction = np.zeros((prediction.shape[0], len(features)))
        padded_prediction[:, 0] = prediction[:, 0]  # Assuming 'Close' is the first feature
        
        # Inverse transform to get actual price
        prediction_rescaled = scaler.inverse_transform(padded_prediction)[0, 0]
        
        return {
            'predicted_price': float(prediction_rescaled),
            'actual_price': latest_price
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
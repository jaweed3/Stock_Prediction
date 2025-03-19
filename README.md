# Tesla Stock Price Prediction (Zero to Deployment)

## Overview
This project builds a **time series forecasting model** to predict Tesla's stock prices using **LSTM (Long Short-Term Memory) networks**. The model is trained on historical stock data and deployed as an API using **FastAPI** with **Uvicorn**.

## Features
- **Data Collection**: Fetching historical Tesla stock prices.
- **Data Preprocessing**: Normalization, feature engineering, and sequence generation.
- **Model Training**: Implementing LSTM with TensorFlow/Keras.
- **Evaluation & Optimization**: Hyperparameter tuning and performance analysis.
- **Deployment**: Serving predictions using FastAPI.

## Dataset
- Source: **Yahoo Finance**
- Data: **Tesla (TSLA) stock prices**
- Features: Date, Open, High, Low, Close, Volume

## Tech Stack
- **Python**
- **TensorFlow & Keras**
- **Pandas & NumPy**
- **Scikit-learn**
- **Matplotlib & Seaborn** (for visualization)
- **FastAPI & Uvicorn** (for API deployment)
- **Docker** (optional for containerization)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/tesla-stock-prediction.git
   cd tesla-stock-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Training
Run the training script:
```bash
python train.py
```
This will:
- Load and preprocess the Tesla stock dataset
- Train an LSTM model
- Save the trained model as `model.h5`

## Deployment
1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
2. API Endpoint:
   ```bash
   GET /predict?days=5
   ```
   Example request:
   ```bash
   curl -X GET "http://127.0.0.1:8000/predict?days=5"
   ```
   Example response:
   ```json
   {
     "predicted_prices": [880.45, 890.12, 905.67, 915.32, 925.89]
   }
   ```

## Future Improvements
- Integrate **Sentiment Analysis** on news headlines for better forecasting.
- Implement **GRU or Transformer-based models** for comparison.
- Deploy on **Cloud (AWS/GCP/Azure)** for real-time inference.

## Contributing
Feel free to fork this repository and submit a pull request if you have any improvements!

## License
This project is licensed under the **MIT License**.


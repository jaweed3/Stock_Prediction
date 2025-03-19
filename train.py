import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import joblib as jb
from model import ComplexLSTMModel
from data_preparation import prepare_train_test_data

def train_model(x_train, y_train, x_test, y_test, batch_size=32, epochs=10):
    _, seq_len, num_features = x_train.shape
    
    model = ComplexLSTMModel(units=50)
    model.build((None, seq_len, num_features))
    model.build_graph((seq_len, num_features)).summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0),
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                verbose=1
            )
        ]
    )
    return model, history

def evaluate_model(model, x_test, y_test, scaler, n_features):
    y_pred = model.predict(x_test)
    
    y_pred_padded = np.concatenate([y_pred, np.zeros((y_pred.shape[0], n_features - 1))], axis=1)
    y_test_padded = np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], n_features - 1))], axis=1)

    y_pred_actual = scaler.inverse_transform(y_pred_padded)[:, 0]
    y_test_actual = scaler.inverse_transform(y_test_padded)[:, 0]

    mse = np.mean((y_pred_actual - y_test_actual) ** 2)
    mae = np.mean(np.abs(y_pred_actual - y_test_actual))

    print(f"Test MSE: {mse:.2f}")
    print(f"Test MAE: {mae:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(y_pred_actual, label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Tesla Stock Price Prediction')

    os.makedirs('models', exist_ok=True)
    plt.savefig('models/prediction_plot.png')
    plt.show()

    return mse, mae

def save_model(model, filename='tesla_prediction_model.keras'):
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', filename)
    model.save(model_path)
    print(f"Model saved at {model_path}")
    return model_path

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scaler, columns = prepare_train_test_data()
    
    model, history = train_model(
        x_train, y_train, x_test, y_test,
        batch_size=32,
        epochs=20)

    mse, mae = evaluate_model(model, x_test, y_test, scaler, len(columns))
    model_path = save_model(model)
    print(f"Training Complete. Model Saved to {model_path}")
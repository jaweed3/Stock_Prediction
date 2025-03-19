import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_preparation import prepare_train_test_data, inspect_data
from model import ComplexLSTMModel

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def debug_model():
    print("Starting debug process...")
    
    try:
        # Prepare data with more verbose output
        x_train, y_train, x_test, y_test, scaler, columns = prepare_train_test_data()
        
        print("\n Training Simple LSTM Model")
        # Try a simpler model first
        simple_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], x_train.shape[2])),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(1)
        ])
        
        # Use a custom callback to monitor training
        class NanCallback(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if logs and np.isnan(logs.get('loss', 0)):
                    print(f"\nNaN loss detected at batch {batch}")
                    self.model.stop_training = True
        
        # Compile with gradient clipping and reduced learning rate
        simple_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        # Print model summary
        simple_model.summary()
        
        # Verify shapes before training
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        # Train with small batch size and monitor closely
        history = simple_model.fit(
            x_train, y_train,
            batch_size=16,  # Smaller batch size
            epochs=5,
            verbose=2,
            validation_split=0.1,
            callbacks=[NanCallback()]
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('simple_model_training.png')
        plt.show()
        
        # If simple model works, try the complex model with adjusted parameters
        if not np.isnan(history.history['loss'][-1]):
            print("\nSimple model trained successfully. Trying complex model with adjusted parameters...")
            
            complex_model = ComplexLSTMModel(units=64)  # Reduced units
            complex_model.build((None, x_train.shape[1], x_train.shape[2]))
            
            complex_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            complex_history = complex_model.fit(
                x_train, y_train,
                batch_size=16,
                epochs=5,
                verbose=2,
                validation_split=0.1,
                callbacks=[NanCallback()]
            )
            
            # Plot complex model training history
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(complex_history.history['loss'])
            plt.plot(complex_history.history['val_loss'])
            plt.title('Complex Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.plot(complex_history.history['mae'])
            plt.plot(complex_history.history['val_mae'])
            plt.title('Complex Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig('complex_model_training.png')
            plt.show()
            
            return complex_model if not np.isnan(complex_history.history['loss'][-1]) else simple_model
        else:
            print("\nSimple model failed. Need to investigate data further.")
            return None
            
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = debug_model()
    if model:
        print("Debug successful! Model trained without NaN loss.")
        # Save the model
        model.save('models/debug_successful_model.keras')
    else:
        print("Debug unsuccessful. Further investigation needed.")
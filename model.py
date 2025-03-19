import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout

@tf.keras.utils.register_keras_serializable()
class ComplexLSTMModel(tf.keras.Model):
    def __init__(self, units, name=None, **kwargs):
        super(ComplexLSTMModel, self).__init__(name=name, **kwargs)
        self.units = units
        self.lstm1 = LSTM(units, return_sequences=True)
        self.lstm2 = LSTM(units, return_sequences=True)
        self.lstm3 = LSTM(units, return_sequences=False)
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.1)
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.lstm3(x)
        output = self.dense(x)
        return output

    def get_config(self):
        config = super(ComplexLSTMModel, self).get_config()
        config.update({
            'units': self.units
        })    
        return config

    @classmethod
    def from_config(cls, config):
        units = config.pop('units')
        return  cls(units=units, **config)

    def build_graph(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
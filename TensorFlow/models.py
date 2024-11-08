import tensorflow as tf

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def compile_and_fit(model, window, max_epochs=20, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


def linear_():
    return tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

def dense_(hidden_units=64, output_units=1):
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=hidden_units, activation='relu'),
        tf.keras.layers.Dense(units=hidden_units, activation='relu'),
        tf.keras.layers.Dense(units=output_units)
    ])
    return dense

def cnn_(filter_size=32, conv_width=3, hidden_units=32, output_units=1):
    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=filter_size,
                            kernel_size=(conv_width,),
                            activation='relu'),
        tf.keras.layers.Dense(units=hidden_units, activation='relu'),
        tf.keras.layers.Dense(units=output_units),
    ])
    return cnn

def lstm_(layers=32, return_sequences=True, output_units=1):
    lstm = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(layers, return_sequences=return_sequences),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=output_units)
    ])
    return lstm


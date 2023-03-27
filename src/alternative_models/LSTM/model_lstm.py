import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Reshape, Dropout

class LSTMModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        # Define model architecture with dropout
        input_layer = Input(shape=self.input_shape)
        reshape_layer = Reshape((1, self.input_shape[0]))(input_layer)
        lstm_layer = LSTM(64, dropout=0.2)(reshape_layer)
        dropout_layer = Dropout(0.2)(lstm_layer)
        output_layer = Dense(3, activation='softmax')(dropout_layer)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        if self.model is None:
            self.build_model()

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        if self.model is None:
            raise Exception('Model not yet built')

        return self.model.predict(X)

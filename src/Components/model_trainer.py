import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self, voc_size=10000, maxlen=30, embedding_features=40):
        self.voc_size = voc_size
        self.maxlen = maxlen
        self.embedding_features = embedding_features
        self.model = None

    def build_model(self):
        model = Sequential([
            Embedding(self.voc_size, self.embedding_features, input_length=self.maxlen),
            LSTM(100),
            Dense(2, activation="softmax")
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
        self.model = model

    def train_model(self, x_train, y_train, x_val, y_val, epochs=5, batch_size=20):
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

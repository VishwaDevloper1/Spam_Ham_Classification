from src.Components.data_ingestion import DataIngestion
from src.Components.data_Preprocessing import DataPreprocessing
from src.Components.model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder

class SpamClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.trained_model = None
        self.label_encoder = None

    def train_model(self):
        # Data ingestion
        ingestion = DataIngestion(file_path=self.data_path)
        data = ingestion.load_data()
        x_train, x_test, y_train, y_test = ingestion.split_data(data)

        # Preprocessing
        preprocessing = DataPreprocessing()
        x_train_seq = preprocessing.preprocess_data(x_train)
        x_test_seq = preprocessing.preprocess_data(x_test)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        # Model training
        trainer = ModelTrainer()
        trainer.build_model()
        trainer.train_model(x_train_seq, y_train_enc, x_test_seq, y_test_enc)

        # Save the trained model
        self.trained_model = trainer.model
        print("Model training complete.")

    def predict_message(self, message):
        if self.trained_model is None or self.label_encoder is None:
            raise ValueError("Model is not trained yet. Please call train_model() first.")

        # Preprocess the input message
        preprocessing = DataPreprocessing()
        preprocessed_text = preprocessing.preprocess_data([message])

        # Predict class
        prediction = self.trained_model.predict(preprocessed_text)
        predicted_label = self.label_encoder.inverse_transform([prediction.argmax()])[0]
        return predicted_label

# Example usage
if __name__ == "__main__":
    # Create an instance of the class
    classifier = SpamClassifier(data_path="data/SMSSpamCollection.csv")

    # Train the model
    classifier.train_model()

    # Predict a sample message
    test_message = "Win a free iPhone! Call now!"
    result = classifier.predict_message(test_message)
    print(f"Prediction: {result}")

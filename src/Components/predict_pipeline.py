import numpy as np

class PredictPipeline:
    def __init__(self, model, preprocessing):
        self.model = model
        self.preprocessing = preprocessing

    def predict(self, text):
        processed_input = self.preprocessing.preprocess_data([text])
        predicted_probs = self.model.predict(processed_input)
        predicted_label = np.argmax(predicted_probs, axis=1)
        return "spam" if predicted_label[0] == 1 else "ham"

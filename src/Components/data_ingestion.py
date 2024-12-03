import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path, names=["label", "message"])
        return data

    def split_data(self, data, test_size=0.2, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(
            data["message"], data["label"], test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test

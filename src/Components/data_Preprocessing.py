import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocessing:
    def __init__(self, voc_size=10000, maxlen=30):
        self.voc_size = voc_size
        self.maxlen = maxlen
        self.lemma = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [self.lemma.lemmatize(word) for word in text if word not in stopwords.words("english")]
        return ' '.join(text)

    def preprocess_data(self, messages):
        corpus = [self.clean_text(message) for message in messages]
        one_hot_rep = [one_hot(text, self.voc_size) for text in corpus]
        padded_seq = pad_sequences(one_hot_rep, padding='pre', maxlen=self.maxlen)
        return padded_seq

import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

model = pickle.load(open('resources/model.pkl', 'rb'))
vectorizer = pickle.load(open('resources/vectorizer.pkl', 'rb'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if not i in stop_words]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)

    return text
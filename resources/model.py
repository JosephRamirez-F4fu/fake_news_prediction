import pickle

model = pickle.load(open('resources/model.pkl', 'rb'))
vectorizer = pickle.load(open('resources/vectorizer.pkl', 'rb'))
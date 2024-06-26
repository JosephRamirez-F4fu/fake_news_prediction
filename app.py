from flask import Flask, render_template, request, jsonify
from resources.model import model, vectorizer,preprocess_text
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news = preprocess_text(news)
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)[0]
    print(prediction)
    response = {
        'prediction': 'FAKE' if prediction < 0.5 else 'REAL',
        'probability': round(float(prediction) * 100, 2)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

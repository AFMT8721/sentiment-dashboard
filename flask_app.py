from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import plotly.express as px
import pandas as pd
import os
from werkzeug.utils import secure_filename
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load models and vectorizer
with open('models/best_logistic_regression_model.pickle', 'rb') as file:
    logistic_regression_model = pickle.load(file)
with open('models/vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    
    # Preprocess and vectorize the text
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    
    # Predict using logistic regression
    prediction = logistic_regression_model.predict(vectorized_text)
    
    sentiment = 'Positive' if prediction[0] == 'positive' else 'Negative'

    return jsonify({'sentiment': sentiment})

@app.route('/visualize', methods=['GET'])
def visualize():
    data = {
        'Text': ['I love this!', 'I hate this!', 'This is amazing!', 'This is terrible!'],
        'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative']
    }
    df = pd.DataFrame(data)

    fig = px.histogram(df, x='Sentiment', title='Sentiment Distribution')
    graph_html = fig.to_html(full_html=False)

    return render_template('visualize.html', graph_html=graph_html)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({'message': 'File uploaded successfully'})
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
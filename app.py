from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure models exist (download if missing)
try:
    from download_models import ensure_models
    ensure_models()
except Exception as e:
    print(f"Note: Could not run download_models: {e}")

# Load models, tokenizer, and vectorizer
lstm_model = load_model('models/lstm_model.h5')
with open('models/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)
with open('models/vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)
with open('models/best_logistic_regression_model.pickle', 'rb') as file:
    logistic_regression_model = pickle.load(file)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess text for LSTM
def preprocess_text_lstm(text):
    max_length = 200
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

# Preprocess text for Logistic Regression
def preprocess_text_lr(text):
    return vectorizer.transform([text])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    model_type = request.form['model_type']

    if model_type == 'lstm':
        preprocessed_text = preprocess_text_lstm(text)
        prediction_prob = lstm_model.predict(preprocessed_text)
        prediction = (prediction_prob > 0.5).astype(int)
    else:
        preprocessed_text = preprocess_text_lr(text)
        prediction = logistic_regression_model.predict(preprocessed_text)

    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

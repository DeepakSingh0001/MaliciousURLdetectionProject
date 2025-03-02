from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models and preprocessing tools
cnn_model = load_model("cnn_model.h5")
with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define constants
max_length = 150  # Must match training config

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        url = request.form["url"]

        # CNN Prediction
        seq = tokenizer.texts_to_sequences([url])
        padded_seq = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
        cnn_pred = cnn_model.predict(padded_seq)
        cnn_label = label_encoder.inverse_transform([np.argmax(cnn_pred)])[0]

        # TF-IDF Predictions
        tfidf_seq = tfidf_vectorizer.transform([url])
        rf_label = label_encoder.inverse_transform([rf_model.predict(tfidf_seq)[0]])[0]
        lr_label = label_encoder.inverse_transform([lr_model.predict(tfidf_seq)[0]])[0]

        return render_template("index.html", url=url, cnn=cnn_label, rf=rf_label, lr=lr_label)

if __name__ == "__main__":
    app.run(debug=True)

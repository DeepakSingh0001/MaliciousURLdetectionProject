from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load saved model & preprocessing objects
model = load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define max length (must be same as training)
max_length = 150

# Function to predict URL attack type
def predict_url(url):
    seq = tokenizer.texts_to_sequences([url])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_seq)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        url = request.form["url"]
        prediction = predict_url(url)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

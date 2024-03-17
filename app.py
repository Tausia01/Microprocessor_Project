

from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.signal import find_peaks
import librosa
import joblib
from pydub import AudioSegment
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/record", methods=["POST"])
def record():
    try:
        audio_blob = request.files['audio']

        # Save the audio file temporarily
        audio_path = 'temp_audio.wav'
        audio_blob.save(audio_path)

        # Read audio data using librosa
        audio_data, sample_rate = librosa.load(audio_path, sr=None)

        # Detect BPM
        bpm = detect_bpm(audio_data, sample_rate)

        # Predict disease
        disease_prediction = predict_disease(audio_data, sample_rate)

        # Remove the temporary audio file
        os.remove(audio_path)

        return jsonify({"bpm": bpm, "disease": disease_prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

def detect_bpm(audio_data, sample_rate):
    try:
        peaks, _ = find_peaks(audio_data, distance=sample_rate//30)
        bpm = len(peaks) * 60 / len(audio_data) * sample_rate
        return bpm
    except Exception as e:
        return str(e)

def predict_disease(audio_data, sample_rate):
    try:
        # Extract features from audio data
        features = extract_features(audio_data, sample_rate)

        # Load pre-trained classifier model
        classifier = joblib.load('your_model.pkl')  # Load your trained model file

        # Make prediction using the model
        disease_prediction = classifier.predict([features])[0]

        return disease_prediction
    except Exception as e:
        return str(e)

def extract_features(audio_data, sample_rate):
    try:
        # Extract Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)

        # Calculate mean of MFCCs along each feature dimension
        mfccs_mean = np.mean(mfccs, axis=1)

        # Combine all features into a single feature vector
        features = np.concatenate((mfccs_mean,))
        return features
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True, port=3000)

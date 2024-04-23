!pip install sounddevice

import sounddevice

import numpy as np
import sounddevice as sd
import librosa
import pickle

# Load the pre-trained model
model = pickle.load(open("m1.pkl", "rb"))

def extract_audio_features(signal, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(signal)), sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)

    mfccs = mfccs[:12]

    return np.hstack((mfccs, chroma, mel))[:52]

def predict_emotion(audio_data, sample_rate):
    audio_features = extract_audio_features(audio_data, sample_rate)
    audio_features = audio_features.reshape(1, -1)
    emotion_prediction = model.predict(audio_features)
    return emotion_prediction[0]

def record_audio(duration=5, sample_rate=22050):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio_data, sample_rate

def main():
    # Record audio
    audio_data, sample_rate = record_audio()

    # Predict emotion
    predicted_emotion = predict_emotion(audio_data[:, 0], sample_rate)

    print("Predicted Emotion:", predicted_emotion)

if __name__ == "__main__":
    main()

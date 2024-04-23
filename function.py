import matplotlib as plt
import librosa
import numpy as np
import plotly.graph_objects as go

def create_heatmap(spectrogram, times, frequencies):
    # Define the trace for the heatmap
    trace = go.Heatmap(
        z=spectrogram,
        x=times,
        y=frequencies,
        colorscale='Viridis'
    )

    # Define layout options
    layout = go.Layout(
        title='Spectrogram',
        xaxis=dict(title='Time (seconds)'),
        yaxis=dict(title='Frequency (Hz)')
    )

def extract_audio_features(signal, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(signal)), sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)
    mfccs = mfccs[:12]
    return np.hstack((mfccs, chroma, mel))[:52]


def create_spectrogram(audio_data, sample_rate):
    # Compute spectrogram using librosa
    spectrogram = np.abs(librosa.stft(audio_data))

    # Convert amplitude to decibels
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Compute time and frequency axis
    times = librosa.times_like(spectrogram_db.shape[1], sr=sample_rate)
    frequencies = librosa.fft_frequencies(sr=sample_rate)

    return spectrogram_db, times, frequencies
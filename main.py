from flask import Flask, render_template, request, redirect, session, flash,jsonify
import mysql.connector
from function import extract_audio_features , create_spectrogram
import numpy as np
import os
import librosa
import soundfile
import pickle
import speech_recognition as sr
import sounddevice as sd


app = Flask(__name__)
app.secret_key = os.urandom(24)

model = pickle.load(open("m1.pkl", "rb"))

conn = mysql.connector.connect(
    host="localhost",
    user="sprite",
    password="12345678",
    database="hoosier-daddy"
)
cursor = conn.cursor()

# Route for login page
@app.route('/')
def login():
    if 'user_id' in session:
        return redirect('/Home')
    return render_template('login.html')

# Route for signup page
@app.route('/signup')
def signup():
    if 'user_id' in session:
        return redirect('/get_started')
    return render_template('signup.html')


@app.route('/Home')
def Home():
    if 'user_id' in session:
        username = session.get('username')
        flash('Welcome, {}'.format(username))
        return render_template('Home.html', username=username)
    else:
        return redirect('/')
    

@app.route('/About')
def about_page():
    if 'user_id' in session:
        return render_template('About.html')
    else:
        return redirect('/')



@app.route('/get_started')
def get_started():
    if 'user_id' in session:
        return render_template('get_started.html')
    else:
        return redirect('/')


@app.route('/predict_emotion', methods=['GET', 'POST'])
def predict_emotion():
    if 'user_id' in session:
        if request.method == 'POST':
            
            audio_file = request.files['audio_file']
            audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
            audio_features = extract_audio_features(audio_data, sample_rate)
            audio_features = audio_features.reshape(1, -1)
            emotion_prediction = model.predict(audio_features)

            
            return render_template('predict_emotion.html', emotion=emotion_prediction[0], audio_data=audio_data.tolist(), audio_features=audio_features.tolist(), 
                                   sample_rate=sample_rate)
    return redirect('/')

@app.route('/compute_spectrogram', methods=['POST'])
def compute_spectrogram():
    data = request.json
    audio_data = data['audio_data']
    sample_rate = data['sample_rate']

    
    spectrogram, times, frequencies = create_spectrogram(audio_data, sample_rate)

    
    return jsonify({
        'spectrogram': spectrogram.tolist(),
        'times': times.tolist(),
        'frequencies': frequencies.tolist()
    })


@app.route('/visualization', methods=['GET', 'POST'])
def visualize_audio():
    if 'user_id' in session:
        if request.method == 'POST':
            audio_data = request.form.get('audio_data')
            audio_features = request.form.get('audio_features')

            if audio_data and audio_features:
                return render_template('visualization.html', audio_data=audio_data, audio_features=audio_features)

        return render_template('visualization.html')
    else:
        return redirect('/')
    
    

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    # Check if an audio file was included in the request
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_file']

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Attempt to transcribe the audio
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return jsonify({'text': text}), 200
    except sr.UnknownValueError:
        return jsonify({'error': 'Speech Recognition could not understand audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results from Google Speech Recognition service: {e}'}), 500
# Route for record page




@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("""SELECT * FROM users WHERE email LIKE '{}' AND password LIKE '{}'""".format(email, password))
    users = cursor.fetchall()
    
    if len(users) > 0:
        session['user_id'] = users[0][0]  
        session['username'] = users[0][1] 
        flash('Logged in successfully!')
        
        return redirect('/Home')
    else:
       
        return render_template('login.html', error_message='Incorrect password')

# Route for handling user signup
@app.route('/add_user', methods=['POST'])
def add_user():
    name = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("""INSERT INTO users(id, Username, Email, Password) VALUES (NULL, '{}', '{}', '{}')""".format(name, email, password))
    conn.commit()

    # Redirect to the login page after successful registration
    return redirect("/")

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session:
        return redirect('/')  
    
    if request.method == 'POST':
        # Process the feedback form submission here
        name = request.form['name']
        email = request.form['email']
        feedback_text = request.form['feedback']
        
        # Insert feedback data into the database
        cursor.execute("INSERT INTO feedback (name, email, feedback_text) VALUES (%s, %s, %s)", (name, email, feedback_text))
        conn.commit()
        
        flash('Feedback submitted successfully!')
        
        return redirect('/About')  
    else:
        return render_template('feedback.html')

# Route for logging out (ends the session)
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')




def calculate_audio_features(signal, sample_rate):
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(signal)), sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)

    mfccs = mfccs[:12]

    return np.hstack((mfccs, chroma, mel))[:52]

def make_emotion_prediction(audio_data, sample_rate):
    audio_features = calculate_audio_features(audio_data, sample_rate)
    audio_features = audio_features.reshape(1, -1)
    emotion_prediction = model.predict(audio_features)
    return emotion_prediction[0]

def start_recording(duration=5, sample_rate=22050):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording stopped.")
    return audio_data, sample_rate

def analyze_recorded_data(duration=5, sample_rate=22050):
    # Record audio
    audio_data, sample_rate = start_recording(duration, sample_rate)

    # Predict emotion
    predicted_emotion = make_emotion_prediction(audio_data[:, 0], sample_rate)

    return predicted_emotion

@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        duration = int(request.form['duration'])
        emotion = analyze_recorded_data(duration)
        return render_template('emotion_prediction.html', predicted_emotion=emotion)
    else:
        
        if 'user_id' in session:
            return render_template('record.html')
        else:
            # Redirect to login page if user is not logged in
            return redirect('/')
        




if __name__== '__main__':
    app.run(debug=True)
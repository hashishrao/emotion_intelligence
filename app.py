import secrets
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import pandas as pd
import io
from PIL import Image
import pickle
import threading
import pyaudio
import wave
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)

# Ensure the database is created within the application context
with app.app_context():
    db.create_all()

# Load the song dataset
song_data = pd.read_csv('/Users/hashishtekkali/Desktop/emotion intelligence/song_data.csv')

# Load the trained model
model = load_model('/Users/hashishtekkali/Desktop/emotion intelligence/emotion_recognition_model.keras')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load the Parkinson's model and scaler within the application context
with app.app_context():
    with open('/Users/hashishtekkali/Desktop/emotion intelligence/parkinsons_model.pkl', 'rb') as model_file:
        parkinsons_model = pickle.load(model_file)
    with open('/Users/hashishtekkali/Desktop/emotion intelligence/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))  # Resize image to 48x48
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add a channel dimension
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    return image

# Function to plot emotion statistics in different chart formats
def plot_emotion_statistics(emotion_counts):
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    # Bar chart
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    
    ax[0].bar(emotions, counts, color='skyblue')
    ax[0].set_title('Emotion Frequency - Bar Chart')
    ax[0].set_xlabel('Emotions')
    ax[0].set_ylabel('Frequency')
    
    # Pie chart
    ax[1].pie(counts, labels=emotions, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(emotions))))
    ax[1].set_title('Emotion Frequency - Pie Chart')
    
    # Line chart
    ax[2].plot(emotions, counts, marker='o', color='skyblue')
    ax[2].set_title('Emotion Frequency - Line Chart')
    ax[2].set_xlabel('Emotions')
    ax[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig.canvas.draw()
    
    # Convert plot to image
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return plot_img

# Function to recommend songs based on the dominant emotion
def recommend_songs(emotion, song_data, top_n=5):
    if emotion == 'Happy':
        filtered_songs = song_data[(song_data['danceability'] > 0.7) & (song_data['energy'] > 0.7) & (song_data['audio_valence'] > 0.7)]
    elif emotion == 'Sad':
        filtered_songs = song_data[(song_data['audio_valence'] < 0.4) & (song_data['tempo'] < 100)]
    elif emotion == 'Angry':
        filtered_songs = song_data[(song_data['energy'] > 0.7) & (song_data['loudness'] > -5)]
    elif emotion == 'Fear':
        filtered_songs = song_data[(song_data['audio_valence'] < 0.4) & (song_data['tempo'] > 100) & (song_data['tempo'] < 140)]
    elif emotion == 'Surprise':
        filtered_songs = song_data[(song_data['audio_valence'] > 0.7) & (song_data['tempo'] > 140)]
    else:  # Neutral or other emotions
        filtered_songs = song_data[(song_data['danceability'] > 0.5) & (song_data['energy'] > 0.5) & (song_data['audio_valence'] > 0.5)]
    
    recommended_songs = filtered_songs.head(top_n)
    return recommended_songs

# Initialize emotion counts
emotion_counts = defaultdict(int)

def generate_frames(user_id):
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect faces in the frame using OpenCV's Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) and preprocess it
            roi = gray_frame[y:y+h, x:x+w]
            processed_roi = preprocess_image(roi)

            # Predict emotion
            prediction = model.predict(processed_roi)
            max_index = np.argmax(prediction)
            emotion = emotion_labels[max_index]

            # Update emotion counts
            emotion_counts[emotion] += 1

            # Draw bounding box and emotion label on the frame
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        elapsed_time = time.time() - start_time
        if elapsed_time > 60:  # Reset emotion counts every minute
            emotion_counts.clear()
            start_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Audio processing
audio_data = deque(maxlen=44100)

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK)
        audio_data.extend(np.frombuffer(data, dtype=np.int16))

# Start recording audio in a separate thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'], method='sha256')
        email = request.form['email']
        
        # Check if the email already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error='Email address already exists. Please choose a different one.')

        # Create a new user
        new_user = User(username=username, password=password, email=email)
        db.session.add(new_user)
        
        try:
            db.session.commit()
        except IntegrityError as e:
            db.session.rollback()
            return render_template('register.html', error='Error creating user. Please try again.')
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(session['user_id']), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_suggestions')
def emotion_suggestions():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Get the dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    # Recommend songs based on the dominant emotion
    recommended_songs = recommend_songs(dominant_emotion, song_data)
    suggestions = [f"{row['title']} by {row['artist']} (Album: {row['album']})" for index, row in recommended_songs.iterrows()]
    return jsonify({'dominant_emotion': dominant_emotion, 'suggestions': suggestions})

@app.route('/emotion_stats')
def emotion_stats():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    plot_img = plot_emotion_statistics(emotion_counts)
    _, buffer = cv2.imencode('.png', plot_img)
    return Response(buffer.tobytes(), mimetype='image/png')

@app.route('/audio_levels')
def audio_levels():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    plt.figure(figsize=(8, 4))
    plt.plot(audio_data)
    plt.ylim(0, 32000)
    plt.title('Live Audio Levels')
    plt.xlabel('Time')
    plt.ylabel('Volume Level')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png')

@app.route('/parkinsons_prediction', methods=['POST'])
def parkinsons_prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Simulated data for demonstration purposes
    features = np.random.rand(1, 22)  # Example features
    scaled_features = scaler.transform(features)
    prediction = parkinsons_model.predict(scaled_features)
    result = "Positive for Parkinson's" if prediction[0] else "Negative for Parkinson's"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

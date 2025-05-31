# 🌟 Edge of Emotion Intelligence 😄😢😡😨

**Project by:** [Tekkali Hashish Raghavendra Rao]  
**Institution:** Gandhi Institute of Technology and Management (GITAM), Visakhapatnam  
**Academic Year:** 2nd Year

---

## 🧠 Project Overview

**Edge of Emotion Intelligence** is a deep learning-based facial emotion recognition system that combines Convolutional Neural Networks (CNNs) with real-time video analysis to detect human emotions. It goes a step further by **recommending personalized music** and **providing psychological guidance** based on the user's **dominant emotion**.

---

## 📌 Key Features

- 🎥 **Real-Time Emotion Detection** using webcam input
- 💻 **CNN-based Classifier** for 7 universal emotions:
  - Angry 😠
  - Disgust 🤢
  - Fear 😨
  - Happy 😄
  - Sad 😢
  - Surprise 😲
  - Neutral 😐
- ⏳ Emotion aggregation over 60 seconds to identify dominant mood
- 🎶 Smart **Music Recommendation System** based on mood
- 🧘‍♀️ Actionable **Emotional Management Advice**
- 🔁 Feedback loop for mood improvement

---

## 🔍 Tech Stack

| Component              | Technology                        |
|------------------------|------------------------------------|
| Emotion Detection      | Python, OpenCV, CNN (Keras/TensorFlow) |
| Dataset Used           | FER-2013                          |
| Recommendation Engine  | Custom rule-based + playlist mapping |
| User Interface         | Tkinter (or Streamlit variant)    |
| Model Evaluation       | Accuracy, Precision, Confusion Matrix |

---

## 📊 Model Performance

The CNN model achieved strong accuracy on validation data, effectively distinguishing between complex facial expressions. Performance metrics are discussed in detail in the paper.

- ✅ Accurate real-time predictions
- 🔁 Handles multiple faces but prioritizes one face per frame for dominant detection
- 🎯 Emotion aggregation increases stability

---

## 📚 System Workflow

```mermaid
flowchart TD
    A[Webcam Input] --> B[Face Detection]
    B --> C[Emotion Recognition (CNN)]
    C --> D[60-sec Emotion Aggregator]
    D --> E[Dominant Emotion Output]
    E --> F[Music Recommender 🎵]
    E --> G[Psychological Advice 🧘‍♂️]

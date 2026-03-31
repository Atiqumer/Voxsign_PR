# SignSpeak AI: Real-Time Sign Language to Speech Translator

SignSpeak AI is a computer vision and machine learning application designed to bridge the communication gap for the hearing and speech-impaired. The system utilizes MediaPipe for high-fidelity hand landmark detection and a custom-trained TensorFlow model to recognize sign language alphabets and convert them into audible speech in real-time.

## ✨ Features
* **Real-Time Detection**: Processes video frames instantly to identify hand gestures.
* **Coordinate Normalization**: Uses wrist-centric and zoom-invariant scaling to ensure high accuracy regardless of hand position or distance from the camera.
* **Text-to-Speech (TTS)**: Automatically announces the recognized letter using the Google Text-to-Speech (gTTS) library.
* **High Accuracy**: Optimized Neural Network architecture achieving over 98% validation accuracy on the alphabet dataset.

## 🛠️ Tech Stack
* **Language**: Python 3.10.11
* **Hand Tracking**: MediaPipe (Vision Tasks API)
* **Deep Learning**: TensorFlow / Keras
* **Computer Vision**: OpenCV
* **Audio**: gTTS (Google Text-to-Speech)

## 🚀 Installation Guide

### Prerequisites
* **Python 3.10.11**: [Download for Windows](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) or [Download for macOS](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg).
* **Webcam**: Needed for real-time sign recognition.

### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Atiqumer/SignSpeakAI](https://github.com/Atiqumer/SignSpeakAI)
   cd SignSpeak-AI


### Create a Virtual Environment:

Windows: python -m venv venv

macOS: python3 -m venv venv

Activate the Environment:

Windows: .\venv\Scripts\activate

macOS: source venv/bin/activate

Install Required Libraries:

Bash
pip install -r requirements.txt
Download the Task File:
Download the Hand Landmarker Task File and place it directly in the project folder.

### 👤 User Guide (For Non-Technical Users)
You don't need to be a coder to use this project! Follow these simple steps to start translating signs and see how accurate the AI is.

## How to use the Translator
Open your Terminal (Command Prompt on Windows or Terminal on Mac) and navigate to the project folder.

Start the program by typing:
python realtime_sign.py

Position your hand in front of your webcam. Make a sign for a letter (like 'A' or 'B').

Watch and Listen:

A green skeleton will appear over your hand on the screen.

The predicted letter and the AI's confidence percentage will appear at the top.

The computer will speak the letter aloud.

To Close: Press the 'q' key on your keyboard.

## How to verify the Model's Accuracy
While using the translator, look at the Percentage (%) next to the predicted letter:

90% - 100%: The AI is extremely confident in its result.

Below 70%: The AI is uncertain. Try improving your lighting or making the sign more clearly.

For a deeper look at accuracy, you can view the training results in the AI_Model.ipynb file, which shows how the model learned to reach a 98.8% success rate during testing.

### 🏗️ Project Structure
train_signspeak.py: The script used to "teach" the AI using the dataset.

realtime_sign.py: The main app that turns your webcam into a translator.

hand_landmarker.task: The "eyes" of the project that find your hand joints.

requirements.txt: The list of "ingredients" (libraries) the project needs to run.

### License
This project is licensed under the MIT License.
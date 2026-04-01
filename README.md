# 🚀 SignSpeak AI – Real-Time Sign Language to Speech

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**SignSpeak AI** is a real-time sign language recognition system that converts hand gestures into **text and speech** using Computer Vision and Machine Learning.

This project helps bridge communication gaps for people with hearing or speech impairments.

---

## ✨ Features

- 🎥 Real-time hand gesture detection (webcam)
- 🤖 Deep learning model (TensorFlow/Keras)
- ✋ Hand tracking using MediaPipe
- 🔊 Text-to-Speech output (gTTS)
- 📊 High accuracy (~98%)
- ⚡ Fast and responsive predictions

---

## 🏗️ How It Works

```text
Webcam → Hand Detection → Landmark Extraction → Model Prediction → Text → Speech
```

---

## 📂 Project Structure

```text
SignSpeakAI/
│── realtime_sign.py        # Run real-time detection
│── train_signspeak.py      # Train the model
│── signspeak_model.keras   # Trained AI model
│── hand_landmarker.task    # MediaPipe model
│── label_map.npy           # Labels
│── signspeak_ui.py         # UI (optional)
│── requirements.txt        # Dependencies
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Atiqumer/SignSpeakAI.git
cd SignSpeakAI
```

---

## 💻 Setup Guide (Windows / Linux / macOS)

| Step | Windows | Linux | macOS |
|------|--------|-------|-------|
| Create venv | `python -m venv venv` | `python3 -m venv venv` | `python3 -m venv venv` |
| Activate venv | `venv\Scripts\activate` | `source venv/bin/activate` | `source venv/bin/activate` |
| Upgrade pip | `pip install --upgrade pip` | `pip install --upgrade pip` | `pip install --upgrade pip` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

---

## 🚀 Quick Start

Run the real-time system:

```bash
python realtime_sign.py
```

---

## ▶️ Usage

1. Open webcam  
2. Show your hand in front of camera  
3. Make a sign (A, B, C...)  
4. System will:
   - Detect the sign  
   - Show prediction on screen  
   - Speak the letter  

👉 Press **`q`** to exit  

---

## ⚙️ Configuration

You can change settings inside the code:

- Detection confidence  
- Tracking confidence  
- Model file (`.keras`)  
- Labels file (`.npy`)  
- Speech delay timing  

---

## 🧠 Train Your Own Model

```bash
python train_signspeak.py
```

---

## 🤝 Contributing

Contributions are welcome!

Steps:
1. Fork the repo  
2. Create a new branch  
3. Make changes  
4. Submit a Pull Request  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ❗ Troubleshooting

| Problem | Solution |
|--------|---------|
| Webcam not working | Check camera permissions |
| Model not loading | Make sure model file exists |
| No detection | Improve lighting |
| Wrong predictions | Use clear hand gestures |

---

## 💡 Future Improvements

- Full sentence recognition  
- Offline speech support  
- Mobile app version  
- Multi-hand detection  

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

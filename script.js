import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

// UI Elements
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");
const alphabetBox = document.getElementById("predictedAlphabet");
const wordBox = document.getElementById("cumulativeWord");
const sentenceBox = document.getElementById("generatedSentence");
const imageUpload = document.getElementById("imageUpload"); // NEW: File Input

// Variables
let handLandmarker;
let model;
let currentPrediction = "";
let lastVideoTime = -1;

// Ensure this matches your specific model's output labels
const labelMap = [
    "A", "B", "Blank", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", 
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
];

async function initialize() {
    try {
        console.log("⏳ Initializing AI Dashboard...");
        
        // 1. Setup MediaPipe
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { 
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" 
            },
            runningMode: "VIDEO", // Keep VIDEO for webcam, we will switch for images
            numHands: 1
        });

        // 2. Load Your AI Model
        model = await tf.loadLayersModel('./web_model/model.json');
        console.log("✅ AI Model Loaded!");

        // 3. Start Camera
        setupCamera();

    } catch (error) {
        console.error("❌ Initialization failed:", error);
        alert("Failed to load AI models. Make sure 'web_model' folder is in the same directory.");
    }
}

async function setupCamera() {
    const constraints = { video: { width: 640, height: 480 } };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        video.play();
        predictWebcam();
    };
}

// Helper to draw UI elements
function drawUI(landmarks) {
    const x = landmarks.map(l => l.x * canvas.width);
    const y = landmarks.map(l => l.y * canvas.height);
    const minX = Math.min(...x); const maxX = Math.max(...x);
    const minY = Math.min(...y); const maxY = Math.max(...y);

    ctx.strokeStyle = "#e74c3c"; 
    ctx.lineWidth = 4;
    ctx.strokeRect(minX - 20, minY - 20, (maxX - minX) + 40, (maxY - minY) + 40);
    
    ctx.fillStyle = "#e74c3c";
    ctx.font = "bold 16px Arial";
    ctx.fillText("HAND DETECTED", minX - 20, minY - 30);
}

// The Normalization
function processLandmarks(landmarks) {
    const wrist = landmarks[0];
    let coords = [];
    for (let i = 0; i < landmarks.length; i++) {
        coords.push(landmarks[i].x - wrist.x);
        coords.push(landmarks[i].y - wrist.y);
        coords.push(landmarks[i].z - wrist.z);
    }
    const maxVal = Math.max(...coords.map(Math.abs)) || 1;
    return coords.map(c => c / maxVal);
}

// Prediction logic shared between Webcam and Uploads
async function runInference(landmarks) {
    const inputData = processLandmarks(landmarks);
    const inputTensor = tf.tensor2d(inputData, [1, 63]);
    const prediction = model.predict(inputTensor);
    const scores = await prediction.data();
    
    let maxIdx = 0;
    let maxScore = -1;
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxIdx = i;
        }
    }

    const detectedLabel = labelMap[maxIdx];

    if (maxScore > 0.75 && detectedLabel !== "Blank") {
        alphabetBox.value = detectedLabel;
        currentPrediction = detectedLabel;
    } else if (detectedLabel === "Blank") {
        alphabetBox.value = "---";
        currentPrediction = "";
    }

    inputTensor.dispose();
    prediction.dispose();
}

// WEBCAM LOOP
async function predictWebcam() {
    if (video.currentTime !== lastVideoTime) {
        canvas.width = video.videoWidth; 
        canvas.height = video.videoHeight;
        lastVideoTime = video.currentTime;
        
        const result = handLandmarker.detectForVideo(video, performance.now());
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (result.landmarks && result.landmarks.length > 0) {
            drawUI(result.landmarks[0]);
            await runInference(result.landmarks[0]);
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

// NEW: IMAGE UPLOAD HANDLER
imageUpload.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        // Stop the webcam loop momentarily by clearing srcObject if you want a freeze frame effect
        // or just draw the image over the canvas
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        // Change mode to IMAGE for better static accuracy
        await handLandmarker.setOptions({ runningMode: "IMAGE" });
        const result = await handLandmarker.detect(img);

        if (result.landmarks && result.landmarks.length > 0) {
            drawUI(result.landmarks[0]);
            await runInference(result.landmarks[0]);
        } else {
            alert("No hand landmarks detected in this image.");
        }

        // Switch back to VIDEO mode for the webcam
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    };
});

// --- BUTTON EVENT LISTENERS ---
document.getElementById("submitBtn").addEventListener("click", () => {
    if (currentPrediction) wordBox.value += currentPrediction;
});

document.getElementById("clearBtn").addEventListener("click", () => {
    wordBox.value = ""; sentenceBox.value = ""; currentPrediction = ""; alphabetBox.value = "";
});

document.getElementById("generateBtn").addEventListener("click", () => {
    const word = wordBox.value;
    if (word) {
        sentenceBox.value = `The user signed: ${word}`;
        const speech = new SpeechSynthesisUtterance(word);
        window.speechSynthesis.speak(speech);
    }
});

initialize();
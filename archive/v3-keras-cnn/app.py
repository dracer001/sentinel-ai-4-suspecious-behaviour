import os
import uuid
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Allows connection from separate frontend servers if needed

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 1. SETTINGS & PATHS MATCHING TRAINING DATA ---
MODEL_PATH = 'baseline_office_model.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Order matches the original training directory layout precisely
NEW_CLASSES = ['Fighting', 'Harvoc', 'NormalVideos', 'Stealing']

if os.path.exists(MODEL_PATH):
    print(f"⚙️ Loading multiclass model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"⚠️ Warning: {MODEL_PATH} not found. Running mock inferences for structural testing.")
    model = None

# In-memory history database
history_db = []

# --- 2. IMAGE PREPROCESSING CONFIGURATION ---
def preprocess_frame(frame):
    """Resizes and converts the frame. 
    Pixel values are kept in the [0, 255] range because the model architecture 
    contains its own built-in preprocess_input functional layer."""
    resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Cast to float32 but DO NOT apply preprocess_input externally
    float_img = np.array(rgb, dtype=np.float32)
    return np.expand_dims(float_img, axis=0)

def run_inference(processed_data):
    """Parses multiclass softmax probability arrays across all target categories."""
    if model:
        # verbose=0 completely removes the prediction progress bar spam from terminal logs
        prediction = model.predict(processed_data, verbose=0)
        
        # Extract the index of the highest softmax probability value
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        label = NEW_CLASSES[class_idx]
    else:
        confidence = float(np.random.uniform(0.6, 0.95))
        label = np.random.choice(NEW_CLASSES)
    return label, confidence

# --- 3. CORE ROUTING ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_asset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file element delivered'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No asset selected'}), 400

    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    file_extension = os.path.splitext(filename)[1].lower()
    is_video = file_extension in ['.mp4', '.avi', '.mov', '.mkv']
    
    highest_confidence = 0.0
    final_label = "NormalVideos"
    frames_processed = 0
    anomaly_frames_count = 0

    if is_video:
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample every 10th frame to maximize processing speed
            if frames_processed % 10 == 0:
                tensor = preprocess_frame(frame)
                lbl, conf = run_inference(tensor)
                
                # Video Aggregation Logic: Prioritize reporting anomalies over normal frames
                if lbl != "NormalVideos":
                    anomaly_frames_count += 1
                    if final_label == "NormalVideos" or conf > highest_confidence:
                        highest_confidence = conf
                        final_label = lbl
                else:
                    if final_label == "NormalVideos" and conf > highest_confidence:
                        highest_confidence = conf
                        
            frames_processed += 1
        cap.release()
    else:
        # Static Image Processing
        img = cv2.imread(file_path)
        tensor = preprocess_frame(img)
        final_label, highest_confidence = run_inference(tensor)

    # Human-friendly label conversions for UI display formatting
    display_label = "Normal Activity" if final_label == "NormalVideos" else f"{final_label} Detected"

    # Compile log payload
    record = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file.filename,
        "type": "Video" if is_video else "Image",
        "result": display_label,
        "confidence": f"{highest_confidence * 100:.2f}%",
        "status_class": "success" if final_label == "NormalVideos" else "danger"
    }
    
    # Prepend record to history log
    history_db.insert(0, record)
    return jsonify(record)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(history_db)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
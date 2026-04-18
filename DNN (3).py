import os
import cv2
import json
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# =========================
# STORAGE CONFIGURATION
# =========================
BASE_LOG_DIR = "herbs_magic_data_collection"
IMAGES_DIR = os.path.join(BASE_LOG_DIR, "captured_faces")
METADATA_DIR = os.path.join(BASE_LOG_DIR, "metadata")

for folder in [IMAGES_DIR, METADATA_DIR]:
    os.makedirs(folder, exist_ok=True)

# =========================
# MULTI-CLASS CONFIGURATION
# =========================
DISEASE_CLASSES = [
    'acne', 'blackheads', 'c', 'dark circle', 'dry', 
    'normal', 'oily', 'pigmentation', 'wrinkles'
]

MODEL_PATH = "herbs_magic_multi_disease_v1.h5"

print("Loading Unified Multi-Disease Model...")
try:
    # Loading without compile to ensure compatibility across environments
    unified_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Unified Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading Unified Model: {e}")

# Load Face Detection
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# =========================
# HELPERS
# =========================
def preprocess(face_rgb):
    """Resizes and scales image for MobileNetV2 (-1 to 1)"""
    face = cv2.resize(face_rgb, (224, 224))
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    return np.expand_dims(face, axis=0)

# =========================
# MAIN API ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    # 1. Face Detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    final_results = []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Crop and Preprocess
            face_bgr = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_input = preprocess(face_rgb)

            # 2. Unified Prediction
            # predictions is an array of 9 probabilities
            predictions = unified_model.predict(face_input, verbose=0)[0]
            
            cnn_scores = {}
            for idx, disease_name in enumerate(DISEASE_CLASSES):
                cnn_scores[disease_name] = round(float(predictions[idx]), 3)

            # Determine the primary disease (the one with the highest probability)
            top_idx = np.argmax(predictions)
            top_disease = DISEASE_CLASSES[top_idx]
            top_confidence = float(predictions[top_idx])

            # 3. Logging for Data Collection
            log_id = str(uuid.uuid4())
            
            # Save the raw face crop (clean, no boxes)
            img_path = os.path.join(IMAGES_DIR, f"{log_id}.jpg")
            cv2.imwrite(img_path, face_bgr)
            
            # Save Metadata JSON
            meta_data = {
                "log_id": log_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "top_prediction": top_disease,
                "confidence": top_confidence,
                "all_scores": cnn_scores,
                "reviewed": False  # Ready for manual labeling later
            }
            
            with open(os.path.join(METADATA_DIR, f"{log_id}.json"), "w") as f:
                json.dump(meta_data, f, indent=4)

            final_results.append({
                "log_id": log_id,
                "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "primary_diagnosis": top_disease,
                "confidence": round(top_confidence, 4),
                "full_report": cnn_scores
            })

    return jsonify({
        "status": "success",
        "faces_detected": len(final_results), 
        "results": final_results
    })

if __name__ == "__main__":
    # Standard Flask port for development
    app.run(debug=True, port=5000)
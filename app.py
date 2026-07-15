# app.py - Flask server for Railway deployment
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import io
import os
import time
import uuid
import json
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "uploads"
MODEL_DIR = "visaia_models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Railway provides PORT environment variable
PORT = int(os.environ.get("PORT", 8000))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenetv3_faw_vs_notfaw.pth")
YOLO_PATH = os.path.join(MODEL_DIR, "best.pt")

# -----------------------------
# DOWNLOAD MODELS FROM URL (if they don't exist)
# -----------------------------
def download_model(url, destination_path):
    """Download a model file from a URL if it doesn't exist locally."""
    if os.path.exists(destination_path):
        print(f"✅ Model already exists: {destination_path}")
        return True
    
    print(f"📥 Downloading model from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get total file size for progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}%", end='\r')
        
        print(f"\n✅ Model downloaded successfully: {destination_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        return False

# Get model URLs from environment variables
# You'll set these in Railway dashboard
MOBILENET_URL = os.environ.get("MOBILENET_URL", "")
YOLO_URL = os.environ.get("YOLO_URL", "")

if not MOBILENET_URL or not YOLO_URL:
    print("⚠️ WARNING: Model URLs not set in environment variables!")
    print("Please set MOBILENET_URL and YOLO_URL in Railway dashboard.")
    print("The server will start but prediction will fail without models.")
else:
    # Download models if they don't exist
    download_model(MOBILENET_URL, MOBILENET_PATH)
    download_model(YOLO_URL, YOLO_PATH)

# -----------------------------
# LOAD MOBILE NET
# -----------------------------
print("🚀 Loading MobileNetV3...")

if not os.path.exists(MOBILENET_PATH):
    raise FileNotFoundError(f"MobileNet model not found at {MOBILENET_PATH}")

mobilenet_model = models.mobilenet_v3_small(pretrained=False)
mobilenet_model.classifier[3] = torch.nn.Linear(
    mobilenet_model.classifier[3].in_features, 2
)

mobilenet_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
mobilenet_model.to(DEVICE)
mobilenet_model.eval()

CLASS_NAMES = ["FAW", "NotFAW"]

mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD YOLO
# -----------------------------
print("🚀 Loading YOLOv8...")

if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f"YOLO model not found at {YOLO_PATH}")

yolo_model = YOLO(YOLO_PATH)

FAW_CLASSES = ["egg", "larva", "pupa", "moth"]

LIFE_STAGE_RISK = {
    "egg": "Low",
    "larva": "High",
    "pupa": "Low",
    "moth": "High"
}

# -----------------------------
# RULE-BASED INSIGHTS
# -----------------------------
def get_rule_based_insights(detected_stages, risk_level):
    """Generate insights based on detected stages without LLM"""
    
    if not detected_stages:
        return {
            "analysis": "No FAW life stages detected in this image.",
            "treatment": "Continue regular monitoring. No immediate action required."
        }
    
    stage_counts = {}
    for stage in detected_stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    analysis = f"Detected {len(detected_stages)} FAW specimens: "
    for stage, count in stage_counts.items():
        analysis += f"{count} {stage}(s), "
    analysis = analysis.rstrip(", ")
    
    if risk_level == "High":
        analysis += ". ⚠️ High risk level due to presence of larvae or moths."
    else:
        analysis += ". ✅ Low to moderate risk level."
    
    # Treatment recommendations
    if "larva" in detected_stages:
        treatment = "IMMEDIATE ACTION: Apply biological control (Bacillus thuringiensis, neem extract) or chemical control (spinosad, chlorantraniliprole) targeting larvae. Follow BPI recommendations for pesticide application."
    elif "moth" in detected_stages:
        treatment = "MONITORING: Use pheromone traps (4-5 per hectare). Replace lures every 6 weeks. Consider mating disruption techniques."
    elif "egg" in detected_stages:
        treatment = "PREVENTION: Remove egg masses manually. Introduce natural predators (Trichogramma wasps). Monitor for hatching."
    elif "pupa" in detected_stages:
        treatment = "SOIL MANAGEMENT: Deep plowing to expose pupae. Maintain crop rotation. Monitor soil conditions."
    else:
        treatment = "Continue regular field scouting using BPI W-pattern method. Maintain monitoring protocols."
    
    return {
        "analysis": analysis,
        "treatment": treatment
    }

# -----------------------------
# IMAGE ANNOTATION
# -----------------------------
def annotate_image(image, detections):
    draw = ImageDraw.Draw(image, "RGBA")
    img_w, img_h = image.size

    COLORS = {
        "egg": (255, 215, 0),
        "larva": (255, 0, 0),
        "pupa": (255, 140, 0),
        "moth": (0, 120, 255)
    }

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for det in detections:
        stage = det["class"]
        conf = det.get("confidence", 0)

        x1 = int(det["x"] * img_w)
        y1 = int(det["y"] * img_h)
        x2 = int((det["x"] + det["width"]) * img_w)
        y2 = int((det["y"] + det["height"]) * img_h)

        color = COLORS.get(stage, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{stage.upper()} {conf*100:.1f}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_y = y1 - text_h - 6
        if label_y < 0:
            label_y = y1 + 4

        draw.rectangle(
            [x1, label_y, x1 + text_w + 6, label_y + text_h + 4],
            fill=(*color, 180)
        )
        draw.text(
            (x1 + 3, label_y + 2),
            label,
            fill=(0, 0, 0),
            font=font
        )

    return image

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return "<h1>FAW Server Running on Railway</h1>"

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "device": str(DEVICE),
        "models_loaded": os.path.exists(MOBILENET_PATH) and os.path.exists(YOLO_PATH)
    })

@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    filename = secure_filename(file.filename)
    unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # MobileNet classification
        tensor = mobilenet_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = mobilenet_model(tensor)
            _, pred = torch.max(outputs, 1)

        prediction = CLASS_NAMES[pred.item()]

        # FAW detected -> YOLO
        if prediction == "FAW":
            results = yolo_model(image)

            detected = []
            boxes = []
            w, h = image.size

            for r in results:
                for box in r.boxes:
                    cls = FAW_CLASSES[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    boxes.append({
                        "class": cls,
                        "x": x1 / w,
                        "y": y1 / h,
                        "width": (x2 - x1) / w,
                        "height": (y2 - y1) / h,
                        "confidence": conf
                    })
                    detected.append(cls)

            # Annotate and save
            annotated = annotate_image(image.copy(), boxes)
            out_name = "annotated_" + unique_name
            annotated.save(os.path.join(UPLOAD_FOLDER, out_name))

            # Determine risk
            risk = "High" if any(LIFE_STAGE_RISK.get(d, "Low") == "High" for d in detected) else "Low"

            # Get insights
            insights = get_rule_based_insights(detected, risk)

            # Generate URL for the annotated image
            railway_url = os.environ.get("RAILWAY_STATIC_URL", f"https://{request.host}")
            image_url = f"{railway_url}/uploads/{out_name}"

            return jsonify({
                "pest": "Fall Army Worm",
                "stages": detected,
                "risk": risk,
                "boxes": boxes,
                "analysis": insights["analysis"],
                "treatment": insights["treatment"],
                "image_url": image_url
            })

        # Not FAW
        else:
            # Get insights for negative case
            insights = get_rule_based_insights([], "Low")
            
            # Generate URL for the annotated image
            railway_url = os.environ.get("RAILWAY_STATIC_URL", f"https://{request.host}")
            image_url = f"{railway_url}/uploads/{unique_name}"

            return jsonify({
                "pest": "Unknown / Not FAW",
                "analysis": "The image does not appear to contain Fall Armyworm (FAW). " + insights["analysis"],
                "treatment": insights["treatment"],
                "image_url": image_url
            })

    except Exception as e:
        print("❌ SERVER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
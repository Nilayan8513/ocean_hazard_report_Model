
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# Load dataset
with open("data/social_media_hazard_feed_3way.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

# Load HuggingFace model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "Nilayan87/ocean_hazard"  # your HuggingFace repo

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# FastAPI app
app = FastAPI(title="INCOIS Hazard API with NLP", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🌊 INCOIS Hazard Reporting API with NLP is running"}

@app.get("/hazards")
def get_hazards():
    hazard_reports = [p for p in posts if p.get("classification") == "Hazard Report"]
    return {"count": len(hazard_reports), "data": hazard_reports}

@app.get("/hazards/heatmap")
def get_heatmap():
    location_coords = {
        "Puri, Odisha": (19.8135, 85.8312),
        "Paradip, Odisha": (20.316, 86.6106),
        "Gopalpur, Odisha": (19.2813, 84.9057),
        "Chennai, Tamil Nadu": (13.0827, 80.2707),
        "Nagapattinam, Tamil Nadu": (10.766, 79.8424),
        "Cuddalore, Tamil Nadu": (11.7447, 79.7680),
        "Vizag, Andhra Pradesh": (17.6868, 83.2185),
        "Nellore, Andhra Pradesh": (14.4426, 79.9865),
        "Machilipatnam, Andhra Pradesh": (16.1875, 81.1389),
        "Kochi, Kerala": (9.9312, 76.2673),
        "Alappuzha, Kerala": (9.4981, 76.3388),
        "Kozhikode, Kerala": (11.2588, 75.7804),
        "Digha, West Bengal": (21.628, 87.507),
        "Sagar Island, West Bengal": (21.646, 88.110),
        "Bakkhali, West Bengal": (21.562, 88.264),
        "Goa": (15.2993, 74.1240),
        "Mumbai": (19.0760, 72.8777),
        "Andaman": (11.7401, 92.6586),
        "India": (22.9734, 78.6569)
    }
    heatmap_points = []
    for p in posts:
        if p.get("classification") != "Hazard Report":
            continue
        loc = p.get("location")
        coords = location_coords.get(loc)
        if coords:
            heatmap_points.append({"lat": coords[0], "lon": coords[1]})
    return {"count": len(heatmap_points), "points": heatmap_points}

@app.post("/classify/")
def classify_post(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "Hazard Report", 1: "Unverified Hazard", 2: "Noise/Irrelevant"}
    return {"text": text, "prediction": label_map.get(pred, "Unknown")}

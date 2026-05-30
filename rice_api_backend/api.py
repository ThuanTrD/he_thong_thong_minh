# api.py
# Điểm vào chính của REST API sử dụng FastAPI

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

from .config import MODEL_PATH
from .image_preprocessing import preprocess_image
from .disease_classifier import DiseaseClassifier
from .ood_detector import OODDetector
from .decision_engine import DecisionEngine
from .recommendation_service import RecommendationService

app = FastAPI(
    title="Rice Disease OOD Detection API",
    description="API Chẩn đoán Bệnh Lúa An Toàn tích hợp phát hiện Out-of-Distribution",
    version="1.0.0"
)

# Load resources globally
print("Loading model and services...")
device_str = "cuda" if torch.cuda.is_available() else "cpu"
classifier = DiseaseClassifier(ckpt_path=MODEL_PATH, device_str=device_str)
ood_detector = OODDetector()
decision_engine = DecisionEngine()
recommender = RecommendationService()
print("All systems ready.")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Rice API Backend is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Preprocessing
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes, img_size=classifier.img_size)
        
        # 2. CNN Classifier
        cnn_result = classifier.predict(tensor)
        
        # 3. OOD Detection
        ood_result = ood_detector.detect(cnn_result["probs_array"], cnn_result["scores"])
        
        # 4. Decision Engine
        decision = decision_engine.make_decision(cnn_result, ood_result)
        
        # 5. Recommendation
        recommendation = recommender.get_recommendation(decision)
        
        # Output JSON
        response = {
            "status": decision["status"],
            "predicted_class": decision["predicted_class"],
            "confidence": decision["confidence"],
            "top_group": decision.get("top_group"),
            "group_confidence": decision.get("group_confidence"),
            "is_ood": decision["is_ood"],
            "message": decision["message"],
            "recommendation": recommendation,
            "debug_info": {
                "max_softmax_probability": cnn_result["confidence"],
                "shannon_entropy": decision["entropy"],
                "raw_cnn_prediction": cnn_result["pred_class"],
                "cnn_scores": cnn_result["scores"]
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("rice_api_backend.api:app", host="0.0.0.0", port=8000, reload=True)

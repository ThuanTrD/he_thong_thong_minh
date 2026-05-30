# disease_classifier.py
# Module chịu trách nhiệm load CNN model và sinh Softmax probabilities

import torch
import torch.nn as nn
from torchvision import models

class DiseaseClassifier:
    def __init__(self, ckpt_path: str, device_str: str = "cpu"):
        self.device = torch.device(device_str)
        self.model, self.classes, self.img_size = self._load_trained_model(ckpt_path, self.device)

    def _load_trained_model(self, ckpt_path: str, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone = ckpt["backbone"]
        num_classes = ckpt["num_classes"]
        classes = ckpt["classes"]
        img_size = ckpt.get("img_size", 224)

        if backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Backbone {backbone} not supported in this API setup yet.")

        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model, classes, img_size

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor):
        """Trả về probability distribution."""
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        
        pred_idx = int(probs.argmax())
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        return {
            "pred_class": pred_class,
            "confidence": confidence,
            "scores": scores,
            "probs_array": probs
        }

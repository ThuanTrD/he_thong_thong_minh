"""
=============================================================================
CNN INFERENCE — BƯỚC 2 (Inference Mode)
=============================================================================
Dùng file trọng số đã train để:
  1. Chạy inference trên 1 ảnh mới  → trả về dict scores cho Fuzzy
  2. Chạy batch inference trên folder → CSV scores

Đây là module tích hợp vào pipeline Hybrid CNN+Fuzzy.
=============================================================================
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
def load_trained_model(ckpt_path: str, device: torch.device):
    """Load model từ file .pt đã train."""
    ckpt = torch.load(ckpt_path, map_location=device)

    backbone    = ckpt["backbone"]
    num_classes = ckpt["num_classes"]
    classes     = ckpt["classes"]
    img_size    = ckpt.get("img_size", 224)

    # Rebuild model architecture
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    print(f"[MODEL] Loaded: {backbone} | Classes: {classes}")
    return model, classes, img_size


# ─────────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────────
def get_inference_transform(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ─────────────────────────────────────────────
#  SINGLE IMAGE INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_single(
    image_path: str,
    model: nn.Module,
    classes: list,
    transform,
    device: torch.device
) -> dict:
    """
    Chạy CNN inference trên 1 ảnh.

    Returns:
        dict: {
            "image_path": str,
            "pred_class": str,
            "pred_confidence": float,
            "scores": {class_name: float, ...}   ← đây là input cho Fuzzy Layer
        }
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    scores = {c: round(float(probs[i]), 6) for i, c in enumerate(classes)}

    result = {
        "image_path"      : str(image_path),
        "pred_class"      : pred_class,
        "pred_confidence" : round(confidence, 6),
        "scores"          : scores,   # ← Fuzzy Layer nhận cái này
    }
    return result


# ─────────────────────────────────────────────
#  BATCH INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_batch(
    image_folder: str,
    model: nn.Module,
    classes: list,
    transform,
    device: torch.device,
    output_csv: str = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Batch inference trên folder ảnh.
    Trả về DataFrame với softmax scores — giao diện với Fuzzy Layer.
    """
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = [
        p for p in Path(image_folder).rglob("*")
        if p.suffix.lower() in EXTS
    ]
    print(f"[BATCH] Tìm thấy {len(img_paths)} ảnh")

    rows = []
    # Mini-batch để tăng tốc
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        tensors = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"  Bỏ qua {p}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

        for j, path in enumerate(valid_paths):
            row = {
                "image_path" : str(path),
                "pred_class" : classes[int(np.argmax(probs[j]))],
                "confidence" : round(float(probs[j].max()), 6),
            }
            for c_idx, c_name in enumerate(classes):
                row[f"score_{c_name}"] = round(float(probs[j, c_idx]), 6)
            rows.append(row)

        print(f"  Processed {min(i+batch_size, len(img_paths))}/{len(img_paths)}")

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"[BATCH] Saved: {output_csv}")

    return df


# ─────────────────────────────────────────────
#  FUZZY LAYER INTERFACE
# ─────────────────────────────────────────────
def get_fuzzy_input(result: dict) -> dict:
    """
    Chuẩn bị dict input cho Fuzzy Reasoning Layer.
    Team Fuzzy gọi hàm này sau khi có kết quả CNN.

    Example output:
    {
        "cnn_scores": {
            "BrownSpot_Mild": 0.12,
            "BrownSpot_Severe": 0.75,
            "Healthy": 0.05,
            ...
        },
        "top_class": "BrownSpot_Severe",
        "top_confidence": 0.75
    }
    """
    return {
        "cnn_scores"     : result["scores"],
        "top_class"      : result["pred_class"],
        "top_confidence" : result["pred_confidence"],
    }


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CNN Inference for Rice Disease")
    parser.add_argument("--ckpt",   type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--image",  type=str, default=None,  help="Single image path")
    parser.add_argument("--folder", type=str, default=None,  help="Folder for batch inference")
    parser.add_argument("--output", type=str, default="cnn_scores_output.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, img_size = load_trained_model(args.ckpt, device)
    transform = get_inference_transform(img_size)

    if args.image:
        result = predict_single(args.image, model, classes, transform, device)
        fuzzy_input = get_fuzzy_input(result)

        print("\n" + "="*50)
        print("  CNN INFERENCE RESULT")
        print("="*50)
        print(f"  Image     : {result['image_path']}")
        print(f"  Prediction: {result['pred_class']} ({result['pred_confidence']:.4f})")
        print(f"\n  Scores (→ Fuzzy Layer):")
        for cls, score in result["scores"].items():
            bar = "█" * int(score * 30)
            print(f"    {cls:<30} {score:.4f}  {bar}")
        print("="*50)
        print("\n[JSON] Fuzzy Input:")
        print(json.dumps(fuzzy_input, indent=2, ensure_ascii=False))

    elif args.folder:
        df = predict_batch(
            args.folder, model, classes, transform,
            device, output_csv=args.output, batch_size=args.batch_size
        )
        print(f"\n[DONE] {len(df)} ảnh đã xử lý. Kết quả: {args.output}")
        print(df.head())

    else:
        print("Dùng --image hoặc --folder để chạy inference.")


if __name__ == "__main__":
    main()

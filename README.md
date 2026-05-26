# he_thong_thong_minh
#  CNN Module — Bước 2: Pipeline Hybrid CNN + Fuzzy (XAI)

## Tổng quan

Module này thực hiện **Bước 2 (CNN Inference)** trong pipeline:

```
[INPUT] → [CNN Inference] → [Fuzzy Reasoning] → [Hybrid Decision] → [Output + XAI]
              ↑ module này
```

Dataset: [Severity-Based Rice Leaf Diseases (Kaggle)](https://www.kaggle.com/datasets/isaacritharson/severity-based-rice-leaf-diseases-dataset)

---

## Cấu trúc thư mục

```
project/
├── train_cnn.py          ← Script train CNN
├── inference_cnn.py      ← Script inference (input cho Fuzzy)
├── requirements.txt
├── dataset/              ← Đặt dataset Kaggle vào đây
│   ├── train/
│   │   ├── BrownSpot/
│   │   ├── Healthy/
│   │   └── ...
│   └── val/              (nếu có, nếu không sẽ auto-split)
├── checkpoints/          ← .pt weights được lưu ở đây
│   ├── best_model.pt
│   └── last_model.pt
└── outputs/              ← CSV scores + metadata cho team Fuzzy
    ├── val_cnn_scores.csv
    ├── class_metadata.json
    ├── training_history.csv
    └── training_report.txt
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## 1. Train CNN

```bash
# Cơ bản (dùng EfficientNet-B0, auto-split 80/20)
python train_cnn.py --data_dir ./dataset --epochs 30

# Tùy chỉnh đầy đủ
python train_cnn.py \
  --data_dir    ./dataset \
  --backbone    efficientnet_b0 \       # hoặc: resnet50, mobilenet_v3_small
  --epochs      30 \
  --batch_size  32 \
  --lr          1e-3 \
  --img_size    224 \
  --freeze_epochs 5 \                   # đóng băng backbone 5 epoch đầu
  --val_split   0.2 \
  --output_dir  ./outputs \
  --ckpt_dir    ./checkpoints
```

### Chiến lược training

| Phase | Epochs | Backbone | Head |
|-------|--------|----------|------|
| Warmup | 1–5 | ❄️ Frozen | ✅ Train (LR × 5) |
| Fine-tune | 6–30 | ✅ Unfreeze | ✅ Train (LR normal) |

---

## 2. Inference (output cho Fuzzy Layer)

### Một ảnh đơn lẻ
```bash
python inference_cnn.py \
  --ckpt  ./checkpoints/best_model.pt \
  --image ./test_leaf.jpg
```

Output:
```
==================================================
  CNN INFERENCE RESULT
==================================================
  Image     : test_leaf.jpg
  Prediction: BrownSpot_Severe (0.7523)

  Scores (→ Fuzzy Layer):
    BrownSpot_Mild            0.0821  ██
    BrownSpot_Moderate        0.1156  ███
    BrownSpot_Severe          0.7523  ██████████████████████
    Healthy                   0.0500  █
==================================================

[JSON] Fuzzy Input:
{
  "cnn_scores": {
    "BrownSpot_Mild": 0.0821,
    "BrownSpot_Moderate": 0.1156,
    "BrownSpot_Severe": 0.7523,
    "Healthy": 0.05
  },
  "top_class": "BrownSpot_Severe",
  "top_confidence": 0.7523
}
```

### Batch inference (nhiều ảnh)
```bash
python inference_cnn.py \
  --ckpt    ./checkpoints/best_model.pt \
  --folder  ./test_images/ \
  --output  ./outputs/batch_scores.csv
```

---

## 3. Tích hợp vào Pipeline (cho team Fuzzy)

```python
# ── Import CNN module ──────────────────────
from inference_cnn import load_trained_model, predict_single, get_fuzzy_input
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model một lần (reuse nhiều lần)
model, classes, img_size = load_trained_model("checkpoints/best_model.pt", device)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Mỗi ca kiểm nghiệm ────────────────────
result = predict_single("leaf_image.jpg", model, classes, transform, device)

# Lấy input cho Fuzzy Layer
fuzzy_input = get_fuzzy_input(result)
# fuzzy_input["cnn_scores"] = {"BrownSpot_Mild": 0.08, "BrownSpot_Severe": 0.75, ...}
# → Truyền vào Fuzzy Reasoning Layer
```

---

## Output Files cho Team Fuzzy

### `val_cnn_scores.csv`
| image_path | true_label | pred_label | correct | score_BrownSpot | score_Healthy | ... |
|---|---|---|---|---|---|---|
| img001.jpg | BrownSpot_Severe | BrownSpot_Severe | 1 | 0.7523 | 0.0500 | ... |

> **Cột `score_*`** là softmax probability ∈ [0, 1], tổng = 1.  
> Đây là input chính cho Fuzzy Reasoning Layer.

### `class_metadata.json`
```json
{
  "backbone": "efficientnet_b0",
  "num_classes": 8,
  "classes": ["BrownSpot_Mild", "BrownSpot_Moderate", ...],
  "class_to_idx": {"BrownSpot_Mild": 0, ...},
  "fuzzy_input_columns": ["score_BrownSpot_Mild", ...],
  "note": "Dùng cột score_* làm input cho Fuzzy Reasoning Layer."
}
```

---

## Ghi chú kỹ thuật

- **Label smoothing = 0.1**: Giúp softmax scores calibrated hơn (không bị cực đoan 0.99/0.01), phù hợp làm input Fuzzy
- **Freeze warmup**: Tránh phá vỡ pretrained features ngay từ đầu
- **AMP (Mixed Precision)**: Tự động bật trên GPU, tăng tốc ~2x
- **Best model**: Lưu theo `val_accuracy` cao nhất


## kết quả
==================================================
  CNN INFERENCE RESULT
==================================================
  Image     : D:\rice_fuzzy\rice_data\Leaf Disease Dataset\validation\Mild 
  
  Blast\BLAST9_087.jpg
  
  Prediction: Mild Blast (0.9747)

  Scores (→ Fuzzy Layer):
    Healthy                        0.0020  
    Mild Bacterial blight          0.0044  
    Mild Blast                     0.9747  █████████████████████████████
    Mild Brownspot                 0.0043  
    Mild Tungro                    0.0019  
    Severe Bacterial blight        0.0026  
    Severe Blast                   0.0070  
    Severe Brownspot               0.0008  
    Severe Tungro                  0.0022  
==================================================

[JSON] Fuzzy Input:
{
  "cnn_scores": {
    "Healthy": 0.001969,
    "Mild Bacterial blight": 0.004422,
    "Mild Blast": 0.974731,
    "Mild Brownspot": 0.004255,
    "Mild Tungro": 0.001899,
    "Severe Bacterial blight": 0.002626,
    "Severe Blast": 0.007046,
    "Severe Brownspot": 0.000837,
    "Severe Tungro": 0.002216
  },
  "top_class": "Mild Blast",
  "top_confidence": 0.974731
}
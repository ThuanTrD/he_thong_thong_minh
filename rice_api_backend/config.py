# config.py
# Cấu hình các tham số và ngưỡng cho OOD Detection Backend

import os

# Ngưỡng (Thresholds) cho hệ thống OOD theo Group Confidence
GROUP_CONFIDENCE_THRESHOLD = 0.60  # Ngưỡng an toàn (KNOWN)
UNCERTAIN_THRESHOLD = 0.40         # Ngưỡng trung gian (UNCERTAIN), dưới mức này là OOD
ENTROPY_THRESHOLD = 0.65           # Ngưỡng Normalized Entropy

# Cấu hình mô hình
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rice_disease_cnn_outputs", "best_model.pt")

# Ánh xạ nhãn bệnh từ mô hình (UI name)
DISEASE_CLASSES_MAP = {
    "Healthy": "Khỏe mạnh",
    "Mild Bacterial blight": "Bạc lá (Bacterial blight)",
    "Severe Bacterial blight": "Bạc lá (Bacterial blight)",
    "Mild Blast": "Đạo ôn (Blast)",
    "Severe Blast": "Đạo ôn (Blast)",
    "Mild Brownspot": "Đốm nâu (Brownspot)",
    "Severe Brownspot": "Đốm nâu (Brownspot)",
    "Mild Tungro": "Vàng lụi (Tungro)",
    "Severe Tungro": "Vàng lụi (Tungro)",
}

# Semantic Groups để gộp xác suất
DISEASE_GROUPS = {
    "Healthy": ["Healthy"],
    "Bacterial blight": ["Mild Bacterial blight", "Severe Bacterial blight"],
    "Blast": ["Mild Blast", "Severe Blast"],
    "Brownspot": ["Mild Brownspot", "Severe Brownspot"],
    "Tungro": ["Mild Tungro", "Severe Tungro"],
}

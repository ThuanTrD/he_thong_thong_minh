# Config cho Package rice_fuzzy_xai

# Giá trị môi trường mặc định nếu người dùng không cung cấp
DEFAULT_TEMP = 25.0      # Nhiệt độ trung bình (°C)
DEFAULT_HUMIDITY = 70.0  # Độ ẩm trung bình (%)

# Ánh xạ các lớp từ mô hình CNN sang nhóm bệnh chính
DISEASE_CLASSES_MAP = {
    "Healthy": "Healthy",
    "Mild Bacterial blight": "Bacterial blight",
    "Severe Bacterial blight": "Bacterial blight",
    "Mild Blast": "Blast",
    "Severe Blast": "Blast",
    "Mild Brownspot": "Brownspot",
    "Severe Brownspot": "Brownspot",
    "Mild Tungro": "Tungro",
    "Severe Tungro": "Tungro",
}

# Ánh xạ mức độ nghiêm trọng từ nhãn của CNN
SEVERITY_CLASSES_MAP = {
    "Healthy": "Healthy",
    "Mild Bacterial blight": "Mild",
    "Severe Bacterial blight": "Severe",
    "Mild Blast": "Mild",
    "Severe Blast": "Severe",
    "Mild Brownspot": "Mild",
    "Severe Brownspot": "Severe",
    "Mild Tungro": "Mild",
    "Severe Tungro": "Severe",
}

# Tên tiếng Việt của các loại bệnh để hiển thị trong báo cáo XAI
DISEASE_NAMES_VI = {
    "Healthy": "Khỏe mạnh (Không nhiễm bệnh)",
    "Bacterial blight": "Bạc lá vi khuẩn (Bacterial blight)",
    "Blast": "Đạo ôn lá (Rice Blast)",
    "Brownspot": "Đốm nâu (Brown spot)",
    "Tungro": "Vàng lụi (Tungro)",
}

# Membership Functions and Fuzzy Sets for rice_fuzzy_xai

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Hàm thành viên hình tam giác (Triangular Membership Function)."""
    assert a <= b <= c, f"Yêu cầu: a <= b <= c. Nhận: {a}, {b}, {c}"
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Hàm thành viên hình hình thang (Trapezoidal Membership Function)."""
    assert a <= b <= c <= d, f"Yêu cầu: a <= b <= c <= d. Nhận: {a}, {b}, {c}, {d}"
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (d - x) / (d - c)


# Định nghĩa các tập mờ (Fuzzy Sets) và các khoảng tham số (Ranges)

# 1. Điểm tự tin CNN / Độ tự tin (0.0 -> 1.0)
CONFIDENCE_MFS = {
    "Low": (0.0, 0.0, 0.25, 0.5),      # Trapmf
    "Medium": (0.3, 0.5, 0.7),         # Trimf
    "High": (0.5, 0.75, 1.0, 1.0)      # Trapmf
}

def fuzzify_confidence(val: float) -> dict:
    return {
        "Low": trapmf(val, *CONFIDENCE_MFS["Low"]),
        "Medium": trimf(val, *CONFIDENCE_MFS["Medium"]),
        "High": trapmf(val, *CONFIDENCE_MFS["High"])
    }


# 2. Khoảng cách Margin giữa Top 1 và Top 2 (0.0 -> 1.0)
# Biểu thị độ mơ hồ của dự đoán (Margin nhỏ = mơ hồ cao)
MARGIN_MFS = {
    "Small": (0.0, 0.0, 0.15, 0.35),   # Trapmf
    "Medium": (0.2, 0.4, 0.6),         # Trimf
    "Large": (0.45, 0.7, 1.0, 1.0)     # Trapmf
}

def fuzzify_margin(val: float) -> dict:
    return {
        "Small": trapmf(val, *MARGIN_MFS["Small"]),
        "Medium": trimf(val, *MARGIN_MFS["Medium"]),
        "Large": trapmf(val, *MARGIN_MFS["Large"])
    }


# 3. Nhiệt độ môi trường (15 -> 45°C)
TEMP_MFS = {
    "Cool": (15.0, 15.0, 20.0, 24.0),  # Trapmf
    "Warm": (20.0, 27.0, 34.0),        # Trimf
    "Hot": (30.0, 36.0, 45.0, 45.0)    # Trapmf
}

def fuzzify_temp(val: float) -> dict:
    return {
        "Cool": trapmf(val, *TEMP_MFS["Cool"]),
        "Warm": trimf(val, *TEMP_MFS["Warm"]),
        "Hot": trapmf(val, *TEMP_MFS["Hot"])
    }


# 4. Độ ẩm môi trường (40% -> 100%)
HUMIDITY_MFS = {
    "Dry": (40.0, 40.0, 55.0, 65.0),   # Trapmf
    "Moderate": (55.0, 70.0, 85.0),    # Trimf
    "Wet": (75.0, 85.0, 100.0, 100.0)  # Trapmf
}

def fuzzify_humidity(val: float) -> dict:
    return {
        "Dry": trapmf(val, *HUMIDITY_MFS["Dry"]),
        "Moderate": trimf(val, *HUMIDITY_MFS["Moderate"]),
        "Wet": trapmf(val, *HUMIDITY_MFS["Wet"])
    }

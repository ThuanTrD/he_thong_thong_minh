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


import json
import os

# Tải cấu hình từ fuzzy_config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "fuzzy_config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = json.load(f)
except Exception as e:
    print(f"Warning: Cannot load fuzzy_config.json ({e}). Using defaults.")
    _config = {
        "CONFIDENCE_MFS": {"Low": [0.0, 0.0, 0.25, 0.5], "Medium": [0.3, 0.5, 0.7], "High": [0.5, 0.75, 1.0, 1.0]},
        "MARGIN_MFS": {"Small": [0.0, 0.0, 0.15, 0.35], "Medium": [0.2, 0.4, 0.6], "Large": [0.45, 0.7, 1.0, 1.0]},
        "TEMP_MFS": {"Cool": [15.0, 15.0, 20.0, 24.0], "Warm": [20.0, 27.0, 34.0], "Hot": [30.0, 36.0, 45.0, 45.0]},
        "HUMIDITY_MFS": {"Dry": [40.0, 40.0, 55.0, 65.0], "Moderate": [55.0, 70.0, 85.0], "Wet": [75.0, 85.0, 100.0, 100.0]}
    }

CONFIDENCE_MFS = _config["CONFIDENCE_MFS"]
MARGIN_MFS = _config["MARGIN_MFS"]
TEMP_MFS = _config["TEMP_MFS"]
HUMIDITY_MFS = _config["HUMIDITY_MFS"]

# 1. Điểm tự tin CNN / Độ tự tin (0.0 -> 1.0)
def fuzzify_confidence(val: float) -> dict:
    return {
        "Low": trapmf(val, *CONFIDENCE_MFS["Low"]),
        "Medium": trimf(val, *CONFIDENCE_MFS["Medium"]),
        "High": trapmf(val, *CONFIDENCE_MFS["High"])
    }


# 2. Khoảng cách Margin giữa Top 1 và Top 2 (0.0 -> 1.0)
def fuzzify_margin(val: float) -> dict:
    return {
        "Small": trapmf(val, *MARGIN_MFS["Small"]),
        "Medium": trimf(val, *MARGIN_MFS["Medium"]),
        "Large": trapmf(val, *MARGIN_MFS["Large"])
    }


# 3. Nhiệt độ môi trường (15 -> 45°C)
def fuzzify_temp(val: float) -> dict:
    return {
        "Cool": trapmf(val, *TEMP_MFS["Cool"]),
        "Warm": trimf(val, *TEMP_MFS["Warm"]),
        "Hot": trapmf(val, *TEMP_MFS["Hot"])
    }


# 4. Độ ẩm môi trường (40% -> 100%)
def fuzzify_humidity(val: float) -> dict:
    return {
        "Dry": trapmf(val, *HUMIDITY_MFS["Dry"]),
        "Moderate": trimf(val, *HUMIDITY_MFS["Moderate"]),
        "Wet": trapmf(val, *HUMIDITY_MFS["Wet"])
    }

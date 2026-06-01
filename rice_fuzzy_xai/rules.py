# Fuzzy Rules and Singletons for rice_fuzzy_xai

# Singletons (hằng số đầu ra cho phương pháp Sugeno)

SUGENO_CONFIDENCE = {
    "Low": 25.0,
    "Medium": 60.0,
    "High": 85.0,
    "VeryHigh": 95.0
}

SUGENO_SEVERITY = {
    "Healthy": 0.0,
    "Mild": 20.0,
    "Moderate": 55.0,
    "Severe": 90.0
}

SUGENO_ENV_RISK = {
    "Low": 15.0,
    "Medium": 50.0,
    "High": 85.0
}

SUGENO_ALERT = {
    "Normal": 10.0,
    "Attention": 40.0,
    "Danger": 70.0,
    "RedAlert": 95.0
}


def evaluate_uncertainty(margin_fuzzy: dict) -> tuple:
    """
    Xác định mức độ bất định và luật tương ứng.
    Độ bất định dựa vào Margin:
      - Margin lớn (Large) -> Bất định Thấp (Low)
      - Margin trung bình (Medium) -> Bất định Vừa (Medium)
      - Margin nhỏ (Small) -> Bất định Cao (High)
    """
    w_low = margin_fuzzy["Large"]
    w_med = margin_fuzzy["Medium"]
    w_high = margin_fuzzy["Small"]
    
    # Tính theo nhãn chiếm ưu thế
    w_max = max(w_low, w_med, w_high)
    if w_max == w_low:
        level = "Low (Thấp)"
    elif w_max == w_med:
        level = "Medium (Trung bình)"
    else:
        level = "High (Cao)"
        
    fired_rules = []
    if w_low > 0:
        fired_rules.append((w_low, "Low", "NẾU Khoảng cách phân biệt (Margin) Lớn THÌ Độ bất định Thấp"))
    if w_med > 0:
        fired_rules.append((w_med, "Medium", "NẾU Khoảng cách phân biệt (Margin) Vừa THÌ Độ bất định Trung bình"))
    if w_high > 0:
        fired_rules.append((w_high, "High", "NẾU Khoảng cách phân biệt (Margin) Nhỏ THÌ Độ bất định Cao"))
        
    return level, fired_rules


def evaluate_diagnostic_confidence(top_conf_fuzzy: dict, margin_fuzzy: dict) -> float:
    """
    Tính toán độ tin cậy chẩn đoán (0% - 100%) bằng Sugeno FIS.
    Kết hợp giữa Top Confidence (C) và Margin (M).
    """
    rules = []
    
    # Rule 1: C High and M Large -> Very High confidence (95%)
    w1 = min(top_conf_fuzzy["High"], margin_fuzzy["Large"])
    if w1 > 0:
        rules.append((w1, SUGENO_CONFIDENCE["VeryHigh"], "NẾU Top Confidence Cao VÀ Margin Lớn THÌ Độ tin cậy chẩn đoán Rất cao"))
        
    # Rule 2: C High and M Medium -> High confidence (85%)
    w2 = min(top_conf_fuzzy["High"], margin_fuzzy["Medium"])
    if w2 > 0:
        rules.append((w2, SUGENO_CONFIDENCE["High"], "NẾU Top Confidence Cao VÀ Margin Vừa THÌ Độ tin cậy chẩn đoán Cao"))
        
    # Rule 3: C Medium and M Large -> High confidence (85%)
    w3 = min(top_conf_fuzzy["Medium"], margin_fuzzy["Large"])
    if w3 > 0:
        rules.append((w3, SUGENO_CONFIDENCE["High"], "NẾU Top Confidence Vừa VÀ Margin Lớn THÌ Độ tin cậy chẩn đoán Cao"))

    # Rule 4: C Medium and M Medium -> Medium confidence (60%)
    w4 = min(top_conf_fuzzy["Medium"], margin_fuzzy["Medium"])
    if w4 > 0:
        rules.append((w4, SUGENO_CONFIDENCE["Medium"], "NẾU Top Confidence Vừa VÀ Margin Vừa THÌ Độ tin cậy chẩn đoán Trung bình"))
        
    # Rule 5: C Low OR M Small -> Low confidence (25%)
    w5 = max(top_conf_fuzzy["Low"], margin_fuzzy["Small"])
    if w5 > 0:
        rules.append((w5, SUGENO_CONFIDENCE["Low"], "NẾU Top Confidence Thấp HOẶC Margin Nhỏ THÌ Độ tin cậy chẩn đoán Thấp"))

    # Dự phòng
    if not rules:
        return 50.0
        
    sum_w_z = sum(w * z for w, z, _ in rules)
    sum_w = sum(w for w, _, _ in rules)
    
    return sum_w_z / sum_w if sum_w > 0 else 50.0


def evaluate_visual_severity(mild_conf_fuzzy: dict, severe_conf_fuzzy: dict, is_healthy: bool) -> tuple:
    """
    Tính toán Chỉ số mức độ bệnh trực quan (Visual Severity Index - VSI) bằng Sugeno FIS.
    """
    if is_healthy:
        return 0.0, [(1.0, SUGENO_SEVERITY["Healthy"], "NẾU Nhận dạng Lành mạnh THÌ Mức độ nghiêm trọng = 0%")]

    rules = []
    
    # Rule 1: Severe High -> Severe severity (90%)
    w1 = severe_conf_fuzzy["High"]
    if w1 > 0:
        rules.append((w1, SUGENO_SEVERITY["Severe"], "NẾU Độ tự tin Severe Cao THÌ Mức độ bệnh Nghiêm trọng"))
        
    # Rule 2: Severe Medium -> Moderate severity (55%)
    w2 = severe_conf_fuzzy["Medium"]
    if w2 > 0:
        rules.append((w2, SUGENO_SEVERITY["Moderate"], "NẾU Độ tự tin Severe Vừa THÌ Mức độ bệnh Trung bình"))
        
    # Rule 3: Severe Low and Mild High -> Mild severity (20%)
    w3 = min(severe_conf_fuzzy["Low"], mild_conf_fuzzy["High"])
    if w3 > 0:
        rules.append((w3, SUGENO_SEVERITY["Mild"], "NẾU Độ tự tin Severe Thấp VÀ Mild Cao THÌ Mức độ bệnh Nhẹ"))
        
    # Rule 4: Severe Low and Mild Medium -> Mild severity (20%)
    w4 = min(severe_conf_fuzzy["Low"], mild_conf_fuzzy["Medium"])
    if w4 > 0:
        rules.append((w4, SUGENO_SEVERITY["Mild"], "NẾU Độ tự tin Severe Thấp VÀ Mild Vừa THÌ Mức độ bệnh Nhẹ"))
        
    # Rule 5: Severe Low and Mild Low -> Mild severity (20%)
    w5 = min(severe_conf_fuzzy["Low"], mild_conf_fuzzy["Low"])
    if w5 > 0:
        rules.append((w5, SUGENO_SEVERITY["Mild"], "NẾU Độ tự tin Severe Thấp VÀ Mild Thấp THÌ Mức độ bệnh Nhẹ"))

    if not rules:
        return 35.0, [(1.0, SUGENO_SEVERITY["Moderate"], "Mặc định THÌ Mức độ bệnh = Trung bình")]
        
    sum_w_z = sum(w * z for w, z, _ in rules)
    sum_w = sum(w for w, _, _ in rules)
    vsi = sum_w_z / sum_w if sum_w > 0 else 35.0
    
    return vsi, rules


def evaluate_environmental_risk(temp_fuzzy: dict, humidity_fuzzy: dict, has_env: bool) -> tuple:
    """
    Tính toán Nguy cơ môi trường (Environmental Risk Index - ERI) bằng Sugeno FIS.
    Nếu không cung cấp T và H, mặc định trả về ERI = 50.0 (Medium) và không kích hoạt luật.
    """
    if not has_env:
        return 50.0, [(1.0, SUGENO_SEVERITY["Moderate"], "NẾU Không cung cấp thông số thời tiết THÌ Nguy cơ môi trường Trung bình (Mặc định)")]
        
    rules = []
    
    # Rule 1: Temp Warm and Humidity Wet -> High risk (85%)
    w1 = min(temp_fuzzy["Warm"], humidity_fuzzy["Wet"])
    if w1 > 0:
        rules.append((w1, SUGENO_ENV_RISK["High"], "NẾU Nhiệt độ Ấm VÀ Độ ẩm Ẩm ướt THÌ Nguy cơ môi trường Cao"))
        
    # Rule 2: Temp Hot and Humidity Wet -> High risk (85%)
    w2 = min(temp_fuzzy["Hot"], humidity_fuzzy["Wet"])
    if w2 > 0:
        rules.append((w2, SUGENO_ENV_RISK["High"], "NẾU Nhiệt độ Nóng VÀ Độ ẩm Ẩm ướt THÌ Nguy cơ môi trường Cao"))

    # Rule 3: Humidity Dry -> Low risk (15%)
    w3 = humidity_fuzzy["Dry"]
    if w3 > 0:
        rules.append((w3, SUGENO_ENV_RISK["Low"], "NẾU Độ ẩm Khô ráo THÌ Nguy cơ môi trường Thấp"))

    # Rule 4: Temp Cool -> Low risk (15%)
    w4 = temp_fuzzy["Cool"]
    if w4 > 0:
        rules.append((w4, SUGENO_ENV_RISK["Low"], "NẾU Nhiệt độ Lạnh THÌ Nguy cơ môi trường Thấp"))
        
    # Rule 5: Temp Warm/Hot and Humidity Moderate -> Medium risk (50%)
    w5 = min(max(temp_fuzzy["Warm"], temp_fuzzy["Hot"]), humidity_fuzzy["Moderate"])
    if w5 > 0:
        rules.append((w5, SUGENO_ENV_RISK["Medium"], "NẾU Nhiệt độ Ấm/Nóng VÀ Độ ẩm Trung bình THÌ Nguy cơ môi trường Trung bình"))

    if not rules:
        return 50.0, [(1.0, SUGENO_ENV_RISK["Medium"], "Mặc định THÌ Nguy cơ môi trường = Trung bình")]
        
    sum_w_z = sum(w * z for w, z, _ in rules)
    sum_w = sum(w for w, _, _ in rules)
    eri = sum_w_z / sum_w if sum_w > 0 else 50.0
    
    return eri, rules


def evaluate_final_alert(vsi: float, eri: float) -> tuple:
    """
    Tính toán Cảnh báo cuối cùng (Final Alert Level - FAI) bằng cách kết hợp VSI và ERI.
    """
    # Mờ hóa đầu vào của FAI: VSI và ERI
    # VSI: Healthy, Mild, Moderate, Severe
    vsi_m = {
        "Healthy": 1.0 if vsi < 10.0 else 0.0,
        "Mild": 1.0 if 10.0 <= vsi < 35.0 else 0.0,
        "Moderate": 1.0 if 35.0 <= vsi < 70.0 else 0.0,
        "Severe": 1.0 if vsi >= 70.0 else 0.0
    }
    # ERI: Low, Medium, High
    eri_m = {
        "Low": 1.0 if eri < 35.0 else 0.0,
        "Medium": 1.0 if 35.0 <= eri < 70.0 else 0.0,
        "High": 1.0 if eri >= 70.0 else 0.0
    }
    
    rules = []
    
    # Rule 1: VSI Healthy -> Normal alert (10%)
    w1 = vsi_m["Healthy"]
    if w1 > 0:
        rules.append((w1, SUGENO_ALERT["Normal"], "NẾU Mức độ bệnh Lành mạnh THÌ Cảnh báo Bình thường"))
        
    # Rule 2: VSI Mild and ERI High/Medium -> Attention alert (40%)
    w2 = min(vsi_m["Mild"], max(eri_m["High"], eri_m["Medium"]))
    if w2 > 0:
        rules.append((w2, SUGENO_ALERT["Attention"], "NẾU Mức độ bệnh Nhẹ VÀ Nguy cơ môi trường Trung bình/Cao THÌ Cảnh báo Chú ý"))
        
    # Rule 3: VSI Mild and ERI Low -> Normal alert (10%)
    w3 = min(vsi_m["Mild"], eri_m["Low"])
    if w3 > 0:
        rules.append((w3, SUGENO_ALERT["Normal"], "NẾU Mức độ bệnh Nhẹ VÀ Nguy cơ môi trường Thấp THÌ Cảnh báo Bình thường"))

    # Rule 4: VSI Moderate and ERI High -> Danger alert (70%)
    w4 = min(vsi_m["Moderate"], eri_m["High"])
    if w4 > 0:
        rules.append((w4, SUGENO_ALERT["Danger"], "NẾU Mức độ bệnh Trung bình VÀ Nguy cơ môi trường Cao THÌ Cảnh báo Nguy hiểm"))
        
    # Rule 5: VSI Moderate and ERI Medium/Low -> Attention alert (40%)
    w5 = min(vsi_m["Moderate"], max(eri_m["Medium"], eri_m["Low"]))
    if w5 > 0:
        rules.append((w5, SUGENO_ALERT["Attention"], "NẾU Mức độ bệnh Trung bình VÀ Nguy cơ môi trường Thấp/Trung bình THÌ Cảnh báo Chú ý"))
        
    # Rule 6: VSI Severe and ERI High -> Red Alert (95%)
    w6 = min(vsi_m["Severe"], eri_m["High"])
    if w6 > 0:
        rules.append((w6, SUGENO_ALERT["RedAlert"], "NẾU Mức độ bệnh Nghiêm trọng VÀ Nguy cơ môi trường Cao THÌ Cảnh báo Báo động đỏ"))
        
    # Rule 7: VSI Severe and ERI Medium/Low -> Danger alert (70%)
    w7 = min(vsi_m["Severe"], max(eri_m["Medium"], eri_m["Low"]))
    if w7 > 0:
        rules.append((w7, SUGENO_ALERT["Danger"], "NẾU Mức độ bệnh Nghiêm trọng VÀ Nguy cơ môi trường Thấp/Trung bình THÌ Cảnh báo Nguy hiểm"))

    if not rules:
        return 40.0, [(1.0, SUGENO_ALERT["Attention"], "Mặc định THÌ Cảnh báo = Chú ý")]
        
    sum_w_z = sum(w * z for w, z, _ in rules)
    sum_w = sum(w for w, _, _ in rules)
    fai = sum_w_z / sum_w if sum_w > 0 else 40.0
    
    return fai, rules

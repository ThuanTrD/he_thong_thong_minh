import numpy as np
from typing import Dict

from .schemas import FuzzyInput, FuzzyOutput
from .config import DISEASE_CLASSES_MAP, SEVERITY_CLASSES_MAP, DEFAULT_TEMP, DEFAULT_HUMIDITY
from .membership import fuzzify_confidence, fuzzify_margin, fuzzify_temp, fuzzify_humidity
from .rules import (
    evaluate_uncertainty,
    evaluate_diagnostic_confidence,
    evaluate_visual_severity,
    evaluate_environmental_risk,
    evaluate_final_alert
)
from .explanation import generate_explanation


class FuzzyEngine:
    def __init__(self):
        pass

    def run(self, inp: FuzzyInput) -> FuzzyOutput:
        # 1. Xác định thời tiết môi trường (có cung cấp hay không)
        has_env = (inp.temperature is not None and inp.humidity is not None)
        temp = inp.temperature if inp.temperature is not None else DEFAULT_TEMP
        humidity = inp.humidity if inp.humidity is not None else DEFAULT_HUMIDITY

        # 2. Tính toán các giá trị trung gian
        # Sắp xếp các lớp CNN theo tự tin giảm dần
        sorted_scores = sorted(inp.cnn_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = inp.top_confidence
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        margin = max_score - second_score

        # Phân loại nhóm bệnh chính từ nhãn lớp hàng đầu
        predicted_disease = DISEASE_CLASSES_MAP.get(inp.top_class, "Unknown")
        is_healthy = (predicted_disease == "Healthy")

        # Gom nhóm điểm số cho từng bệnh chính
        grouped_disease_scores: Dict[str, float] = {}
        for cls, score in inp.cnn_scores.items():
            disease = DISEASE_CLASSES_MAP.get(cls, "Unknown")
            grouped_disease_scores[disease] = grouped_disease_scores.get(disease, 0.0) + score

        # Lấy điểm số của dạng Mild và Severe tương ứng đối với bệnh được dự đoán
        mild_score = 0.0
        severe_score = 0.0
        if not is_healthy:
            for cls, score in inp.cnn_scores.items():
                if predicted_disease in cls:
                    if "Mild" in cls:
                        mild_score = score
                    elif "Severe" in cls:
                        severe_score = score

        # 3. Mờ hóa (Fuzzification)
        top_conf_fuzzy = fuzzify_confidence(max_score)
        margin_fuzzy   = fuzzify_margin(margin)
        mild_fuzzy     = fuzzify_confidence(mild_score)
        severe_fuzzy   = fuzzify_confidence(severe_score)
        temp_fuzzy     = fuzzify_temp(temp)
        humidity_fuzzy = fuzzify_humidity(humidity)

        # 4. Suy diễn mờ (Rule Evaluation) & Giải mờ (Defuzzification)
        
        # A. Độ bất định (Uncertainty Level)
        uncertainty_level, uncertainty_rules = evaluate_uncertainty(margin_fuzzy)

        # B. Độ tin cậy chẩn đoán (Diagnostic Confidence)
        diagnostic_confidence = evaluate_diagnostic_confidence(top_conf_fuzzy, margin_fuzzy)

        # C. Mức độ nghiêm trọng trực quan (Visual Severity Index - VSI)
        vsi, vsi_rules = evaluate_visual_severity(mild_fuzzy, severe_fuzzy, is_healthy)
        
        if vsi < 10.0:
            visual_severity_level = "Healthy (Lành mạnh)"
        elif vsi < 35.0:
            visual_severity_level = "Mild (Nhẹ)"
        elif vsi < 70.0:
            visual_severity_level = "Moderate (Trung bình)"
        else:
            visual_severity_level = "Severe (Nghiêm trọng)"

        # D. Nguy cơ môi trường (Environmental Risk Index - ERI)
        eri, eri_rules = evaluate_environmental_risk(temp_fuzzy, humidity_fuzzy, has_env)
        
        if eri < 35.0:
            environmental_risk_level = "Low (Thấp)"
        elif eri < 70.0:
            environmental_risk_level = "Medium (Trung bình)"
        else:
            environmental_risk_level = "High (Cao)"

        # E. Mức độ cảnh báo cuối cùng (Final Alert Level - FAI)
        fai, alert_rules = evaluate_final_alert(vsi, eri)
        
        if fai < 15.0:
            final_alert_level = "Normal (Bình thường)"
        elif fai < 45.0:
            final_alert_level = "Attention (Chú ý)"
        elif fai < 80.0:
            final_alert_level = "Danger (Nguy hiểm)"
        else:
            final_alert_level = "Red Alert (Báo động đỏ)"

        # 5. Sinh giải thích (XAI) và khuyến nghị nông nghiệp
        explanation, recommendation = generate_explanation(
            predicted_disease=predicted_disease,
            top_class=inp.top_class,
            top_confidence=max_score,
            margin=margin,
            uncertainty_level=uncertainty_level,
            diagnostic_confidence=diagnostic_confidence,
            visual_severity_level=visual_severity_level,
            environmental_risk_level=environmental_risk_level,
            final_alert_level=final_alert_level,
            temp=temp,
            humidity=humidity,
            has_env=has_env,
            vsi_rules=vsi_rules,
            eri_rules=eri_rules,
            alert_rules=alert_rules
        )

        return FuzzyOutput(
            predicted_disease=predicted_disease,
            visual_severity_level=visual_severity_level,
            diagnostic_confidence=round(float(diagnostic_confidence), 2),
            uncertainty_level=uncertainty_level,
            environmental_risk_level=environmental_risk_level,
            final_alert_level=final_alert_level,
            explanation=explanation,
            recommendation=recommendation
        )

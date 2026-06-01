import math
import numpy as np
from typing import Dict, Tuple

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


def apply_expert_guided_reasoning(
    predicted_disease: str,
    max_score: float,
    uncertainty_level: str,
    normalized_entropy: float,
    margin: float,
    snail_density: float,
    final_alert_level: str,
    visual_severity_level: str
) -> Tuple[str, str, float, str, str]:
    expert_signal = 0.0
    if snail_density is not None and snail_density > 0:
        expert_signal = min(snail_density / 10.0, 1.0)
        
    inference_mode = "AI_CONFIDENT"
    fused_confidence = max_score
    
    if expert_signal > 0:
        # Conditions for AI_CONFIDENT
        is_ai_very_confident = (max_score >= 0.80 and margin >= 0.2 and normalized_entropy <= 0.4 and expert_signal < 0.8)
        
        # Conditions for EXPERT_GUIDED_MODE
        is_uncertain = (max_score <= 0.65 or margin < 0.1 or normalized_entropy >= 0.6 or "High" in uncertainty_level)
        
        if is_ai_very_confident:
            inference_mode = "AI_CONFIDENT"
            fused_confidence = max_score
        elif expert_signal >= 0.5 and is_uncertain:
            inference_mode = "EXPERT_GUIDED_MODE"
            predicted_disease = "Golden Apple Snail"
            fused_confidence = min(0.60 + expert_signal * 0.39, 0.99)
            final_alert_level = "Red Alert (Báo động đỏ)"
            visual_severity_level = "Severe (Nghiêm trọng)"
        else:
            inference_mode = "HYBRID_WARNING"
            fused_confidence = 0.7 * max_score + 0.3 * expert_signal
            if "Normal" in final_alert_level:
                final_alert_level = "Attention (Chú ý)"

    return predicted_disease, inference_mode, fused_confidence, final_alert_level, visual_severity_level


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

        # Tính entropy / OOD signal
        eps = 1e-9
        entropy = -sum(score * math.log(score + eps) for score in inp.cnn_scores.values())
        num_classes = len(inp.cnn_scores) if len(inp.cnn_scores) > 1 else 2
        normalized_entropy = entropy / math.log(num_classes)

        # Phân loại nhóm bệnh chính từ nhãn lớp hàng đầu
        predicted_disease = DISEASE_CLASSES_MAP.get(inp.top_class, "Unknown")
        
        # Gom nhóm điểm số cho từng bệnh chính
        grouped_disease_scores: Dict[str, float] = {}
        for cls, score in inp.cnn_scores.items():
            disease = DISEASE_CLASSES_MAP.get(cls, "Unknown")
            grouped_disease_scores[disease] = grouped_disease_scores.get(disease, 0.0) + score

        # Tìm best_group_disease và best_group_confidence
        best_group_disease = max(grouped_disease_scores, key=grouped_disease_scores.get) if grouped_disease_scores else "Unknown"
        best_group_confidence = grouped_disease_scores.get(best_group_disease, 0.0)

        # Củng cố bệnh dựa trên grouped confidence
        if max_score < 0.8 and best_group_confidence > 0.6 and best_group_disease != predicted_disease:
            predicted_disease = best_group_disease
            
        is_healthy = (predicted_disease == "Healthy")

        # Lấy điểm số của dạng Mild và Severe tương ứng đối với bệnh được dự đoán
        mild_score = 0.0
        severe_score = 0.0
        if not is_healthy:
            for cls, score in inp.cnn_scores.items():
                if predicted_disease in cls:
                    if "Mild" in cls:
                        mild_score += score
                    elif "Severe" in cls:
                        severe_score += score

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
        if normalized_entropy > 0.6 and "High" not in uncertainty_level:
            uncertainty_level = "High Uncertainty (Dựa trên Entropy)"

        # B. Độ tin cậy chẩn đoán (Diagnostic Confidence)
        diagnostic_confidence = evaluate_diagnostic_confidence(top_conf_fuzzy, margin_fuzzy)

        # C. Mức độ nghiêm trọng trực quan (Visual Severity Index - VSI)
        is_explicit_severe = ("Severe" in inp.top_class)
        vsi, vsi_rules = evaluate_visual_severity(mild_fuzzy, severe_fuzzy, is_healthy, is_explicit_severe, top_conf_fuzzy)
        
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

        # [TÍN HIỆU CHUYÊN GIA] Expert-Assisted Confidence Fusion
        predicted_disease, inference_mode, fused_confidence, final_alert_level, visual_severity_level = apply_expert_guided_reasoning(
            predicted_disease=predicted_disease,
            max_score=max_score,
            uncertainty_level=uncertainty_level,
            normalized_entropy=normalized_entropy,
            margin=margin,
            snail_density=inp.snail_density if inp.snail_density is not None else 0.0,
            final_alert_level=final_alert_level,
            visual_severity_level=visual_severity_level
        )

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
            alert_rules=alert_rules,
            snail_density=inp.snail_density if inp.snail_density is not None else 0.0,
            inference_mode=inference_mode,
            fused_confidence=fused_confidence,
            normalized_entropy=normalized_entropy,
            best_group_disease=best_group_disease,
            best_group_confidence=best_group_confidence
        )

        return FuzzyOutput(
            predicted_disease=predicted_disease,
            visual_severity_level=visual_severity_level,
            diagnostic_confidence=round(float(diagnostic_confidence), 2),
            uncertainty_level=uncertainty_level,
            environmental_risk_level=environmental_risk_level,
            final_alert_level=final_alert_level,
            inference_mode=inference_mode,
            fused_confidence=round(float(fused_confidence), 2),
            explanation=explanation,
            recommendation=recommendation,
            normalized_entropy=round(float(normalized_entropy), 4),
            best_group_disease=best_group_disease,
            best_group_confidence=round(float(best_group_confidence), 4)
        )

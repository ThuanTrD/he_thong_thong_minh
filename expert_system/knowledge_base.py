# expert_system/knowledge_base.py
# This module acts as the Expert Knowledge Base, separating hardcoded rules, messages,
# and thresholds from the UI and inference pipelines.

def get_uncertainty_assessment(entropy: float) -> dict:
    """
    Evaluates Shannon entropy and returns the structured dictionary for UI display.
    """
    if entropy < 0.5:
        return {
            "text": "Phân phối dự đoán tập trung và đáng tin cậy.<br><i style='color:#94a3b8;font-size:0.8rem;'>Prediction distribution is concentrated and considered reliable.</i>",
            "color": "#10b981",
            "bg": "rgba(16, 185, 129, 0.1)"
        }
    elif entropy < 1.0:
        return {
            "text": "Tồn tại mức độ bất định nhất định giữa các lớp bệnh.<br><i style='color:#94a3b8;font-size:0.8rem;'>Some uncertainty exists between disease classes.</i>",
            "color": "#f59e0b",
            "bg": "rgba(245, 158, 11, 0.1)"
        }
    else:
        return {
            "text": "Dự đoán không rõ ràng và cần được kiểm tra kỹ.<br><i style='color:#94a3b8;font-size:0.8rem;'>Prediction is ambiguous and should be reviewed carefully.</i>",
            "color": "#ef4444",
            "bg": "rgba(239, 68, 68, 0.1)"
        }

def get_expert_guided_assessment(inference_mode: str, is_ood: bool) -> dict:
    """
    Evaluates the fuzzy inference mode and OOD flag to return the appropriate expert advice.
    """
    if inference_mode in ["EXPERT_GUIDED_MODE", "HYBRID_WARNING"] or is_ood:
        return {
            "text": "Mẫu có thể chứa các đặc điểm ngoại lai (OOD) hoặc dấu hiệu bệnh không chắc chắn. Khuyến nghị có sự kiểm tra từ chuyên gia.<br><i style='color:#94a3b8;font-size:0.8rem;'>The sample may contain out-of-distribution patterns or uncertain disease characteristics. Expert review is recommended.</i>",
            "color": "#f59e0b",
            "bg": "rgba(245, 158, 11, 0.1)"
        }
    else:
        return {
            "text": "Hệ thống đánh giá dự đoán này phù hợp để tự động kết luận.<br><i style='color:#94a3b8;font-size:0.8rem;'>The system considers this prediction suitable for automatic inference.</i>",
            "color": "#10b981",
            "bg": "rgba(16, 185, 129, 0.1)"
        }

def get_confidence_category(confidence: float) -> dict:
    """
    Categorizes the final fused confidence into HIGH, MODERATE, or LOW ranges.
    """
    conf_percent = confidence * 100
    
    if conf_percent > 85:
        return {
            "category": "Độ tin cậy cao<br><span style='font-size:0.8rem;color:#94a3b8;'>(High Confidence)</span>",
            "description": "> 85%",
            "color": "#10b981"
        }
    elif conf_percent >= 60:
        return {
            "category": "Độ tin cậy trung bình<br><span style='font-size:0.8rem;color:#94a3b8;'>(Moderate Confidence)</span>",
            "description": "60–85%",
            "color": "#f59e0b"
        }
    else:
        return {
            "category": "Độ tin cậy thấp<br><span style='font-size:0.8rem;color:#94a3b8;'>(Low Confidence)</span>",
            "description": "< 60%",
            "color": "#ef4444"
        }

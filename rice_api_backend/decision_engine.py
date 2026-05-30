# decision_engine.py
# Quyết định trạng thái cuối cùng dựa trên Classifier và OOD Detector

from typing import Dict, Any

class DecisionEngine:
    def __init__(self):
        pass

    def make_decision(self, cnn_result: Dict[str, Any], ood_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đưa ra phán quyết cuối cùng (KNOWN, UNCERTAIN, OOD)
        """
        is_unknown = ood_result["is_unknown"]
        is_ood = ood_result["is_ood"]
        top_group = ood_result.get("top_group", "Unknown")
        
        status = "KNOWN"
        message = "Chẩn đoán thành công."
        
        if is_ood:
            status = "OOD"
            message = "Phát hiện hình ảnh bất thường (Out-of-Distribution). Hình ảnh không rõ ràng hoặc không thuộc nhóm bệnh lá lúa đã học."
        elif is_unknown:
            status = "UNCERTAIN"
            message = f"Hệ thống nhận diện ảnh có đặc trưng thuộc nhóm {top_group}, tuy nhiên độ chắc chắn giữa các mức độ bệnh còn phân tán."
            
        return {
            "status": status,
            "message": message,
            "predicted_class": cnn_result["pred_class"] if status == "KNOWN" else None,
            "confidence": cnn_result["confidence"] if status == "KNOWN" else None,
            "top_group": top_group,
            "group_confidence": ood_result.get("group_confidence", 0.0),
            "is_ood": is_ood,
            "entropy": ood_result["entropy"]
        }

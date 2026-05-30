# ood_detector.py
# Module phát hiện hình ảnh OOD (Out-of-Distribution) dựa vào Softmax Output

import numpy as np
from typing import Dict, Any
from .config import GROUP_CONFIDENCE_THRESHOLD, UNCERTAIN_THRESHOLD, ENTROPY_THRESHOLD, DISEASE_GROUPS

class OODDetector:
    def __init__(self):
        pass

    def calculate_entropy(self, probs_array: np.ndarray) -> float:
        """
        Tính toán Normalized Shannon Entropy (Uncertainty Score).
        Mức độ 'phân vân' của mô hình được chuẩn hóa về thang [0, 1].
        """
        # Số lượng class của mô hình là 9 (Healthy + 8 classes bệnh)
        N_CLASSES = 9
        
        # Thêm epsilon nhỏ để tránh log(0)
        epsilon = 1e-10
        probs = np.clip(probs_array, epsilon, 1.0)
        
        # Tính Shannon Entropy theo cơ số 2 (Information bits)
        raw_entropy = -np.sum(probs * np.log2(probs))
        
        # Chuẩn hóa (Normalize) về thang [0, 1] bằng cách chia cho entropy cực đại log2(N)
        max_entropy = np.log2(N_CLASSES)
        normalized_entropy = raw_entropy / max_entropy
        
        return float(normalized_entropy)
        
    def calculate_group_confidence(self, cnn_scores: Dict[str, float]) -> dict:
        """
        Gộp xác suất theo nhóm ngữ nghĩa (Semantic Group Aggregation)
        để giải quyết hiện tượng Probability Splitting.
        """
        group_probs = {}
        for group_name, classes in DISEASE_GROUPS.items():
            # Tổng xác suất của các class trong nhóm
            total_prob = sum(cnn_scores.get(cls_name, 0.0) for cls_name in classes)
            group_probs[group_name] = total_prob
            
        # Tìm nhóm có độ tự tin cao nhất
        top_group = max(group_probs, key=group_probs.get)
        top_group_confidence = group_probs[top_group]
        
        return {
            "group_probs": group_probs,
            "top_group": top_group,
            "top_group_confidence": top_group_confidence
        }

    def detect(self, probs_array: np.ndarray, cnn_scores: Dict[str, float]) -> dict:
        """
        Đánh giá xem kết quả có phải là Out-of-Distribution không,
        sử dụng Semantic Group Confidence thay vì Top-1 MSP.
        """
        entropy = self.calculate_entropy(probs_array)
        group_info = self.calculate_group_confidence(cnn_scores)
        
        top_group = group_info["top_group"]
        group_conf = group_info["top_group_confidence"]
        
        is_unknown = False
        is_ood = False
        
        # Logic 1: Đánh giá theo Group Confidence
        if group_conf < UNCERTAIN_THRESHOLD:
            # Tự tin rất thấp -> Hoàn toàn không biết -> OOD
            is_ood = True
        elif group_conf < GROUP_CONFIDENCE_THRESHOLD:
            # Tự tin lưng chừng -> Không chắc chắn (Uncertain)
            is_unknown = True
            
        # Logic 2: Nếu Entropy quá cao (phân bố quá đều) -> OOD
        if entropy > ENTROPY_THRESHOLD:
            is_ood = True
            # Nếu đã OOD vì entropy thì không cần ghi là unknown nữa
            if is_ood: 
                is_unknown = False
            
        return {
            "entropy": entropy,
            "is_unknown": is_unknown,
            "is_ood": is_ood,
            "top_group": top_group,
            "group_confidence": group_conf
        }

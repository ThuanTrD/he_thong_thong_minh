"""
=============================================================================
HYBRID CNN + FUZZY PIPELINE — BƯỚC 3 & 4: FUZZY REASONING & XAI MODULE
=============================================================================
Nhận đầu vào:
  - CNN scores (Tỷ lệ dự đoán bệnh)
  - Nhiệt độ môi trường (Temperature - T, 15 - 45°C)
  - Độ ẩm không khí (Humidity - H, 40 - 100%)
Đầu ra:
  - Chỉ số mức độ bệnh liên tục (Disease Severity Index - DSI, 0% - 100%)
  - Giải thích XAI ngôn ngữ tự nhiên dựa trên các luật mờ được kích hoạt.
=============================================================================
"""

import os
import json
import numpy as np

# ─────────────────────────────────────────────
#  HÀM THÀNH VIÊN (MEMBERSHIP FUNCTIONS)
# ─────────────────────────────────────────────
def trimf(x, params):
    """Hàm thành viên hình tam giác."""
    a, b, c = params
    assert a <= b <= c, f"Yêu cầu: a <= b <= c. Nhận: {a}, {b}, {c}"
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def trapmf(x, params):
    """Hàm thành viên hình hình thang."""
    a, b, c, d = params
    assert a <= b <= c <= d, f"Yêu cầu: a <= b <= c <= d. Nhận: {a}, {b}, {c}, {d}"
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (d - x) / (d - c)


class RiceFuzzySystem:
    def __init__(self):
        # Định nghĩa các tập mờ cho đầu vào

        # 1. CNN Confidence (C): 0.0 -> 1.0
        self.c_mfs = {
            "Low": [0.0, 0.0, 0.25, 0.5],       # Trapmf
            "Medium": [0.3, 0.5, 0.7],          # Trimf
            "High": [0.5, 0.75, 1.0, 1.0]       # Trapmf
        }

        # 2. Nhiệt độ (T): 15 -> 45°C
        # Nhiệt độ khoảng 22-32°C (Warm) là thích hợp nhất cho nấm phát triển
        self.t_mfs = {
            "Cool": [15.0, 15.0, 20.0, 24.0],   # Trapmf
            "Warm": [20.0, 27.0, 34.0],         # Trimf
            "Hot": [30.0, 36.0, 45.0, 45.0]     # Trapmf
        }

        # 3. Độ ẩm (H): 40% -> 100%
        # Độ ẩm >80% (Wet) là điều kiện tối ưu cho bệnh đạo ôn và đốm nâu bùng phát
        self.h_mfs = {
            "Dry": [40.0, 40.0, 55.0, 65.0],    # Trapmf
            "Moderate": [55.0, 70.0, 85.0],     # Trimf
            "Wet": [75.0, 85.0, 100.0, 100.0]   # Trapmf
        }

        # Định nghĩa các hằng số đầu ra cho bộ suy diễn Sugeno (Disease Severity Index - DSI)
        self.dsi_singletons = {
            "Healthy": 0.0,      # Khỏe mạnh (0%)
            "Mild": 20.0,        # Bệnh nhẹ (20%)
            "Moderate": 55.0,    # Bệnh trung bình (55%)
            "Severe": 90.0       # Bệnh nặng (90%)
        }

    def fuzzify_c(self, val):
        """Mờ hóa điểm số tự tin CNN."""
        val = np.clip(val, 0.0, 1.0)
        return {
            "Low": trapmf(val, self.c_mfs["Low"]),
            "Medium": trimf(val, self.c_mfs["Medium"]),
            "High": trapmf(val, self.c_mfs["High"])
        }

    def fuzzify_t(self, val):
        """Mờ hóa nhiệt độ."""
        val = np.clip(val, 15.0, 45.0)
        return {
            "Cool": trapmf(val, self.t_mfs["Cool"]),
            "Warm": trimf(val, self.t_mfs["Warm"]),
            "Hot": trapmf(val, self.t_mfs["Hot"])
        }

    def fuzzify_h(self, val):
        """Mờ hóa độ ẩm."""
        val = np.clip(val, 40.0, 100.0)
        return {
            "Dry": trapmf(val, self.h_mfs["Dry"]),
            "Moderate": trimf(val, self.h_mfs["Moderate"]),
            "Wet": trapmf(val, self.h_mfs["Wet"])
        }

    def evaluate_rules(self, c_scores, temp, humidity, predicted_class):
        """
        Nhận đầu vào rõ, mờ hóa, áp dụng các luật mờ Sugeno.
        Đầu ra là DSI rõ (%) và XAI logs.
        """
        # Nếu CNN đoán là Lành mạnh (Healthy) và độ tự tin cao
        is_healthy = (predicted_class == "Healthy")
        
        # Lấy độ tự tin cho Mild và Severe của lớp bệnh được dự đoán
        # Nếu là Healthy thì lấy điểm số tự tin của lớp Healthy
        if is_healthy:
            c_val_healthy = c_scores.get("Healthy", 1.0)
            c_val_mild = 0.0
            c_val_severe = 0.0
        else:
            # Xác định nhóm bệnh (ví dụ: 'Blast', 'Brownspot', 'Bacterial blight', 'Tungro')
            # Lớp dự đoán có thể dạng 'Mild Blast' hoặc 'Severe Blast'
            disease_name = predicted_class.replace("Mild ", "").replace("Severe ", "")
            c_val_healthy = c_scores.get("Healthy", 0.0)
            
            # Tìm score của dạng Mild và Severe tương ứng của nhóm bệnh đó
            c_val_mild = 0.0
            c_val_severe = 0.0
            for cls, score in c_scores.items():
                if disease_name in cls:
                    if "Mild" in cls:
                        c_val_mild = score
                    elif "Severe" in cls:
                        c_val_severe = score

        # Mờ hóa các giá trị tự tin đầu vào
        c_m_healthy = self.fuzzify_c(c_val_healthy)
        c_m_mild    = self.fuzzify_c(c_val_mild)
        c_m_severe  = self.fuzzify_c(c_val_severe)
        
        t_m = self.fuzzify_t(temp)
        h_m = self.fuzzify_h(humidity)

        rules_fired = []
        
        # ────────────────────────────────────────────────────────
        #  HỆ LUẬT MỜ (FUZZY RULES)
        # ────────────────────────────────────────────────────────
        
        # Nhóm 1: Các luật cho cây khỏe mạnh (Healthy)
        if is_healthy:
            # R1: CNN tự tin Healthy cao -> Khỏe mạnh
            w = c_m_healthy["High"]
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán Khỏe mạnh với độ tự tin Cao THÌ cây Lành mạnh",
                    "w": w, "output": self.dsi_singletons["Healthy"],
                    "desc": "Mô hình CNN nhận dạng lá cây hoàn toàn bình thường và khỏe mạnh."
                })
            # R2: CNN tự tin Healthy Medium, nhưng độ ẩm khô ráo -> Rất an toàn
            w = min(c_m_healthy["Medium"], h_m["Dry"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán Khỏe mạnh vừa phải VÀ thời tiết Khô ráo THÌ cây Lành mạnh",
                    "w": w, "output": self.dsi_singletons["Healthy"],
                    "desc": "Lá lúa bình thường, thời tiết khô ráo giúp hạn chế tối đa nguy cơ mầm bệnh phát sinh."
                })
        
        # Nhóm 2: Các luật cho bệnh nhẹ (Mild)
        if not is_healthy:
            # R3: CNN dự đoán Mild cao/vừa + thời tiết khô -> Mức độ Nhẹ
            w = min(c_m_mild["High"], h_m["Dry"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán nhiễm bệnh Nhẹ cao VÀ thời tiết Khô ráo THÌ mức độ bệnh Nhẹ",
                    "w": w, "output": self.dsi_singletons["Mild"],
                    "desc": "CNN ghi nhận bệnh nhẹ, thời tiết khô ráo hạn chế tối đa nấm bệnh phát triển."
                })
            w = min(c_m_mild["Medium"], h_m["Dry"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán nhiễm bệnh Nhẹ trung bình VÀ thời tiết Khô ráo THÌ mức độ bệnh Nhẹ",
                    "w": w, "output": self.dsi_singletons["Mild"],
                    "desc": "Cảnh báo bệnh ở mức trung bình kết hợp thời tiết khô hạn giúp kìm hãm vết bệnh lan rộng."
                })
            # R4: CNN dự đoán Mild Medium + thời tiết lạnh -> Mức độ Nhẹ
            w = min(c_m_mild["Medium"], t_m["Cool"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán nhiễm bệnh Nhẹ vừa VÀ nhiệt độ Lạnh THÌ mức độ bệnh Nhẹ",
                    "w": w, "output": self.dsi_singletons["Mild"],
                    "desc": "Nhiệt độ mát mẻ/lạnh làm chậm quá trình nhân lên và phát tán của bào tử nấm."
                })
            w = min(c_m_mild["Medium"], h_m["Moderate"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán nhiễm bệnh Nhẹ vừa VÀ độ ẩm Trung bình THÌ mức độ bệnh Nhẹ",
                    "w": w, "output": self.dsi_singletons["Mild"],
                    "desc": "Mô hình CNN dự đoán vết bệnh nhẹ và độ ẩm ở mức bình thường, bệnh ít có khả năng bùng phát nhanh."
                })

            # Nhóm 3: Các luật cho bệnh trung bình (Moderate)
            # R5: CNN dự đoán Mild High/Medium, nhưng nhiệt độ Ấm và ẩm cao -> Bệnh chuyển biến thành Trung bình
            w = min(c_m_mild["High"], t_m["Warm"], h_m["Wet"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán bệnh Nhẹ cao VÀ thời tiết Ấm nóng + Ẩm ướt THÌ mức độ bệnh Trung bình",
                    "w": w, "output": self.dsi_singletons["Moderate"],
                    "desc": "Mặc dù triệu chứng bên ngoài còn nhẹ, thời tiết nóng ẩm hiện tại cực kỳ lý tưởng để nấm bệnh tiến triển nhanh."
                })
            w = min(c_m_mild["Medium"], t_m["Warm"], h_m["Wet"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán bệnh Nhẹ vừa VÀ thời tiết Ấm nóng + Ẩm ướt THÌ mức độ bệnh Trung bình",
                    "w": w, "output": self.dsi_singletons["Moderate"],
                    "desc": "Triệu chứng bệnh vừa phải kết hợp với thời tiết nóng ẩm thuận lợi thúc đẩy vết bệnh phát triển nhanh hơn."
                })
            # R6: CNN dự đoán Severe Medium + thời tiết trung bình -> Mức độ bệnh Trung bình
            w = min(c_m_severe["Medium"], h_m["Moderate"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán bệnh Nặng vừa VÀ độ ẩm Trung bình THÌ mức độ bệnh Trung bình",
                    "w": w, "output": self.dsi_singletons["Moderate"],
                    "desc": "Ghi nhận triệu chứng nhiễm bệnh ở mức độ trung bình với độ ẩm không khí vừa phải."
                })

            # Nhóm 4: Các luật cho bệnh nặng (Severe)
            # R7: CNN dự đoán Severe High -> Mức độ bệnh Nặng
            w = c_m_severe["High"]
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán bệnh Nặng cao THÌ mức độ bệnh Nghiêm trọng",
                    "w": w, "output": self.dsi_singletons["Severe"],
                    "desc": "CNN nhận dạng vết bệnh lớn và nghiêm trọng trên bề mặt lá với độ chính xác cao."
                })
            # R8: CNN dự đoán Severe Medium + thời tiết Ấm nóng ẩm ướt -> Đẩy lên mức Nghiêm trọng
            w = min(c_m_severe["Medium"], t_m["Warm"], h_m["Wet"])
            if w > 0:
                rules_fired.append({
                    "rule": "NẾU CNN đoán bệnh Nặng vừa VÀ thời tiết Ấm nóng + Ẩm ướt THÌ mức độ bệnh Nghiêm trọng",
                    "w": w, "output": self.dsi_singletons["Severe"],
                    "desc": "Triệu chứng bệnh rõ ràng kết hợp thời tiết nóng ẩm tạo điều kiện cực kỳ nguy hiểm, có thể bùng phát dịch."
                })

        # Trường hợp dự phòng nếu không luật nào được kích hoạt (tránh chia cho 0)
        if not rules_fired:
            # Lấy mặc định theo kết quả CNN
            default_out = self.dsi_singletons["Healthy"] if is_healthy else self.dsi_singletons["Moderate"]
            rules_fired.append({
                "rule": "Luật mặc định theo phân loại CNN",
                "w": 1.0, "output": default_out,
                "desc": "Không có luật thời tiết nào kích hoạt, suy diễn dựa trên điểm tự tin CNN."
            })

        # ─────────────────────────────────────────────
        #  GIẢI MỜ (DEFUZZIFICATION - Weighted Average)
        # ─────────────────────────────────────────────
        sum_w_z = sum(r["w"] * r["output"] for r in rules_fired)
        sum_w   = sum(r["w"] for r in rules_fired)
        
        dsi = sum_w_z / sum_w if sum_w > 0 else 0.0

        # Phân lớp mức độ bệnh dựa trên chỉ số DSI rõ đầu ra
        if dsi < 10.0:
            severity_class = "Healthy (Lành mạnh)"
        elif dsi < 35.0:
            severity_class = "Mild (Nhẹ)"
        elif dsi < 70.0:
            severity_class = "Moderate (Trung bình)"
        else:
            severity_class = "Severe (Nghiêm trọng)"

        # ─────────────────────────────────────────────
        #  BÁO CÁO GIẢI THÍCH XAI (EXPLAINABLE AI)
        # ─────────────────────────────────────────────
        xai_report = []
        xai_report.append(f"Kết quả phân loại CNN: {predicted_class}")
        if not is_healthy:
            disease_name = predicted_class.replace("Mild ", "").replace("Severe ", "")
            xai_report.append(f"  - Độ tự tin bệnh Nhẹ (Mild): {c_val_mild:.4f}")
            xai_report.append(f"  - Độ tự tin bệnh Nặng (Severe): {c_val_severe:.4f}")
        else:
            xai_report.append(f"  - Độ tự tin Lành mạnh (Healthy): {c_val_healthy:.4f}")
            
        xai_report.append(f"Thông số môi trường: Nhiệt độ = {temp}°C, Độ ẩm = {humidity}%")
        xai_report.append("\nCác luật mờ được kích hoạt (Fired Rules):")
        
        # Sắp xếp các luật theo trọng số kích hoạt giảm dần
        sorted_rules = sorted(rules_fired, key=lambda x: x["w"], reverse=True)
        for idx, r in enumerate(sorted_rules, 1):
            xai_report.append(f"  {idx}. [{r['rule']}] (Trọng số kích hoạt = {r['w']:.2f})")
            xai_report.append(f"     => Lý giải: {r['desc']}")

        xai_report.append(f"\nKết quả giải mờ:")
        xai_report.append(f"  - Chỉ số mức độ bệnh (Disease Severity Index - DSI): {dsi:.2f}%")
        xai_report.append(f"  - Đánh giá cuối cùng: {severity_class}")

        report_str = "\n".join(xai_report)

        return {
            "dsi": round(dsi, 2),
            "severity_class": severity_class,
            "fired_rules": sorted_rules,
            "xai_report": report_str
        }


# Chạy test độc lập nếu cần
if __name__ == "__main__":
    fs = RiceFuzzySystem()
    # Test case: ảnh nhiễm bệnh Mild Blast nhưng thời tiết nóng ẩm
    cnn_test_scores = {
        "Healthy": 0.05,
        "Mild Blast": 0.85,
        "Severe Blast": 0.10
    }
    result = fs.evaluate_rules(cnn_test_scores, temp=28.0, humidity=90.0, predicted_class="Mild Blast")
    print(result["xai_report"])

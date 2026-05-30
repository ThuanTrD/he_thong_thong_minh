# recommendation_service.py
# Dịch vụ cung cấp khuyến nghị dựa trên trạng thái (KNOWN/UNKNOWN/OOD)

from .config import DISEASE_CLASSES_MAP

class RecommendationService:
    def __init__(self):
        pass

    def get_recommendation(self, decision: dict) -> str:
        """
        Trả về khuyến nghị xử lý.
        TUYỆT ĐỐI không đưa ra lời khuyên dùng thuốc nếu trạng thái là UNKNOWN hoặc OOD.
        """
        status = decision["status"]
        
        if status in ["UNKNOWN", "OOD"]:
            return (
                "⚠️ Không khuyến nghị thuốc khi chưa xác định chắc chắn.\n"
                "Gợi ý hành động:\n"
                "- Vui lòng chụp lại ảnh rõ hơn, tập trung vào vùng lá bị bệnh.\n"
                "- Chắc chắn rằng ảnh không chứa các sinh vật lạ (như ốc bươu vàng, sâu cuốn lá) vì hệ thống AI này chỉ chuyên chẩn đoán nấm/vi khuẩn.\n"
                "- Gửi mẫu bệnh phẩm cho chuyên gia nông nghiệp địa phương để kiểm tra thủ công."
            )
            
        # Nếu là KNOWN, đưa ra lời khuyên (giả lập đơn giản)
        pred_class = decision["predicted_class"]
        disease_vi = DISEASE_CLASSES_MAP.get(pred_class, pred_class)
        
        rec = f"✅ Bệnh được xác định: {disease_vi}.\nKhuyến nghị xử lý:\n"
        if "Blast" in pred_class:
            rec += "- Sử dụng thuốc trị nấm đặc trị Đạo ôn (vd: Tricyclazole).\n- Dừng bón phân đạm ngay lập tức."
        elif "Bacterial blight" in pred_class:
            rec += "- Sử dụng các dòng thuốc sát khuẩn gốc đồng.\n- Rút bớt nước trên ruộng."
        elif "Brownspot" in pred_class:
            rec += "- Bón bổ sung lân và kali vì bệnh thường xuất hiện trên đất nghèo dinh dưỡng."
        elif "Tungro" in pred_class:
            rec += "- Phun thuốc diệt rầy xanh đuôi đen (vật chủ trung gian truyền virus).\n- Tiêu hủy khóm lúa bệnh nặng."
        else:
            rec = "✅ Cây lúa khỏe mạnh. Tiếp tục duy trì chế độ chăm sóc định kỳ."
            
        return rec

from typing import List, Tuple
from .config import DISEASE_NAMES_VI

def generate_explanation(
    predicted_disease: str,
    top_class: str,
    top_confidence: float,
    margin: float,
    uncertainty_level: str,
    diagnostic_confidence: float,
    visual_severity_level: str,
    environmental_risk_level: str,
    final_alert_level: str,
    temp: float,
    humidity: float,
    has_env: bool,
    vsi_rules: List[Tuple],
    eri_rules: List[Tuple],
    alert_rules: List[Tuple]
) -> Tuple[str, str]:
    """
    Sinh báo cáo giải thích ngôn ngữ tự nhiên (XAI) và các khuyến nghị nông nghiệp.
    """
    disease_vi = DISEASE_NAMES_VI.get(predicted_disease, predicted_disease)
    
    # ── 1. PHẦN GIẢI THÍCH (EXPLANATION) ──
    exp_parts = []
    exp_parts.append(f"=== BÁO CÁO GIẢI THÍCH CHẨN ĐOÁN (XAI REPORT) ===")
    exp_parts.append(f"1. Phân tích Hình ảnh (CNN):")
    exp_parts.append(f"   - Mô hình CNN chẩn đoán lá lúa nhiễm bệnh: {disease_vi}")
    exp_parts.append(f"   - Phân loại chi tiết của CNN: {top_class} (Độ tự tin: {top_confidence*100:.2f}%)")
    exp_parts.append(f"   - Khoảng cách phân biệt (Margin) so với lớp thứ 2: {margin:.4f}")
    exp_parts.append(f"   - Đánh giá độ bất định chẩn đoán: {uncertainty_level}")
    exp_parts.append(f"   - Độ tin cậy chẩn đoán tổng hợp của hệ thống: {diagnostic_confidence:.2f}%")
    
    exp_parts.append(f"\n2. Phân tích Tác động Môi trường (Fuzzy Logic):")
    if has_env:
        exp_parts.append(f"   - Thông số môi trường ghi nhận: Nhiệt độ = {temp}°C, Độ ẩm = {humidity}%")
        exp_parts.append(f"   - Đánh giá nguy cơ bùng phát do thời tiết: {environmental_risk_level}")
    else:
        exp_parts.append(f"   - Không có thông số thời tiết thực tế được cung cấp. Hệ thống sử dụng cấu hình mặc định.")
        exp_parts.append(f"   - Nguy cơ môi trường mặc định: {environmental_risk_level}")
        
    exp_parts.append(f"\n3. Mức độ Cảnh báo Tổng hợp: {final_alert_level}")
    exp_parts.append(f"   - Đánh giá mức độ tổn thương trên bề mặt lá (Visual Severity): {visual_severity_level}")
    
    exp_parts.append(f"\n4. Các Luật Mờ được kích hoạt (Fuzzy Rules Fired):")
    rule_idx = 1
    # Thêm các luật VSI
    for rule_txt, w, out in vsi_rules:
        if w > 0:
            exp_parts.append(f"   {rule_idx}. [Luật Mức độ bệnh] {rule_txt} (Trọng số kích hoạt: {w:.2f})")
            rule_idx += 1
    # Thêm các luật ERI
    for rule_txt, w, out in eri_rules:
        if w > 0:
            exp_parts.append(f"   {rule_idx}. [Luật Thời tiết] {rule_txt} (Trọng số kích hoạt: {w:.2f})")
            rule_idx += 1
    # Thêm các luật FAI
    for rule_txt, w, out in alert_rules:
        if w > 0:
            exp_parts.append(f"   {rule_idx}. [Luật Cảnh báo] {rule_txt} (Trọng số kích hoạt: {w:.2f})")
            rule_idx += 1
            
    explanation_str = "\n".join(exp_parts)

    # ── 2. PHẦN KHUYẾN NGHỊ (RECOMMENDATION) ──
    rec_parts = []
    
    if predicted_disease == "Healthy":
        rec_parts.append("Cây lúa khỏe mạnh. Không cần phun thuốc bảo vệ thực vật.")
        rec_parts.append("Khuyến nghị: Tiếp tục duy trì chế độ bón phân cân đối và kiểm tra đồng ruộng định kỳ.")
    else:
        rec_parts.append(f"Đã phát hiện bệnh hại: {disease_vi} ở mức độ nghiêm trọng trực quan: {visual_severity_level}.")
        
        # Đưa ra khuyến nghị theo mức độ nguy hiểm
        if "Red Alert" in final_alert_level:
            rec_parts.append("⚠️ CẢNH BÁO KHẨN CẤP: Dịch bệnh có nguy cơ bùng phát rất cao do điều kiện nóng ẩm thích hợp.")
            rec_parts.append("Hành động ngay lập tức:")
            rec_parts.append("  1. Phun thuốc bảo vệ thực vật đặc trị cho loại bệnh này ngay lập tức.")
            rec_parts.append("  2. Ngừng bón phân đạm (nitrogen) vì đạm làm vết bệnh (đặc biệt là đạo ôn và bạc lá) phát triển nhanh hơn.")
            rec_parts.append("  3. Tháo bớt nước trên ruộng nếu ruộng đang ngập nước sâu để giảm độ ẩm tiểu khí hậu.")
        elif "Danger" in final_alert_level:
            rec_parts.append("👉 CẦN CAN THIỆP: Mức độ nguy hiểm cao.")
            rec_parts.append("Hành động khuyến nghị:")
            rec_parts.append("  1. Tiến hành phun thuốc đặc trị khoanh vùng nhiễm bệnh để tránh lây lan ra diện rộng.")
            rec_parts.append("  2. Kiểm tra nguồn nước tưới, tránh tưới nước vào ban đêm khi độ ẩm không khí đang cao.")
        elif "Attention" in final_alert_level:
            rec_parts.append("ℹ️ CHÚ Ý THEO DÕI: Vết bệnh nhẹ nhưng môi trường ẩm ướt hoặc có nguy cơ nhẹ.")
            rec_parts.append("Hành động khuyến nghị:")
            rec_parts.append("  1. Theo dõi sát sao các ruộng lân cận trong vòng 2-3 ngày tới.")
            rec_parts.append("  2. Chỉ nên bón phân kali để tăng cường sức đề kháng cho cây, chưa cần phun thuốc hóa học diện rộng.")
        else:
            rec_parts.append("ℹ️ THEO DÕI THÔNG THƯỜNG: Vết bệnh nhẹ và thời tiết khô ráo hạn chế dịch lan nhanh.")
            rec_parts.append("Khuyến nghị: Cắt tỉa các lá bị bệnh nặng đơn lẻ để tiêu hủy, chưa cần can thiệp hóa chất diện rộng.")
            
        # Khuyến nghị cụ thể cho từng loại bệnh
        if predicted_disease == "Blast":
            rec_parts.append("  * Đặc thù Bệnh Đạo Ôn: Sử dụng các hoạt chất như Tricyclazole, Fenoxanil hoặc Isoprothiolane. Tránh bón thêm phân đạm.")
        elif predicted_disease == "Bacterial blight":
            rec_parts.append("  * Đặc thù Bệnh Bạc Lá: Sử dụng các dòng thuốc sát khuẩn đồng (Copper) hoặc thuốc kháng sinh nông nghiệp. Ngừng tưới tràn.")
        elif predicted_disease == "Brownspot":
            rec_parts.append("  * Đặc thù Bệnh Đốm Nâu: Thường phát sinh trên đất nghèo dinh dưỡng hoặc chua. Cần bón lân và kali kết hợp làm cỏ.")
        elif predicted_disease == "Tungro":
            rec_parts.append("  * Đặc thù Bệnh Tungro: Do virus truyền bởi rầy xanh đuôi đen. Cần tập trung phun thuốc diệt rầy để cắt đứt nguồn truyền bệnh.")

    recommendation_str = "\n".join(rec_parts)
    
    return explanation_str, recommendation_str

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
    alert_rules: List[Tuple],
    snail_density: float = 0.0,
    inference_mode: str = "AI_CONFIDENT",
    fused_confidence: float = 0.0
) -> Tuple[str, str]:
    """
    Sinh báo cáo giải thích ngôn ngữ tự nhiên (XAI) và các khuyến nghị nông nghiệp.
    """
    disease_vi = DISEASE_NAMES_VI.get(predicted_disease, predicted_disease)
    
    # ── 1. PHẦN GIẢI THÍCH (EXPLANATION) ──
    exp_parts = []
    exp_parts.append(f"=== BÁO CÁO GIẢI THÍCH CHẨN ĐOÁN (XAI REPORT) ===")
    exp_parts.append(f"1. Phân tích Hình ảnh và Suy luận Lai (Hybrid Inference):")
    
    if inference_mode == "AI_CONFIDENT":
        exp_parts.append(f"   - Mô hình CNN chẩn đoán lá lúa nhiễm bệnh: {disease_vi}")
        exp_parts.append(f"   - Phân loại chi tiết của CNN: {top_class} (Độ tự tin: {top_confidence*100:.2f}%)")
        if snail_density > 0:
            exp_parts.append(f"   - Nhận xét: Mô hình nhận diện khá rõ đặc trưng bệnh trên lá. Tín hiệu ngoại lệ từ thực địa ({snail_density} con/m2) chưa đủ mạnh để thay đổi kết luận chính.")
    elif inference_mode == "HYBRID_WARNING":
        exp_parts.append(f"   - Mô hình CNN chẩn đoán bệnh ưu tiên: {disease_vi} (Độ tự tin: {top_confidence*100:.2f}%)")
        exp_parts.append(f"   - Nhận xét: Hệ thống ghi nhận đồng thời đặc trưng bệnh lá và tín hiệu ngoại lệ từ thực địa (Mật độ ốc: {snail_density} con/m2).")
        exp_parts.append(f"   - Khuyến nghị: Kiểm tra bổ sung để phân biệt bệnh và tác nhân gây hại phối hợp.")
    elif inference_mode == "EXPERT_GUIDED_MODE":
        exp_parts.append(f"   - Trạng thái: Chế độ Suy luận Hỗ trợ Chuyên gia (Expert-Guided Mode)")
        exp_parts.append(f"   - Nhận xét: Hệ thống chuyển sang chế độ suy luận có hỗ trợ tín hiệu thực địa do dữ liệu ảnh đầu vào có độ bất định cao và mật độ ốc bươu vàng ghi nhận ở mức cao ({snail_density} con/m2).")
        exp_parts.append(f"   - Hệ thống tăng trọng số tín hiệu chuyên gia để giảm rủi ro suy luận sai trong trường hợp dữ liệu hình ảnh không chắc chắn.")
    
    exp_parts.append(f"   - Khoảng cách phân biệt (Margin) so với lớp thứ 2: {margin:.4f}")
    exp_parts.append(f"   - Đánh giá độ bất định chẩn đoán hình ảnh: {uncertainty_level}")
    exp_parts.append(f"   - Độ tin cậy chẩn đoán tổng hợp sau kết hợp (Fused Confidence): {fused_confidence*100:.2f}%")
    
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
    # Tính tổng (w * z) cho từng nhóm luật để tính phần trăm đóng góp chính xác theo phương pháp Sugeno
    total_wz_vsi = sum(w * z for w, z, txt in vsi_rules)
    total_wz_eri = sum(w * z for w, z, txt in eri_rules)
    total_wz_alert = sum(w * z for w, z, txt in alert_rules)
    
    # Hàm phụ trợ tính đóng góp
    def calc_contrib(w, z, total_wz, rule_list):
        if total_wz > 0:
            return ((w * z) / total_wz) * 100
        else:
            # Fallback nếu tất cả z = 0 (vd: Healthy severity = 0)
            total_w = sum(rw for rw, rz, rtxt in rule_list)
            return (w / total_w) * 100 if total_w > 0 else 0

    # Thêm các luật VSI
    for w, z, rule_txt in vsi_rules:
        if w > 0:
            contrib = calc_contrib(w, z, total_wz_vsi, vsi_rules)
            exp_parts.append(f"   {rule_idx}. [Luật Mức độ bệnh] {rule_txt} (Kích hoạt: {w:.2f}, Đóng góp: {contrib:.1f}%)")
            rule_idx += 1
            
    # Thêm các luật ERI
    for w, z, rule_txt in eri_rules:
        if w > 0:
            contrib = calc_contrib(w, z, total_wz_eri, eri_rules)
            exp_parts.append(f"   {rule_idx}. [Luật Thời tiết] {rule_txt} (Kích hoạt: {w:.2f}, Đóng góp: {contrib:.1f}%)")
            rule_idx += 1
            
    # Thêm các luật FAI
    for w, z, rule_txt in alert_rules:
        if w > 0:
            contrib = calc_contrib(w, z, total_wz_alert, alert_rules)
            exp_parts.append(f"   {rule_idx}. [Luật Cảnh báo] {rule_txt} (Kích hoạt: {w:.2f}, Đóng góp: {contrib:.1f}%)")
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
            rec_parts.append("⚠️ CẢNH BÁO KHẨN CẤP: Dịch bệnh/Dịch hại có nguy cơ bùng phát rất cao do điều kiện đang vô cùng thuận lợi.")
            rec_parts.append("Hành động ngay lập tức:")
            if predicted_disease == "Tungro":
                rec_parts.append("  1. Lập tức phun thuốc diệt rầy xanh đuôi đen (vật chủ truyền bệnh).")
                rec_parts.append("  2. Tiêu hủy các khóm lúa đã nhiễm bệnh nặng để cắt đứt nguồn lây virus.")
            elif predicted_disease == "Golden Apple Snail":
                rec_parts.append("  1. Xử lý triệt để bằng thuốc diệt ốc đặc trị (hoặc bắt thủ công nếu diện tích nhỏ).")
            else:
                rec_parts.append("  1. Phun thuốc bảo vệ thực vật đặc trị cho loại bệnh này ngay lập tức.")
                rec_parts.append("  2. Ngừng bón phân đạm (nitrogen) vì đạm làm vết bệnh nấm/vi khuẩn phát triển nhanh hơn.")
            rec_parts.append("  3. Tháo bớt nước trên ruộng nếu đang ngập sâu để giảm độ ẩm tiểu khí hậu (trừ trường hợp dùng thuốc diệt ốc cần giữ nước).")
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
        elif predicted_disease == "Golden Apple Snail":
            if snail_density > 3:
                rec_parts.append("  * Đặc thù Ốc bươu vàng: Mật độ cao nguy hiểm. Dùng thuốc Metaldehyde (nếu không tháo được nước) hoặc Niclosamide (nếu có lớp nước 2-3cm).")
            else:
                rec_parts.append("  * Đặc thù Ốc bươu vàng: Bắt ốc thủ công, thu trứng đẻ sẵn trên bẹ lá, cắm cọc làm giá thể dụ ốc đẻ trứng. Rút nước giữ ở 2-3cm.")

    recommendation_str = "\n".join(rec_parts)
    
    return explanation_str, recommendation_str

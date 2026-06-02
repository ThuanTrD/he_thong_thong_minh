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
    fused_confidence: float = 0.0,
    normalized_entropy: float = 0.0,
    best_group_disease: str = "",
    best_group_confidence: float = 0.0
) -> Tuple[str, str]:
    """
    Sinh báo cáo giải thích ngôn ngữ tự nhiên (XAI) và các khuyến nghị nông nghiệp.
    """
    disease_vi = DISEASE_NAMES_VI.get(predicted_disease, predicted_disease)
    best_group_disease_vi = DISEASE_NAMES_VI.get(best_group_disease, best_group_disease)
    
    # ── 1. PHẦN GIẢI THÍCH (EXPLANATION) ──
    exp_parts = []
    exp_parts.append(f"=== BÁO CÁO GIẢI THÍCH CHẨN ĐOÁN (XAI REPORT) ===")
    exp_parts.append(f"1. Phân tích Hình ảnh và Suy luận Lai (Hybrid Inference):")
    
    if inference_mode == "AI_CONFIDENT":
        exp_parts.append(f"   - Mô hình CNN chẩn đoán lá lúa có khả năng nhiễm bệnh: {disease_vi}")
        exp_parts.append(f"   - Phân loại chi tiết của CNN: {top_class} (Độ tự tin: {top_confidence*100:.2f}%)")
        if best_group_confidence > top_confidence:
            exp_parts.append(f"   - Nhóm bệnh tốt nhất: {best_group_disease_vi} (Độ tự tin: {best_group_confidence*100:.2f}%)")
            exp_parts.append(f"   - Gợi ý: Grouped confidence giúp củng cố dự đoán ở mức bệnh chính dù phân loại chi tiết có phần phân tán.")
        exp_parts.append(f"   - Nhận xét: Hệ thống ưu tiên giữ trạng thái AI_CONFIDENT do độ tự tin cao, khoảng cách phân biệt (margin) rõ ràng, mức độ bất định (entropy) thấp và chưa ghi nhận tín hiệu thực địa đủ mạnh để thay đổi quyết định.")
    elif inference_mode == "HYBRID_WARNING":
        exp_parts.append(f"   - Mô hình CNN chẩn đoán bệnh ưu tiên: {disease_vi} (Độ tự tin: {top_confidence*100:.2f}%)")
        exp_parts.append(f"   - Nhóm bệnh tốt nhất: {best_group_disease_vi} (Độ tự tin: {best_group_confidence*100:.2f}%)")
        exp_parts.append(f"   - Nhận xét: Hệ thống chưa phủ quyết kết quả CNN, nhưng nâng mức cảnh báo do có dấu hiệu bất định nhẹ hoặc tín hiệu thực địa bổ sung (Mật độ ốc: {snail_density} con/m2).")
        exp_parts.append(f"   - Mức độ tự tin tổng hợp (Fused Confidence: {fused_confidence*100:.2f}%) là kết quả kết hợp hài hòa giữa độ tự tin của CNN và trọng số rủi ro từ tín hiệu chuyên gia.")
        exp_parts.append(f"   - Hỗ trợ quyết định: Gợi ý cần kiểm tra bổ sung tại thực địa để phân biệt bệnh lý và tác nhân gây hại phối hợp.")
    elif inference_mode == "EXPERT_GUIDED_MODE":
        exp_parts.append(f"   - Trạng thái: Chế độ Suy luận Hỗ trợ Chuyên gia (Expert-Guided Mode)")
        exp_parts.append(f"   - Thông tin nhận thức ban đầu từ CNN:")
        exp_parts.append(f"       + Phân loại Top-1: {top_class} ({top_confidence*100:.2f}%)")
        exp_parts.append(f"       + Nhóm bệnh tốt nhất: {best_group_disease_vi} ({best_group_confidence*100:.2f}%)")
        exp_parts.append(f"       + Khoảng cách phân biệt (Margin): {margin:.4f}")
        exp_parts.append(f"       + Entropy chuẩn hóa: {normalized_entropy:.4f}")
        exp_parts.append(f"   - Nhận xét: CNN đưa ra dự đoán ban đầu như trên, tuy nhiên do entropy cao / margin thấp kết hợp với tín hiệu thực địa rất mạnh ({snail_density} con/m2), hệ thống quyết định chuyển sang Expert-Guided Mode.")
        exp_parts.append(f"   - Hệ thống ưu tiên trọng số tín hiệu chuyên gia để cảnh báo nguy cơ ốc bươu vàng, nhằm hỗ trợ quyết định quản trị rủi ro an toàn nhất.")
    
    if inference_mode != "EXPERT_GUIDED_MODE":
        exp_parts.append(f"   - Khoảng cách phân biệt (Margin) so với lớp thứ 2: {margin:.4f}")
        exp_parts.append(f"   - Entropy chuẩn hóa (Normalized Entropy / OOD signal): {normalized_entropy:.4f}")
    
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

    fired_rules_log = []

    # Thêm các luật VSI
    for w, z, rule_txt in vsi_rules:
        if w > 0:
            contrib = calc_contrib(w, z, total_wz_vsi, vsi_rules)
            fired_rules_log.append(f"   {rule_idx}. [Luật Mức độ bệnh] {rule_txt} (Kích hoạt: {w:.2f}, Đóng góp: {contrib:.1f}%)")
            rule_idx += 1
            
    # Thêm các luật ERI
    for w, z, rule_txt in eri_rules:
        if w > 0:
            contrib = calc_contrib(w, z, total_wz_eri, eri_rules)
            fired_rules_log.append(f"   {rule_idx}. [Luật Thời tiết] {rule_txt} (Kích hoạt: {w:.2f}, Đóng góp: {contrib:.1f}%)")
            rule_idx += 1
            
    # Thêm các luật FAI
    for w, z, rule_txt in alert_rules:
        if w > 0:
            contrib = calc_contrib(w, z, total_wz_alert, alert_rules)
            fired_rules_log.append(f"   {rule_idx}. [Luật Cảnh báo] {rule_txt} (Kích hoạt: {w:.2f}, Đóng góp: {contrib:.1f}%)")
            rule_idx += 1

    if fired_rules_log:
        exp_parts.append("   Các luật dưới đây là những luật có mức kích hoạt > 0, thể hiện quá trình suy diễn mờ dẫn tới kết quả cuối cùng:")
        exp_parts.extend(fired_rules_log)
    else:
        exp_parts.append("   Không có luật mờ đáng kể nào được kích hoạt.")
            
    explanation_str = "\n".join(exp_parts)

    # ── 2. PHẦN KHUYẾN NGHỊ (RECOMMENDATION) ──
    rec_parts = []
    has_chemical_recommendation = False
    
    if predicted_disease == "Healthy":
        rec_parts.append("Hệ thống nhận diện cây lúa có khả năng đang khỏe mạnh. Chưa gợi ý can thiệp thuốc bảo vệ thực vật.")
        rec_parts.append("Hỗ trợ quyết định: Tiếp tục duy trì chế độ bón phân cân đối và kiểm tra đồng ruộng định kỳ.")
    else:
        rec_parts.append(f"Đã phát hiện dấu hiệu có khả năng là bệnh hại: {disease_vi} ở mức độ nghiêm trọng trực quan: {visual_severity_level}.")
        
        # Đưa ra khuyến nghị theo mức độ nguy hiểm
        if "Red Alert" in final_alert_level:
            rec_parts.append("⚠️ CẢNH BÁO CAO: Gợi ý có nguy cơ bùng phát dịch bệnh/dịch hại do điều kiện đang thuận lợi.")
            rec_parts.append("Hành động đề xuất:")
            if predicted_disease == "Tungro":
                rec_parts.append("  1. Ưu tiên xử lý rầy xanh đuôi đen (vật chủ truyền bệnh).")
                rec_parts.append("  2. Cân nhắc tiêu hủy các khóm lúa đã nhiễm bệnh nặng để hạn chế lây lan virus.")
            elif predicted_disease == "Golden Apple Snail":
                rec_parts.append("  1. Gợi ý xử lý bằng biện pháp đặc trị ốc bươu vàng (hoặc bắt thủ công nếu diện tích nhỏ).")
            else:
                rec_parts.append("  1. Cân nhắc sử dụng thuốc bảo vệ thực vật đặc trị phù hợp.")
                rec_parts.append("  2. Khuyến cáo tạm ngừng bón phân đạm (nitrogen) vì đạm có khả năng làm vết bệnh phát triển nhanh hơn.")
            rec_parts.append("  3. Tháo bớt nước trên ruộng nếu đang ngập sâu để giảm độ ẩm tiểu khí hậu (trừ trường hợp dùng thuốc diệt ốc cần giữ nước).")
        elif "Danger" in final_alert_level:
            rec_parts.append("👉 CẦN LƯU Ý: Mức độ rủi ro được đánh giá ở mức cao.")
            rec_parts.append("Hành động đề xuất:")
            rec_parts.append("  1. Có thể tiến hành can thiệp khoanh vùng nhiễm bệnh để tránh lây lan ra diện rộng.")
            rec_parts.append("  2. Kiểm tra nguồn nước tưới, hạn chế tưới nước vào ban đêm khi độ ẩm không khí đang cao.")
        elif "Attention" in final_alert_level:
            rec_parts.append("ℹ️ CHÚ Ý THEO DÕI: Vết bệnh nhẹ nhưng môi trường ẩm ướt hoặc có nguy cơ tiềm ẩn.")
            rec_parts.append("Hành động đề xuất:")
            rec_parts.append("  1. Theo dõi sát sao các ruộng lân cận trong vòng 2-3 ngày tới.")
            rec_parts.append("  2. Gợi ý bón bổ sung phân kali để tăng cường sức đề kháng cho cây, cân nhắc kỹ trước khi dùng hóa chất diện rộng.")
        else:
            rec_parts.append("ℹ️ THEO DÕI THÔNG THƯỜNG: Vết bệnh nhẹ và thời tiết khô ráo có khả năng hạn chế dịch lan nhanh.")
            rec_parts.append("Hỗ trợ quyết định: Cắt tỉa các lá bị bệnh nặng đơn lẻ để tiêu hủy, chưa thật sự cần thiết can thiệp hóa chất diện rộng.")
            
        # Khuyến nghị cụ thể cho từng loại bệnh
        if predicted_disease == "Blast":
            rec_parts.append("  * Đặc thù Bệnh Đạo Ôn: Thường cân nhắc các hoạt chất như Tricyclazole, Fenoxanil hoặc Isoprothiolane. Khuyến cáo tránh bón thêm đạm.")
            has_chemical_recommendation = True
        elif predicted_disease == "Bacterial blight":
            rec_parts.append("  * Đặc thù Bệnh Bạc Lá: Gợi ý các dòng thuốc sát khuẩn gốc đồng (Copper) hoặc thuốc kháng sinh nông nghiệp. Khuyến cáo ngừng tưới tràn.")
            has_chemical_recommendation = True
        elif predicted_disease == "Brownspot":
            rec_parts.append("  * Đặc thù Bệnh Đốm Nâu: Thường phát sinh trên đất nghèo dinh dưỡng hoặc chua. Gợi ý bón lân và kali kết hợp làm cỏ.")
        elif predicted_disease == "Tungro":
            rec_parts.append("  * Đặc thù Bệnh Tungro: Do virus truyền bởi rầy xanh đuôi đen. Cần tập trung kiểm soát rầy để cắt đứt nguồn truyền bệnh.")
        elif predicted_disease == "Golden Apple Snail":
            if snail_density > 3:
                rec_parts.append("  * Đặc thù Ốc bươu vàng: Mật độ cao nguy hiểm. Có thể dùng hoạt chất Metaldehyde (nếu không tháo được nước) hoặc Niclosamide (nếu có lớp nước 2-3cm).")
                has_chemical_recommendation = True
            else:
                rec_parts.append("  * Đặc thù Ốc bươu vàng: Ưu tiên bắt thủ công, thu trứng trên bẹ lá, cắm cọc dụ ốc đẻ trứng. Giữ mực nước ở 2-3cm.")

    if has_chemical_recommendation:
        rec_parts.append("\n⚠️ LƯU Ý QUAN TRỌNG: Việc sử dụng thuốc/hoạt chất hóa học cần tuân thủ nghiêm ngặt hướng dẫn của cán bộ bảo vệ thực vật địa phương và các quy định an toàn nông nghiệp.")

    recommendation_str = "\n".join(rec_parts)
    
    return explanation_str, recommendation_str


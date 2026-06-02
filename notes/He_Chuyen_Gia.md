# Báo cáo Kiến trúc: Tái cấu trúc Hệ Chuyên Gia (Hybrid Expert Decision Support System)

## 1. Mục đích và Động lực (Purpose & Motivation)
Mục tiêu ban đầu của hệ thống là đánh giá mức độ nghiêm trọng của bệnh cây trồng (Plant disease severity assessment). Tuy nhiên, trong quá trình phát triển, nhóm nhận thấy rủi ro "quá tự tin" (overconfident) của mô hình học sâu (CNN) khi đối mặt với dữ liệu nhiễu, ảnh chứa điểm ngoại lai (Out-of-Distribution - OOD), hoặc các điều kiện không chắc chắn.

Để giải quyết vấn đề này và nâng tầm đồ án, nhóm đã thiết kế thêm một **lớp xử lý bất định và tri thức chuyên gia (expert-guided uncertainty handling layer)**. 
Gần đây nhất, hệ thống đã được tái cấu trúc (refactor) để tách bạch hoàn toàn phần Giao diện (UI) và phần Tri thức Chuyên gia (Expert Knowledge Base), tạo nên một kiến trúc Hệ hỗ trợ Quyết định (Decision Support System) đúng nghĩa.

Việc tách biệt này mang lại các lợi ích:
- **Minh bạch hóa logic**: UI không còn chứa các đoạn mã `if/else` quy định luật hay câu chữ giải thích (hardcoded strings).
- **Tuân thủ nguyên lý thiết kế**: Hệ thống mô phỏng cấu trúc kinh điển của một Hệ chuyên gia: `Knowledge Base + Inference Engine + User Interface`.
- **Dễ dàng mở rộng**: Có thể thêm mới các tập luật nông nghiệp, khuyến nghị phun thuốc, và mô tả bệnh học vào Knowledge Base mà không cần chạm vào luồng nhận dạng ảnh CNN hay Hệ suy luận Mờ (Fuzzy Inference).

## 2. Kiến trúc Hệ thống Hiện tại (Conceptual Architecture)
Hệ thống tuân theo luồng xử lý (pipeline) chuẩn xác như sau:

1. **CNN Predictor**: Trích xuất đặc trưng và xuất ra phân phối xác suất (Softmax Distribution).
2. **Uncertainty & OOD Analysis**: Tính toán Entropy, nhận diện OOD để xác định độ phân tán và tính hợp lệ của dữ liệu.
3. **Fuzzy Expert Reasoning (Inference Engine)**: Kết hợp độ tự tin của CNN với các yếu tố ngoại cảnh (Nhiệt độ, Độ ẩm) thông qua logic mờ để đưa ra độ tự tin tổng hợp (Fused Confidence) và chế độ suy luận (Inference Mode).
4. **Expert Knowledge Base (Lớp Tri thức)**: Áp dụng các luật chuyên gia để diễn dịch các chỉ số kỹ thuật khô khan (Entropy, OOD, Fused Confidence) thành ngôn ngữ tự nhiên, cảnh báo rủi ro và phân cấp độ tin cậy.
5. **UI Explanation / Recommendation**: Trình bày thông tin cho người dùng cuối qua một giao diện Dashboard chuyên nghiệp (Dark AI style).

## 3. Chi tiết Mô-đun `expert_system` (Lớp Tri thức)
Mô-đun `expert_system/knowledge_base.py` được xây dựng như một kho tri thức độc lập. Giao diện (UI) sẽ gọi các hàm helper từ mô-đun này để lấy thông điệp hiển thị.

### a. `get_uncertainty_assessment(entropy: float)`
- **Nhiệm vụ**: Đánh giá chỉ số Shannon Entropy.
- **Quy tắc**:
  - `Entropy < 0.5`: Dự đoán tập trung, an toàn (Màu xanh).
  - `0.5 <= Entropy < 1.0`: Có sự phân vân giữa các lớp (Màu cam).
  - `Entropy >= 1.0`: Mơ hồ, cần xem xét lại (Màu đỏ).

### b. `get_expert_guided_assessment(inference_mode: str, is_ood: bool)`
- **Nhiệm vụ**: Đưa ra lời khuyên chuyên gia dựa trên trạng thái của Hệ mờ và trạng thái OOD.
- **Quy tắc**:
  - Nếu chế độ suy luận là `EXPERT_GUIDED_MODE`, `HYBRID_WARNING` hoặc phát hiện `OOD` -> Đưa ra cảnh báo mẫu ảnh có thể chứa dấu hiệu ngoại lai hoặc không chắc chắn, khuyến nghị chuyên gia can thiệp.
  - Ngược lại -> Hệ thống đánh giá mẫu ảnh phù hợp để tự động kết luận.

### c. `get_confidence_category(confidence: float)`
- **Nhiệm vụ**: Phân dải băng tin cậy cuối cùng.
- **Quy tắc**:
  - `> 85%`: High Confidence.
  - `60% - 85%`: Moderate Confidence.
  - `< 60%`: Low Confidence.

## 4. Hướng mở rộng Tri thức Chuyên gia (Future Extension)
Cấu trúc thư mục hiện tại (`expert_system/`) cho phép mở rộng nhanh chóng trong tương lai để bổ sung hàm lượng chuyên môn cho báo cáo:

- **`recommendation_rules.py`**: Tích hợp các quy tắc khuyến nghị nông nghiệp. Ví dụ: Dựa vào nhãn "Đạo ôn", hệ thống có thể truy xuất cơ sở dữ liệu chuyên gia để đề xuất lượng nước tưới, loại phân bón và thuốc bảo vệ thực vật cần thiết.
- **`disease_dictionary.py`**: Lưu trữ từ điển mô tả chi tiết nguyên nhân, biểu hiện, và điều kiện phát sinh của từng loại bệnh.
- **`thresholds.py`**: Quản lý tập trung các ngưỡng ra quyết định (ví dụ ngưỡng Entropy, ngưỡng OOD, ngưỡng Confidence) để các chuyên gia nông nghiệp có thể dễ dàng tinh chỉnh (fine-tune) logic cảnh báo mà không cần phải biết lập trình.

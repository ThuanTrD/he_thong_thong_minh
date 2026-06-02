# Báo cáo Cập nhật: Bổ sung Giao diện Phân tích Độ tin cậy (Confidence Analysis) và Tích hợp Fuzzy Logic

## 1. Mục tiêu và Triết lý Thiết kế
- **Triết lý**: Chuyển đổi hệ thống từ một mô hình nhận dạng hình ảnh tĩnh (Hard Prediction) sang một hệ thống hỗ trợ ra quyết định (Decision Support System) theo chuẩn XAI. Hệ thống sẽ không "ép" ra kết luận khi dữ liệu có độ bất định cao.
- **Tiêu điểm**: Kết hợp Explainable AI (XAI) và Logic Mờ (Fuzzy Logic) để giải thích chi tiết quá trình mô hình đưa ra suy luận. Giao diện được thiết kế mang phong cách hiện đại (Dark AI Dashboard), sử dụng màu Cyan/Teal để tạo cảm giác chuyên nghiệp giống các hệ thống y tế, khoa học.

## 2. Chi tiết Thành phần UI Mới (Confidence Analysis Dialog)
Một nút bấm mới `[ 📊 Analyze Confidence ]` đã được thêm vào bên dưới dự đoán hàng đầu của CNN. Khi người dùng click, một cửa sổ popup (dialog) sẽ hiện ra chứa các thông tin phân tích chuyên sâu:

### a. Raw CNN Confidence (Độ tin cậy gốc)
- Hiển thị nhãn dự đoán có xác suất cao nhất và điểm tự tin thô (Softmax Score) do mạng CNN xuất ra trước khi hệ mờ can thiệp.

### b. Top-5 Softmax Distribution
- Biểu diễn Top 5 lớp dự đoán có xác suất cao nhất dưới dạng các thanh tiến trình ngang (Horizontal bars) hiện đại.
- Cung cấp góc nhìn về mức độ "phân tán" của mô hình, giúp người dùng biết được mô hình đang phân vân giữa các lớp bệnh lý nào (ví dụ: Đạo ôn nhẹ vs. Đốm nâu nhẹ).

### c. Uncertainty Assessment (Đánh giá Bất định dựa trên Shannon Entropy)
- Thông qua chỉ số phân tán Shannon Entropy tính toán từ chuỗi Softmax:
  - **Entropy thấp**: Phân phối xác suất hội tụ tốt -> "Dự đoán tập trung và đáng tin cậy". (Màu xanh)
  - **Entropy trung bình**: Có sự phân vân giữa một vài lớp -> "Tồn tại một mức độ bất định nhất định giữa các lớp bệnh". (Màu cam)
  - **Entropy cao**: Mô hình mơ hồ -> "Dự đoán không rõ ràng, dữ liệu có độ bất định cao và cần được con người đánh giá lại". (Màu đỏ)

### d. Expert-Guided Assessment (Đánh giá theo Hướng dẫn của Chuyên gia)
- Phản ánh trực tiếp trạng thái của **Fuzzy Engine**.
- Nếu ảnh rơi vào trạng thái ngoại lai (Out-of-Distribution - OOD) hoặc kích hoạt ngoại lệ từ thực địa (VD: Phát hiện ốc bươu vàng, thời tiết khắc nghiệt), hệ thống sẽ cảnh báo: *The sample may contain out-of-distribution patterns... Expert review is recommended.*
- Ngược lại, hệ thống sẽ báo kết quả đủ an toàn để suy luận tự động.

### e. Phân loại Cấp độ Tự tin (Confidence Interpretation Block)
- Hệ thống tự động phân dải băng tin cậy:
  - **High Confidence** (> 85%): An toàn để tự động đưa ra kết luận và khuyến nghị.
  - **Moderate Confidence** (60% - 85%): Cần chú ý theo dõi.
  - **Low Confidence** (< 60%): Độ tin cậy thấp, rủi ro cao, bắt buộc cần chuyên gia kiểm chứng.

### f. Advanced Confidence Metrics (Các chỉ số XAI Nâng cao)
- Một panel mở rộng (Expander) dành riêng cho nghiên cứu viên hoặc chuyên gia muốn theo dõi sát các thông số kỹ thuật nội bộ:
  - Điểm Shannon Entropy thực tế.
  - Trạng thái nhận diện Out-of-Distribution (OOD).
  - Độ tự tin tổng hợp (Fused Confidence) sau khi áp dụng các cơ chế phạt (Penalty adjustments) từ Hệ mờ.
  - Chế độ suy luận thực tế (Inference Mode): `NORMAL`, `EXPERT_GUIDED_MODE`, `HYBRID_WARNING`.

## 3. Giá trị Mang lại đối với Hệ Chuyên gia Mờ (Fuzzy Expert System)
- **Tăng tính minh bạch (Transparency)**: Nhờ có XAI Modal, người nông dân hoặc chuyên gia nông nghiệp có thể thấu hiểu rõ *tại sao* hệ thống lại đưa ra mức cảnh báo đó, thay vì chỉ nhận một kết quả dự đoán "hộp đen".
- **Hỗ trợ xử lý Ngoại lệ hiệu quả**: Việc phân tách các chỉ số phân tán và độ tự tin giúp người dùng yên tâm hơn khi kết hợp các thông số ngoại cảnh (Nhiệt độ, Độ ẩm, Mật độ sinh vật hại) trong Module Logic Mờ cùng với phán đoán của AI.
- **Tính liền mạch**: Tách biệt luồng hiển thị (UI Layer) với logic tính toán (Backend Layer) giúp ứng dụng nhẹ nhàng, tương tác mượt mà trong khi vẫn truyền tải đủ hàm lượng khoa học dữ liệu.

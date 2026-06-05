"Severity Assessment" (Đánh giá mức độ nghiêm trọng)

Đây là điểm sáng giá. Rất nhiều dự án chỉ dừng lại ở bài toán Classification (Phân loại: Có bệnh/Không bệnh). Tuy nhiên, dự án của nhóm:

AI (CNN) xuất ra xác suất cho các lớp kèm mức độ (Mild/Severe).
Sau đó, hệ thống tính toán ra chỉ số VSI (Visual Severity Index) và xếp loại mức độ tổn thương: Nhẹ (Mild), Trung bình (Moderate), Nghiêm trọng (Severe).
Đúng vậy, bạn nhận xét rất chuẩn xác!

Ngay từ khâu đầu vào, mô hình CNN (EfficientNet-B0) đã được huấn luyện với tập dữ liệu phân chia sẵn theo mức độ bệnh. Cụ thể, thay vì chỉ có 5 lớp bệnh cơ bản, bộ dữ liệu (dataset) và mô hình AI đã được thiết kế mở rộng thành 9 lớp (classes), bao gồm:

Khỏe mạnh (Healthy)
Bạc lá nhẹ (Mild Bacterial blight)
Bạc lá nặng (Severe Bacterial blight)
Đạo ôn nhẹ (Mild Blast)
Đạo ôn nặng (Severe Blast)
Đốm nâu nhẹ (Mild Brownspot)
Đốm nâu nặng (Severe Brownspot)
Vàng lụi nhẹ (Mild Tungro)
Vàng lụi nặng (Severe Tungro)
Tuy nhiên, điểm "ăn tiền" và tạo nên sự khác biệt của hệ thống (lý do cần thêm Fuzzy Logic) nằm ở chỗ này:

Nếu chỉ dùng CNN đơn thuần, hệ thống sẽ đưa ra một quyết định "cứng" (Hard label) - tức là bệnh 1 là Nhẹ, 2 là Nặng. Nhưng trong thực tế nông nghiệp, ranh giới giữa "Nhẹ" và "Nặng" rất mong manh và không rõ ràng.

Vì vậy, kiến trúc của nhóm đã làm một việc thông minh hơn:

Lấy điểm số mềm (Softmax Scores): Hệ thống lấy xác suất của cả hai lớp (ví dụ: AI báo Đạo ôn nhẹ 60%, Đạo ôn nặng 30%).
Nội suy qua Fuzzy Logic (VSI): Thuật toán Mờ (Fuzzy Engine) sẽ trộn hai điểm số xác suất này lại để tính ra một thang điểm liên tục gọi là Visual Severity Index (VSI) chạy từ 0 đến 100. Điều này giúp hệ thống đánh giá được các trạng thái "lửng lơ" (VD: "Bệnh đang ở mức Trung Bình - Moderate, chuẩn bị chuyển sang Nặng").
Kết hợp yếu tố Ngoại cảnh: Đánh giá trên lá (CNN) là chưa đủ. Fuzzy Logic sẽ tiếp tục lấy mức độ bệnh (VSI) kết hợp với Nhiệt độ & Độ ẩm để đưa ra Cảnh báo cuối cùng (Final Alert Level). (VD: Bệnh trên lá mới chớm Nhẹ, nhưng độ ẩm đang 90% rất thuận lợi cho nấm phát triển $\rightarrow$ Hệ thống tự động nâng mức cảnh báo lên Danger).
Có thể tóm tắt là: CNN làm nhiệm vụ "nhìn" và "ước lượng thô" mức độ bệnh, còn Fuzzy Logic đóng vai trò là vị chuyên gia nông nghiệp ngồi "cân nhắc, dung hòa" các yếu tố để đưa ra đánh giá tinh tế và thực tế nhất.

6:21 PM
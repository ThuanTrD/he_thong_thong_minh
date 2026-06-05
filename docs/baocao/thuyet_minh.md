# THUYẾT MINH GIAO DIỆN HỆ THỐNG CHẨN ĐOÁN BỆNH LÚA (NHÓM 14)

Giao diện hệ thống chẩn đoán bệnh lúa là một Web Dashboard tương tác được xây dựng bằng Streamlit, thể hiện một luồng xử lý lai (Hybrid Pipeline) kết hợp giữa Deep Learning (CNN) và Hệ chuyên gia mờ (Fuzzy Logic), cùng khả năng giải thích (Explainable AI - XAI). 

Dưới đây là phần thuyết minh chi tiết cho từng khu vực trên giao diện:

## 1. Tiêu đề và Thanh điều hướng (Sidebar)
*   **Tiêu đề chính:** "BÁO CÁO MÔN HỌC HỆ THỐNG THÔNG MINH - NHÓM 14", với phụ đề nhấn mạnh công nghệ cốt lõi "CNN + FUZZY EXPERT SYSTEM + EXPLAINABLE AI (XAI) FOR SMART AGRICULTURE".
*   **Luồng xử lý (Pipeline):** Được trực quan hóa ở ngay dưới tiêu đề, cho thấy 5 bước: `Image Input` $\rightarrow$ `CNN Classifier` $\rightarrow$ `Softmax Distribution` $\rightarrow$ `Fuzzy Inference` $\rightarrow$ `XAI Recommendation`.
*   **Thanh điều hướng bên trái (Sidebar):**
    *   **System Status (Trạng thái hệ thống):** Cho biết các động cơ (Engine) CNN và Fuzzy đều đang hoạt động (Active). Mô hình đang chạy trên CPU, sử dụng checkpoint `best_model.pt`.
    *   **Quick Operations:** Hướng dẫn sử dụng nhanh qua 4 bước: (1) Tải ảnh lên, (2) Chỉnh thời tiết, (3) Xem chẩn đoán, (4) Xuất báo cáo.

## 2. Khu vực Nhập liệu & Môi trường (INPUT & ENVIRONMENT)
Nằm ở cột ngoài cùng bên trái của phần thân chính, khu vực này cho phép người dùng cung cấp dữ liệu đầu vào cho hệ thống:
*   **Tải ảnh lá lúa:** Người dùng tải lên một hình ảnh (ví dụ trong hình là lá lúa bị các đốm bệnh đạo ôn). Hệ thống sẽ hiển thị ảnh xem trước ngay lập tức.
*   **Cảm biến môi trường giả lập:** Gồm hai thanh trượt (slider) cho phép điều chỉnh:
    *   **Nhiệt độ thời tiết ($^\circ$C):** Đang thiết lập ở mức 28.00$^\circ$C.
    *   **Độ ẩm không khí (%):** Đang thiết lập ở mức 45.00%.
    *(Hai thông số môi trường này sẽ được đưa thẳng vào bộ suy diễn mờ Fuzzy Engine để đánh giá nguy cơ lây lan).*
*   **Nhập liệu từ Nông dân (Manual Input):** Một thanh trượt đặc biệt cho phép người dùng nhập *Mật độ Ốc bươu quan sát được* ngoài thực địa. Đây là cơ chế Human-in-the-loop giúp hệ thống vượt qua điểm mù của Camera (CNN) khi sâu hại không xuất hiện trực tiếp trên lá.

## 3. Khu vực Nhận dạng & Suy luận Lai (CNN & HYBRID INFERENCE)
Nằm ở cột giữa, khu vực này trình bày kết quả đầu ra kết hợp giữa mạng nơ-ron (EfficientNet-B0) và Tín hiệu chuyên gia (Expert Signal):
*   **Trạng thái Suy luận (Inference Mode):** Hệ thống đánh giá đồng thời độ tự tin của CNN và mật độ ốc bươu vàng để hiển thị 1 trong 3 trạng thái:
    *   🟢 **AI Confident:** CNN tự tin cao, giữ nguyên kết quả nhận diện bệnh lá lúa.
    *   🟡 **Hybrid Warning:** Kết hợp cảnh báo bệnh lá và tín hiệu ngoại lệ thực địa.
    *   🔴 **Expert-Guided Mode:** Khi ảnh đầu vào bất định (OOD) và tín hiệu ốc bươu vàng mạnh, hệ thống sẽ ưu tiên cảnh báo Ốc bươu vàng để đảm bảo an toàn.
*   **Lớp bệnh nhận dạng & Fused Confidence:** Hiển thị kết luận cuối cùng và **Độ tự tin tổng hợp** (Fused Confidence) đã được tính toán lại thông qua cơ chế lai ghép.
*   **Phân phối Softmax (Top-5):** Biểu đồ thanh ngang hiển thị 5 lớp có khả năng cao nhất gốc từ mạng CNN, giúp người dùng thấy được sự "phân vân" (bất định) của mô hình học sâu trước khi có sự can thiệp của chuyên gia.

## 4. Khu vực Suy diễn Mờ (FUZZY EXPERT REASONING)
Nằm ở cột bên phải, khu vực này là kết quả đánh giá tổng hợp của Hệ chuyên gia (Fuzzy Engine) dựa trên cả ảnh chụp (CNN) và điều kiện môi trường:
*   **Mức cảnh báo (Alert):** Hệ thống đưa ra mức cảnh báo chung là **Attention (Chú ý)**.
*   **Nguy cơ thời tiết:** Đánh giá mức độ thuận lợi cho bệnh phát triển dựa trên nhiệt độ (28$^\circ$C) và độ ẩm (45%). Kết quả là **Low (Thấp)**.
*   **Độ bất định chẩn đoán:** Đánh giá sự phân tán trong biểu đồ Softmax của CNN. Kết quả là **Medium (Trung bình)**.
*   **Nghiêm trọng trực quan:** Tính toán mức độ nghiêm trọng dựa trên loại bệnh và độ tự tin của CNN. Kết quả là **Moderate (Trung bình)**.
*   **Độ tin cậy hệ thống:** Điểm số tổng hợp đánh giá độ đáng tin cậy của toàn bộ kết luận, đạt **69.44%**.

## 5. Khu vực Tabs Chức năng Chi tiết (Bottom Section)
Phần dưới cùng chia thành 3 Tabs (thẻ) để người dùng xem chi tiết kết quả:
*   **Báo cáo giải thích (XAI REASONING):** Nơi hệ thống sẽ diễn dịch bằng ngôn ngữ tự nhiên lý do tại sao đưa ra mức cảnh báo như trên (dựa trên việc kích hoạt các luật mờ nào).
*   **Khuyến nghị chuyên gia (AI EXPERT ADVICE):** Cung cấp các hành động nông nghiệp cụ thể (ví dụ: cách ly, xịt thuốc, bón phân) tương ứng với tình trạng bệnh.
*   **Trạng thái hệ thống & Export (Tab đang mở):**
    *   **Xuất báo cáo:** Cung cấp nút tải xuống toàn bộ dữ liệu suy diễn dưới định dạng JSON, phục vụ cho việc lưu trữ hoặc tích hợp vào hệ thống khác.
    *   **Chi tiết hệ thống:** Hiển thị thông số kỹ thuật (Mô hình EfficientNet-B0, động cơ Sugeno Hybrid, API Streamlit 1.37.0).

---
**Tổng kết:** Giao diện được thiết kế trực quan, chia tách rõ ràng vai trò của AI nhận thức (Deep Learning) và AI logic (Fuzzy Expert System), qua đó cung cấp cho người nông dân hoặc chuyên gia nông nghiệp không chỉ kết quả "Bệnh gì?" mà còn là "Tại sao?" và "Cần làm gì tiếp theo?".

# Báo cáo Nâng cấp Hệ thống Học thuật (Academic Upgrade Walkthrough)

Dự án đã được nâng cấp toán học thành công, đáp ứng hoàn toàn các tiêu chuẩn chặt chẽ của một đồ án thạc sĩ về Trí tuệ nhân tạo.

## Các thay đổi đã thực hiện

### 1. Nâng cấp OOD Detector (Normalized Entropy)
- **File sửa đổi:** `rice_api_backend/ood_detector.py` và `rice_api_backend/config.py`
- Thuật toán Entropy đã được chuyển từ cơ số $e$ sang cơ số 2 ($\log_2$) để tính toán lượng thông tin (bits) chuẩn mực.
- Hệ thống áp dụng phép chia cho $\log_2(N)$ (với $N=9$ classes) để tính ra **Normalized Shannon Entropy (Uncertainty Score)**.
- Kết quả được chuẩn hóa về dải $[0, 1]$, và ngưỡng chặn được cập nhật chặt chẽ xuống `0.35`.

### 2. Định lượng báo cáo XAI (Quantitative XAI)
- **File sửa đổi:** `rice_fuzzy_xai/explanation.py`
- Thay vì chỉ in ra *Trọng số kích hoạt ($w$)* vô hồn, hệ thống hiện đã có thuật toán tính toán độc lập **Phần trăm đóng góp (Contribution %)** cho mỗi luật.
- Giải thích sinh ra ở Frontend bây giờ sẽ chi tiết đến từng số thập phân, ví dụ: *"Luật Mức độ bệnh (Kích hoạt: 0.80, Đóng góp: 66.7%)"*.
- Điểm nhấn này minh chứng tuyệt đối cho tính minh bạch (Traceability) của hệ mờ.

### 3. Tinh chỉnh Học thuật trên UI
- **File sửa đổi:** `app_streamlit.py`
- Đã loại bỏ các từ ngữ dân dã và thay bằng thuật ngữ chuyên ngành:
  - *"Nghiêm trọng trực quan"* $\rightarrow$ **"Chỉ số tổn thương trực quan (VSI)"**
  - *"Độ tin cậy hệ thống"* $\rightarrow$ **"Độ chắc chắn chẩn đoán"**

## Hướng dẫn Kiểm tra
Bạn có thể trực tiếp mở giao diện Streamlit hiện tại (hệ thống đã tự động reload). 
Hãy thử tải lên một ảnh bệnh đạo ôn và nhìn sang cột `FUZZY EXPERT REASONING` bên phải, kéo xuống phần giải thích XAI. Bạn sẽ thấy ngay các phần trăm đóng góp được liệt kê vô cùng chuyên nghiệp!

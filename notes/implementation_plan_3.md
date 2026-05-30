# Kế hoạch Nâng cấp Học thuật (Academic Upgrade Plan)

Dựa trên yêu cầu tăng cường tính học thuật cho đồ án thạc sĩ mà không làm vỡ luồng chạy ổn định hiện tại, tôi đề xuất thực hiện các nâng cấp toán học sau. Các nâng cấp này giúp hệ thống chặt chẽ hơn về mặt lý thuyết và khớp hoàn toàn với những luận điểm đã viết trong Cheat Sheet.

## User Review Required

> [!IMPORTANT]
> Việc nâng cấp này sẽ tinh chỉnh lại công thức toán học lõi của OOD và XAI. Vui lòng xem xét các thay đổi dưới đây và xác nhận (Approve) để tôi tiến hành sửa code.

## Proposed Changes

### 1. Nâng cấp Toán học: Normalized Shannon Entropy
- **Vấn đề:** Shannon Entropy hiện tại tính theo cơ số $e$ (Natural log) và không chuẩn hóa, dẫn đến giá trị phụ thuộc vào số lượng class (9 classes). Giá trị `0.75` là một con số "magic number" thực nghiệm, khó giải thích về mặt học thuật.
- **Giải pháp (OOD Detector):**
  - Chuyển sang cơ số 2 ($\log_2$) để tính theo đơn vị bits (chuẩn Information Theory).
  - Áp dụng **Normalized Entropy**: chia cho $\log_2(N)$ (với $N=9$).
  - Công thức: $H_{norm} = \frac{-\sum p_i \log_2 p_i}{\log_2(N)} \in [0, 1]$.
  - Lợi ích: Biến Entropy thành một thang đo chuẩn xác từ 0 đến 1 (100%), đại diện cho "Uncertainty Score".
- **Cập nhật Threshold:** Ngưỡng `0.75` cũ sẽ tương đương với `0.34` trên thang Normalized mới. Sẽ cập nhật `ENTROPY_THRESHOLD = 0.35` trong `config.py`.

### 2. Nâng cấp XAI: Rule Contribution Percentage
- **Vấn đề:** Báo cáo XAI hiện tại chỉ in ra *Fire Strength* ($w_i$), không phản ánh được mức độ chi phối thực tế của luật đó vào giá trị giải mờ cuối cùng.
- **Giải pháp (Explanation Module):**
  - Cập nhật hàm sinh giải thích để tính toán chính xác phần trăm đóng góp của mỗi luật dựa trên công thức giải mờ Sugeno.
  - Công thức Contribution của luật thứ $i$: $$ C_i = \frac{w_i \cdot z_i}{\sum_{k=1}^n w_k \cdot z_k} \times 100\% $$
  - Lợi ích: Báo cáo sẽ hiển thị rõ "Luật A đóng góp 75%, Luật B đóng góp 25%", biến XAI thành một thước đo định lượng (Quantitative XAI) cực kỳ thuyết phục cho hội đồng.

### 3. Nâng cấp Giao diện học thuật (Streamlit UI)
- Đổi tên các nhãn trên UI cho chuẩn thuật ngữ học thuật:
  - `Shannon Entropy` $\rightarrow$ `Normalized Entropy (Độ bất định)`
  - `Nghiêm trọng trực quan` $\rightarrow$ `Chỉ số tổn thương trực quan (Visual Severity Index - VSI)`
  - `Độ tin cậy hệ thống` $\rightarrow$ `Độ chắc chắn chẩn đoán (Diagnostic Certainty)`

## Verification Plan

### Automated Tests
- Chạy lại API `uvicorn` và giao diện `streamlit`.
- Upload ảnh ốc bươu vàng để kiểm tra OOD Detector với Normalized Entropy mới có bắt được ảnh rác không.
- Upload ảnh hợp lệ (Tungro) và xem báo cáo XAI ở cột bên phải để xác minh Phần trăm đóng góp có cộng lại đúng bằng 100% không.
- Đảm bảo tính năng Expert Override (Snail) vẫn hoạt động hoàn hảo.

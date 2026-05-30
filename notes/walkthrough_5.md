# Báo cáo Nâng cấp: Semantic Group Confidence Aggregation

Chúng ta vừa thực hiện một cú "lột xác" hoàn toàn về mặt kiến trúc cho lớp bảo vệ OOD (Out-of-Distribution Safeguard) của hệ thống, xử lý triệt để bài toán **Probability Splitting (Phân mảnh xác suất)** kinh điển trong AI đa nhãn.

## Các thay đổi cốt lõi

### 1. Nâng cấp OOD Detector (`rice_api_backend/ood_detector.py`)
- Thay vì lấy "bù nhìn" là lớp có xác suất cao nhất (Top-1 MSP), hệ thống giờ đây đã thông minh hơn khi biết gom nhóm các lớp có chung bản chất (Semantic Groups).
- **Thuật toán mới:** Tính tổng xác suất của các lớp anh em (Ví dụ: `Blast = Mild Blast + Severe Blast`).
- Lợi ích: Bức ảnh lá đạo ôn trong ví dụ của bạn sẽ không còn bị chặn oan vì sự phân chia xác suất $41.87\% + 22.13\%$, mà sẽ được tính gộp thành một khối thống nhất là $64\%$.

### 2. Decision Engine Đa Trạng Thái (`rice_api_backend/decision_engine.py`)
Hệ thống không còn cứng nhắc giữa 2 bờ vực Sống (KNOWN) và Chết (OOD) nữa, mà đã có vùng đệm học thuật:
- **KNOWN (Group Confidence >= 0.60):** Khẳng định tự tin, cho phép vào Hệ Mờ.
- **UNCERTAIN (0.40 <= Group Confidence < 0.60):** Trạng thái lưỡng lự. Hệ thống sẽ báo cáo *"Hệ thống nhận diện ảnh có đặc trưng thuộc nhóm X, tuy nhiên độ chắc chắn giữa các mức độ bệnh còn phân tán."* 
- **OOD (Group Confidence < 0.40 hoặc Entropy > 0.65):** Rác rưởi thật sự, chặn đứng!

### 3. Tối ưu Giao diện Streamlit (`app_streamlit.py`)
- Khi rơi vào trạng thái UNCERTAIN, UI không còn báo dòng chữ "Out-of-Distribution/Unknown" thô kệch nữa.
- UI sẽ vinh danh **"Nhóm bệnh nghi ngờ (Group Confidence)"** và hiển thị tên nhóm chung chung (ví dụ: `Nhóm bệnh: Blast`) kèm tổng độ tự tin, thay vì cố chấp in ra một mức độ nặng/nhẹ thiếu tin cậy.

### 4. Bồi đắp Tài liệu Học thuật (`notes/baocao/cheat_sheet_bao_ve.md`)
- Kịch bản bảo vệ đã được bổ sung nguyên một chuyên mục riêng (Câu số 1) để đàm đạo về thuật ngữ **Probability Splitting** và **Semantic Group Confidence Aggregation**. Đây sẽ là ngón đòn quyết định giúp bạn lấy trọn vẹn điểm số phản biện từ hội đồng giám khảo khó tính nhất!

## Bước tiếp theo
Hãy F5 và upload lại tấm ảnh lá Blast đó lên Streamlit để chứng kiến hệ thống mới nuốt chửng nó và nôn ra báo cáo XAI hoàn hảo nhé!

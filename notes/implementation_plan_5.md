# Kế hoạch Refactor OOD Detection (Semantic Group Confidence Aggregation)

Dựa trên yêu cầu học thuật của bạn, hệ thống cần giải quyết hiện tượng **Probability Splitting** (Softmax fragmentation) khi mô hình CNN phân mảnh xác suất vào các nhãn có cùng chung thuộc tính ngữ nghĩa (ví dụ: Đạo ôn nhẹ và Đạo ôn nặng).

## User Review Required

> [!IMPORTANT]
> Việc refactor này sẽ thay đổi luồng đầu ra của API `predict`, thay vì trả về `predicted_class` là nhãn cụ thể, hệ thống sẽ sử dụng `top_group` làm mỏ neo chính cho đánh giá OOD. Vui lòng xem xét các thay đổi dưới đây và xác nhận (Approve) để tôi tiến hành sửa code.

## Proposed Changes

### 1. Cấu hình Semantic Groups (`rice_api_backend/config.py`)
Định nghĩa một hằng số `DISEASE_GROUPS` ánh xạ từ tên nhóm bệnh sang các nhãn cụ thể của CNN.
```python
DISEASE_GROUPS = {
    "Healthy": ["Healthy"],
    "Bacterial blight": ["Mild Bacterial blight", "Severe Bacterial blight"],
    "Blast": ["Mild Blast", "Severe Blast"],
    "Brownspot": ["Mild Brownspot", "Severe Brownspot"],
    "Tungro": ["Mild Tungro", "Severe Tungro"],
}
# Ngưỡng mới
GROUP_CONFIDENCE_THRESHOLD = 0.65
UNCERTAIN_THRESHOLD = 0.40
```

### 2. Cập nhật OOD Detector (`rice_api_backend/ood_detector.py`)
- Thêm phương thức `calculate_group_confidence(cnn_scores)` để tính tổng xác suất (sum of probabilities) cho mỗi nhóm `DISEASE_GROUPS`.
- Thay đổi logic `detect` để đánh giá dựa trên `group_confidence` thay vì `max_confidence` (Top-1 MSP).
- Logic đánh giá 3 trạng thái:
  - `KNOWN`: `group_confidence >= 0.65`
  - `UNCERTAIN`: `0.40 <= group_confidence < 0.65`
  - `OOD`: `group_confidence < 0.40` HOẶC `entropy > ENTROPY_THRESHOLD`

### 3. Cập nhật Decision Engine (`rice_api_backend/decision_engine.py`)
- Sửa đổi message trả về để phù hợp với ngữ cảnh Group Confidence.
- Thay vì "Hình ảnh không thuộc 4 nhóm bệnh", thông báo UNCERTAIN sẽ là: *"Hệ thống nhận diện ảnh có đặc trưng thuộc nhóm {top_group}, tuy nhiên độ chắc chắn giữa các mức độ bệnh còn phân tán."*

### 4. Cập nhật API Backend (`rice_api_backend/api.py`)
- Trả về `top_group` và `group_confidence` trong JSON response.

### 5. Cập nhật Frontend (`app_streamlit.py`)
- Hiển thị "Nhóm bệnh nghi ngờ cao nhất" thay vì Top-1 Class khi hệ thống bị kẹt ở trạng thái UNCERTAIN.
- Sửa đổi cách bắt lỗi OOD từ Backend.

### 6. Cập nhật Cheat Sheet (`notes/baocao/cheat_sheet_bao_ve.md`)
- Thêm luận điểm học thuật về `Probability Splitting`, `Softmax fragmentation` và `Semantic Group Confidence Aggregation`.

## Verification Plan

### Automated Tests
- Chạy lại API Backend và Streamlit.
- Sử dụng chính tấm ảnh `BLAST9_068.JPG` trong ví dụ của bạn:
  - Đạo ôn nặng (22.13%) + Đạo ôn nhẹ (41.87%) = Tổng Blast Confidence: 64.00%
  - Với ngưỡng mới 0.65, nó sẽ rơi vào `UNCERTAIN` thay vì bị chặn hoàn toàn, hoặc nếu bạn muốn ngưỡng thấp hơn, tôi sẽ cấu hình để nó thành `KNOWN`.

### Open Question
Bạn có muốn `GROUP_CONFIDENCE_THRESHOLD` được giữ ở mức `0.65` (như ví dụ) hay một mức thấp hơn (ví dụ `0.50`) để cho phép ảnh `BLAST9_068.JPG` vượt qua rào cản và đi vào Hệ Mờ?

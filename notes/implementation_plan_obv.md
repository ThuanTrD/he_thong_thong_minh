# Bổ sung Ngoại lệ "Ốc bươu vàng" vào Hệ thống chẩn đoán

Do mô hình Deep Learning (CNN) hiện tại chưa được huấn luyện để nhận diện hình ảnh ốc bươu vàng, chúng ta sẽ áp dụng phương án **"Ngoại lệ chuyên gia" (Expert Exception)**. Hệ thống sẽ bỏ qua kết quả hình ảnh của CNN nếu người nông dân chủ động báo cáo có sự xuất hiện của ốc bươu vàng trên ruộng với mật độ đáng kể.

## User Review Required

> [!WARNING]
> Vì CNN không thể tự động đếm ốc, người dùng sẽ phải tự đánh giá mật độ ốc trên ruộng và nhập vào thông qua thanh kéo (slider). Phương pháp này giúp hệ thống phản ứng ngay lập tức với dịch hại mà không cần tốn kém thời gian thu thập dữ liệu và huấn luyện lại (retrain) mô hình AI nặng.

## Proposed Changes

### Giao diện Điều khiển (UI)

#### [MODIFY] `app_streamlit.py`
- Bổ sung thanh trượt (slider): **"Mật độ Ốc bươu vàng (con/m2)"** vào khu vực **INPUT & ENVIRONMENT**. Mặc định là `0.0`.
- Chuyền giá trị `snail_density` này vào đối tượng `FuzzyInput` để bộ suy diễn mờ xử lý.

---

### Cấu trúc Dữ liệu

#### [MODIFY] `rice_fuzzy_xai/schemas.py`
- Thêm trường `snail_density: Optional[float] = 0.0` vào class `FuzzyInput`.

#### [MODIFY] `rice_fuzzy_xai/config.py`
- Bổ sung định nghĩa ánh xạ tiếng Việt: `"Golden Apple Snail": "Ốc bươu vàng"`.

---

### Bộ Suy diễn Mờ (Fuzzy Logic)

#### [MODIFY] `rice_fuzzy_xai/engine.py`
- Cập nhật hàm `run()`: Nếu `snail_density > 0`, kích hoạt cơ chế "Bypass" (Vượt quyền):
  - Ép `predicted_disease` thành `"Golden Apple Snail"`.
  - Nếu `snail_density > 3` (Mật độ cao): Ép Mức độ cảnh báo (`final_alert_level`) thành **"Red Alert (Báo động đỏ)"**.
  - Nếu `0 < snail_density <= 3` (Mật độ thấp): Ép Mức độ cảnh báo thành **"Attention (Chú ý)"**.
  - Chuyền `snail_density` vào module XAI `generate_explanation`.

#### [MODIFY] `rice_fuzzy_xai/explanation.py`
- Cập nhật hàm `generate_explanation()` nhận thêm tham số `snail_density`.
- Nếu bệnh là `"Golden Apple Snail"`, bộ XAI sẽ tạo ra các báo cáo giải thích và khuyến nghị chuyên biệt:
  - **Khuyến nghị mật độ nhẹ:** Bắt ốc thủ công, rút nước, thả vịt.
  - **Khuyến nghị mật độ nặng:** Dùng thuốc *Niclosamide* hoặc *Metaldehyde* theo đúng hướng dẫn của cơ quan Nông nghiệp.

## Verification Plan
1. Khởi chạy giao diện Streamlit.
2. Tải lên một hình ảnh lá lúa bất kỳ.
3. Kéo thanh trượt "Mật độ Ốc bươu vàng" lên `4.0` con/m2.
4. Kiểm tra xem hệ thống có báo động đỏ "Ốc bươu vàng" và bỏ qua chẩn đoán bệnh lý của CNN hay không. Khuyến nghị thuốc hóa học có xuất hiện đúng như kịch bản hay không.

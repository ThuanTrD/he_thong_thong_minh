# Walkthrough — Kết Quả Triển Khai Package `rice_fuzzy_xai`

Tôi đã hoàn tất việc thiết kế, xây dựng và kiểm thử thành công package độc lập `rice_fuzzy_xai` cùng kịch bản chạy end-to-end trực tiếp trên ảnh lá lúa trên nhánh `khanhtrang`.

## Cấu Trúc Các Tệp Tin Đã Triển Khai

Package đã được tổ chức khoa học để tách biệt các trách nhiệm:
1. **[`rice_fuzzy_xai/schemas.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/rice_fuzzy_xai/schemas.py):** Định nghĩa rõ ràng cấu trúc dữ liệu đầu vào `FuzzyInput` (scores, top_class, top_confidence, và optional temperature/humidity) cùng đầu ra `FuzzyOutput` thông qua Python `dataclass`.
2. **[`rice_fuzzy_xai/config.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/rice_fuzzy_xai/config.py):** Chứa các hằng số cấu hình mặc định (nhiệt độ, độ ẩm) và ánh xạ lớp phân loại từ CNN sang các nhóm bệnh chính (Blast, Brownspot, v.v.).
3. **[`rice_fuzzy_xai/membership.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/rice_fuzzy_xai/membership.py):** Định nghĩa hàm tam giác/hình thang và mờ hóa điểm tự tin, margin phân tách giữa lớp 1 & lớp 2, nhiệt độ, độ ẩm.
4. **[`rice_fuzzy_xai/rules.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/rice_fuzzy_xai/rules.py):** Định nghĩa hệ quy tắc mờ Sugeno và giải mờ cho: độ bất định chẩn đoán, độ tin cậy hệ thống, mức độ nghiêm trọng trực quan, nguy cơ môi trường và mức độ cảnh báo tổng hợp.
5. **[`rice_fuzzy_xai/engine.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/rice_fuzzy_xai/engine.py):** Bộ máy suy diễn chính tính toán các giá trị trung gian (hiệu số margin, Mild/Severe score) và chạy qua hệ luật mờ.
6. **[`rice_fuzzy_xai/explanation.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/rice_fuzzy_xai/explanation.py):** Sinh tự động báo cáo XAI bằng ngôn ngữ tự nhiên tiếng Việt và đưa ra các khuyến nghị chăm sóc lúa tương ứng.

## Kịch Bản Chạy & Tài Liệu & Kiểm Thử
- **CLI Runner kết nối trực tiếp ảnh [`scripts/run_hybrid_pipeline.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/scripts/run_hybrid_pipeline.py):** Tải mô hình CNN, chạy phân loại ảnh lá lúa, trích xuất điểm tự tin và chuyển trực tiếp sang bộ máy suy diễn mờ XAI của package `rice_fuzzy_xai`.
- **CLI Runner độc lập từ điểm số/JSON [`scripts/run_fuzzy_inference.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/scripts/run_fuzzy_inference.py):** Cho phép chạy suy diễn mờ trực tiếp từ tham số CLI hoặc từ file JSON kết quả CNN.
- **Tài liệu Thiết Kế [`docs/fuzzy_design.md`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/docs/fuzzy_design.md):** Mô tả toán học chi tiết về các hàm liên thuộc, singletons và tập luật phục vụ báo cáo môn học của bạn.
- **Unit Tests [`tests/test_fuzzy_engine.py`](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/2.BaitapNhom/code/tests/test_fuzzy_engine.py):** Chứa 5 ca kiểm thử bao phủ toàn bộ các phương pháp tính toán mờ hóa, giải mờ và trường hợp cây khỏe mạnh/bệnh nặng nguy cơ cao.

## Kết Quả Xác Minh Luồng Chạy Ảnh Thực Tế (End-to-End Image)

Khi chạy kiểm thử tích hợp trên tệp ảnh lá lúa thật `demo_assets/demo_mild_blast.jpg` cùng thời tiết ẩm ướt (`--temp 28 --humidity 85`):

```text
===========================================================================
  BẮT ĐẦU CHẠY PIPELINE LAI END-TO-END (IMAGE -> CNN -> FUZZY XAI)
===========================================================================
  [Đầu vào 1] Ảnh lá lúa    : datatest/demo_mild_blast.jpg
  [Đầu vào 2] Nhiệt độ      : 28.0
  [Đầu vào 3] Độ ẩm         : 85.0
---------------------------------------------------------------------------
[LUỒNG CHẠY] Bước 1: Đang tải mô hình CNN...
[MODEL] Loaded: efficientnet_b0 | Classes: ['Healthy', 'Mild Bacterial blight', 'Mild Blast', 'Mild Brownspot', 'Mild Tungro', 'Severe Bacterial blight', 'Severe Blast', 'Severe Brownspot', 'Severe Tungro']
[LUỒNG CHẠY] Bước 2: Đang nhận dạng hình ảnh qua mô hình CNN...
  ✓ Nhận diện từ CNN: Mild Blast (Độ tự tin: 64.51%)
[LUỒNG CHẠY] Bước 3: Đang thực hiện suy diễn mờ và sinh giải thích XAI...
  ✓ Hoàn tất suy diễn mờ.

===========================================================================
  BÁO CÁO PHÂN TÍCH TỔNG HỢP & GIẢI THÍCH (HYBRID REPORT & XAI)
===========================================================================
=== BÁO CÁO GIẢI THÍCH CHẨN ĐOÁN (XAI REPORT) ===
1. Phân tích Hình ảnh (CNN):
   - Mô hình CNN chẩn đoán lá lúa nhiễm bệnh: Đạo ôn lá (Rice Blast)
   - Phân loại chi tiết của CNN: Mild Blast (Độ tự tin: 64.51%)
   - Khoảng cách phân biệt (Margin) so với lớp thứ 2: 0.4421
   - Đánh giá độ bất định chẩn đoán: Medium (Trung bình)
   - Độ tin cậy chẩn đoán tổng hợp của hệ thống: 76.97%

2. Phân tích Tác động Môi trường (Fuzzy Logic):
   - Thông số môi trường ghi nhận: Nhiệt độ = 28.0°C, Độ ẩm = 85.0%
   - Đánh giá nguy cơ bùng phát do thời tiết: High (Cao)

3. Mức độ Cảnh báo Tổng hợp: Attention (Chú ý)
   - Đánh giá mức độ tổn thương trên bề mặt lá (Visual Severity): Mild (Nhẹ)

4. Các Luật Mờ được kích hoạt (Fuzzy Rules Fired):
   1. [Luật Mức độ bệnh] 0.580228 (Trọng số kích hoạt: 20.00)
   2. [Luật Mức độ bệnh] 0.2747149999999999 (Trọng số kích hoạt: 20.00)
   3. [Luật Thời tiết] 0.8571428571428571 (Trọng số kích hoạt: 85.00)
   4. [Luật Cảnh báo] 1.0 (Trọng số kích hoạt: 40.00)

Khuyến nghị hành động nông nghiệp (Agricultural Recommendations):
Đã phát hiện bệnh hại: Đạo ôn lá (Rice Blast) ở mức độ nghiêm trọng trực quan: Mild (Nhẹ).
ℹ️ CHÚ Ý THEO DÕI: Vết bệnh nhẹ nhưng môi trường ẩm ướt hoặc có nguy cơ nhẹ.
Hành động khuyến nghị:
  1. Theo dõi sát sao các ruộng lân cận trong vòng 2-3 ngày tới.
  2. Chỉ nên bón phân kali để tăng cường sức đề kháng cho cây, chưa cần phun thuốc hóa học diện rộng.
  * Đặc thù Bệnh Đạo Ôn: Sử dụng các hoạt chất như Tricyclazole, Fenoxanil hoặc Isoprothiolane. Tránh bón thêm phân đạm.
===========================================================================
```

---
Toàn bộ mã nguồn và tệp tích hợp đã được commit và push lên remote branch `khanhtrang` một cách an toàn và sạch sẽ.

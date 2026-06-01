Tôi muốn tổ chức phần Fuzzy Logic & XAI thành một package riêng, tách biệt khỏi code CNN hiện có để tránh lẫn trách nhiệm giữa CNN baseline và fuzzy reasoning.

Yêu cầu kiến trúc:

1. Không sửa sâu vào code CNN hiện có trong thư mục `rice/`, trừ khi cần thêm hook/export rất nhỏ.
2. Tạo package riêng tên `rice_fuzzy_xai/`.
3. Package này nhận input từ CNN inference dưới dạng JSON/dict gồm:

   * cnn_scores
   * top_class
   * top_confidence
4. Package tự tính thêm:

   * max_score
   * second_score
   * margin
   * grouped_disease_scores
   * mild_score
   * severe_score
5. Output của package gồm:

   * predicted_disease
   * visual_severity_level
   * diagnostic_confidence
   * uncertainty_level
   * environmental_risk_level
   * final_alert_level
   * explanation
   * recommendation

Cấu trúc thư mục đề xuất:

rice_fuzzy_xai/
**init**.py
config.py
membership.py
rules.py
engine.py
explanation.py
schemas.py

scripts/
run_fuzzy_inference.py

docs/
fuzzy_design.md

tests/
test_fuzzy_engine.py

Vai trò từng file:

* `schemas.py`: định nghĩa input/output schema bằng dataclass hoặc dict validation đơn giản.
* `membership.py`: định nghĩa hàm thành viên fuzzy cho confidence, margin, severity, uncertainty, và optional environment inputs.
* `rules.py`: chứa các luật mờ.
* `engine.py`: fuzzy inference engine chính, nhận CNN JSON và trả kết quả.
* `explanation.py`: sinh giải thích XAI dạng tiếng Việt/tiếng Anh.
* `config.py`: ngưỡng và mapping class disease/severity.
* `scripts/run_fuzzy_inference.py`: CLI để chạy thử từ JSON hoặc từ output inference CNN.
* `docs/fuzzy_design.md`: mô tả thiết kế phục vụ báo cáo môn học.
* `tests/test_fuzzy_engine.py`: unit test đơn giản cho các case chính.

Nguyên tắc thiết kế:

* Fuzzy Layer là phần chính trên branch `khanhtrang`, không phải future work.
* Temperature/humidity chỉ là optional manual inputs, không bắt buộc.
* Nếu không có temperature/humidity, hệ thống vẫn chạy bình thường bằng CNN scores.
* Environment inputs chỉ ảnh hưởng `environmental_risk_level` và `final_alert_level`, không trực tiếp đổi `visual_severity_level`.
* Không merge vào `main`.
* Trước khi tạo/sửa file, hãy hiển thị kế hoạch file và chờ tôi duyệt.

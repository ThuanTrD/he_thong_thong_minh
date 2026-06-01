Hãy chỉnh code FuzzyEngine hiện tại theo hướng tăng rõ vai trò “Fuzzy + Expert-Guided Reasoning”, nhưng không phá vỡ logic đang chạy.

Yêu cầu chính:

1. Bổ sung entropy/OOD signal
- Tính entropy từ inp.cnn_scores:
  entropy = -sum(p * log(p + 1e-9))
- Chuẩn hóa entropy theo số lớp:
  normalized_entropy = entropy / log(num_classes)
- Dùng entropy như một tín hiệu uncertainty bổ sung bên cạnh margin.
- Nếu normalized_entropy cao, hoặc margin thấp, xem là CNN không chắc chắn.

2. Khai thác grouped_disease_scores
Code hiện đã gom Mild/Severe theo bệnh chính nhưng chưa dùng đủ.
Hãy:
- Tìm best_group_disease và best_group_confidence.
- Nếu top_class confidence thấp nhưng best_group_confidence cao, dùng best_group_disease làm disease-level prediction.
- Ví dụ Mild Brownspot + Severe Brownspot cùng cao thì hệ thống hiểu là Brownspot, dù từng lớp riêng lẻ không quá cao.
- Vẫn giữ visual severity dựa trên Mild/Severe score.

3. Tách Expert-Guided Reasoning thành hàm riêng
Tạo hàm riêng, ví dụ:

def apply_expert_guided_reasoning(...):
    ...

Hàm này nhận:
- predicted_disease
- max_score
- uncertainty_level
- normalized_entropy
- margin
- snail_density
- final_alert_level
- visual_severity_level

và trả về:
- predicted_disease
- inference_mode
- fused_confidence
- final_alert_level
- visual_severity_level

Mục tiêu là để phần expert reasoning rõ ràng, dễ giải thích trong báo cáo.

4. Điều kiện chuyển mode
Giữ 3 mode:
- AI_CONFIDENT
- HYBRID_WARNING
- EXPERT_GUIDED_MODE

Gợi ý logic:

AI_CONFIDENT:
- max_score >= 0.80
- margin đủ lớn
- normalized_entropy thấp
- expert_signal yếu hoặc không có

HYBRID_WARNING:
- Có tín hiệu chuyên gia nhưng chưa đủ mạnh
- Hoặc CNN hơi bất định nhưng chưa đủ để override
- Không đổi hẳn bệnh, chỉ nâng mức cảnh báo nếu cần

EXPERT_GUIDED_MODE:
- expert_signal >= 0.5
- Và một trong các điều kiện:
  + max_score <= 0.65
  + margin thấp
  + normalized_entropy cao
  + uncertainty_level là High
- Khi đó predicted_disease = "Golden Apple Snail"
- fused_confidence = min(0.60 + expert_signal * 0.39, 0.99)
- final_alert_level = "Red Alert (Báo động đỏ)"
- visual_severity_level = "Severe (Nghiêm trọng)"

5. Bổ sung explanation
Cập nhật generate_explanation hoặc tham số truyền vào để explanation có thể nói rõ:
- CNN top confidence
- margin
- entropy chuẩn hóa
- grouped disease confidence
- lý do hệ thống giữ AI_CONFIDENT / chuyển HYBRID_WARNING / chuyển EXPERT_GUIDED_MODE

6. Không làm quá rộng
Không thêm database, không thêm UI lớn, không đổi schema nếu không cần.
Nếu cần đổi schema thì chỉ thêm các field có ích:
- normalized_entropy
- best_group_disease
- best_group_confidence

7. Yêu cầu chất lượng code
- Giữ backward compatibility tối đa.
- Không làm hỏng Streamlit app hiện tại.
- Viết comment rõ ràng bằng tiếng Việt/Anh.
- Sau khi sửa, chạy test nhanh với 3 case:
  a) CNN rất chắc bệnh Blast, không có snail_density → AI_CONFIDENT
  b) CNN phân vân Mild/Severe cùng một bệnh → dùng grouped confidence để ổn định disease-level prediction
  c) Ảnh ốc bươu vàng/snail_density cao + CNN bất định → EXPERT_GUIDED_MODE

Mục tiêu cuối cùng:
Code phải thể hiện rõ luận điểm báo cáo:
“CNN nhận diện mẫu hình ảnh, Fuzzy đánh giá uncertainty/severity/risk, Expert-Guided Reasoning can thiệp khi có tín hiệu thực địa hoặc OOD để tránh kết luận sai quá tự tin.”
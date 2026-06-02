Hãy điều chỉnh file explanation.py để làm rõ vai trò XAI, Fuzzy Rules Fired và Expert-Guided Reasoning, nhưng không phá vỡ logic hiện có.

Yêu cầu cụ thể:

1. Với EXPERT_GUIDED_MODE
- Bổ sung thêm thông tin CNN ban đầu:
  + top_class
  + top_confidence
  + best_group_disease
  + best_group_confidence
  + margin
  + normalized_entropy
- Mục tiêu: cho thấy hệ thống không bỏ qua CNN, mà đánh giá CNN trước rồi mới ưu tiên tín hiệu chuyên gia/thực địa.
- Viết explanation theo hướng:
  “CNN đưa ra dự đoán ban đầu..., tuy nhiên entropy cao / margin thấp / tín hiệu thực địa mạnh nên hệ thống chuyển sang Expert-Guided Mode.”

2. Với HYBRID_WARNING
- Làm rõ đây không phải phủ quyết CNN.
- Thêm câu:
  “Hệ thống chưa phủ quyết kết quả CNN, nhưng nâng mức cảnh báo do có dấu hiệu bất định hoặc tín hiệu thực địa bổ sung.”
- Giải thích rõ fused_confidence là kết quả kết hợp giữa CNN confidence và expert signal.

3. Với AI_CONFIDENT
- Thêm câu giải thích vì sao giữ AI_CONFIDENT:
  + top_confidence cao
  + margin đủ rõ
  + normalized_entropy thấp
  + không có expert signal mạnh
- Nếu best_group_confidence cao hơn top_confidence thì giải thích:
  “Grouped confidence giúp củng cố dự đoán ở mức bệnh chính.”

4. Phần Fuzzy Rules Fired
- Giữ nguyên logic tính đóng góp hiện tại.
- Nhưng bổ sung dòng mở đầu:
  “Các luật dưới đây là những luật có mức kích hoạt > 0, thể hiện quá trình suy diễn mờ dẫn tới kết quả cuối cùng.”
- Nếu không có rule nào kích hoạt thì in:
  “Không có luật mờ đáng kể nào được kích hoạt.”

5. Phần Recommendation
- Với các khuyến nghị dùng tên thuốc hoặc hoạt chất như Tricyclazole, Copper, Metaldehyde, Niclosamide:
  thêm câu an toàn:
  “Việc sử dụng thuốc/hoạt chất cần tuân thủ hướng dẫn của cán bộ bảo vệ thực vật địa phương và quy định an toàn nông nghiệp.”
- Không xóa các khuyến nghị hiện có, chỉ bổ sung disclaimer mềm để tránh bị bắt bẻ khi bảo vệ.

6. Cải thiện ngôn ngữ báo cáo
- Giữ tiếng Việt rõ ràng, học thuật vừa phải.
- Không dùng từ quá tuyệt đối như “chắc chắn”, “đúng tuyệt đối”.
- Ưu tiên các cụm:
  + “có khả năng”
  + “gợi ý”
  + “hệ thống ưu tiên”
  + “cần kiểm tra bổ sung”
  + “hỗ trợ quyết định”

7. Không thay đổi interface nếu không cần
- Giữ nguyên chữ ký hàm generate_explanation nếu có thể.
- Không làm hỏng các nơi đang gọi hàm này.
- Không đổi tên field output.

Mục tiêu cuối cùng:
explanation.py phải thể hiện rõ rằng hệ thống không chỉ trả nhãn bệnh, mà giải thích được:
CNN dự đoán gì,
mức bất định ra sao,
grouped confidence hỗ trợ thế nào,
luật fuzzy nào được kích hoạt,
vì sao chọn AI_CONFIDENT / HYBRID_WARNING / EXPERT_GUIDED_MODE,
và khuyến nghị nông nghiệp được đưa ra như một hỗ trợ quyết định chứ không phải kết luận tuyệt đối.
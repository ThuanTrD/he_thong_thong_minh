
VSI -> nhóm tự định nghĩa
FAI -> Final Alert Index
XAI = 3 tầng giải thích

Nếu thầy hỏi sâu - Bạn đã có câu trả lời rồi:

Về Fuzzy
min() là toán tử AND (T-norm).
VSI được giải mờ bằng trung bình trọng số Sugeno.

Về XAI
Hệ thống giải thích được chế độ suy luận, các luật kích hoạt và tỷ lệ đóng góp của từng luật.

Về OOD
Entropy và Margin được dùng để phát hiện mức độ bất định của CNN.

===========

Hệ thống thể hiện khả năng giải thích thông qua ba cơ chế chính.

Thứ nhất, hệ thống minh bạch hóa luồng suy luận bằng cách xác định chế độ ra quyết định: AI Confident, Hybrid Warning hoặc Expert-Guided Mode. Nhờ đó, người dùng biết hệ thống đang dựa chủ yếu vào kết quả CNN, kết hợp thêm tín hiệu bất định, hay cần ưu tiên tín hiệu chuyên gia.

Thứ hai, hệ thống truy xuất được các luật mờ đã được kích hoạt trong quá trình suy luận. Mỗi luật IF–THEN được hiển thị kèm mức kích hoạt và tỷ lệ đóng góp vào kết quả cuối cùng, giúp làm rõ vì sao hệ thống đưa ra chỉ số VSI, ERI hoặc FAI tương ứng.

Thứ ba, hệ thống chuyển các chỉ số kỹ thuật thành báo cáo giải thích bằng ngôn ngữ tự nhiên và khuyến nghị hành động, giúp người sử dụng dễ hiểu, dễ kiểm tra và dễ ra quyết định.


Dạ, XAI của nhóm không chỉ nằm ở việc giải thích ảnh đầu vào, mà nằm ở tầng ra quyết định của hệ thống. Cụ thể, hệ thống giải thích được chế độ suy luận, truy xuất các luật mờ đã kích hoạt, tính mức đóng góp của từng luật và sinh báo cáo bằng ngôn ngữ tự nhiên. Vì vậy, tầng CNN có thể là black-box, nhưng tầng suy luận và ra quyết định cuối cùng được thiết kế theo hướng white-box và có thể truy vết.

CNN có thể là black-box, nhưng tầng Fuzzy Inference và Decision Layer là white-box, có luật, có trọng số, có truy vết và có giải thích.


=========================================


Chúng ta hãy "mổ xẻ" từ khóa "Explainable" (Có thể giải thích được - XAI) nhé! Đây chính là vũ khí hạng nặng giúp dự án của nhóm thoát khỏi cái dớp "hộp đen" (black-box) muôn thuở của Deep Learning.

Trong các bài báo khoa học hiện đại, XAI không chỉ là việc vẽ cái biểu đồ, mà là việc hệ thống có khả năng "nói cho con người biết TẠI SAO nó lại ra quyết định như vậy". Nhóm đã hiện thực hóa điều này một cách cực kỳ xuất sắc thông qua file rice_fuzzy_xai/explanation.py.

Dưới đây là 3 điểm "ăn tiền" chứng minh tính Explainable của hệ thống mà bạn có thể dùng để trình bày:

1. Minh bạch hóa luồng suy luận (Reasoning Transparency)
Hệ thống không chỉ in ra kết quả cuối cùng mà còn giải thích rõ nó đang ở trạng thái suy luận nào (Inference Mode). Ví dụ, nếu nó rơi vào trạng thái EXPERT_GUIDED_MODE (Chế độ chuyên gia can thiệp), báo cáo sẽ ghi rõ lý do:

"CNN đưa ra dự đoán ban đầu là Đạo ôn (60%), tuy nhiên do Entropy cao (mức độ bất định lớn) kết hợp với tín hiệu thực địa rất mạnh (Ốc bươu vàng 5 con/m2), hệ thống quyết định ưu tiên trọng số tín hiệu chuyên gia để cảnh báo nguy cơ ốc bươu vàng, nhằm hỗ trợ quyết định quản trị rủi ro an toàn nhất."

2. Truy xuất ngược Luật Mờ (Rule Traceability) - Điểm sáng giá nhất 🌟
Trong mạng Nơ-ron (CNN), bạn không thể biết nơ-ron nào quyết định lá này bị bệnh Nặng. Nhưng với Fuzzy Logic, nhóm đã thiết kế để truy xuất ngược (Traceback) chính xác các luật nào đã được kích hoạt.

Trong code explanation.py, nhóm đã viết một hàm để tính phần trăm đóng góp của từng luật:

python
# Hàm phụ trợ tính đóng góp của từng luật vào quyết định cuối
    def calc_contrib(w, z, total_wz, rule_list):
        if total_wz > 0:
            return ((w * z) / total_wz) * 100
        else:
            total_w = sum(rw for rw, rz, rtxt in rule_list)
            return (w / total_w) * 100 if total_w > 0 else 0
Nhờ đó, Báo cáo XAI sẽ in ra màn hình cho nông dân (hoặc giám khảo) thấy rõ ràng các dòng như:

[Luật Mức độ bệnh] NẾU Độ tự tin Severe Thấp VÀ Mild Cao THÌ Mức độ bệnh Nhẹ (Kích hoạt: 0.8, Đóng góp: 80%)
[Luật Thời tiết] NẾU Nhiệt độ Ấm VÀ Độ ẩm Ẩm ướt THÌ Nguy cơ môi trường Cao (Kích hoạt: 0.9, Đóng góp: 90%)
=> Thầy giáo hỏi: "Làm sao biết máy không đoán mò?" => Bạn trả lời: "Dạ hệ thống của bọn em là White-box ở tầng quyết định cuối cùng. Thầy có thể nhìn vào Báo cáo XAI, nó list ra đúng 2 luật logic IF-THEN đã cấu thành nên quyết định đó, kèm theo phần trăm tỷ trọng đóng góp của từng luật thưa thầy." 😎

3. Giải thích bằng Ngôn ngữ Tự nhiên (Natural Language Generation)
Thay vì bắt người dùng (nông dân) phải đọc các chỉ số toán học khô khan (VSI = 85.5, FAI = 92), file XAI đã dịch toàn bộ các số liệu này thành ngôn ngữ đời thường kèm theo Khuyến nghị hành động (Actionable Recommendations):

Nếu là bệnh Bạc lá: "Khuyến cáo ngừng tưới tràn, sử dụng thuốc gốc Đồng."
Nếu có Ốc bươu vàng: "Ưu tiên bắt thủ công, giữ mực nước 2-3cm..."
Tổng kết lại: Cụm từ "Explainable" trong tên đề tài đã được nhóm chứng minh bằng hành động code thực tế chứ không chỉ là lý thuyết. Sự kết hợp giữa AI nhận diện (CNN) và một hệ thống biết giải thích bằng lời (Fuzzy XAI) chính là xu hướng cốt lõi của Trí tuệ nhân tạo Y tế và Nông nghiệp hiện đại!
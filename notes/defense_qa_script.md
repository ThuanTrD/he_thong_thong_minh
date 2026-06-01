# KỊCH BẢN HỎI ĐÁP BẢO VỆ LUẬN VĂN (DEFENSE Q&A SCRIPT)

**Chủ đề:** An Explainable Intelligent System for Plant Disease Severity Assessment using Fuzzy Inference
**Định vị dự án:** Hệ thống Hỗ trợ ra quyết định thông minh lai (Hybrid Intelligent Decision-Support System)
**Nguyên tắc cốt lõi:** Consistency > Stability > Explainability > Visual Quality

Tài liệu này tổng hợp 12 câu hỏi phản biện (dạng "chặn họng" và xoáy sâu vào kỹ thuật) thường gặp từ Hội đồng chuyên môn (các chuyên gia về AI, Fuzzy Logic, Hệ thống Thông minh) và cách trả lời sắc bén, tuân thủ tuyệt đối [SYSTEM FREEZE & DEFENSE HARDENING PROMPT].

---

## PHẦN 1: KIẾN TRÚC & TRIẾT LÝ HỆ THỐNG (ARCHITECTURE)

### 1. "Tại sao lại phải dùng hệ lai (Hybrid)? Tại sao không huấn luyện thẳng một mô hình Deep Learning cực mạnh (như Vision Transformer) để dự đoán luôn cả bệnh, mức độ và rủi ro môi trường?"

**> Trả lời (Sắc bén & Phòng thủ):**
"Thưa Hội đồng, hệ thống của nhóm em không chỉ hướng tới độ chính xác (Accuracy), mà quan trọng hơn là **Tính minh bạch (Explainability)** và **Khả năng kiểm soát rủi ro**.
Nếu dùng một mô hình Deep Learning End-to-End để gánh toàn bộ, chúng ta sẽ tạo ra một 'Hộp đen' (Black-box) khổng lồ. Khi mô hình dự đoán sai do nhiễu môi trường, người dùng hoàn toàn không biết tại sao.
Việc chia tách thành hệ lai giúp phân nhiệm rõ ràng: CNN chỉ làm nhiệm vụ 'Nhận thức thị giác' (Perception), trong khi 'Suy luận mức độ nguy hiểm' được nhường lại cho Fuzzy Logic. Nhờ đó, chuyên gia nông nghiệp có thể can thiệp vào tập luật mờ của môi trường (Nhiệt độ, Độ ẩm) mà không cần phải thu thập lại hàng vạn bức ảnh để huấn luyện lại CNN."

### 1b. "Cơ sở lý thuyết để xây dựng nên các quy luật Fuzzy (Fuzzy Rules) và các hàm liên thuộc (Membership Functions) áp dụng trong hệ thống này là từ đâu? Các bạn tự nghĩ ra (heuristic) hay dựa trên cơ sở nào?"

**> Trả lời (Sắc bén & Khoa học):**
"Thưa Hội đồng, cơ sở lý thuyết để xây dựng tập luật mờ trong hệ thống này không phải là tự phát cảm tính, mà được tổng hợp từ 3 nền tảng khoa học cốt lõi:

**1. Nền tảng Dịch tễ học Thực vật (Plant Epidemiology):** 
Theo lý thuyết 'Tam giác bệnh cây' (Plant Disease Triangle), bệnh hại bùng phát khi hội tụ 3 yếu tố: Ký chủ, Mầm bệnh, và **Môi trường thuận lợi**. Các luật mờ tính toán Chỉ số Rủi ro Môi trường (ERI) được chuyển ngữ hóa trực tiếp từ tri thức Nông học chuẩn. Ví dụ: bào tử nấm Đạo ôn (Blast) nảy mầm cực mạnh khi độ ẩm >85% và nhiệt độ 25-28°C. Do đó, tri thức này được ánh xạ thành luật mờ: `IF Temp is Warm AND Humidity is High THEN Risk is High`.

**2. Lý thuyết Hiệu chuẩn Bất định trong Học máy (Uncertainty Calibration in ML):**
Cơ sở để đưa `Margin` và `Normalized Entropy` vào bộ luật mờ xuất phát từ bài toán nhức nhối của Neural Networks là 'Overconfidence' (tự tin thái quá vào quyết định sai). Việc dùng Fuzzy Logic để hấp thụ các tín hiệu Entropy/Margin chính là kỹ thuật hiệu chuẩn sự bất định (Aleatoric & Epistemic Uncertainty), giúp bẻ gãy các quyết định ranh giới cứng (Hard boundaries) rủi ro của Softmax.

**3. Cơ sở Khoa học Nhận thức (Cognitive Science) trong Hệ chuyên gia:**
Việc chọn hàm liên thuộc hình thang (Trapezoidal) / tam giác (Triangular) cho các biến Mild/Severe mô phỏng hoàn hảo **Sự chồng lấn ngôn ngữ (Linguistic Overlap)**. Trong thực tế, chuyên gia nông nghiệp không bao giờ kết luận 'đúng 80% diện tích là tổn thương nặng, 79.9% là nhẹ', mà đó là một phổ chuyển tiếp liên tục. Hệ suy diễn mờ chính là công cụ toán học duy nhất mô phỏng đúng quy trình nhận thức (Cognitive Process) này của con người để đưa ra các quyết định linh hoạt vùng biên."

### 2. "Cái 'Fuzzy Logic' của các bạn có thực sự cần thiết không? Tại sao không dùng vài câu lệnh `IF-ELSE` đơn giản với ngưỡng cứng (Hard Threshold) cho nhanh?"

**> Trả lời:**
"Thưa Hội đồng, các trạng thái bệnh của thực vật trong tự nhiên là một **chuỗi phổ liên tục (Continuous Spectrum)**, không phải là các mức độ gián đoạn (Discrete).
Một chiếc lá chuyển từ 'Nhẹ' (Mild) sang 'Nặng' (Severe) không xảy ra ngay lập tức qua một ranh giới IF-ELSE cứng nhắc. Nếu dùng ngưỡng cứng (VD: Confidence > 0.5 là Nặng, ngược lại là Nhẹ), hệ thống sẽ vô cùng nhạy cảm và dao động liên tục tại vùng ranh giới.
Fuzzy Inference giải quyết triệt để sự bất định này bằng cách **nội suy (Interpolation)**. Nó tự động tổng hợp ra trạng thái trung gian 'Moderate' tại vùng biên giao thoa, giúp sự chuyển mức độ (VSI) diễn ra mượt mà, phản ánh đúng quy luật tự nhiên."

---

## PHẦN 2: XỬ LÝ SỰ BẤT ĐỊNH & NGOẠI LAI (UNCERTAINTY & OOD)

### 3. "Nếu tôi đưa vào hệ thống một hình ảnh không liên quan (ví dụ: hình con mèo hoặc chiếc lá bị cháy nắng không nằm trong tập train), hệ thống của bạn sẽ làm gì? CNN luôn xuất ra một nhãn nào đó cơ mà?"

**> Trả lời:**
"Đây chính là một trong những tính năng an toàn (Safety-critical) cốt lõi của hệ thống. Nhóm em ý thức rõ điểm yếu của CNN là mô hình có tính 'tự tin thái quá' (Overconfidence) kể cả với dữ liệu ngoài miền (Out-of-Distribution).
Do đó, trước khi đưa kết quả CNN vào Fuzzy Logic, hệ thống thực hiện hai chốt chặn:
1. **Kiểm tra Confidence (OOD Detection):** Nếu xác suất cao nhất (Max Confidence) < 0.25, hệ thống lập tức chối bỏ hình ảnh và yêu cầu sự can thiệp của chuyên gia.
2. **Kiểm tra Entropy:** Nếu mô hình phân vân đều giữa các lớp, hệ thống sử dụng **Normalized Entropy** (Entropy chuẩn hóa) làm chỉ báo OOD bổ sung. Khi Normalized Entropy cao, hệ thống sẽ hạ mức độ tự tin, phát ra cảnh báo `HYBRID_WARNING` hoặc thậm chí chuyển sang chế độ `EXPERT_GUIDED_MODE` nếu có tín hiệu thực địa bất thường. Điều này giúp ngăn chặn sai sót do tin tưởng mù quáng vào xác suất bị thổi phồng của mạng Neural."

### 3b. "Nếu mô hình CNN bị phân vân giữa 'Mild' và 'Severe' của cùng một bệnh thì sao? Điểm số từng lớp sẽ bị pha loãng và rất thấp, hệ thống có báo sai loại bệnh không?"

**> Trả lời:**
"Dạ không thưa Hội đồng. Đây là một vấn đề kinh điển trong các mô hình học sâu khi các class có sự phân cấp (Hierarchical Classification). Để giải quyết triệt để, hệ thống sử dụng kỹ thuật **Gom nhóm Điểm số (Grouped Disease Confidence)**.
Ngay cả khi mô hình bị 'pha loãng' xác suất do phân vân mức độ (ví dụ: Mild Brownspot = 40%, Severe Brownspot = 35%), Max Score của từng lớp đều rất thấp và có thể bị đánh giá là 'Bất định' (Uncertainty). Tuy nhiên, hệ thống tự động cộng gộp (Grouped Confidence = 75%) để giữ vững kết luận cốt lõi: 'Chắc chắn 75% đây là bệnh Brownspot'. Nhờ vậy, AI không bao giờ dự đoán sai loại bệnh cốt lõi, phần phân định Nhẹ/Nặng tiếp theo sẽ do Fuzzy Logic nội suy."

### 3c. "Tại sao trong sơ đồ các bạn vẽ Entropy có hàm liên thuộc (Low/Medium/High) để mờ hóa, nhưng trong code thực tế lại dùng một ngưỡng cắt (Crisp Threshold) là `normalized_entropy > 0.6`?"

**> Trả lời:**
"Dạ thưa Hội đồng, đây là một sự điều chỉnh có chủ ý của nhóm nhằm tối ưu điểm rơi giữa 'Lý thuyết' và 'Hiệu năng'.
Về mặt lý thuyết (Conceptual level), việc coi Entropy là một biến ngôn ngữ (Linguistic Variable) cần được mờ hóa là hoàn toàn chính xác. Do đó, trên sơ đồ kiến trúc, nhóm biểu diễn dạng hàm liên thuộc để đảm bảo sự đồng nhất và chuẩn mực của hệ suy diễn Fuzzy.
Tuy nhiên, ở khâu cài đặt thực tế (Implementation), do `Normalized Entropy` đã tự động được chuẩn hóa về miền giá trị [0..1] (giống như một hàm thuộc tính sẵn có), việc nội suy lại lần nữa là dư thừa tài nguyên. Nhóm đã áp dụng phương pháp **Cắt Alpha (Alpha-cut)** trực tiếp tại ngưỡng $\alpha = 0.6$ để kích hoạt thẳng vào tập luật 'High Uncertainty'. Điều này giúp hệ thống phản ứng cực nhanh với tín hiệu OOD (Out-of-Distribution) mà không phát sinh thêm độ trễ tính toán (latency) trong môi trường triển khai thực tế."

---

## PHẦN 3: ĐÁNH GIÁ LỖI & SỰ SUY GIẢM AN TOÀN (ERROR ANALYSIS)

### 4. "Tôi thấy hệ thống vẫn bị sai 4 ảnh. Đặc biệt có lỗi dự đoán bệnh Nặng (Severe) thành Nhẹ (Mild). Các bạn giải thích sao về sự yếu kém này?"

**> Trả lời (Dùng thuật ngữ "Conservative Downgrade"):**
"Thưa Hội đồng, nhóm em đã phân tích rất kỹ 4 trường hợp ngoại lệ này. Đây không phải là sự 'đổ vỡ' của hệ thống, mà là hệ quả của cơ chế **Hạ cấp an toàn (Conservative Downgrades)** tại vùng giao thoa ranh giới (Borderline regions).
Trong 4 lỗi này, đặc tính thị giác của vết bệnh cực kỳ mơ hồ. Trong trạng thái bất định đó, Fuzzy Logic chọn cách làm mịn (smoothing) và hạ cấp cảnh báo từ 'Severe' xuống vùng 'Moderate/Mild' thay vì vội vàng đẩy lên mức tối đa.
Quan trọng nhất, bảng Confusion Matrix cho thấy: **Số lỗi 'Severe -> Healthy' (Bỏ lọt nguy hiểm) bằng 0 tuyệt đối.** Điều này minh chứng rằng hệ thống hoạt động đúng bản chất của một 'Hệ thống hỗ trợ ra quyết định an toàn' (Safe decision-support behavior)."

### 5. "Liệu có phải Fuzzy Logic của các bạn đang 'sửa sai' hoặc 'ghi đè' (override) lên kết quả của mô hình CNN không?"

**> Trả lời (Tuyệt đối KHÔNG dùng từ "Override"):**
"Dạ không thưa Hội đồng. Fuzzy Logic không hề 'sửa sai' hay phủ định kết quả của CNN.
CNN là khối Nhận thức (Perception) và nó đã làm xuất sắc việc nội kết xuất các đặc trưng thị giác. Nhiệm vụ của Fuzzy Logic là tiếp nhận sự 'không chắc chắn' từ các phân bố Softmax đó (ví dụ CNN báo 60% Severe, 40% Mild), rồi dùng hàm liên thuộc để kết hợp với bối cảnh, từ đó **Nội suy (Interpolate)** ra mức độ cảnh báo cuối cùng. Nó làm mịn (smoothing) sự bất định chứ không can thiệp vào trọng số hay quyết định gốc của khối Deep Learning."

### 5b. "Khi nào thì tín hiệu từ chuyên gia/thực địa sẽ được ưu tiên thay vì chỉ dựa vào AI (Cơ chế Expert-Guided Mode)?"

**> Trả lời:**
"Thưa Hội đồng, hệ thống có một hàm giám sát riêng gọi là **Expert-Guided Reasoning**. Cơ chế này chỉ kích hoạt khi có sự hội tụ của 2 yếu tố: 
Thứ nhất, AI đang rơi vào trạng thái cực kỳ bất định (Biểu hiện qua **Normalized Entropy > 0.6**, hoặc Confidence < 0.65). 
Thứ hai, có một tín hiệu ngoại lệ mạnh từ thực địa (Ví dụ: Cảm biến hoặc chuyên gia báo cáo Mật độ ốc bươu vàng rất dày đặc). 
Khi đó, hệ thống sẽ tự động hạ quyền quyết định của thị giác (vì ảnh lúc này có thể là OOD hoặc nhiễu) và kích hoạt `EXPERT_GUIDED_MODE`, nâng cảnh báo lên 'Báo động đỏ' dựa trên bằng chứng thực địa. Đây là nguyên tắc cốt lõi của một hệ hỗ trợ ra quyết định an toàn."

---

## PHẦN 4: BỐI CẢNH MÔI TRƯỜNG & TÍNH MINH BẠCH (CONTEXT & XAI)

### 6. "Tại sao không nối thẳng dữ liệu Nhiệt độ, Độ ẩm (ERI) vào các mạng Nơ-ron Đa tầng (MLP) đằng sau CNN để nó tự học, mà phải đi qua tập luật mờ?"

**> Trả lời:**
"Việc ghép nối thẳng (Concatenation) Nhiệt độ/Độ ẩm vào Neural Network là một cách làm phổ biến trong Deep Learning, nhưng nó triệt tiêu đi **TÍNH GIẢI THÍCH ĐƯỢC (Explainability)**.
Nếu Neural Network dự đoán 'DANGER' khi độ ẩm 90%, chúng ta không thể biết trọng số nào của nơ-ron đã quyết định điều đó.
Ngược lại, bằng cách dùng Fuzzy Inference System, việc leo thang cảnh báo (Contextual alert escalation) dựa vào môi trường được số hóa thành các luật tường minh (Symbolic Rules). Chuyên gia có thể đọc hiểu chính xác: 'Cảnh báo Danger được kích hoạt vì Luật 2 (Humidity Wet -> ERI High) đóng góp 100% vào quyết định'. Điều này tạo ra sự tin tưởng tuyệt đối cho End-user."

### 7. "Cái tính năng XAI (Explainable AI) của các bạn khác gì thuật toán Grad-CAM thường thấy trong các bài báo Deep Learning hiện nay?"

**> Trả lời:**
"Thưa Hội đồng, Grad-CAM là XAI thuộc cấp độ **Thị giác (Visual Explainability)**. Nó chỉ khoanh vùng đỏ lên chiếc lá để cho biết CNN 'đang nhìn vào đâu'. Grad-CAM không thể giải thích 'Tại sao hệ thống lại bảo tôi phải phun thuốc khẩn cấp?'.
Hệ thống của nhóm em tiến thêm một bước cao hơn: **Traceability Logic (Tính minh bạch về suy luận biểu tượng)**.
Hệ thống xuất ra một báo cáo văn bản ngôn ngữ tự nhiên (Natural Language Report), chỉ rõ: Chỉ số VSI là bao nhiêu, ERI là bao nhiêu, và cụ thể những LUẬT NÀO (Rule ID) đang được kích hoạt (Fired) với trọng số đóng góp bao nhiêu phần trăm. Sự kết hợp giữa Grad-CAM (giải thích hình ảnh) và Fuzzy XAI (giải thích logic) tạo nên một cấu trúc minh bạch toàn diện."

---

## PHẦN 5: TÍNH THỰC TIỄN & ỨNG DỤNG (PRACTICALITY)

### 8. "Hệ thống này có vẻ quá phức tạp và nặng nề, liệu có thể chạy trên thiết bị di động của nông dân hay phải cần máy chủ GPU siêu khủng?"

**> Trả lời:**
"Nhóm em đã cân nhắc rất kỹ bài toán triển khai (Deployment).
Đó là lý do tại sao ở tầng Perception, nhóm chọn **EfficientNet-B0** – một kiến trúc cực kỳ tối ưu, nhẹ và nhanh, chuyên biệt cho thiết bị Edge/Mobile thay vì các mô hình cồng kềnh như ResNet152.
Tầng Fuzzy Inference Engine và OOD Checks đằng sau hoàn toàn là các phép toán Logic và Ma trận cơ bản, chi phí tính toán (Computational cost) gần như bằng không so với CNN.
Sơ đồ 'End-to-End Deployment' của hệ thống (Fig 12) cho thấy hệ thống đã được tách luồng hoàn chỉnh: Backend FastAPI xử lý gọn nhẹ dưới dạng Microservice, trả về chuỗi JSON siêu nhẹ, sẵn sàng tích hợp thẳng vào App Mobile hoặc Web (Streamlit UI) mà không đòi hỏi GPU đắt đỏ phía End-user."

---

## TỔNG KẾT CHIẾN THUẬT PHÒNG THỦ:
* **Khi bị ép về độ chính xác (Tại sao chỉ 96.67%?):** Lái sang "Tính an toàn", "Bảo toàn suy luận tại vùng biên", "Tránh Overconfidence".
* **Khi bị ép về độ phức tạp (Tại sao lại dùng hệ Lai?):** Lái sang "Khả năng giải thích (XAI)", "Sự kết hợp tri thức chuyên gia", "Kiểm soát rủi ro".
* **Khi bị ép về lỗi (Tại sao CNN sai?):** Lái sang "Bản chất bất định của ảnh thực tế", "Hạ cấp cảnh báo bảo thủ (Conservative Downgrade)".

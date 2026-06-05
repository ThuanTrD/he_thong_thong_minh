# TÀI LIỆU KỊCH BẢN BẢO VỆ ĐỒ ÁN (CODE DEFENSE CHEAT SHEET)
*(Tài liệu tham khảo chuyên sâu phục vụ công tác bảo vệ hội đồng)*

Dưới đây là hệ thống hóa các luận điểm học thuật nhằm hỗ trợ quá trình bảo vệ đồ án, đã được **đối chiếu chính xác 100% với mã nguồn hiện tại**.

---

### 1. Cơ chế phát hiện dữ liệu ngoài phân phối (Out-of-Distribution - OOD)

**Câu hỏi:** Hệ thống kiểm soát rủi ro từ dữ liệu hình ảnh ngoài phân phối (OOD) bằng phương pháp nào?

**Trả lời:**
Dạ thưa hội đồng, logic này được triển khai tại phân hệ API Backend (`rice_api_backend/ood_detector.py` và `decision_engine.py`).
Hệ thống sử dụng kỹ thuật **Semantic Group Confidence Aggregation** kết hợp với **Normalized Shannon Entropy** để giải quyết triệt để hiện tượng phân mảnh xác suất (Probability Splitting) thường gặp trong phân loại đa nhãn có phân cấp mức độ (Nặng/Nhẹ).

1. **Hiện tượng Probability Splitting (Softmax Fragmentation):** 
   - Khi nhận diện một ảnh "Đạo ôn", hàm Softmax thường chia sẻ xác suất cho cả "Đạo ôn nhẹ" và "Đạo ôn nặng". Hậu quả là xác suất Top-1 (MSP - Maximum Softmax Probability) bị sụt giảm mạnh, dẫn đến việc nhận diện sai thành OOD nếu chỉ dùng ngưỡng Top-1 cứng nhắc.
2. **Semantic Group Confidence Aggregation:**
   - Hệ thống của em không đánh giá OOD dựa trên Top-1 class. Thay vào đó, em thiết kế `DISEASE_GROUPS` để gom tổng xác suất (Sum of Probabilities) của các lớp có chung ngữ nghĩa (ví dụ: `Blast = Mild Blast + Severe Blast`).
   - Nếu `Group_Confidence >= 0.60`, hệ thống coi là `KNOWN`. Nếu nằm trong khoảng `[0.40, 0.60)`, hệ thống chuyển sang trạng thái `UNCERTAIN` (Không chắc chắn).
3. **Normalized Shannon Entropy (Đánh giá độ nhiễu loạn):**
   - Song song với Group Confidence, hệ thống tính toán Entropy theo cơ số 2: $H_{norm} = \frac{-\sum p_i \log_2 p_i}{\log_2(N)}$, chuẩn hóa về thang [0, 1]. Nếu $H_{norm} > 0.65$, hệ thống chặn lại vì phân bố quá phân tán, đóng vai trò như lớp OOD Safeguard cuối cùng.

---

### 2. Kiến trúc Hệ chuyên gia Mờ (Fuzzy Expert System) và Thuật toán Giải mờ (Defuzzification)

**Câu hỏi:** Hệ chuyên gia Mờ được thiết kế theo kiến trúc nào? Cơ chế giải mờ hoạt động ra sao?

**Trả lời:**
Dạ thưa hội đồng, hệ thống triển khai kiến trúc **Zero-order Sugeno (Takagi-Sugeno-Kang)**, được lập trình tường minh (explicit coding) tại thư mục `rice_fuzzy_xai/`.
- **Tập luật (`rules.py`):** Khai báo các hàm kích hoạt (fire strength - $w_i$) và gán các hằng số đầu ra (crisp output - $z_i$) tương ứng (ví dụ: `Severe = 90`, `Mild = 30`).
- **Cơ chế Giải mờ (`engine.py`):**
  Khác với kiến trúc Mamdani thường yêu cầu bước centroid defuzzification trên miền đầu ra liên tục, Sugeno định nghĩa đầu ra là một hằng số. Do đó, thuật toán tính toán bằng phương pháp **Trung bình có trọng số (Weighted Average)**.
  
  **Công thức (áp dụng chính xác trong code):**
  $$ \text{Output} = \frac{\sum_{i=1}^n w_i \cdot z_i}{\sum_{i=1}^n w_i} $$
  *(Lưu ý: Mã nguồn quy định nếu tổng $\sum w_i = 0$, Output sẽ trả về giá trị mặc định an toàn là 50.0).*
  
  *Ví dụ:* Luật 1 ($w_1 = 0.8, z_1 = 90$) và Luật 2 ($w_2 = 0.4, z_2 = 50$). 
  Giải mờ = $(0.8 \times 90 + 0.4 \times 50) / 1.2 = 76.67$.
  
  **Đánh giá:** Weighted Average có độ phức tạp tính toán thấp hơn nhiều so với Mamdani, rất phù hợp cho các DSS (Decision Support System) yêu cầu realtime hoặc triển khai trên thiết bị edge.

---

### 3. Cơ chế Ghi đè Tri thức Chuyên gia (Expert Override)

**Câu hỏi:** Tính năng xử lý ngoại lệ (ví dụ: Ốc bươu vàng) hoạt động theo cơ chế nào? Liệu việc này có phá vỡ luồng suy diễn của AI không?

**Trả lời:**
Dạ thưa hội đồng, đây là cơ chế tạo ra một nhánh suy luận ngoại lệ:
*   **Vấn đề:** Mạng CNN là "hộp đen" (black-box), đôi khi rất tự tin một cách vô lý (Overconfident) vào dữ liệu nhiễu, nhưng lại mù lòa trước các tác nhân thực địa (vd: ốc bươu vàng dưới nước).
*   **Giải pháp - Expert-Assisted Confidence Fusion:** 
    * Thay vì dùng cơ chế "Ghi đè tuyệt đối" (Hard Override) thô sơ, hệ thống áp dụng cơ chế "Kết hợp độ tự tin" (Confidence Fusion).
    * Tín hiệu chuyên gia (Expert Signal) được lấy từ quan sát thực địa (vd: kéo thanh trượt mật độ ốc).
    * Hệ thống đánh giá đồng thời: Độ tự tin của CNN, Độ bất định (Uncertainty/Entropy), và Cường độ tín hiệu chuyên gia.
    * Phân loại thành 3 chế độ suy luận:
      1. **AI_CONFIDENT:** Khi CNN cực kỳ chắc chắn và tín hiệu ngoại lệ yếu. (Giữ nguyên CNN).
      2. **HYBRID_WARNING:** Khi CNN mức trung bình, tín hiệu thực địa xuất hiện. (Phát cảnh báo kép).
      3. **EXPERT_GUIDED_MODE:** Khi CNN bất định cao (OOD) và tín hiệu thực địa mạnh. (Ưu tiên kết luận theo chuyên gia để đảm bảo an toàn).
*   **Chốt hạ với thầy cô:** *"Dạ thưa thầy, thiết kế này tuân thủ nguyên tắc Trustworthy AI và Human-in-the-loop. Thay vì để AI quyết định độc đoán, hệ thống cho phép con người đóng vai trò hoa tiêu trong những tình huống dữ liệu đầu vào mập mờ, giúp giảm thiểu rủi ro nông nghiệp đến mức thấp nhất."*

---

### 4. Khả năng Giải thích của Mô hình (Explainable AI - XAI)

**Câu hỏi:** Báo cáo XAI của hệ thống được xây dựng dựa trên nguyên lý nào?

**Trả lời:**
Dạ, báo cáo XAI thuộc nhóm **Model-Specific Explanation**, dựa trên tính **truy vết (traceability)** và độ kích hoạt (rule activation) của Hệ mờ.
- Tại `explanation.py`, hệ thống duyệt mảng `rules_fired` (các luật tham gia vào giải mờ).
- Báo cáo sinh ra liệt kê minh bạch từng quy tắc mờ (ví dụ: "NẾU Top Confidence Vừa VÀ Margin Lớn...") đi kèm với **Trọng số kích hoạt (Fire Strength - $w_i$)** tương ứng. 
- *(Về mặt lý thuyết, hệ thống có thể mở rộng để tính chính xác Phần trăm đóng góp (Contribution %) của mỗi luật bằng công thức $w_i \cdot z_i / \sum(w \cdot z)$, tuy nhiên mã nguồn hiện tại ưu tiên hiển thị raw activation weight để đảm bảo tính nguyên bản của quá trình mờ hóa).*

---

### 5. CÁC CÂU HỎI PHẢN BIỆN CHUYÊN SÂU (Q&A BACKUP)

**Q1. Vì sao Shannon Entropy phản ánh được độ bất định (uncertainty)?**
$\rightarrow$ Entropy là thước đo sự hỗn loạn. Nếu mô hình chắc chắn, xác suất tập trung vào một lớp, Entropy thấp. Nếu mô hình bối rối (overconfident chệch), xác suất rải đều, Entropy đạt giá trị cao. 

**Q2. Vì sao Softmax dễ overconfident với dữ liệu OOD?**
$\rightarrow$ Hàm $e^x$ dồn ép không gian xác suất khiến tổng luôn bằng 1. Mô hình buộc phải chia xác suất cho các lớp tĩnh (Closed-set) dù ảnh đầu vào là rác, dẫn đến hiện tượng overconfidence.

**Q3. Điều kiện chặn OOD của nhóm là AND hay OR?**
$\rightarrow$ Dạ là **OR**. Nếu ảnh vi phạm ngưỡng Entropy (quá hỗn loạn) HOẶC vi phạm ngưỡng MSP (quá thiếu tự tin), hệ thống sẽ gán trạng thái `OOD` hoặc `UNKNOWN` tương ứng và kích hoạt OOD Safeguard.

**Q4. Điều kiện 0 chia 0 trong giải mờ được xử lý thế nào?**
$\rightarrow$ Dạ mã nguồn `rules.py` có hàm kiểm tra an toàn: `if sum_w > 0 else 50.0`. Nếu không có luật nào được kích hoạt, hệ thống trả về mức 50.0 (Nguy cơ trung bình/an toàn) để tránh exception sập hệ thống.

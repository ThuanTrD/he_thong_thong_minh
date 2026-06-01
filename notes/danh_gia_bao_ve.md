# Đánh giá Kiến trúc Hệ thống — Đối chiếu với Tên Đề tài Đăng ký

> **Tên đề tài:** *An Explainable Intelligent System for Plant Disease Severity Assessment using Fuzzy Inference*
> **Nhóm:** 14 — Môn Hệ thống Thông minh (Cao học)

---

## I. Mức độ khớp với Tên Đề tài (Alignment with Research Topic)

Kiến trúc hiện tại được thiết kế bám sát các từ khóa học thuật của đề tài:

| Từ khóa | Hiện thực trong hệ thống (Implementation) |
|---------|---------------------------------------------|
| **Explainable** | Tầng **Explainability Layer** hỗ trợ *symbolic reasoning traceability* bằng cách truy xuất các luật mờ được kích hoạt và gán trọng số đóng góp chính xác theo công thức tỷ trọng Sugeno: $(w_i \times z_i) / \sum(w_i \times z_i)$, từ đó sinh ra *interpretable recommendations*. |
| **Intelligent System** | Áp dụng mô hình *Neuro-symbolic reasoning pipeline* (kết hợp deep learning và logic mờ) hướng tới *decision-support oriented*. |
| **Plant Disease** | The primary evaluation focuses on four rice leaf diseases and healthy leaves. The system additionally supports abnormal/OOD agricultural threat scenarios such as Golden Apple Snail as part of uncertainty-aware expert-guided reasoning. |
| **Severity Assessment** | Sử dụng VSI (Visual Severity Index) với 4 mức (Healthy, Mild, Moderate, Severe). *The fuzzy inference layer does not replace the CNN classifier. Instead, it refines disease severity assessment by integrating CNN confidence, uncertainty information, and contextual reasoning into a more interpretable decision-support process.* |
| **Fuzzy Inference** | Kiến trúc Sugeno Zero-order FIS, xử lý bất định sinh học và thông tin môi trường. Mạng tham số mờ được tham số hóa (externalized) qua cấu hình JSON (`fuzzy_config.json`) giúp hệ thống mang tính *data-driven* và dễ dàng *fine-tune*. |

---

## II. Đóng góp Nghiên cứu (Research Contributions / Strengths)

Hệ thống được thiết kế theo mô hình **Explainable Hybrid Intelligent System** với các thành phần cốt lõi:

*   **CNN (Visual feature extraction and disease recognition):** Mạng EfficientNet-B0 đóng vai trò trích xuất đặc trưng hình ảnh. Phương pháp Label Smoothing (0.1) được áp dụng để tránh phân phối phổ điểm cứng nhắc.
*   **Fuzzy Inference (Severity reasoning and contextual decision support):** Kiến trúc mờ Sugeno hoạt động trên miền đầu ra tĩnh, sử dụng *Weighted Average* defuzzification, phù hợp cho các bài toán hỗ trợ quyết định có tính diễn dịch cao. Đặc biệt, hệ thống cung cấp tính năng **Parameter Externalization** qua `fuzzy_config.json`, hỗ trợ chuyên gia nông nghiệp điều chỉnh hệ mờ mà không cần can thiệp mã nguồn.
*   **OOD / Entropy (Uncertainty handling and abnormal-case detection):** Việc sử dụng *Semantic Group Confidence Aggregation* và *Normalized Shannon Entropy* giúp kiểm soát hiện tượng phân mảnh xác suất (*Probability Splitting*). The system incorporates uncertainty-aware detection mechanisms to reduce silent failure risk and support safer decision-making.
*   **Expert Fusion (Human-in-the-loop adjustment for agricultural field conditions):** Việc phân chia thành ba chế độ suy luận (`AI_CONFIDENT`, `HYBRID_WARNING`, `EXPERT_GUIDED_MODE`) *improves robustness* trước các tình huống thực địa phức tạp.
*   **Explainable AI tinh chỉnh:** Công thức XAI tính toán chính xác phần trăm đóng góp của từng tập luật dựa trên trọng số giải mờ $(w_i \times z_i) / \sum(w_i \times z_i)$, mang lại tính truy vết logic biểu tượng sắc nét thay vì xấp xỉ tỷ lệ kích hoạt đơn thuần.

---

## III. Các Hạn chế Hiện tại (Limitations / Weaknesses)

Bên cạnh những điểm mạnh, hệ thống mang những giới hạn nhất định của AI và kiến trúc lai:

*   **Sự phụ thuộc vào bộ quy tắc mờ (Rule Base Dependency):** Dù đã tách cấu hình JSON, quá trình *Severity Assessment* vẫn phụ thuộc vào việc định nghĩa *Membership functions* và tập luật chuyên gia ban đầu. Trong các hướng nghiên cứu xa hơn, việc tự động tinh chỉnh (Tuning) hàm thuộc bằng học máy (ví dụ: mô hình ANFIS) có thể được cân nhắc.
*   **Phạm vi đánh giá định lượng:** The current dataset provides labeled severity levels for Healthy, Mild, and Severe. The Moderate level is retained as an intermediate fuzzy reasoning and decision-support state, but quantitative evaluation is conducted using the available labeled classes.
*   **Giới hạn về Khả năng Tổng quát hóa (Generalizability):** Hệ thống hiện đang thiết kế dựa trên tập nhãn bệnh cụ thể của cây lúa. Để áp dụng cho các giống cây trồng khác, cần cập nhật mô hình thị giác máy tính cũng như *Knowledge Base* của Fuzzy System.

---

## IV. Đánh giá Định lượng (Severity Evaluation Results)

Quá trình đánh giá được thực hiện trên tập dữ liệu kiểm thử (180 mẫu) nhằm đo lường khả năng kết hợp giữa nhận dạng của CNN và cơ chế suy diễn mờ:

*   **Fuzzy Severity Accuracy:** 98.33% (177/180 mẫu)
*   **Macro F1-score (Severity):** 98.75%
*   **Disease-level consistency:** The system achieved 100% disease-level consistency on the current validation dataset.

**Phân tích lỗi (Error Analysis):**
*   Sau khi tích hợp luật tăng cường (Boost Rule), số lượng mẫu Severe bị hạ cấp không mong muốn giảm thiểu đáng kể (chỉ còn 2 ca).
*   Most errors are conservative downgrades from Severe to Mild.
*   No Severe samples were misclassified as Healthy.
*   *This suggests the system behaves conservatively in borderline cases rather than aggressively overestimating severity.* Phản xạ này tuân thủ đúng nguyên lý xây dựng một hệ thống *decision-support oriented* nhằm giảm thiểu báo động giả thái quá.

---

## V. Câu hỏi Hội đồng dự đoán & Luận điểm Bảo vệ

| Câu hỏi | Luận điểm Bảo vệ (Academic Defense) |
|---------|-------------------------------------|
| **"Tại sao dùng Sugeno thay vì Mamdani?"** | Sugeno phù hợp cho DSS: không đòi hỏi *centroid defuzzification* trên miền liên tục, độ trễ thấp $O(n)$, và đầu ra *crisp* rất thuận lợi để giải thích nguyên nhân. |
| **"XAI của hệ thống khác gì LIME/SHAP?"** | LIME/SHAP tiếp cận theo hướng *post-hoc approximation* để giải thích mô hình hộp đen. Tầng Explainability Layer của hệ thống là *model-intrinsic / symbolic reasoning traceability*: trực tiếp truy vết luật mờ và tính toán % đóng góp theo công thức giải mờ. |
| **"Hệ mờ đóng góp gì so với chỉ dùng CNN?"** | The fuzzy inference layer does not replace the CNN classifier. Nó tinh chỉnh mức độ bệnh (*Severity Assessment*) bằng cách tổng hợp độ tự tin (confidence), thông tin môi trường và bối cảnh đa chiều. |
| **"OOD Detector hoạt động thế nào?"** | Sử dụng Semantic Grouping để giảm rủi ro Softmax Fragmentation, đồng thời tính Shannon Entropy để *handle uncertainty*, qua đó *reduce silent failure risk* trên mẫu lạ. |

---

## VI. Kết luận (Conclusion)

The system demonstrates that combining deep learning with fuzzy inference can provide a more interpretable and context-aware approach to plant disease severity assessment. The proposed architecture shows promising results for intelligent agricultural decision support while remaining extensible for future multi-label and real-world uncertainty scenarios.

Hệ thống này đại diện cho một **Uncertainty-aware agricultural AI system** và một **Severity-oriented decision support framework**, chứng minh nhóm nghiên cứu thấu hiểu các giới hạn của mô hình phân loại học sâu thuần túy và biết cách ứng dụng suy diễn logic mờ để ra quyết định an toàn, minh bạch hơn.

## 5.1. Đánh giá độ chính xác

### 5.1.1. Kết quả huấn luyện CNN

Mô hình CNN (EfficientNet-B0) được huấn luyện trên dataset **Severity-Based Rice Leaf Diseases** (Kaggle) gồm 9 lớp, qua 10 epoch (5 epoch freeze + 5 epoch fine-tune). Kết quả:

- **Best Validation Accuracy: 92.04%** (đạt tại epoch 7)
- **Macro Average F1-score: 0.9190**
- **Weighted Average F1-score: 0.9191**

**Bảng Classification Report chi tiết:**

| Lớp bệnh | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Healthy | 1.0000 | 1.0000 | 1.0000 | 51 |
| Mild Bacterial blight | 0.9500 | 0.9048 | 0.9268 | 63 |
| Mild Blast | 0.9385 | 0.9104 | 0.9242 | 67 |
| Mild Brownspot | 0.7561 | 0.9538 | 0.8435 | 65 |
| Mild Tungro | 0.9516 | 1.0000 | 0.9752 | 59 |
| Severe Bacterial blight | 0.9143 | 0.9412 | 0.9275 | 68 |
| Severe Blast | 0.9118 | 0.9394 | 0.9254 | 66 |
| Severe Brownspot | 0.9250 | 0.6491 | 0.7629 | 57 |
| Severe Tungro | 1.0000 | 0.9710 | 0.9853 | 69 |
| **Tổng / Trung bình** | **0.9258** | **0.9204** | **0.9191** | **565** |

**Nhận xét:**
- Lớp **Healthy** đạt F1 = 1.0 (nhận diện hoàn hảo cây khỏe mạnh)
- Lớp **Severe Tungro** và **Mild Tungro** đạt F1 > 0.97
- Lớp **Severe Brownspot** có Recall thấp nhất (0.6491) do bị nhầm lẫn với Mild Brownspot (20/57 mẫu) — đây là thách thức vì hai mức độ của cùng một bệnh có đặc trưng thị giác gần nhau
- Nhờ **Label Smoothing = 0.1**, phân phối softmax đầu ra được calibrated tốt (không bị cực đoan), phù hợp làm input cho Fuzzy Layer

### 5.1.2. Lịch sử huấn luyện

| Epoch | Pha | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----|-----------|-----------|---------|---------|
| 1 | Freeze | 1.2914 | 0.6488 | 1.0159 | 0.7752 |
| 2 | Freeze | 1.0034 | 0.7943 | 0.9063 | 0.8354 |
| 5 | Freeze | 0.9435 | 0.8200 | 0.8789 | 0.8673 |
| 7 | Fine-tune | 0.7996 | 0.8881 | 0.7053 | **0.9204** |
| 10 | Fine-tune | 0.6618 | 0.9368 | 0.6712 | 0.9186 |

→ Validation accuracy tăng mạnh từ 77.5% (epoch 1) lên 92.04% (epoch 7) sau khi chuyển sang pha fine-tune.

## 5.2. Đánh giá tính dễ hiểu (Explainability)

Hệ thống đảm bảo tính giải thích được (XAI) thông qua các cơ chế sau:

**1. Báo cáo bằng ngôn ngữ tự nhiên tiếng Việt:**
Thay vì chỉ trả về con số, hệ thống sinh báo cáo chi tiết với các mệnh đề dạng: "Mô hình CNN chẩn đoán lá lúa nhiễm bệnh: Đạo ôn lá (Rice Blast)" kèm phân tích từng yếu tố.

**2. Truy vết luật mờ và Chế độ suy luận (Traceability):**
Mỗi kết luận đều liệt kê rõ: Hệ thống đang chạy ở chế độ nào (`AI_CONFIDENT`, `HYBRID_WARNING` hay `EXPERT_GUIDED_MODE`), tại sao lại chọn chế độ đó (dựa trên Entropy, Margin, Grouped Confidence), và luật IF-THEN nào đã kích hoạt với trọng số bao nhiêu. Người nông dân có thể đối chiếu trực tiếp với kinh nghiệm thực tế.

**3. Trực quan hóa trên Dashboard:**
Giao diện Streamlit hiển thị:
- Biểu đồ thanh (bar chart) phân phối softmax Top-5 lớp CNN
- Badge màu sắc theo mức cảnh báo (xanh/vàng/đỏ)
- Thanh tiến trình cho VSI và Diagnostic Confidence
- Card hiển thị từng luật mờ kích hoạt

**4. Xuất báo cáo JSON:**
Toàn bộ kết quả (CNN scores, thông số môi trường, chỉ số Fuzzy, giải thích, khuyến nghị) được đóng gói thành file JSON để lưu trữ và truy vết.

## 5.3. Ưu điểm và Nhược điểm

### Ưu điểm:
1. **Tính giải thích cao (XAI):** Khác với mô hình black-box, hệ thống trình bày rõ ràng từng bước suy luận thông qua luật mờ IF-THEN, cho phép người nông dân hiểu và tin tưởng kết quả.
2. **Kiến trúc Hybrid hiệu quả:** Kết hợp sức mạnh nhận diện hình ảnh của CNN với tri thức chuyên gia nông nghiệp trong Fuzzy Logic, tận dụng ưu thế của cả hai.
3. **Tích hợp yếu tố môi trường và thực địa:** Hệ thống không chỉ dựa vào hình ảnh mà còn kết hợp nhiệt độ, độ ẩm (đánh giá nguy cơ) và tín hiệu thực địa (như mật độ ốc bươu vàng) để ra quyết định toàn diện.
4. **Xử lý An toàn (Safety-critical OOD Handling):** Kỹ thuật sử dụng *Normalized Entropy* và *Grouped Confidence* giúp phát hiện dữ liệu bất định, chống lại sự "tự tin thái quá" của CNN, tự động nhường quyền cho chế độ `EXPERT_GUIDED_MODE` khi cần thiết.
5. **Label Smoothing → Calibrated Softmax:** Kỹ thuật này giúp phân phối xác suất đầu ra CNN phản ánh đúng mức độ chắc chắn thực tế, phù hợp làm input cho Fuzzy.
6. **Khuyến nghị cụ thể theo loại bệnh:** Mỗi loại bệnh có hướng dẫn thuốc đặc trị và biện pháp canh tác riêng, kèm theo các cảnh báo an toàn hóa chất.
7. **Giao diện trực quan:** Dashboard Streamlit dark theme hiện đại, dễ sử dụng cho cả người không chuyên IT.

### Nhược điểm:
1. **Luật mờ thiết kế thủ công:** Cơ sở luật và hàm thuộc hiện được xác định bởi chuyên gia, chưa có cơ chế tự học (adaptive) từ dữ liệu thực tế.
2. **Giới hạn số lớp bệnh:** Chỉ hỗ trợ 4 loại bệnh lá lúa (Bacterial blight, Blast, Brownspot, Tungro) + Healthy. Chưa bao phủ các bệnh khác như Sheath blight, False smut.
3. **Severe Brownspot bị nhầm lẫn:** Recall = 64.91%, do đặc trưng thị giác của Mild và Severe Brownspot rất gần nhau. Cần bổ sung data augmentation hoặc attention mechanism.
4. **Chưa tích hợp IoT thời gian thực:** Thông số nhiệt độ và độ ẩm hiện do người dùng nhập tay, chưa kết nối cảm biến tự động.
5. **Chỉ hỗ trợ ảnh đơn:** Chưa xử lý video hoặc ảnh đa góc chụp để tăng độ chính xác chẩn đoán.

---

# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## Kết luận

Đề tài đã thiết kế và xây dựng thành công một **Hệ thống thông minh có thể giải thích được (Explainable Intelligent System)** cho bài toán đánh giá mức độ nghiêm trọng bệnh lá lúa, đạt được các kết quả chính:

1. **Về mặt Tri thức:** Đã số hóa thành công tri thức chuyên gia nông nghiệp vào 20+ luật mờ IF-THEN, 4 nhóm hàm thuộc (Confidence, Margin, Temperature, Humidity), và bảng khuyến nghị đặc thù cho 4 loại bệnh lúa.

2. **Về mặt Xử lý:** Bộ suy diễn mờ Sugeno bậc 0 hoạt động ổn định, tích hợp thông suốt với mô hình CNN EfficientNet-B0 (Accuracy 92.04%). Đặc biệt, **Lớp Quyết định Hỗ trợ Chuyên gia (Expert-Guided Decision Layer)** giúp hệ thống xử lý hoàn hảo sự bất định của AI thông qua Entropy, đảm bảo an toàn tuyệt đối. Pipeline Hybrid xử lý trơn tru từ ảnh đầu vào đến kết luận + XAI report.

3. **Về mặt Giải thích (XAI):** Hệ thống sinh báo cáo ngôn ngữ tự nhiên tiếng Việt, truy vết luật mờ kích hoạt với trọng số, cung cấp khuyến nghị nông nghiệp cụ thể — đáp ứng yêu cầu minh bạch cho người nông dân.

4. **Về mặt Hệ thống:** Giao diện Streamlit trực quan, module hóa rõ ràng (rice/, rice_fuzzy_xai/), hỗ trợ xuất JSON, sẵn sàng mở rộng.

## Hướng phát triển

1. **Tự động hóa luật mờ (ANFIS):** Áp dụng Adaptive Neuro-Fuzzy Inference System để tự học tham số hàm thuộc từ dữ liệu thực tế, giảm phụ thuộc vào chuyên gia.
2. **Mở rộng loại bệnh:** Bổ sung thêm các bệnh lúa phổ biến khác (Sheath blight, False smut, Leaf scald) và mở rộng sang cây trồng khác (ngô, cà phê).
3. **Tích hợp IoT:** Kết nối cảm biến nhiệt độ, độ ẩm, lượng mưa thời gian thực từ đồng ruộng thông qua MQTT/REST API.
4. **Ứng dụng di động:** Phát triển ứng dụng Android/iOS cho phép nông dân chụp ảnh tại ruộng và nhận kết quả chẩn đoán trực tiếp.
5. **Cải thiện CNN:** Thử nghiệm các kiến trúc mạnh hơn (EfficientNet-B3, Vision Transformer) hoặc bổ sung Attention mechanism để cải thiện phân biệt Mild/Severe.

---

## TÀI LIỆU THAM KHẢO

[1] Bài giảng Hệ thống Thông minh — Trường Đại học Thủy Lợi.
[2] L.A. Zadeh, "Fuzzy Sets," Information and Control, vol. 8, no. 3, pp. 338–353, 1965.
[3] E.H. Mamdani, S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1–13, 1975.
[4] T. Takagi, M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," IEEE Transactions on Systems, Man, and Cybernetics, vol. 15, no. 1, pp. 116–132, 1985.
[5] M. Tan, Q.V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML 2019.
[6] A. Barredo Arrieta et al., "Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI," Information Fusion, vol. 58, pp. 82–115, 2020.
[7] Kaggle Dataset: "Severity-Based Rice Leaf Diseases Dataset" by Isaac Ritharson. https://www.kaggle.com/datasets/isaacritharson/severity-based-rice-leaf-diseases-dataset
[8] PyTorch Documentation. https://pytorch.org/docs/
[9] Streamlit Documentation. https://docs.streamlit.io/

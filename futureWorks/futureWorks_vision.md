# Định hướng cải tiến Kiến trúc Hệ thống (Future Works)

Tài liệu này ghi lại toàn bộ các ý tưởng, đánh giá khách quan và định hướng chiến lược để nâng cấp dự án "Đánh giá mức độ nghiêm trọng của bệnh trên lá lúa sử dụng Logic Mờ (Fuzzy Logic) và AI có thể giải thích được (XAI)" trong tương lai.

---

## PHẦN I: TẦM NHÌN VÀ CÁC ĐỊNH HƯỚNG CẢI TIẾN TỔNG THỂ

Để đưa hệ thống lên một tầm cao mới, dưới đây là các nhóm giải pháp chiến lược có thể áp dụng:

### 1. Cải tiến mô hình cốt lõi (Core Model Enhancements)
*   **Neuro-Fuzzy Systems (ANFIS):** Thay vì thiết kế các hàm liên thuộc (membership functions) và tập luật (rules) thủ công, hệ thống có thể sử dụng mạng Nơ-ron để *tự động học và điều chỉnh* các tham số này từ dữ liệu thực tế. Điều này kết hợp sức mạnh học hỏi của Deep Learning và tính minh bạch của Fuzzy Logic.
*   **Tối ưu hóa tiến hóa:** Sử dụng Thuật toán Di truyền (GA) hoặc Tối ưu hóa bầy đàn (PSO) để tự động tinh chỉnh (fine-tune) các luật mờ sao cho độ chính xác cao nhất.

### 2. Nâng cấp khả năng giải thích (XAI - Explainability)
*   **Giải thích bằng Ngôn ngữ Tự nhiên (Natural Language Explanations - NLG):** Xây dựng module dịch các kết quả luật mờ thành câu văn thân thiện. Ví dụ: *"Hệ thống đánh giá bệnh đang ở mức NẶNG vì phát hiện 40% diện tích lá bị cháy vàng và có nhiều đốm nâu tập trung."*
*   **Trực quan hóa (Visual Explanations):** Tích hợp Grad-CAM hoặc SHAP để tạo các *heatmap*, giúp người dùng *nhìn thấy* trực tiếp vùng bệnh nào trên ảnh đã khiến mô hình quyết định mức độ bệnh.

### 3. Mở rộng Hệ thống và Dữ liệu (Scaling & Scope)
*   **Đa dạng hóa loại bệnh:** Mở rộng nhận diện nhiều loại bệnh khác nhau (Đạo ôn, Bạc lá, Đốm vằn...).
*   **Tích hợp dữ liệu Đa phương thức (Multi-modal input):** Đưa thêm các biến môi trường (nhiệt độ, độ ẩm) vào hệ thống Fuzzy để dự báo thực tế và chính xác hơn.
*   **Học liên tục (Continuous Learning):** Cơ chế nhận feedback từ người dùng để hệ thống tự động cập nhật và tinh chỉnh tập luật mờ.

### 4. Triển khai Ứng dụng thực tế (Deployment)
*   **Edge Computing / Mobile App:** Đóng gói mô hình thành ứng dụng di động chạy offline để nông dân có thể chụp ảnh và nhận kết quả trực tiếp ngoài đồng ruộng.

---

## PHẦN II: PHÂN TÍCH KIẾN TRÚC XỬ LÝ ẢNH ĐẦU VÀO (SSFC vs YOLO)

Một trong những bài toán quan trọng nhất là nâng cấp khâu xử lý ảnh đầu vào (Region Proposal) để thay thế cho các phương pháp cũ kỹ, tốn kém. Dưới đây là phân tích dựa trên 2 mô hình kiến trúc:

### 1. Phương án truyền thống bằng SSFC (Hình `SSFC.jpg`)
*   **Cách tiếp cận:** Sử dụng phương pháp SSFC-FS để phân đoạn toàn bộ bức ảnh thành các segment nhỏ $C^1, C^2,..., C^n$. Sau đó trích xuất đặc trưng và phân cụm từng segment.
*   **Ưu điểm:** Bám sát lý thuyết logic mờ từ đầu đến cuối; không cần dữ liệu gán nhãn lớn.
*   **Nhược điểm:** Tạo ra quá nhiều segment dư thừa (background, lá khỏe) gây nghẽn cổ chai hiệu năng. Khó chạy real-time và dễ bị nhiễu bởi môi trường thực tế.

### 2. Phương án Kiến trúc Hybrid với YOLO (Hình `YOLO.png`)
*   **Cách tiếp cận:** Dùng Deep Learning (YOLOv8-seg/YOLOv11-seg) làm "Bộ lọc thô" ở đầu vào để bắt đúng các vùng tổn thương (ROI). Sau đó, các vùng này mới được đẩy vào Hệ chuyên gia mờ để suy luận.
*   **Ưu điểm:** Tập trung đúng trọng tâm, loại bỏ nhiễu background ngay lập tức. Hiệu năng cực cao (chạy được luồng video). Dễ trực quan hóa.
*   **Nhược điểm:** Yêu cầu gán nhãn dữ liệu khổng lồ (vẽ bounding box) cho quá trình huấn luyện ban đầu.

---

## PHẦN III: ĐỀ XUẤT 2 PHƯƠNG ÁN FUTURE WORKS ĐỂ BÓC TÁCH VÙNG BỆNH

Để giải quyết triệt để khâu "Đầu vào" (Front-end) mà vẫn khắc phục được bài toán gán nhãn dữ liệu, chúng ta có 2 phương án song song cực kỳ khả thi (kế thừa triết lý Hybrid AI):

### Hướng 1: YOLO-based Lesion Proposal (Khuyến nghị cho Sản phẩm thực tế)
*   **Cách tiếp cận:** Sử dụng YOLO fine-tune với 1 class duy nhất (Vết bệnh / Disease Spot). Nó đóng vai trò như chiếc kéo cắt các vết bệnh ra, đẩy vào mạng CNN rút trích đặc trưng và Hệ chuyên gia mờ.
*   **Ưu điểm:** Tốc độ cực nhanh (Real-time). Bóc tách vùng bệnh gọn gàng, viền sắc nét. Chuẩn công nghiệp (hiển thị bounding box chuyên nghiệp).
*   **Nhược điểm:** Đòi hỏi nguồn lực lớn để vẽ hàng ngàn bounding box quanh từng vết bệnh nhỏ trên tập dữ liệu train.

### Hướng 2: XAI/Grad-CAM-based Weakly Supervised Segmentation (Khuyến nghị cho NCKH & Học thuật)
*   **Cách tiếp cận:** Huấn luyện một mô hình CNN (như ResNet) chỉ để phân loại toàn ảnh (Ảnh Có bệnh vs Ảnh Không bệnh). Dùng thuật toán XAI (Grad-CAM) để sinh Bản đồ nhiệt (Heatmap) chỉ ra vùng mô hình đang "nhìn". Cuối cùng dùng đặt ngưỡng (Thresholding) trên Heatmap để cắt ra vết bệnh.
*   **Ưu điểm:** Cực kỳ nhàn lúc làm dữ liệu (chỉ cần gán nhãn mức độ ảnh, không cần vẽ box). Có tính giải thích cao (XAI), gây ấn tượng thị giác (WOW effect) cực mạnh khi báo cáo hội đồng.
*   **Nhược điểm:** Tốc độ chậm hơn YOLO một chút do phải tính toán ngược gradient, vùng viền đôi khi bị lem, không sắc nét bằng YOLO.

**Kết luận chung:**
*   Làm App/Sản phẩm thực tế $\rightarrow$ Chọn **YOLO**.
*   Làm Đồ án/NCKH và thiếu nhân sự làm Data $\rightarrow$ Chọn **Grad-CAM**.

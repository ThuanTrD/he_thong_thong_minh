# Danh mục Ảnh minh họa Học thuật (Academic Figures Index)

Thư mục này chứa 12 ảnh minh họa vector hóa được lập trình (programmatically generated) theo đúng kiến trúc của dự án, độ phân giải 1920x1080 (16:9), phù hợp tuyệt đối cho **Slide thuyết trình** và **Báo cáo (Report)** bảo vệ đồ án Cao học. 

Tất cả các hình ảnh đều bám sát concept **Intelligent Decision-Support System**, hoàn toàn không chứa các yếu tố marketing hay khoa học viễn tưởng.

## Hướng dẫn Sử dụng (Recommended Usage)

| Tên File | Mô tả Nội dung | Mục đích / Section Slide đề xuất |
| :--- | :--- | :--- |
| `fig_01_system_overview.png` | **Sơ đồ kiến trúc toàn cảnh**: Hiển thị pipeline 5 bước (CNN -> OOD -> Fuzzy -> ERI -> XAI). | Slide Tổng quan Kiến trúc (System Architecture) |
| `fig_02_severity_pipeline.png` | **Luồng suy diễn Severity**: Phân tích chi tiết đường đi của Softmax score qua các hàm mờ để tạo ra VSI và Final Alert. | Slide Phương pháp cốt lõi (Core Methodology) |
| `fig_03_membership_functions.png` | **Đồ thị hàm thuộc (Membership Functions)**: Hiển thị các tập mờ Healthy, Mild, Moderate, Severe và vùng giao thoa. | Slide Cơ sở lý thuyết Logic mờ (Fuzzy Logic Basis) |
| `fig_04_vsi_transition.png` | **Quang phổ VSI (Continuous Spectrum)**: Minh họa sự chuyển tiếp mượt mà và nhấn mạnh "Moderate" là một trạng thái nội suy. | Slide Xử lý nội suy (Severity Interpolation) |
| `fig_05_rule_activation.png` | **Bảng Truy vết Luật (Rule Traceability)**: Hiển thị Firing Strength ($w_i$) và % đóng góp của từng tập luật. | Slide Explainable AI (XAI Mechanism) |
| `fig_06_ood_uncertainty.png` | **Luồng xử lý Bất định (Uncertainty Handling)**: Phân nhánh giữa Known, High Entropy và chế độ Hybrid_Warning. | Slide Xử lý Ngoại lệ / OOD Detection |
| `fig_07_contextual_alert.png` | **Nâng cấp Cảnh báo theo Ngữ cảnh**: Minh họa 2 trường hợp cùng VSI nhưng môi trường khác nhau sinh ra cảnh báo khác nhau. | Slide Trí tuệ ngữ cảnh (Contextual Decision Support) |
| `fig_08_xai_report.png` | **Ví dụ Báo cáo XAI bằng ngôn ngữ tự nhiên**: Kết xuất reasoning rule thành câu chữ giải thích. | Slide Demo Output / Explainability |
| `fig_09_error_analysis.png` | **Phân tích sai số học thuật**: Minh họa lý do vì sao hệ thống có hành vi "Conservative Downgrade" (Hạ cấp thận trọng). | Slide Đánh giá Lỗi (Error Analysis) |
| `fig_10_system_contributions.png` | **Bảng Đóng góp thành phần**: Liệt kê rõ CNN làm gì, Fuzzy làm gì, ERI làm gì... | Slide Đóng góp của Đề tài (Contributions) |
| `fig_11_hybrid_ai_architecture.png` | **Ngăn xếp Công nghệ (Architecture Stack)**: Lớp CNN dưới cùng, đến Fuzzy, Expert, Environment, và XAI trên cùng. | Slide Tóm tắt Phương pháp (Methodology Summary) |
| `fig_12_deployment_pipeline.png` | **Luồng Triển khai End-to-End**: Cách FastAPI kết nối với Fuzzy Engine và đưa lên Web Streamlit. | Slide Triển khai Hệ thống (System Deployment) |

> **Lưu ý cho Sinh viên**: Khi chèn các ảnh này vào Word (Report) hoặc PowerPoint (Slide), hãy sử dụng nền tối hoặc tùy chọn viền hợp lý vì bộ ảnh được thiết kế theo tone màu **Dark Academic** (Nền tối #0f172a, chữ sáng, màu điểm nhấn là Xanh ngọc Teal và Đỏ san hô). Điều này giúp trình bày trên máy chiếu cực kỳ chuyên nghiệp và sang trọng.

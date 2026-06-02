# Sơ đồ Kiến trúc Hệ thống Suy diễn Lai (Hybrid Fuzzy-CNN)

Đoạn mã Mermaid dưới đây biểu diễn toàn bộ luồng suy luận của hệ thống, được chú thích rõ **3 Cơ sở lý thuyết khoa học** cốt lõi. Bạn có thể chèn đoạn mã này vào file Markdown, GitHub, Notion, hoặc các công cụ hỗ trợ Mermaid để lấy ảnh đưa vào Slide Thuyết trình.

```mermaid
graph TD
    classDef theoretical fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef input fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef cnn fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef fuzzy fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef expert fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000
    classDef output fill:#eceff1,stroke:#263238,stroke-width:2px,color:#000

    %% Inputs
    I1[📸 Ảnh bệnh trên lá] ::: input
    I2[🌤️ Thời tiết: Nhiệt độ, Độ ẩm] ::: input
    I3[🐌 Tín hiệu thực địa: Mật độ ốc] ::: input

    %% CNN & Signals
    CNN[🧠 Mạng tích chập CNN <br/> Visual Perception Layer] ::: cnn
    I1 --> CNN

    subgraph "Lý thuyết 1: Uncertainty in ML (Phát hiện OOD & Bất định)"
        S1[Max Confidence & Margin] ::: cnn
        S2[Normalized Entropy <br/> OOD Signal] ::: cnn
        S3[Grouped Confidence <br/> Bệnh ưu tiên] ::: cnn
    end
    CNN --> S1 & S2 & S3

    %% Fuzzy Logic
    subgraph "Lý thuyết 2: Cognitive Science (Sự chồng lấn nhận thức)"
        F1[Mờ hóa: Điểm số Mild & Severe <br/> Hàm liên thuộc Hình thang/Tam giác] ::: fuzzy
    end
    CNN --> F1

    subgraph "Lý thuyết 3: Plant Epidemiology (Dịch tễ học Thực vật)"
        F2[Mờ hóa: Temperature & Humidity <br/> Điều kiện mầm bệnh bùng phát] ::: fuzzy
    end
    I2 --> F2

    F3{⚙️ Fuzzy Inference Engine <br/> Sugeno (Zero-order)} ::: fuzzy
    F1 --> F3
    F2 --> F3
    S1 --> F3

    F3 --> |Suy diễn nội suy| O1[Mức độ bệnh tổn thương - VSI] ::: fuzzy
    F3 --> |Cảnh báo môi trường| O2[Nguy cơ môi trường - ERI] ::: fuzzy
    F3 --> |Đánh giá độ tin cậy| O3[Uncertainty Level] ::: fuzzy

    %% Expert System
    subgraph "Lý thuyết 4: Decision Support System (Quản trị rủi ro & An toàn)"
        E1{⚖️ Bộ định tuyến Chế độ <br/> Inference Mode Router} ::: expert
        S2 --> |High Entropy > 0.6| E1
        I3 --> |Tín hiệu ngoại lệ cực đoan| E1
        O3 --> E1
        S1 --> E1
        S3 --> E1

        E1 --> |AI Confident| E2[AI_CONFIDENT <br/> Tin tưởng thị giác máy tính] ::: expert
        E1 --> |Hybrid Warning| E3[HYBRID_WARNING <br/> Kết hợp cảnh báo] ::: expert
        E1 --> |Expert Guided Mode| E4[EXPERT_GUIDED_MODE <br/> Phủ quyết AI, Ưu tiên thực địa] ::: expert
    end

    O1 --> E1
    O2 --> E1

    %% Final Output
    E2 --> Out[📄 Báo cáo XAI - Explainable Report <br/> Truy vết luật & Thông số] ::: output
    E3 --> Out
    E4 --> Out
```

### 💡 Hướng dẫn sử dụng sơ đồ này khi Bảo vệ:

1. **Hiệu ứng thị giác:** Hãy render sơ đồ này (có thể copy mã vào https://mermaid.live/ để xuất ra ảnh PNG nét căng) và đưa lên 1 slide riêng biệt mang tên **"Architecture & Theoretical Foundations"**.
2. **Kịch bản thuyết trình (Power Move):** 
   - Khi Thầy cô hỏi xoáy: *"Các bạn dựa vào đâu để viết các luật IF/ELSE này? Nó có phải cảm tính không?"* 
   - Ngay lập tức, bạn bấm chuyển sang Slide chứa Sơ đồ này và nói: *"Dạ thưa Thầy/Cô, sơ đồ luồng hệ thống trên màn hình minh họa chính xác 4 trụ cột lý thuyết mà nhóm đã áp dụng..."*. Việc chuẩn bị sẵn sơ đồ để "counter" một câu hỏi phản biện sẽ khiến hội đồng đánh giá cực kỳ cao sự chuẩn bị và hàm lượng khoa học của bạn.

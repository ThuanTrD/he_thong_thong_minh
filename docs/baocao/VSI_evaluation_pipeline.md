# Sơ đồ Kiến trúc Đánh giá Mức độ Bệnh (CNN + Fuzzy Logic)

Dưới đây là sơ đồ luồng dữ liệu (Data Flow) chi tiết, mô tả cách hệ thống tiếp nhận hình ảnh đầu vào, đi qua mạng nơ-ron (CNN) để lấy xác suất mềm (Softmax Scores), sau đó kết hợp với thông số môi trường đi vào Bộ động cơ mờ (Fuzzy Engine) để đưa ra các chỉ số và cảnh báo cuối cùng.

```mermaid
graph TD
    %% Định nghĩa các Style
    classDef inputNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef cnnNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef fuzzyNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef expertNode fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef outputNode fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000;

    %% 1. Inputs
    I_Img["📸 Hình ảnh lá lúa"] ::: inputNode
    I_Env["🌡️ Môi trường (Nhiệt độ, Độ ẩm)"] ::: inputNode
    I_Exp["🐌 Tín hiệu thực địa (Mật độ ốc...)"] ::: inputNode

    %% 2. CNN Stage
    subgraph Mạng Nơ-ron Tích chập (CNN - EfficientNet-B0)
        CNN_Ext["Trích xuất đặc trưng (Feature Extraction)"] ::: cnnNode
        CNN_Soft["Softmax Layer (Xuất xác suất 9 lớp)"] ::: cnnNode
        
        CNN_Scores["Xác suất thô (Hard/Soft Labels):<br/>- Healthy<br/>- Mild (Nhẹ)<br/>- Severe (Nặng)"] ::: cnnNode
        CNN_Uncert["Tính toán Độ bất định:<br/>- Margin (Khoảng cách)<br/>- Entropy (OOD Signal)"] ::: cnnNode
    end

    I_Img --> CNN_Ext --> CNN_Soft --> CNN_Scores
    CNN_Soft --> CNN_Uncert

    %% 3. Pre-Fuzzy (Trích xuất điểm Mild/Severe)
    subgraph Tiền xử lý Dữ liệu Mờ
        Pre_Extract["Trích xuất Softmax Scores:<br/>mild_score & severe_score<br/>cho bệnh được dự đoán"] ::: fuzzyNode
    end
    
    CNN_Scores --> Pre_Extract

    %% 4. Fuzzy Engine Stage
    subgraph Hệ Suy diễn Mờ (Fuzzy Inference Engine)
        Fuz_Fuzzification["Mờ hóa (Fuzzification):<br/>Biến đổi giá trị rõ (Crisp) thành Tập mờ"] ::: fuzzyNode
        
        %% Các luồng nội suy
        Fuz_VSI["Đánh giá VSI<br/>(Visual Severity Index)"] ::: fuzzyNode
        Fuz_ERI["Đánh giá ERI<br/>(Environmental Risk Index)"] ::: fuzzyNode
        Fuz_FAI["Đánh giá FAI<br/>(Final Alert Index)"] ::: fuzzyNode
        
        %% Defuzzification
        Fuz_Defuzz["Giải mờ (Defuzzification) & Phân cấp độ"] ::: fuzzyNode
    end

    Pre_Extract --> Fuz_Fuzzification
    I_Env --> Fuz_Fuzzification
    
    Fuz_Fuzzification -- "mild_fuzzy, severe_fuzzy" --> Fuz_VSI
    Fuz_Fuzzification -- "temp_fuzzy, hum_fuzzy" --> Fuz_ERI
    
    Fuz_VSI --> Fuz_FAI
    Fuz_ERI --> Fuz_FAI
    
    Fuz_VSI --> Fuz_Defuzz
    Fuz_ERI --> Fuz_Defuzz
    Fuz_FAI --> Fuz_Defuzz

    %% 5. Expert Mode & Fusion
    subgraph Xử lý Tín hiệu Chuyên gia (Hybrid Fusion)
        Exp_Mode{"Kiểm tra Entropy/Margin<br/>+ Tín hiệu thực địa"} ::: expertNode
        Exp_Action["Phủ quyết/Nâng cảnh báo<br/>(Expert-Guided Mode)"] ::: expertNode
    end

    CNN_Uncert --> Exp_Mode
    I_Exp --> Exp_Mode
    Fuz_Defuzz --> Exp_Mode
    Exp_Mode -- Bất định cao / Có tín hiệu --> Exp_Action
    Exp_Mode -- AI tự tin --> Out_Output

    %% 6. Output Stage
    subgraph Kết quả Đầu ra (Outputs & XAI)
        Out_Output["Kết quả Tổng hợp:<br/>- VSI (Mức độ tổn thương lá)<br/>- Fused Confidence<br/>- Mức Cảnh báo cuối cùng"] ::: outputNode
        Out_XAI["📝 Báo cáo XAI:<br/>Truy xuất các Luật Mờ (Fuzzy Rules)<br/>đã được kích hoạt"] ::: outputNode
        Out_Rec["👨‍🌾 Khuyến nghị Nông nghiệp"] ::: outputNode
    end

    Exp_Action --> Out_Output
    Out_Output --> Out_XAI
    Out_Output --> Out_Rec

```

### Diễn giải chi tiết các bước trong sơ đồ:

1. **Đầu vào (Inputs):** Hệ thống nhận ảnh chụp lá lúa, thông số môi trường (nhiệt độ, độ ẩm), và có thể có thêm các tín hiệu từ thực địa do nông dân cung cấp (như mật độ ốc bươu vàng).
2. **Xử lý CNN:** Ảnh đi qua mạng EfficientNet-B0. Thay vì chỉ xuất ra kết quả cuối cùng, mạng trả về xác suất mềm (Softmax Scores) cho cả 9 lớp. Đồng thời, hệ thống tính toán luôn độ phân tán (Entropy) để biết AI đang "tự tin" hay "bối rối".
3. **Trích xuất điểm số:** Lấy riêng xác suất của dạng `Mild` (Nhẹ) và `Severe` (Nặng) của loại bệnh đang được chẩn đoán để chuẩn bị đưa vào hệ mờ.
4. **Hệ Suy diễn Mờ (Fuzzy Engine):**
   * **Mờ hóa:** Biến các con số (VD: xác suất 0.6, nhiệt độ 30°C) thành các mức độ mờ (Cao, Thấp, Trung bình).
   * **VSI (Visual Severity Index):** Tính toán độ nghiêm trọng trên lá dựa vào sự pha trộn giữa điểm `mild` và `severe`.
   * **ERI (Environmental Risk Index):** Tính toán nguy cơ bùng phát dịch dựa vào nhiệt độ và độ ẩm.
   * **FAI (Final Alert Index):** Kết hợp VSI và ERI để ra mức độ cảnh báo tổng.
5. **Chế độ Chuyên gia (Expert Fusion):** Đây là chốt chặn an toàn. Nếu AI phân tích ảnh thấy bất định (Entropy cao, Margin thấp), hệ thống sẽ xem xét các thông tin thực địa (như có ốc bươu vàng) để ép chuyển sang `Expert-Guided Mode`, có thể nâng mức cảnh báo lên Báo động Đỏ kể cả khi lá lúa trông có vẻ bình thường.
6. **Đầu ra & Giải thích (Outputs & XAI):** Xuất ra kết quả chẩn đoán, mức độ nghiêm trọng, và quan trọng nhất là tạo Báo cáo XAI giải thích rõ ràng tại sao hệ thống lại ra quyết định đó (nhờ truy xuất ngược lại các luật mờ đã được kích hoạt).

## 4.2. Minh họa kịch bản chạy thử (Use Case / Simulation)

### 4.2.1. Kịch bản: Chẩn đoán bệnh Đạo ôn nhẹ (Mild Blast)

**Dữ liệu đầu vào:**
- Ảnh lá lúa: Ảnh chụp lá lúa có vết đốm đạo ôn (file BLAST9_087.jpg)
- Nhiệt độ: 28.0°C (Warm)
- Độ ẩm: 85.0% (Wet)

**Bước 1 — CNN Inference:** Mô hình EfficientNet-B0 phân tích ảnh và trả về phân phối Softmax 9 lớp:

| Lớp bệnh | Xác suất Softmax |
|-----------|-----------------|
| Mild Blast | **0.9747** |
| Severe Blast | 0.0070 |
| Mild Bacterial blight | 0.0044 |
| Mild Brownspot | 0.0043 |
| Severe Bacterial blight | 0.0026 |
| Severe Tungro | 0.0022 |
| Healthy | 0.0020 |
| Mild Tungro | 0.0019 |
| Severe Brownspot | 0.0008 |

→ **Top-1 Prediction**: Mild Blast (Đạo ôn nhẹ) với **Confidence = 97.47%**
→ **Margin** (Top1 - Top2) = 0.9747 - 0.0070 = **0.9677** (rất lớn → bất định rất thấp)

**Bước 2 — Mờ hóa (Fuzzification):**

| Biến | Giá trị rõ | → Low/Cool/Dry/Small | → Medium/Warm/Moderate | → High/Hot/Wet/Large |
|------|-----------|---------------------|----------------------|---------------------|
| CNN Confidence | 0.9747 | 0.00 | 0.00 | 1.00 |
| Margin | 0.9677 | 0.00 | 0.00 | 1.00 |
| Temperature (28°C) | 28.0 | 0.00 | 0.86 (Warm) | 0.00 |
| Humidity (85%) | 85.0 | 0.00 | 0.00 | 1.00 (Wet) |

**Bước 3 — Đánh giá luật mờ (Rule Evaluation) & Giải mờ (Defuzzification Sugeno):**

*A. Độ tin cậy chẩn đoán (Diagnostic Confidence):*
- Luật kích hoạt: "NẾU Top Confidence Cao VÀ Margin Lớn THÌ Độ tin cậy Rất cao" → w = min(1.0, 1.0) = 1.0
- Giải mờ Sugeno: (1.0 × 95.0) / 1.0 = **95.00%**

*B. Mức độ nghiêm trọng trực quan (VSI):*
- Mild score = 0.9747, Severe score = 0.0070
- Luật kích hoạt: "NẾU Độ tự tin Severe Thấp VÀ Mild Cao THÌ Mức độ bệnh Nhẹ" → w = min(1.0, 1.0) = 1.0
- Giải mờ Sugeno: VSI = **20.0** → "Mild (Nhẹ)"

*C. Nguy cơ môi trường (ERI):*
- Luật kích hoạt: "NẾU Nhiệt độ Ấm VÀ Độ ẩm Ẩm ướt THÌ Nguy cơ môi trường Cao" → w = min(0.86, 1.0) = 0.86
- Giải mờ Sugeno: ERI = **85.0** → "High (Cao)"

*D. Cảnh báo cuối cùng (FAI):*
- VSI = 20.0 (Mild) + ERI = 85.0 (High)
- Luật kích hoạt: "NẾU Mức độ bệnh Nhẹ VÀ Nguy cơ môi trường Cao THÌ Cảnh báo Chú ý" → w = 1.0
- Giải mờ Sugeno: FAI = **40.0** → "Attention (Chú ý)"

**Bước 4 — Kết luận & XAI Output:**

Hệ thống kết luận: Bệnh **Đạo ôn lá (Rice Blast)** ở mức nhẹ nhưng điều kiện thời tiết ấm ẩm (28°C, 85% RH) tạo nguy cơ bùng phát cao.

*Khuyến nghị nông nghiệp được sinh tự động:*
- "CHÚ Ý THEO DÕI: Vết bệnh nhẹ nhưng môi trường ẩm ướt có nguy cơ."
- "Theo dõi sát sao các ruộng lân cận trong vòng 2-3 ngày tới."
- "Đặc thù Bệnh Đạo Ôn: Sử dụng hoạt chất Tricyclazole, Fenoxanil hoặc Isoprothiolane. Tránh bón thêm phân đạm."

### 4.2.2. Sơ đồ luồng xử lý Pipeline hoàn chỉnh

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│  INPUTS         │    │  CNN INFERENCE       │    │  FUZZY REASONING        │
│                 │    │                      │    │                         │
│ • Ảnh lá lúa    │───→│ • EfficientNet-B0    │───→│ • Mờ hóa 4 biến         │
│ • Nhiệt độ °C   │    │ • Softmax 9 lớp      │    │ • 20+ luật IF-THEN      │
│ • Độ ẩm %       │    │ • Top-1, Margin      │    │ • Sugeno Defuzzification│
│ • Tín hiệu ốc   │    │ • Entropy, Grouped   │    └───────────┬─────────────┘
└────────┬────────┘    └──────────┬───────────┘                │
         │                        │                            │
         │             ┌──────────▼───────────┐    ┌───────────▼─────────────┐
         └────────────►│ EXPERT DECISION LAYER│◄───│  HYBRID DECISION        │
                       │                      │    │                         │
                       │ • Mode Router        │    │ • Diagnostic Confidence │
                       │ • AI_CONFIDENT       │    │ • Visual Severity (VSI) │
                       │ • HYBRID_WARNING     │    │ • Env Risk (ERI)        │
                       │ • EXPERT_GUIDED_MODE │    │ • Final Alert (FAI)     │
                       └──────────┬───────────┘    └───────────┬─────────────┘
                                  │                            │
                       ┌──────────▼────────────────────────────▼─────────────┐
                       │  XAI REPORT (BÁO CÁO GIẢI THÍCH)                    │
                       │                                                     │
                       │ • Lý do chọn Inference Mode & Thông số CNN          │
                       │ • Báo cáo ngôn ngữ TN & Luật mờ kích hoạt           │
                       │ • Khuyến nghị NN hỗ trợ quyết định an toàn          │
                       └─────────────────────────────────────────────────────┘
```

## 4.3. Thành phần giải thích (Explainable Component)

Module XAI (explanation.py) sinh báo cáo giải thích tự động gồm 4 phần:

**Phần 1 — Phân tích Hình ảnh và Suy luận Lai (Hybrid Inference):** 
Trình bày kết quả CNN gồm lớp bệnh dự đoán, nhóm bệnh ưu tiên (Grouped Confidence), độ tự tin, khoảng cách margin so với lớp thứ 2, và Entropy chuẩn hóa. Đặc biệt, giải thích lý do hệ thống chọn chế độ suy luận (`AI_CONFIDENT`, `HYBRID_WARNING` hay `EXPERT_GUIDED_MODE`).

**Phần 2 — Phân tích Tác động Môi trường:** Hiển thị thông số thời tiết (nhiệt độ, độ ẩm) và đánh giá nguy cơ bùng phát dịch bệnh do điều kiện tự nhiên.

**Phần 3 — Mức độ Cảnh báo Tổng hợp:** Kết hợp VSI và ERI để đưa ra 4 cấp cảnh báo: Normal → Attention → Danger → Red Alert.

**Phần 4 — Các Luật Mờ được Kích hoạt:** Liệt kê chi tiết từng luật IF-THEN đã kích hoạt kèm trọng số, chia thành 3 nhóm: Luật Mức độ bệnh, Luật Thời tiết, Luật Cảnh báo. Đây chính là yếu tố XAI cốt lõi — cho phép người dùng đối chiếu và kiểm chứng từng bước suy luận của hệ thống.

**Khuyến nghị nông nghiệp** được sinh tự động theo 4 mức:
- **Red Alert**: Phun thuốc đặc trị ngay, ngừng bón đạm, tháo nước ruộng
- **Danger**: Phun thuốc khoanh vùng, kiểm tra nguồn nước tưới
- **Attention**: Theo dõi 2-3 ngày, bón kali tăng sức đề kháng
- **Normal**: Cắt tỉa lá bệnh đơn lẻ, chưa cần can thiệp hóa chất

Mỗi loại bệnh còn có khuyến nghị đặc thù riêng (ví dụ: Đạo ôn → Tricyclazole; Bạc lá → thuốc sát khuẩn đồng; Tungro → diệt rầy xanh; Ốc bươu vàng → bắt thủ công hoặc dùng Metaldehyde). Đặc biệt, hệ thống luôn chèn cảnh báo an toàn: *"Việc sử dụng thuốc/hoạt chất cần tuân thủ hướng dẫn của cán bộ BVTV địa phương"*.

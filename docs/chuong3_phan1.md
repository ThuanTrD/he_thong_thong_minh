# CHƯƠNG 3: CÀI ĐẶT VÀ THỬ NGHIỆM (IMPLEMENTATION AND RESULTS)

## 4.1. Công nghệ sử dụng

Hệ thống được xây dựng hoàn toàn trên nền tảng Python, tận dụng hệ sinh thái phong phú của các thư viện mã nguồn mở cho Học sâu và Xử lý dữ liệu. Bảng dưới đây tổng hợp các công nghệ cốt lõi:

| STT | Công nghệ | Phiên bản | Vai trò trong hệ thống |
|-----|-----------|-----------|------------------------|
| 1 | Python | 3.10+ | Ngôn ngữ lập trình chính |
| 2 | PyTorch | ≥ 2.0.0 | Framework Học sâu, huấn luyện và suy diễn CNN |
| 3 | TorchVision | ≥ 0.15.0 | Cung cấp mô hình pretrained (EfficientNet-B0), các phép biến đổi ảnh (transforms) |
| 4 | NumPy | ≥ 1.24.0 | Tính toán ma trận, xử lý mảng số |
| 5 | Pandas | ≥ 2.0.0 | Quản lý dữ liệu dạng bảng (DataFrame), xuất CSV |
| 6 | Scikit-learn | ≥ 1.3.0 | Đánh giá mô hình (classification_report, confusion_matrix) |
| 7 | Pillow (PIL) | ≥ 10.0.0 | Đọc và xử lý ảnh đầu vào |
| 8 | Streamlit | Latest | Xây dựng giao diện web demo tương tác |

**Kiến trúc mã nguồn** được tổ chức theo mô hình module hóa:

```
project/
├── rice/                          ← Module CNN (Bước 2)
│   ├── train_cnn.py               ← Script huấn luyện CNN
│   ├── inference_cnn.py           ← Script suy diễn CNN
│   └── requirements.txt
├── rice_fuzzy_xai/                ← Module Fuzzy + XAI (Bước 3-5)
│   ├── __init__.py                ← Export: FuzzyInput, FuzzyOutput, FuzzyEngine
│   ├── schemas.py                 ← Định nghĩa cấu trúc dữ liệu I/O
│   ├── config.py                  ← Cấu hình ánh xạ bệnh, tên tiếng Việt
│   ├── membership.py              ← Hàm thuộc tam giác và hình thang
│   ├── rules.py                   ← Cơ sở luật mờ Sugeno (20+ luật)
│   ├── engine.py                  ← Bộ suy diễn mờ chính (FuzzyEngine)
│   └── explanation.py             ← Module sinh báo cáo XAI
├── rice_disease_cnn_outputs/      ← Kết quả huấn luyện CNN
│   ├── best_model.pt              ← Trọng số tốt nhất (16MB)
│   ├── class_metadata.json        ← Metadata lớp bệnh
│   ├── training_report.txt        ← Báo cáo kết quả huấn luyện
│   ├── training_history.csv       ← Lịch sử loss/accuracy theo epoch
│   ├── confusion_matrix.png       ← Ma trận nhầm lẫn
│   └── val_cnn_scores.csv         ← Softmax scores trên tập validation
├── app_streamlit.py               ← Ứng dụng demo Streamlit
└── demo_assets/                   ← Ảnh demo minh họa
```

### 4.1.1. Chi tiết Module CNN (rice/)

**Mô hình CNN**: Sử dụng kiến trúc **EfficientNet-B0** (pretrained ImageNet, fine-tuned) với các đặc điểm kỹ thuật:
- **Kích thước ảnh đầu vào**: 224×224 pixels
- **Số lớp phân loại**: 9 lớp (1 Healthy + 4 bệnh × 2 mức độ Mild/Severe)
- **Chiến lược huấn luyện**: Transfer Learning 2 pha:
  - Pha 1 (Freeze, 5 epoch đầu): Đóng băng backbone, chỉ huấn luyện classifier head với learning rate cao (5×lr)
  - Pha 2 (Fine-tune, epoch 6-10): Mở toàn bộ backbone, huấn luyện end-to-end với CosineAnnealingLR
- **Loss function**: CrossEntropyLoss với **Label Smoothing = 0.1** — đặc biệt quan trọng vì giúp phân phối softmax không bị cực đoan (0.99/0.01), phù hợp làm đầu vào cho tầng Fuzzy
- **Optimizer**: AdamW (weight_decay = 1e-4)
- **Data Augmentation**: RandomCrop, RandomFlip, RandomRotation(30°), ColorJitter, RandomErasing (giả lập che khuất lá)
- **AMP (Automatic Mixed Precision)**: Tự động bật trên GPU để tăng tốc ~2x

**9 lớp phân loại**:

| STT | Tên lớp (English) | Tên tiếng Việt | Nhóm bệnh |
|-----|-------------------|----------------|------------|
| 0 | Healthy | Khỏe mạnh | — |
| 1 | Mild Bacterial blight | Bạc lá nhẹ | Bacterial blight |
| 2 | Mild Blast | Đạo ôn nhẹ | Blast |
| 3 | Mild Brownspot | Đốm nâu nhẹ | Brownspot |
| 4 | Mild Tungro | Tungro nhẹ | Tungro |
| 5 | Severe Bacterial blight | Bạc lá nặng | Bacterial blight |
| 6 | Severe Blast | Đạo ôn nặng | Blast |
| 7 | Severe Brownspot | Đốm nâu nặng | Brownspot |
| 8 | Severe Tungro | Tungro nặng | Tungro |

### 4.1.2. Chi tiết Module Fuzzy + XAI (rice_fuzzy_xai/)

Bộ suy diễn mờ được thiết kế theo mô hình **Sugeno bậc 0** với 4 nhóm biến đầu vào và 4 chỉ số đầu ra:

**Các biến đầu vào được mờ hóa:**

| Biến | Miền giá trị | Tập mờ | Dạng hàm thuộc |
|------|-------------|--------|----------------|
| CNN Confidence (Top-1) | [0, 1] | Low, Medium, High | Trapezoidal / Triangular |
| Margin (Top1 - Top2) | [0, 1] | Small, Medium, Large | Trapezoidal / Triangular |
| Temperature | [15, 45°C] | Cool, Warm, Hot | Trapezoidal / Triangular |
| Humidity | [40, 100%] | Dry, Moderate, Wet | Trapezoidal / Triangular |

**Các chỉ số đầu ra (Sugeno Singletons):**
1. **Diagnostic Confidence** (Độ tin cậy chẩn đoán): 25% / 60% / 85% / 95%
2. **Visual Severity Index - VSI** (Chỉ số nghiêm trọng): 0% / 20% / 55% / 90%
3. **Environmental Risk Index - ERI** (Nguy cơ môi trường): 15% / 50% / 85%
8. **Final Alert Index - FAI** (Cảnh báo cuối cùng): 10% / 40% / 70% / 95%

**Lớp Quyết định Hỗ trợ Chuyên gia (Decision Support Layer):**
Bên cạnh bộ suy diễn mờ, module tích hợp thêm một **Bộ định tuyến Chế độ Suy luận (Inference Mode Router)** để quản trị rủi ro, sử dụng các chỉ số:
- **Normalized Entropy**: Đo lường sự bất định (Out-of-Distribution). Nếu $Entropy > 0.6$, hệ thống đánh giá CNN đang phân vân.
- **Grouped Disease Confidence**: Gom nhóm điểm số của các lớp cùng loại bệnh (vd: Mild Blast + Severe Blast) để chốt bệnh ưu tiên, chống lại hiện tượng pha loãng xác suất.
- **Expert Signal (Tín hiệu thực địa)**: Ví dụ như mật độ ốc bươu vàng, đóng vai trò phủ quyết.
Từ đó, định tuyến luồng ra 1 trong 3 chế độ:
- `AI_CONFIDENT`: Tin tưởng hoàn toàn mô hình hình ảnh.
- `HYBRID_WARNING`: Kết hợp CNN và cảnh báo rủi ro bổ sung.
- `EXPERT_GUIDED_MODE`: Nhường quyền quyết định cho tín hiệu thực địa khi AI có độ bất định cao.

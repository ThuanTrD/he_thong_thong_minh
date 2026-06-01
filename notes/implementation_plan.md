# Kế hoạch phát triển Tích hợp Kịch bản chạy ảnh đầu vào (Nhánh `khanhtrang`)

Kế hoạch này đề xuất thiết kế cho tệp `scripts/run_hybrid_pipeline.py` để kết nối trực tiếp mô hình CNN hiện có với package `rice_fuzzy_xai/` mới, cho phép người dùng truyền trực tiếp 1 ảnh lá lúa để nhận kết quả Fuzzy XAI đầy đủ.

## User Review Required

> [!IMPORTANT]
> **Phê duyệt file mới:** Tôi đề xuất thiết kế tệp `scripts/run_hybrid_pipeline.py` như bên dưới. Tôi sẽ tạm dừng và chờ bạn xác nhận trước khi tạo tệp này.

## Thiết kế tệp `scripts/run_hybrid_pipeline.py`

Tệp này sẽ thực hiện luồng tích hợp end-to-end:

1. **Khởi tạo và Tham số:** Nhận đường dẫn ảnh (`--image`), checkpoint CNN (`--ckpt`), nhiệt độ (`--temp`), độ ẩm (`--humidity`) và file kết xuất (`--output`).
2. **CNN Inference (Trích xuất):**
   - Import hàm `load_trained_model`, `predict_single`, `get_inference_transform` từ module CNN cũ `rice/inference_cnn.py`.
   - Load mô hình CNN `best_model.pt` và chạy dự đoán trên ảnh lá lúa đầu vào để lấy:
     - `cnn_scores`
     - `top_class`
     - `top_confidence`
3. **Fuzzy XAI Reasoning (Suy diễn):**
   - Import `FuzzyEngine` và `FuzzyInput` từ package `rice_fuzzy_xai/`.
   - Truyền điểm số CNN cùng thông số thời tiết vào `FuzzyEngine`.
4. **Hiển thị báo cáo:** In toàn bộ báo cáo XAI giải thích và khuyến nghị hành động ra Terminal, đồng thời lưu file JSON nếu yêu cầu.

## Sơ đồ luồng hoạt động từ ảnh đầu vào

```
[Đường dẫn ảnh: la_lua_benh.jpg]
              │
              ▼
    [load_trained_model]
    [predict_single] ────────► Lấy cnn_scores, top_class, top_confidence
              │
              ▼
  [FuzzyEngine.run(FuzzyInput)]
              │
              ▼
   Báo cáo giải thích XAI & Khuyến nghị nông nghiệp
```

---

## Kế hoạch kiểm tra (Verification Plan)

### Kiểm tra tự động
Chạy thử nghiệm tích hợp trên ảnh mẫu:
`$env:PYTHONIOENCODING="utf-8"; .\venv\Scripts\python scripts/run_hybrid_pipeline.py --ckpt rice_disease_cnn_outputs/best_model.pt --image test_dummy.jpg --temp 28 --humidity 90`

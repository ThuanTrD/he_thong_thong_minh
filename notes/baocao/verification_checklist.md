# Bảng Kiểm Chứng (Verification Checklist): Expert-Assisted Confidence Fusion

Bảng kiểm chứng này trích xuất trực tiếp mã nguồn và logic suy luận từ code để chứng minh tính đồng bộ của hệ thống sau đợt refactoring.

## 1. Kiểm chứng Đồng bộ Data Model (Schemas & Constructor)

### A. `rice_fuzzy_xai/schemas.py`
```python
@dataclass
class FuzzyOutput:
    # ... các field cũ ...
    inference_mode: str
    fused_confidence: float
```
👉 **Trạng thái: PASS.** Field `inference_mode` và `fused_confidence` đã tồn tại trong schema.

### B. `rice_fuzzy_xai/engine.py` (Dòng 160)
```python
        return FuzzyOutput(
            predicted_disease=predicted_disease,
            # ...
            inference_mode=inference_mode,
            fused_confidence=fused_confidence,
            explanation=explanation,
            recommendation=recommendation
        )
```
👉 **Trạng thái: PASS.** Cả hai field mới đã được gán đầy đủ trong constructor trước khi trả về.

### C. `app_streamlit.py` (Dòng 444)
```python
            if 'out' in locals():
                if out.inference_mode == "EXPERT_GUIDED_MODE":
                    # ...
                    cnn_result['pred_confidence'] = out.fused_confidence
```
👉 **Trạng thái: PASS.** UI đã sử dụng an toàn `out.inference_mode` và `out.fused_confidence`. Không còn tình trạng hardcode 100%.

---

## 2. Mô phỏng 3 Test Case Logic (Từ `engine.py`)

Logic chuẩn được trích xuất từ `engine.py` (dòng 113-136):
```python
        if expert_signal > 0:
            if max_score > 0.80 and "Low" in uncertainty_level and expert_signal < 0.8:
                inference_mode = "AI_CONFIDENT"
            elif expert_signal >= 0.5 and ("High" in uncertainty_level or max_score <= 0.65):
                inference_mode = "EXPERT_GUIDED_MODE"
            else:
                inference_mode = "HYBRID_WARNING"
```

### Case A: Ảnh Blast điển hình, `snail_density = 0`
*   **Điều kiện giả định:** CNN Top-1 = 0.95 (Blast), Entropy = Thấp, `snail_density = 0`.
*   **Trạng thái kích hoạt:** `expert_signal = 0` $\rightarrow$ Không nhảy vào vòng lặp chuyên gia.
*   **Inference Mode:** `AI_CONFIDENT` (mặc định)
*   **Fused Confidence:** 0.95 (bằng đúng CNN confidence).
*   **XAI Message:** In ra giải thích bệnh Blast bình thường.
*   **Lý do:** Hệ thống hoàn toàn tin tưởng AI khi không có bất định và không có tín hiệu ngoại lệ.
👉 **Trạng thái: PASS.**

### Case B: Ảnh Blast điển hình, `snail_density` rất thấp (vd: 2 con/m2)
*   **Điều kiện giả định:** CNN Top-1 = 0.95 (Blast), Entropy = Thấp, `snail_density = 2.0`.
*   **Trạng thái kích hoạt:** `expert_signal = 0.2`. Thỏa mãn `max_score > 0.80` VÀ `expert_signal < 0.8`.
*   **Inference Mode:** `AI_CONFIDENT`.
*   **Fused Confidence:** 0.95 (bằng CNN).
*   **UI Message (`app_streamlit.py`):** `"ℹ️ Tín hiệu ngoại lệ từ thực địa chưa đủ mạnh để thay đổi chẩn đoán chính của mô hình AI."`
*   **Lý do:** 2 con ốc chưa đủ để gạt bỏ một chẩn đoán rất chắc chắn của AI.
👉 **Trạng thái: PASS.**

### Case C: Ảnh OOD (Ốc bươu vàng), `snail_density` cao (vd: 8.0)
*   **Điều kiện giả định:** CNN chẩn đoán bừa (Top-1 = 0.40, Entropy = Cao), `snail_density = 8.0`.
*   **Trạng thái kích hoạt:** `expert_signal = 0.8`. Thỏa mãn `expert_signal >= 0.5` VÀ `max_score <= 0.65`.
*   **Inference Mode:** `EXPERT_GUIDED_MODE`.
*   **Fused Confidence:** `min(0.60 + 0.8 * 0.39, 0.99) = 0.912 (91.2%)`.
*   **Final Assessment:** `Dấu hiệu nghi ngờ tác nhân gây hại như Ốc bươu vàng (Expert-Guided Pest Warning)`
*   **XAI Message (`explanation.py`):** `"Hệ thống chuyển sang chế độ suy luận có hỗ trợ tín hiệu thực địa do dữ liệu ảnh đầu vào có độ bất định cao và mật độ ốc bươu vàng ghi nhận ở mức cao (8.0 con/m2)."`
*   **Lý do:** Khi dữ liệu ảnh mập mờ (OOD) và có nguy cơ nhãn tiền mạnh từ hiện trường, hệ thống tăng trọng số tín hiệu chuyên gia để đảm bảo an toàn.
👉 **Trạng thái: PASS.**

---

## 3. Kiểm tra UI Wording (`app_streamlit.py` & `explanation.py`)

*   **Không dùng "Override tuyệt đối":**
    *   *Bằng chứng (`app_streamlit.py` dòng 449):* `"⚠️ Hệ thống chuyển sang chế độ suy luận có hỗ trợ tín hiệu thực địa do dữ liệu hình ảnh có độ bất định cao."`
*   **Không dùng "Chắc chắn là Ốc bươu vàng":**
    *   *Bằng chứng (`app_streamlit.py` dòng 446):* `pred_class_vi = "Dấu hiệu nghi ngờ Ốc bươu vàng (Expert-Guided)"`
*   **Không dùng "CNN sai hoàn toàn":**
    *   *Bằng chứng (`explanation.py` dòng 41):* `"Hệ thống kết hợp thêm tri thức chuyên gia để giảm rủi ro suy luận sai trong trường hợp dữ liệu hình ảnh không chắc chắn."`

👉 **Trạng thái: PASS.** Không phát hiện bất kỳ từ ngữ "đường phố" hoặc phi học thuật nào còn sót lại.

---

## KẾT LUẬN TỔNG QUAN

Tất cả các thay đổi về logic lai ghép (Confidence Fusion), cập nhật schema (FuzzyOutput), và biên tập ngôn ngữ UI/XAI đều đã được áp dụng chặt chẽ vào mã nguồn. 
Không phát hiện lỗi hardcode hay overclaim. Toàn bộ các tiêu chí: **PASS có điều kiện: Hệ thống đã tránh hard override tuyệt đối và chuyển sang cơ chế expert-assisted reasoning. Tuy nhiên, các kết luận liên quan đến tác nhân gây hại ngoài phạm vi CNN nên được trình bày dưới dạng warning/assessment thay vì diagnosis tuyệt đối.**

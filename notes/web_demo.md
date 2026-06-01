Hãy tạo một Streamlit demo app cho branch `khanhtrang`.

Mục tiêu:
Cho phép người dùng upload ảnh lá lúa, nhập optional temperature/humidity, sau đó chạy CNN inference + Fuzzy/XAI và hiển thị kết quả trực quan.

Yêu cầu file:

* Tạo file mới: `app_streamlit.py`
* Không sửa logic CNN/Fuzzy core nếu không cần thiết.
* Không thay đổi branch `main`.

Chức năng app:

1. Upload ảnh `.jpg/.jpeg/.png`
2. Hiển thị ảnh đầu vào
3. Chạy CNN inference bằng checkpoint:

   * `rice_disease_cnn_outputs/best_model.pt`
4. Hiển thị:

   * predicted class
   * top confidence
   * bar chart CNN softmax scores
5. Cho phép nhập optional:

   * temperature, mặc định 28°C
   * humidity, mặc định 85%
6. Truyền CNN JSON sang package `rice_fuzzy_xai`
7. Hiển thị Fuzzy/XAI output:

   * predicted_disease
   * visual_severity_level
   * diagnostic_confidence
   * uncertainty_level
   * environmental_risk_level
   * final_alert_level
   * explanation
   * recommendation
8. Tạo layout trực quan:

   * cột trái: ảnh + input môi trường
   * cột phải: kết quả CNN + Fuzzy
   * phía dưới: XAI report chi tiết
9. Thêm nút export kết quả JSON.
10. App chạy bằng:
    `streamlit run app_streamlit.py`

Nguyên tắc:

* App chỉ là demo UI, không thay đổi model.
* Nếu thiếu checkpoint thì báo lỗi rõ ràng.
* Nếu ảnh upload không hợp lệ thì báo lỗi thân thiện.
* Không scan filesystem.
* Không tự tìm dataset.
* Commit trên branch `khanhtrang`.
* Trước khi sửa/tạo file, hiển thị kế hoạch ngắn và chờ tôi duyệt.

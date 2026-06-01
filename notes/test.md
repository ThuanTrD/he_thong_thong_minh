Tôi muốn thay `test_dummy.jpg` bằng một ảnh lá lúa bị bệnh thật từ dataset để demo thuyết phục hơn.

Yêu cầu:

1. Không dùng ảnh dummy nữa trong báo cáo/demo chính.
2. Hãy chọn một ảnh thật từ dataset thuộc một trong các class:

   * Mild Blast
   * Severe Blast
   * Mild Brownspot
   * Severe Brownspot
3. Ưu tiên ảnh mà CNN dự đoán đúng hoặc có confidence trung bình-cao để output dễ giải thích.
4. Copy ảnh demo đó vào thư mục riêng:

   * `demo_assets/`
5. Đặt tên ảnh rõ nghĩa, ví dụ:

   * `demo_mild_blast.jpg`
   * `demo_severe_brownspot.jpg`
6. Chạy lại CNN inference trên ảnh demo thật đó.
7. Sau đó truyền JSON CNN output sang package `rice_fuzzy_xai`.
8. Xuất kết quả demo gồm:

   * ảnh đầu vào
   * CNN scores
   * Fuzzy/XAI output
   * recommendation
9. Không quét toàn bộ ổ đĩa; chỉ tìm ảnh trong `dataset_path` tôi cung cấp.
10. Mọi thay đổi thực hiện trên branch `khanhtrang`.
11. Trước khi copy ảnh hoặc sửa file, hiển thị kế hoạch và chờ tôi duyệt.

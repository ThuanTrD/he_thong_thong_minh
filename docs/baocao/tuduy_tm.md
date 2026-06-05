# TƯ DUY THUYẾT MINH: XỬ LÝ NGOẠI LỆ KHI AI NHẬN DIỆN SAI (OUT-OF-DISTRIBUTION)

Dựa vào tình huống thực tế đang hiển thị trên màn hình (ảnh trứng ốc bươu vàng bị mô hình nhận diện nhầm thành bệnh Tungro nặng), đây là một "kịch bản vàng" (golden scenario) để bạn thuyết trình trước hội đồng về sự kết hợp giữa Deep Learning và Hệ chuyên gia (Fuzzy Logic). 

Dưới đây là kịch bản và tư duy thuyết minh chi tiết:

## 1. Nêu vấn đề (Trình bày giới hạn của Deep Learning)
*   **Thao tác trên UI:** Upload bức ảnh có ổ trứng ốc bươu vàng màu hồng (`1649839...o_vang.jpg`), giữ nguyên thanh trượt Mật độ ốc ở mức `0.0`.
*   **Lời thoại:** 
    > *"Thưa hội đồng, như quý vị đang thấy trên màn hình, khi chúng ta đưa vào một bức ảnh chứa ổ trứng của ốc bươu vàng, mạng nơ-ron CNN ngay lập tức đưa ra dự đoán sai lệch là **'Tungro nặng'** với độ tự tin lên tới **77.40%**."*
*   **Phân tích kỹ thuật (Tại sao lại sai?):**
    > *"Lý do cốt lõi là vì mô hình AI của chúng ta bị giới hạn bởi tập dữ liệu huấn luyện (Training Data). Mô hình chỉ được học 4 loại bệnh nấm/vi sinh (Đạo ôn, Bạc lá, Đốm nâu, Tungro). Trứng ốc bươu vàng nằm ngoài phân phối dữ liệu học (Out-of-Distribution). Trong trường hợp này, CNN đã cố gắng 'ép' hình ảnh lạ này vào một trong các nhãn mà nó biết. Sự nhầm lẫn này vô cùng nguy hiểm nếu hệ thống chỉ có một lớp AI duy nhất, vì nó sẽ khuyên nông dân đi mua thuốc trừ rầy (để trị Tungro) thay vì thuốc trị ốc."*

## 2. Giải pháp (Sức mạnh của Hybrid System & Expert Exception)
*   **Thao tác trên UI:** Kéo thanh trượt **"Mật độ Ốc bươu vàng (con/m²)"** ở góc trái lên mức `4.0` (Mức nguy hiểm).
*   **Lời thoại:**
    > *"Để khắc phục triệt để 'điểm mù' này của các mô hình học sâu truyền thống, nhóm chúng em đã thiết kế hệ thống theo kiến trúc lai (Hybrid). Trong đó, Hệ chuyên gia mờ (Fuzzy Expert System) đóng vai trò kiểm soát cuối cùng. Chúng em tích hợp tính năng **'Ngoại lệ chuyên gia' (Expert Exception)**."*
    > 
    > *"Khi người nông dân hoặc chuyên viên quan sát thấy ốc/trứng ốc thực tế trên ruộng và kéo thanh mật độ ốc lên mức 4 con/m² (vượt ngưỡng 3 con/m² theo chuẩn của Cục BVTV), lập tức có hai điều xảy ra:"*
*   **Nhấn mạnh sự thay đổi trên màn hình (Sau khi kéo slider):**
    1.  **Tính năng Vượt quyền (Bypass):** Hệ logic mờ sẽ **hủy bỏ/bỏ qua** hoàn toàn kết quả chẩn đoán sai lệch (Tungro) của CNN.
    2.  **Cảnh báo Đỏ (Red Alert):** Mức cảnh báo được đẩy lên tối đa. Báo cáo XAI sẽ chuyển hướng, tuyên bố đang xử lý ngoại lệ Ốc bươu vàng. Lời khuyên cũng lập tức thay đổi: Yêu cầu sử dụng thuốc *Metaldehyde* hoặc *Niclosamide* đúng chuẩn.

## 3. Tổng kết giá trị khoa học và thực tiễn
*   **Lời thoại:**
    > *"Thiết kế này chứng minh rằng: AI không nhất thiết phải hoàn hảo 100% trong mọi trường hợp. Bằng cách kết hợp Trí tuệ nhân tạo (CNN) để xử lý các bài toán tự động hóa ở tầng nhận thức, và Trí tuệ con người/Chuyên gia (Fuzzy Logic Rules) ở tầng ra quyết định, chúng ta tạo ra một **Hệ thống Hỗ trợ Ra Quyết Định (DSS) thực dụng, linh hoạt và an toàn hơn rất nhiều**. Đây chính là cốt lõi của nông nghiệp thông minh: Công nghệ phục vụ con người, nhưng con người vẫn giữ quyền làm chủ."*

---
**💡 Mẹo thuyết trình:** 
Cố tình để thanh ốc ở mức 0 lúc đầu để hội đồng thấy mô hình AI đang bị "ngáo". Sau đó bạn cười nhẹ và kéo thanh ốc lên, mọi thứ thay đổi 180 độ. Đó sẽ là khoảnh khắc cực kỳ ấn tượng (WOW moment) cho bài bảo vệ của bạn!

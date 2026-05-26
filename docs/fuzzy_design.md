# Tài Liệu Thiết Kế Hệ Suy Diễn Mờ & XAI (Fuzzy Logic & XAI Design)

Tài liệu này mô tả chi tiết kiến trúc, các hàm liên thuộc (Membership Functions), hệ quy tắc (Rules) và giải thuật giải mờ (Defuzzification) của package `rice_fuzzy_xai`. Thiết kế này phục vụ trực tiếp cho báo cáo học tập/bài tập nhóm của bạn.

---

## 1. Kiến Trúc Tổng Quan (Pipeline Hybrid CNN + Fuzzy)

Hệ thống kết hợp sức mạnh trích xuất đặc trưng hình ảnh của CNN với khả năng xử lý độ bất định và suy luận theo luật của Fuzzy Logic:

```
[Ảnh Đầu Vào] + [Nhiệt Độ, Độ Ẩm]
      │
      ▼ (Bước 1: CNN Baseline)
  Softmax Probability Scores ──► Tính toán: Margin, Mild Score, Severe Score
                                      │
                                      ▼ (Bước 2: Fuzzification)
                               Các tập mờ đầu vào (Low, Medium, High...)
                                      │
                                      ▼ (Bước 3: Fuzzy Inference Engine)
                               Đánh giá Hệ Luật Mờ Sugeno
                                      │
                                      ▼ (Bước 4: Defuzzification)
                           Weighted Average (Trung bình trọng số rõ)
                                      │
                                      ▼ (Bước 5: XAI & Output)
                       Chỉ số DSI, Cảnh báo cảnh ruộng, Khuyến nghị nông nghiệp
```

---

## 2. Các Biến Đầu Vào và Hàm Liên Thuộc (Fuzzy Inputs & MFs)

### 2.1. Độ tự tin dự đoán ($Confidence$)
- **Miền giá trị:** $[0.0, 1.0]$
- **Các tập mờ:**
  - `Low`: Hình thang $[0.0, 0.0, 0.25, 0.5]$
  - `Medium`: Hình tam giác $[0.3, 0.5, 0.7]$
  - `High`: Hình thang $[0.5, 0.75, 1.0, 1.0]$

### 2.2. Khoảng cách phân biệt ($Margin$)
Được tính bằng: $Margin = Score_{Top1} - Score_{Top2}$
- **Miền giá trị:** $[0.0, 1.0]$
- **Các tập mờ:**
  - `Small` (Bất định cao): Hình thang $[0.0, 0.0, 0.15, 0.35]$
  - `Medium`: Hình tam giác $[0.2, 0.4, 0.6]$
  - `Large` (Bất định thấp): Hình thang $[0.45, 0.7, 1.0, 1.0]$

### 2.3. Nhiệt độ môi trường ($Temperature$)
- **Miền giá trị:** $[15.0, 45.0]^\circ\text{C}$
- **Các tập mờ:**
  - `Cool` (Lạnh): Hình thang $[15.0, 15.0, 20.0, 24.0]$
  - `Warm` (Ấm): Hình tam giác $[20.0, 27.0, 34.0]$ (Tối ưu cho nấm phát triển)
  - `Hot` (Nóng): Hình thang $[30.0, 36.0, 45.0, 45.0]$

### 2.4. Độ ẩm không khí ($Humidity$)
- **Miền giá trị:** $[40\%, 100\%]$
- **Các tập mờ:**
  - `Dry` (Khô): Hình thang $[40.0, 40.0, 55.0, 65.0]$
  - `Moderate` (Vừa): Hình tam giác $[55.0, 70.0, 85.0]$
  - `Wet` (Ẩm ướt): Hình thang $[75.0, 85.0, 100.0, 100.0]$ (Tối ưu cho bào tử nảy mầm)

---

## 3. Hệ Luật Mờ Sugeno (Sugeno Fuzzy Rule Base)

Hệ thống sử dụng các hằng số đầu ra (Singletons) cho từng đầu ra mục tiêu:

### Hằng số đầu ra (Singletons)
- **Độ tin cậy chẩn đoán:** `Low` = 25, `Medium` = 60, `High` = 85, `VeryHigh` = 95
- **Mức độ nghiêm trọng trực quan (VSI):** `Healthy` = 0, `Mild` = 20, `Moderate` = 55, `Severe` = 90
- **Nguy cơ môi trường (ERI):** `Low` = 15, `Medium` = 50, `High` = 85
- **Cảnh báo cuối cùng (FAI):** `Normal` = 10, `Attention` = 40, `Danger` = 70, `RedAlert` = 95

### Các quy tắc điển hình (Fuzzy Rules)
1. **Luật Bất định:**
   - NẾU $Margin$ là `Small` THÌ Độ bất định là `High`.
   - NẾU $Margin$ là `Large` THÌ Độ bất định là `Low`.
2. **Luật Mức độ nghiêm trọng:**
   - NẾU $C_{severe}$ là `High` THÌ Mức độ nghiêm trọng là `Severe` ($90\%$).
   - NẾU $C_{severe}$ là `Low` VÀ $C_{mild}$ là `High` THÌ Mức độ nghiêm trọng là `Mild` ($20\%$).
3. **Luật Nguy cơ thời tiết:**
   - NẾU $Temperature$ là `Warm` VÀ $Humidity$ là `Wet` THÌ Nguy cơ môi trường là `High` ($85\%$).
   - NẾU $Humidity$ là `Dry` THÌ Nguy cơ môi trường là `Low` ($15\%$).
4. **Luật Cảnh báo kết hợp:**
   - NẾU Mức độ nghiêm trọng là `Severe` VÀ Nguy cơ môi trường là `High` THÌ Cảnh báo là `Red Alert` ($95\%$).
   - NẾU Mức độ nghiêm trọng là `Healthy` THÌ Cảnh báo là `Normal` ($10\%$).

---

## 4. Giải Mờ (Defuzzification)

Giải mờ được thực hiện bằng phương thức tính trung bình trọng số (Weighted Average):

$$Output_{crisp} = \frac{\sum_{i=1}^{N} w_i \cdot z_i}{\sum_{i=1}^{N} w_i}$$

Trong đó:
- $w_i$ là độ kích hoạt của luật $i$ (tính bằng hàm $\min$ hoặc $\max$ của các điều kiện đầu vào).
- $z_i$ là hằng số singleton tương ứng của luật $i$.

Chỉ số đầu ra rõ sau đó được ánh xạ ngược lại nhãn ngôn ngữ tự nhiên phục vụ chẩn đoán trực quan và XAI.

# Báo cáo Kiểm tra & Sửa lỗi

> Sau merge nhánh `khanhtrang` (`91bbe1c` → `a45033b`)  
> **Kết quả cuối: 34/34 tests PASS ✅**

---

## Vấn đề 1: Tuple Mismatch — `rules.py` vs `explanation.py`

### 🔴 LỖI THỰC SỰ — ĐÃ SỬA

**Root cause:**

| Vị trí | Format tuple | Đúng/Sai |
|--------|-------------|----------|
| Các rule sinh động trong `rules.py` | `(w, z, rule_txt)` | ✅ |
| **Hardcoded fallback returns (6 chỗ)** | `(rule_txt, w, z)` | ❌ SAI |
| `explanation.py` unpack | `for w, z, rule_txt in rules` | ✅ |

**Lỗi xảy ra khi nào?** Khi rơi vào fallback (không có rule nào kích hoạt):
- `evaluate_visual_severity(is_healthy=True)` → hardcoded tuple
- `evaluate_environmental_risk(has_env=False)` → không có T/H
- Bất kỳ `evaluate_*` nào khi `rules = []`

**Error message thực tế:**
```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
# explanation.py dòng 67: sum(w for w, z, txt in eri_rules)
# vì w thực ra là str (rule_txt bị nhầm vị trí)
```

**Fix:** [`rules.py`](file:///T:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/5.BaitapNhom/code/rice_fuzzy_xai/rules.py) — sửa 6 hardcoded tuples:

```diff
- fired_rules.append(("NẾU Margin Lớn THÌ Bất định Thấp", w_low, "Low"))
+ fired_rules.append((w_low, "Low", "NẾU Margin Lớn THÌ Bất định Thấp"))

- return 0.0, [("NẾU Nhận dạng Lành mạnh...", 1.0, SUGENO_SEVERITY["Healthy"])]
+ return 0.0, [(1.0, SUGENO_SEVERITY["Healthy"], "NẾU Nhận dạng Lành mạnh...")]

# + 4 fallback tương tự trong evaluate_visual_severity, evaluate_environmental_risk, evaluate_final_alert
```

---

## Vấn đề 2: EXPERT_GUIDED_MODE ghi đè CNN

### ✅ Logic đúng — Không có bug runtime

**Sơ đồ quyết định (đã xác minh với 20 test cases):**

```
expert_signal = min(snail_density / 10.0, 1.0)

snail = 0 / None → AI_CONFIDENT (CNN quyết định hoàn toàn)

snail > 0:
  ├─ [AI_CONFIDENT]: max_score > 0.80 VÀ Low uncertainty VÀ expert_signal < 0.8
  │    → CNN rất tự tin, ốc chưa đủ mạnh, giữ nguyên kết quả CNN
  │
  ├─ [EXPERT_GUIDED_MODE]: expert_signal >= 0.5 VÀ (High uncertainty HOẶC max_score <= 0.65)
  │    → snail phải >= 5.0 (để expert_signal >= 0.5)
  │    → GHI ĐÈ predicted_disease = "Golden Apple Snail"
  │    → snail > 3: Red Alert + Severe  (luôn đúng vì snail >= 5)
  │    → snail <= 3: DEAD CODE (không thể đạt khi snail >= 5)
  │
  └─ [HYBRID_WARNING]: else
       → CNN giữ nguyên, fused = 0.7*CNN + 0.3*expert, alert nâng lên Attention
```

**Boundary conditions đã test:**

| snail | max_score | uncertainty | Mode | Ghi đè? |
|-------|-----------|-------------|------|---------|
| 0 | 0.70 | any | AI_CONFIDENT | Không |
| 5 | 0.90 | Low | AI_CONFIDENT | Không (CNN quá mạnh) |
| 5 | 0.42 | High | EXPERT_GUIDED | ✅ → Golden Apple Snail |
| 5 | 0.65 | any | EXPERT_GUIDED | ✅ (đúng ngưỡng) |
| 4.9 | 0.42 | High | HYBRID_WARNING | Không (dưới ngưỡng 5) |
| 3 | 0.70 | Low | HYBRID_WARNING | Không |
| 100 | 0.42 | High | EXPERT_GUIDED | ✅ cap tại 0.99 |

### ⚠️ Phát hiện: Dead Code trong EXPERT_GUIDED_MODE

```python
# engine.py dòng 125-130
if inp.snail_density > 3:      # ← luôn True khi snail >= 5
    final_alert_level = "Red Alert"
else:
    # ← DEAD CODE: không bao giờ chạy được
    final_alert_level = "Attention"
```

Nhánh `else` không bao giờ chạy vì để vào `EXPERT_GUIDED_MODE` cần `snail >= 5 > 3`.  
Không phải lỗi runtime, nhưng có thể cần xem xét lại ý đồ thiết kế.

---

## Kết quả Test

```
Ran 34 tests in 0.006s — OK ✅

  TestTupleOrder (9):        format (w,z,txt) nhất quán toàn bộ
  TestExpertGuidedMode (20): tất cả boundary & edge cases
  TestFuzzyEngine (5):       regression — code cũ không bị ảnh hưởng
```

---

## Files thay đổi

| File | Thay đổi |
|------|----------|
| [rules.py](file:///T:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/5.BaitapNhom/code/rice_fuzzy_xai/rules.py) | **Bug fix**: 6 hardcoded tuples `(txt,w,z)` → `(w,z,txt)` |
| [test_tuple_and_expert_mode.py](file:///T:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/5.BaitapNhom/code/tests/test_tuple_and_expert_mode.py) | **Mới**: 29 test cases |

"""
=============================================================================
EVALUATE FUZZY SEVERITY ASSESSMENT — Đánh giá định lượng hệ mờ
=============================================================================
Mục tiêu: Tính accuracy/F1 của Fuzzy Engine trên bộ data_test/

Pipeline:
  Ảnh (data_test/<label>/*.jpg)
    → CNN predict_single()
    → FuzzyEngine.run()
    → So sánh visual_severity_level với ground truth từ tên thư mục

Ground truth:
  - Thư mục "Healthy"                → "Healthy"
  - Thư mục "Mild <disease>"         → "Mild"
  - Thư mục "Severe <disease>"       → "Severe"

Output:
  - Bảng kết quả chi tiết (CSV)
  - Confusion matrix (Healthy / Mild / Moderate / Severe)
  - Accuracy, Per-class Precision/Recall/F1
  - Alert level distribution
=============================================================================
"""
import sys
import os
import io
import time
from pathlib import Path
from collections import defaultdict, Counter

# Fix Unicode output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import pandas as pd
import numpy as np

# Setup paths
project_root = str(Path(__file__).parent.parent)   # scripts/ → project root
rice_dir     = os.path.join(project_root, "rice")
for p in [project_root, rice_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference_cnn import load_trained_model, get_inference_transform, predict_single
from rice_fuzzy_xai import FuzzyEngine, FuzzyInput

# ─────────────────────────────────────────────
#  CẤU HÌNH
# ─────────────────────────────────────────────
CKPT_PATH   = os.path.join(project_root, "rice_disease_cnn_outputs", "best_model.pt")
DATA_DIR    = os.path.join(project_root, "data_test")
OUTPUT_CSV  = os.path.join(project_root, "notes", "evaluation_results.csv")

# Thông số môi trường mặc định (trung tính)
DEFAULT_TEMP     = 28.0   # °C
DEFAULT_HUMIDITY = 75.0   # %

# Ánh xạ tên thư mục → ground truth severity
def get_gt_severity(folder_name: str) -> str:
    """Trích xuất ground truth severity từ tên thư mục."""
    fn = folder_name.lower()
    if fn == "healthy":
        return "Healthy"
    elif fn.startswith("mild"):
        return "Mild"
    elif fn.startswith("severe"):
        return "Severe"
    else:
        return "Unknown"

# Ánh xạ tên thư mục → ground truth disease
def get_gt_disease(folder_name: str) -> str:
    """Trích xuất ground truth disease từ tên thư mục."""
    fn = folder_name
    if fn == "Healthy":
        return "Healthy"
    # Remove "Mild " / "Severe " prefix
    for prefix in ["Mild ", "Severe "]:
        if fn.startswith(prefix):
            return fn[len(prefix):]
    return fn

# Chuẩn hóa output của Fuzzy về 3 nhóm chính để so sánh
def normalize_fuzzy_severity(vsi_level: str) -> str:
    """
    Gom Moderate về Mild để đơn giản hóa phân tích
    (Moderate là trung gian giữa Mild và Severe).
    Trả về: Healthy / Mild / Severe
    """
    lvl = vsi_level.lower()
    if "healthy" in lvl:
        return "Healthy"
    elif "severe" in lvl:
        return "Severe"
    else:
        return "Mild"  # Mild + Moderate gộp chung


# ─────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  FUZZY SEVERITY ASSESSMENT EVALUATION")
    print("=" * 65)

    # Load CNN model
    print(f"\n[1/4] Load CNN model từ: {CKPT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")
    model, classes, img_size = load_trained_model(CKPT_PATH, device)
    transform = get_inference_transform(img_size)

    # Init Fuzzy Engine
    engine = FuzzyEngine()

    # Scan data_test/
    print(f"\n[2/4] Scan dữ liệu từ: {DATA_DIR}")
    data_dir = Path(DATA_DIR)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    folders = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"      Tìm thấy {len(folders)} thư mục lớp:")
    for f in sorted(folders):
        n_imgs = len([x for x in f.iterdir() if x.suffix.lower() in image_exts])
        gt = get_gt_severity(f.name)
        print(f"        {f.name:<30} → GT={gt:<8} ({n_imgs} ảnh)")

    # Evaluation loop
    print(f"\n[3/4] Chạy CNN + Fuzzy cho từng ảnh...")
    rows = []
    total = 0
    t_start = time.time()

    for folder in sorted(folders):
        gt_severity = get_gt_severity(folder.name)
        gt_disease  = get_gt_disease(folder.name)
        if gt_severity == "Unknown":
            continue

        img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in image_exts])

        for img_path in img_paths:
            total += 1
            try:
                # --- Bước 1: CNN Inference ---
                cnn_res = predict_single(str(img_path), model, classes, transform, device)

                # --- Bước 2: Fuzzy Inference ---
                inp = FuzzyInput(
                    cnn_scores     = cnn_res["scores"],
                    top_class      = cnn_res["pred_class"],
                    top_confidence = cnn_res["pred_confidence"],
                    temperature    = DEFAULT_TEMP,
                    humidity       = DEFAULT_HUMIDITY,
                    snail_density  = 0.0
                )
                out = engine.run(inp)

                # Normalize để so sánh
                fuzzy_sev_norm = normalize_fuzzy_severity(out.visual_severity_level)
                gt_sev_norm    = gt_severity  # Healthy / Mild / Severe

                is_correct_sev  = (fuzzy_sev_norm == gt_sev_norm)
                is_correct_dis  = (out.predicted_disease == gt_disease)

                rows.append({
                    "image"             : img_path.name,
                    "folder"            : folder.name,
                    "gt_disease"        : gt_disease,
                    "gt_severity"       : gt_sev_norm,
                    "cnn_pred_class"    : cnn_res["pred_class"],
                    "cnn_confidence"    : round(cnn_res["pred_confidence"], 4),
                    "fuzzy_disease"     : out.predicted_disease,
                    "fuzzy_vsi_raw"     : out.visual_severity_level,
                    "fuzzy_severity"    : fuzzy_sev_norm,
                    "fuzzy_alert"       : out.final_alert_level,
                    "uncertainty"       : out.uncertainty_level,
                    "diag_confidence"   : round(out.diagnostic_confidence, 2),
                    "correct_severity"  : is_correct_sev,
                    "correct_disease"   : is_correct_dis,
                    "inference_mode"    : out.inference_mode,
                })

            except Exception as e:
                print(f"  ⚠ Lỗi: {img_path.name} → {e}")

            if total % 50 == 0:
                elapsed = time.time() - t_start
                print(f"     {total} ảnh đã xử lý... ({elapsed:.1f}s)")

    elapsed_total = time.time() - t_start
    print(f"\n  ✓ Hoàn thành: {total} ảnh trong {elapsed_total:.1f}s")

    # ─────────────────────────────────────────
    #  PHÂN TÍCH KẾT QUẢ
    # ─────────────────────────────────────────
    print(f"\n[4/4] Phân tích và in kết quả...")
    df = pd.DataFrame(rows)

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Kết quả chi tiết đã lưu: {OUTPUT_CSV}")

    # ── ACCURACY TỔNG QUAN ──
    n_total  = len(df)
    n_sev_ok = df["correct_severity"].sum()
    n_dis_ok = df["correct_disease"].sum()
    acc_sev  = n_sev_ok / n_total * 100
    acc_dis  = n_dis_ok / n_total * 100

    print("\n" + "=" * 65)
    print("  KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN")
    print("=" * 65)
    print(f"  Tổng số ảnh đánh giá          : {n_total}")
    print(f"  ✅ Severity Accuracy (Fuzzy)   : {acc_sev:.2f}%  ({n_sev_ok}/{n_total})")
    print(f"  ✅ Disease Accuracy (Fuzzy)    : {acc_dis:.2f}%  ({n_dis_ok}/{n_total})")

    # ── CONFUSION MATRIX SEVERITY ──
    print("\n" + "─" * 65)
    print("  CONFUSION MATRIX — Fuzzy Severity (GT rows × Predicted cols)")
    print("─" * 65)
    labels = ["Healthy", "Mild", "Severe"]
    conf = pd.crosstab(
        df["gt_severity"],
        df["fuzzy_severity"],
        rownames=["GT\\Pred"],
        colnames=[""]
    ).reindex(index=labels, columns=labels, fill_value=0)
    print(conf.to_string())

    # ── PER-CLASS METRICS ──
    print("\n" + "─" * 65)
    print("  PER-CLASS METRICS — Severity Assessment")
    print("─" * 65)
    print(f"  {'Class':<10} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    f1_list = []
    for cls in labels:
        tp = ((df["gt_severity"] == cls) & (df["fuzzy_severity"] == cls)).sum()
        fp = ((df["gt_severity"] != cls) & (df["fuzzy_severity"] == cls)).sum()
        fn = ((df["gt_severity"] == cls) & (df["fuzzy_severity"] != cls)).sum()
        support = (df["gt_severity"] == cls).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)

        print(f"  {cls:<10} {support:>8} {precision:>9.1%} {recall:>8.1%} {f1:>8.1%}")

    macro_f1 = np.mean(f1_list)
    print(f"\n  {'Macro F1':<10}            {macro_f1:>27.1%}")

    # ── PHÂN TÍCH THEO BỆNH ──
    print("\n" + "─" * 65)
    print("  ACCURACY THEO TỪNG BỆNH")
    print("─" * 65)
    print(f"  {'Disease':<22} {'N':>5} {'Sev.Acc':>9} {'Dis.Acc':>9} {'Avg Conf':>10}")
    print(f"  {'─'*22} {'─'*5} {'─'*9} {'─'*9} {'─'*10}")
    for disease in sorted(df["gt_disease"].unique()):
        sub = df[df["gt_disease"] == disease]
        n   = len(sub)
        sa  = sub["correct_severity"].mean() * 100
        da  = sub["correct_disease"].mean() * 100
        avg_conf = sub["diag_confidence"].mean()
        print(f"  {disease:<22} {n:>5} {sa:>8.1f}% {da:>8.1f}% {avg_conf:>9.1f}%")

    # ── PHÂN BỐ ALERT LEVEL ──
    print("\n" + "─" * 65)
    print("  PHÂN BỐ ALERT LEVEL (Fuzzy Output)")
    print("─" * 65)
    alert_counts = df["fuzzy_alert"].value_counts()
    for alert, count in alert_counts.items():
        pct  = count / n_total * 100
        bar  = "█" * int(pct / 2)
        print(f"  {alert:<35} {count:>4} ({pct:>5.1f}%)  {bar}")

    # ── PHÂN BỐ UNCERTAINTY ──
    print("\n" + "─" * 65)
    print("  PHÂN BỐ UNCERTAINTY LEVEL")
    print("─" * 65)
    unc_counts = df["uncertainty"].value_counts()
    for unc, count in unc_counts.items():
        pct  = count / n_total * 100
        bar  = "█" * int(pct / 2)
        print(f"  {unc:<25} {count:>4} ({pct:>5.1f}%)  {bar}")

    # ── PHÂN TÍCH LỖI ──
    print("\n" + "─" * 65)
    print("  PHÂN TÍCH LỖI — Top sai lầm phổ biến (Severity)")
    print("─" * 65)
    wrong = df[~df["correct_severity"]]
    if len(wrong) > 0:
        error_pairs = wrong.groupby(["gt_severity", "fuzzy_severity"]).size().sort_values(ascending=False)
        for (gt, pred), count in error_pairs.items():
            pct = count / n_total * 100
            print(f"  GT={gt:<8} → Predicted={pred:<8} : {count:>4} lần ({pct:.1f}%)")
    else:
        print("  Không có lỗi! Accuracy 100%")

    # ── TÓM TẮT CHO BÁO CÁO ──
    print("\n" + "=" * 65)
    print("  TÓM TẮT ĐỂ TRÌNH BÀY TRONG BÁO CÁO")
    print("=" * 65)
    print(f"""
  Hệ thống được đánh giá trên {n_total} ảnh trong bộ data_test.

  📊 KẾT QUẢ CHÍNH:
  ┌─────────────────────────────────────┬──────────────┐
  │ Chỉ số                              │ Kết quả      │
  ├─────────────────────────────────────┼──────────────┤
  │ Fuzzy Severity Accuracy             │ {acc_sev:>6.2f}%      │
  │ Fuzzy Disease Accuracy              │ {acc_dis:>6.2f}%      │
  │ Macro F1-score (Severity)           │ {macro_f1*100:>6.2f}%      │
  │ Tổng số ảnh kiểm tra                │ {n_total:>6}       │
  └─────────────────────────────────────┴──────────────┘

  💾 File chi tiết: {OUTPUT_CSV}
""")


if __name__ == "__main__":
    main()

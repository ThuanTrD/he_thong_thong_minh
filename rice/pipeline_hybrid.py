"""
=============================================================================
HYBRID CNN + FUZZY PIPELINE — SYSTEM INTEGRATION SCRIPT
=============================================================================
Kịch bản chạy tích hợp toàn bộ pipeline:
  [Ảnh đầu vào] + [Thời tiết]
       │
       ▼ (Bước 2)
  [CNN Inference] ──> Softmax Scores
                           │
                           ▼ (Bước 3 & 4)
                    [Fuzzy Reasoning] ──> Disease Severity Index (DSI) + XAI
=============================================================================
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

# Thêm thư mục hiện tại (rice) vào path để import
sys.path.append(str(Path(__file__).parent))

from inference_cnn import load_trained_model, get_inference_transform, predict_single
from fuzzy_inference import RiceFuzzySystem


def get_args():
    parser = argparse.ArgumentParser(description="Hybrid CNN + Fuzzy Pipeline for Rice Disease")
    parser.add_argument("--ckpt",     type=str, required=True,
                        help="Path to trained CNN .pt weight file")
    parser.add_argument("--image",    type=str, required=True,
                        help="Path to input rice leaf image")
    parser.add_argument("--temp",     type=float, default=25.0,
                        help="Current ambient temperature in °C (15 - 45)")
    parser.add_argument("--humidity", type=float, default=70.0,
                        help="Current relative humidity in % (40 - 100)")
    parser.add_argument("--output",   type=str, default=None,
                        help="Path to save output JSON report")
    return parser.parse_args()


def main():
    args = get_args()

    # 1. Khởi tạo device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Kiểm tra các tệp đầu vào có tồn tại hay không
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Không tìm thấy file trọng số CNN: {args.ckpt}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"[ERROR] Không tìm thấy ảnh đầu vào: {args.image}")
        sys.exit(1)

    print("\n" + "="*70)
    print("  BẮT ĐẦU CHẠY PIPELINE LAI (HYBRID CNN + FUZZY PIPELINE)")
    print("="*70)
    print(f"  Ảnh đầu vào  : {args.image}")
    print(f"  Trọng số CNN  : {args.ckpt}")
    print(f"  Môi trường    : Nhiệt độ = {args.temp}°C | Độ ẩm = {args.humidity}%")
    print("-"*70)

    # 2. Tải mô hình CNN
    print("[PIPELINE] Bước 1: Đang tải mô hình CNN...")
    try:
        model, classes, img_size = load_trained_model(args.ckpt, device)
        transform = get_inference_transform(img_size)
        print(f"  ✓ Tải mô hình thành công. Nhận dạng được {len(classes)} lớp bệnh.")
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải mô hình: {e}")
        sys.exit(1)

    # 3. Chạy CNN Inference
    print("[PIPELINE] Bước 2: Đang phân tích hình ảnh (CNN Inference)...")
    try:
        cnn_result = predict_single(args.image, model, classes, transform, device)
        print(f"  ✓ Dự đoán CNN: {cnn_result['pred_class']} (Độ tự tin: {cnn_result['pred_confidence']:.4f})")
    except Exception as e:
        print(f"[ERROR] Lỗi khi chạy inference hình ảnh: {e}")
        sys.exit(1)

    # 4. Chạy suy diễn mờ (Fuzzy Logic) & Giải thích XAI
    print("[PIPELINE] Bước 3: Đang tính toán suy diễn mờ (Fuzzy Reasoning)...")
    try:
        fuzzy_sys = RiceFuzzySystem()
        hybrid_result = fuzzy_sys.evaluate_rules(
            c_scores=cnn_result["scores"],
            temp=args.temp,
            humidity=args.humidity,
            predicted_class=cnn_result["pred_class"]
        )
        print("  ✓ Suy diễn mờ hoàn tất.")
    except Exception as e:
        print(f"[ERROR] Lỗi khi thực hiện suy diễn mờ: {e}")
        sys.exit(1)

    # 5. Hiển thị báo cáo chi tiết
    print("\n" + "="*70)
    print("  KẾT QUẢ PHÂN TÍCH TỔNG HỢP (HYBRID DIAGNOSIS REPORT)")
    print("="*70)
    print(hybrid_result["xai_report"])
    print("="*70 + "\n")

    # 6. Xuất báo cáo JSON nếu được yêu cầu
    output_data = {
        "input_image": args.image,
        "environment": {
            "temperature_C": args.temp,
            "humidity_percent": args.humidity
        },
        "cnn_outputs": {
            "prediction": cnn_result["pred_class"],
            "confidence": cnn_result["pred_confidence"],
            "all_scores": cnn_result["scores"]
        },
        "fuzzy_outputs": {
            "disease_severity_index_percent": hybrid_result["dsi"],
            "severity_class": hybrid_result["severity_class"]
        }
    }

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"[PIPELINE] Báo cáo kết quả chi tiết đã được xuất ra: {args.output}")
        except Exception as e:
            print(f"[WARNING] Không thể ghi file kết quả: {e}")


if __name__ == "__main__":
    main()

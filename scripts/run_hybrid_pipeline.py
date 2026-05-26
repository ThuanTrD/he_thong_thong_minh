import os
import sys
import json
import argparse
import torch
from pathlib import Path

# Thêm thư mục gốc vào path để import các package
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Thêm thư mục rice vào path để import inference_cnn
sys.path.append(os.path.join(project_root, "rice"))

from inference_cnn import load_trained_model, get_inference_transform, predict_single
from rice_fuzzy_xai import FuzzyEngine, FuzzyInput, FuzzyOutput


def get_args():
    parser = argparse.ArgumentParser(description="Run End-to-End Hybrid CNN + Fuzzy Pipeline")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Đường dẫn đến file trọng số CNN (.pt)")
    parser.add_argument("--image", type=str, required=True,
                        help="Đường dẫn đến ảnh lá lúa đầu vào")
    parser.add_argument("--temp", type=float, default=None,
                        help="Nhiệt độ môi trường (°C) - Tùy chọn")
    parser.add_argument("--humidity", type=float, default=None,
                        help="Độ ẩm không khí (%) - Tùy chọn")
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn lưu kết quả JSON - Tùy chọn")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Kiểm tra sự tồn tại của tệp tin
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Không tìm thấy tệp trọng số CNN: {args.ckpt}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"[ERROR] Không tìm thấy ảnh lá lúa đầu vào: {args.image}")
        sys.exit(1)

    print("\n" + "="*75)
    print("  BẮT ĐẦU CHẠY PIPELINE LAI END-TO-END (IMAGE -> CNN -> FUZZY XAI)")
    print("="*75)
    print(f"  [Đầu vào 1] Ảnh lá lúa    : {args.image}")
    print(f"  [Đầu vào 2] Nhiệt độ      : {args.temp if args.temp is not None else 'Không nhập (Mặc định)'}")
    print(f"  [Đầu vào 3] Độ ẩm         : {args.humidity if args.humidity is not None else 'Không nhập (Mặc định)'}")
    print("-"*75)

    # 1. Chạy CNN Inference
    print("[LUỒNG CHẠY] Bước 1: Đang tải mô hình CNN...")
    try:
        model, classes, img_size = load_trained_model(args.ckpt, device)
        transform = get_inference_transform(img_size)
    except Exception as e:
        print(f"[ERROR] Không thể tải mô hình CNN: {e}")
        sys.exit(1)

    print("[LUỒNG CHẠY] Bước 2: Đang nhận dạng hình ảnh qua mô hình CNN...")
    try:
        cnn_result = predict_single(args.image, model, classes, transform, device)
        print(f"  ✓ Nhận diện từ CNN: {cnn_result['pred_class']} (Độ tự tin: {cnn_result['pred_confidence']*100:.2f}%)")
    except Exception as e:
        print(f"[ERROR] Lỗi khi nhận dạng ảnh lá lúa: {e}")
        sys.exit(1)

    # 2. Tạo FuzzyInput và chạy FuzzyEngine
    print("[LUỒNG CHẠY] Bước 3: Đang thực hiện suy diễn mờ và sinh giải thích XAI...")
    inp = FuzzyInput(
        cnn_scores=cnn_result["scores"],
        top_class=cnn_result["pred_class"],
        top_confidence=cnn_result["pred_confidence"],
        temperature=args.temp,
        humidity=args.humidity
    )
    
    engine = FuzzyEngine()
    out: FuzzyOutput = engine.run(inp)
    print("  ✓ Hoàn tất suy diễn mờ.")

    # 3. Hiển thị báo cáo trực quan ra màn hình
    print("\n" + "="*75)
    print("  BÁO CÁO PHÂN TÍCH TỔNG HỢP & GIẢI THÍCH (HYBRID REPORT & XAI)")
    print("="*75)
    print(out.explanation)
    print("\nKhuyến nghị hành động nông nghiệp (Agricultural Recommendations):")
    print(out.recommendation)
    print("="*75 + "\n")

    # 4. Lưu kết quả ra file JSON
    if args.output:
        out_dict = {
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
                "predicted_disease": out.predicted_disease,
                "visual_severity_level": out.visual_severity_level,
                "diagnostic_confidence_percent": out.diagnostic_confidence,
                "uncertainty_level": out.uncertainty_level,
                "environmental_risk_level": out.environmental_risk_level,
                "final_alert_level": out.final_alert_level,
                "explanation": out.explanation,
                "recommendation": out.recommendation
            }
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_dict, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Đã ghi báo cáo tích hợp ra file: {args.output}")
        except Exception as e:
            print(f"[WARNING] Không thể ghi file báo cáo: {e}")


if __name__ == "__main__":
    main()

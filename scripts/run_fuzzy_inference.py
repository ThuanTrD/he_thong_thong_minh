import os
import sys
import json
import argparse
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path để import package
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from rice_fuzzy_xai import FuzzyEngine, FuzzyInput, FuzzyOutput

def get_args():
    parser = argparse.ArgumentParser(description="Run Rice Disease Fuzzy Inference & XAI")
    
    # Cách 1: Truyền trực tiếp qua tham số dòng lệnh (CLI)
    parser.add_argument("--top_class", type=str, default=None,
                        help="Tên lớp dự đoán chính của CNN (ví dụ: 'Mild Blast')")
    parser.add_argument("--top_confidence", type=float, default=None,
                        help="Độ tự tin của lớp chính (0.0 - 1.0)")
    parser.add_argument("--second_class", type=str, default=None,
                        help="Tên lớp dự đoán lớn thứ hai (ví dụ: 'Severe Blast')")
    parser.add_argument("--second_confidence", type=float, default=0.0,
                        help="Độ tự tin của lớp thứ hai")
    
    # Cách 2: Đọc từ tệp kết quả JSON của CNN
    parser.add_argument("--cnn_json", type=str, default=None,
                        help="Đường dẫn đến file JSON đầu ra của CNN inference")
                        
    # Các tham số môi trường
    parser.add_argument("--temp", type=float, default=None,
                        help="Nhiệt độ thực tế ngoài ruộng lúa (°C)")
    parser.add_argument("--humidity", type=float, default=None,
                        help="Độ ẩm không khí thực tế (%)")
                        
    parser.add_argument("--output", type=str, default=None,
                        help="Đường dẫn lưu tệp JSON kết quả lai")
                        
    return parser.parse_args()


def main():
    args = get_args()
    engine = FuzzyEngine()

    cnn_scores = {}
    top_class = ""
    top_confidence = 0.0

    # 1. Thu thập dữ liệu đầu vào CNN
    if args.cnn_json:
        if not os.path.exists(args.cnn_json):
            print(f"[ERROR] Không tìm thấy tệp JSON: {args.cnn_json}")
            sys.exit(1)
        try:
            with open(args.cnn_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Xử lý các định dạng JSON khác nhau có thể có
            if "scores" in data:  # Định dạng từ predict_single
                cnn_scores = data["scores"]
                top_class = data["pred_class"]
                top_confidence = data["pred_confidence"]
            elif "cnn_scores" in data:  # Định dạng từ get_fuzzy_input
                cnn_scores = data["cnn_scores"]
                top_class = data["top_class"]
                top_confidence = data["top_confidence"]
            else:
                print("[ERROR] Tệp JSON không đúng định dạng đầu ra của CNN.")
                sys.exit(1)
            print(f"[INFO] Đọc dữ liệu thành công từ file JSON: {args.cnn_json}")
        except Exception as e:
            print(f"[ERROR] Lỗi khi đọc file JSON: {e}")
            sys.exit(1)
            
    else:
        # Sử dụng các tham số CLI truyền trực tiếp
        if not args.top_class or args.top_confidence is None:
            print("[ERROR] Bạn cần cung cấp (--top_class và --top_confidence) HOẶC --cnn_json để chạy.")
            sys.exit(1)
            
        top_class = args.top_class
        top_confidence = args.top_confidence
        
        # Mô phỏng một dict cnn_scores đơn giản dựa vào tham số đầu vào
        cnn_scores = {
            "Healthy": 0.0,
            "Mild Bacterial blight": 0.0, "Severe Bacterial blight": 0.0,
            "Mild Blast": 0.0, "Severe Blast": 0.0,
            "Mild Brownspot": 0.0, "Severe Brownspot": 0.0,
            "Mild Tungro": 0.0, "Severe Tungro": 0.0
        }
        
        # Gán giá trị
        if top_class in cnn_scores:
            cnn_scores[top_class] = top_confidence
        else:
            cnn_scores[top_class] = top_confidence
            
        if args.second_class:
            cnn_scores[args.second_class] = args.second_confidence

    # 2. Xây dựng FuzzyInput
    inp = FuzzyInput(
        cnn_scores=cnn_scores,
        top_class=top_class,
        top_confidence=top_confidence,
        temperature=args.temp,
        humidity=args.humidity
    )

    # 3. Chạy suy diễn mờ (Fuzzy Inference)
    print("\n" + "="*70)
    print("  CHẠY BỘ SUY DIỄN MỜ RICE_FUZZY_XAI")
    print("="*70)
    out: FuzzyOutput = engine.run(inp)

    # 4. Hiển thị báo cáo giải thích & Khuyến nghị
    print(out.explanation)
    print("\nKhuyến nghị hành động nông nghiệp (Agricultural Recommendations):")
    print(out.recommendation)
    print("="*70 + "\n")

    # 5. Lưu báo cáo JSON
    if args.output:
        out_dict = {
            "predicted_disease": out.predicted_disease,
            "visual_severity_level": out.visual_severity_level,
            "diagnostic_confidence_percent": out.diagnostic_confidence,
            "uncertainty_level": out.uncertainty_level,
            "environmental_risk_level": out.environmental_risk_level,
            "final_alert_level": out.final_alert_level,
            "explanation": out.explanation,
            "recommendation": out.recommendation
        }
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_dict, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Đã xuất báo cáo mờ ra file: {args.output}")
        except Exception as e:
            print(f"[WARNING] Không thể ghi file báo cáo: {e}")


if __name__ == "__main__":
    main()

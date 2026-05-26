import unittest
import sys
from pathlib import Path

# Thêm thư mục gốc vào path để import
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from rice_fuzzy_xai import FuzzyEngine, FuzzyInput, FuzzyOutput
from rice_fuzzy_xai.membership import trimf, trapmf, fuzzify_confidence, fuzzify_margin


class TestFuzzyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FuzzyEngine()

    def test_membership_functions(self):
        # Test hàm liên thuộc tam giác
        self.assertEqual(trimf(0.5, 0.3, 0.5, 0.7), 1.0)
        self.assertEqual(trimf(0.2, 0.3, 0.5, 0.7), 0.0)
        self.assertAlmostEqual(trimf(0.4, 0.3, 0.5, 0.7), 0.5)

        # Test hàm liên thuộc hình thang
        self.assertEqual(trapmf(0.1, 0.0, 0.0, 0.25, 0.5), 1.0)
        self.assertEqual(trapmf(0.6, 0.0, 0.0, 0.25, 0.5), 0.0)
        self.assertAlmostEqual(trapmf(0.375, 0.0, 0.0, 0.25, 0.5), 0.5)

    def test_fuzzification(self):
        # Mờ hóa độ tự tin 0.5
        c_fuzzy = fuzzify_confidence(0.5)
        self.assertAlmostEqual(c_fuzzy["Medium"], 1.0)
        self.assertAlmostEqual(c_fuzzy["Low"], 0.0)
        self.assertAlmostEqual(c_fuzzy["High"], 0.0)

        # Mờ hóa margin 0.1 (Small margin -> Small membership should be > 0)
        m_fuzzy = fuzzify_margin(0.1)
        self.assertGreater(m_fuzzy["Small"], 0.0)
        self.assertEqual(m_fuzzy["Large"], 0.0)

    def test_engine_healthy(self):
        # Trường hợp cây lúa hoàn toàn khỏe mạnh
        inp = FuzzyInput(
            cnn_scores={
                "Healthy": 0.95,
                "Mild Blast": 0.01,
                "Severe Blast": 0.01,
                "Mild Brownspot": 0.01,
                "Severe Brownspot": 0.01,
                "Mild Bacterial blight": 0.01,
                "Severe Bacterial blight": 0.00,
                "Mild Tungro": 0.00,
                "Severe Tungro": 0.00
            },
            top_class="Healthy",
            top_confidence=0.95,
            temperature=25.0,
            humidity=70.0
        )
        
        out = self.engine.run(inp)
        self.assertEqual(out.predicted_disease, "Healthy")
        self.assertEqual(out.visual_severity_level, "Healthy (Lành mạnh)")
        self.assertEqual(out.final_alert_level, "Normal (Bình thường)")

    def test_engine_disease_high_risk(self):
        # Bệnh nhẹ nhưng môi trường nóng ẩm thuận lợi bùng phát dịch
        inp = FuzzyInput(
            cnn_scores={
                "Healthy": 0.05,
                "Mild Blast": 0.70,
                "Severe Blast": 0.05,
                "Mild Brownspot": 0.10,
                "Severe Brownspot": 0.05,
                "Mild Bacterial blight": 0.05,
                "Severe Bacterial blight": 0.00,
                "Mild Tungro": 0.00,
                "Severe Tungro": 0.00
            },
            top_class="Mild Blast",
            top_confidence=0.70,
            temperature=28.0,   # Ấm (Warm)
            humidity=90.0       # Ẩm ướt (Wet)
        )
        
        out = self.engine.run(inp)
        self.assertEqual(out.predicted_disease, "Blast")
        self.assertEqual(out.visual_severity_level, "Mild (Nhẹ)")
        self.assertEqual(out.environmental_risk_level, "High (Cao)")
        # Cảnh báo cuối cùng bị đẩy lên do thời tiết nóng ẩm nguy cơ cao
        self.assertEqual(out.final_alert_level, "Attention (Chú ý)")

    def test_engine_high_uncertainty(self):
        # Độ tự tin giữa lớp 1 và lớp 2 rất gần nhau (Margin nhỏ -> Bất định Cao)
        inp = FuzzyInput(
            cnn_scores={
                "Healthy": 0.02,
                "Mild Blast": 0.42,
                "Severe Blast": 0.40,
                "Mild Brownspot": 0.08,
                "Severe Brownspot": 0.08,
                "Mild Bacterial blight": 0.00,
                "Severe Bacterial blight": 0.00,
                "Mild Tungro": 0.00,
                "Severe Tungro": 0.00
            },
            top_class="Mild Blast",
            top_confidence=0.42,
            temperature=22.0,
            humidity=60.0
        )
        
        out = self.engine.run(inp)
        self.assertEqual(out.uncertainty_level, "High (Cao)")
        self.assertLess(out.diagnostic_confidence, 50.0)


if __name__ == "__main__":
    unittest.main()

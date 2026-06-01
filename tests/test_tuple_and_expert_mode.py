"""
=============================================================================
TEST SUITE: Kiểm tra 2 vấn đề nghi ngờ sau khi merge nhánh khanhtrang
=============================================================================
VẤN ĐỀ 1: Thứ tự tuple rule (explanation.py vs rules.py) có khớp không?
VẤN ĐỀ 2: EXPERT_GUIDED_MODE ghi đè CNN - kiểm tra tất cả edge cases
=============================================================================
"""
import unittest
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rice_fuzzy_xai import FuzzyEngine, FuzzyInput, FuzzyOutput
from rice_fuzzy_xai.rules import (
    evaluate_uncertainty,
    evaluate_visual_severity,
    evaluate_environmental_risk,
    evaluate_final_alert
)
from rice_fuzzy_xai.membership import fuzzify_confidence, fuzzify_margin, fuzzify_temp, fuzzify_humidity


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
FULL_SCORES_HEALTHY = {
    "Healthy": 0.95, "Mild Blast": 0.01, "Severe Blast": 0.00,
    "Mild Brownspot": 0.01, "Severe Brownspot": 0.00,
    "Mild Bacterial blight": 0.01, "Severe Bacterial blight": 0.00,
    "Mild Tungro": 0.01, "Severe Tungro": 0.01,
}

FULL_SCORES_UNCERTAIN = {
    "Healthy": 0.02, "Mild Blast": 0.42, "Severe Blast": 0.40,
    "Mild Brownspot": 0.08, "Severe Brownspot": 0.08,
    "Mild Bacterial blight": 0.00, "Severe Bacterial blight": 0.00,
    "Mild Tungro": 0.00, "Severe Tungro": 0.00,
}

FULL_SCORES_MILD_BLAST = {
    "Healthy": 0.05, "Mild Blast": 0.70, "Severe Blast": 0.05,
    "Mild Brownspot": 0.10, "Severe Brownspot": 0.05,
    "Mild Bacterial blight": 0.05, "Severe Bacterial blight": 0.00,
    "Mild Tungro": 0.00, "Severe Tungro": 0.00,
}

FULL_SCORES_HIGH_CONF_BLAST = {
    "Healthy": 0.01, "Mild Blast": 0.90, "Severe Blast": 0.03,
    "Mild Brownspot": 0.02, "Severe Brownspot": 0.01,
    "Mild Bacterial blight": 0.01, "Severe Bacterial blight": 0.01,
    "Mild Tungro": 0.01, "Severe Tungro": 0.00,
}


# ═════════════════════════════════════════════════════════════
# VẤN ĐỀ 1: Kiểm tra thứ tự tuple rule
# ═════════════════════════════════════════════════════════════
class TestTupleOrder(unittest.TestCase):
    """
    Mục tiêu: Xác minh thứ tự (w, z, rule_txt) trong rules.py khớp 100%
    với cách explanation.py unpack: `for w, z, rule_txt in vsi_rules`.
    """

    def _assert_tuple_format(self, rules, label):
        """Kiểm tra từng tuple (w, z, txt): w∈[0,1], z∈[0,100], txt là str."""
        self.assertIsInstance(rules, list, f"[{label}] Phải là list")
        for i, item in enumerate(rules):
            self.assertEqual(len(item), 3,
                f"[{label}] Tuple #{i} phải có đúng 3 phần tử, nhận: {item}")
            w, z, txt = item
            self.assertIsInstance(w, (int, float),
                f"[{label}] Phần tử 0 (w) phải là số, nhận: {type(w).__name__} = {w!r}")
            self.assertIsInstance(z, (int, float),
                f"[{label}] Phần tử 1 (z) phải là số, nhận: {type(z).__name__} = {z!r}")
            self.assertIsInstance(txt, str,
                f"[{label}] Phần tử 2 (txt) phải là str, nhận: {type(txt).__name__} = {txt!r}")
            self.assertGreaterEqual(w, 0.0, f"[{label}] w phải >= 0")
            self.assertLessEqual(w, 1.0,    f"[{label}] w phải <= 1")
            self.assertGreaterEqual(z, 0.0, f"[{label}] z (singleton) phải >= 0")

    def test_vsi_rules_format(self):
        """evaluate_visual_severity → vsi_rules có đúng format (w, z, txt)?"""
        mild_fuzzy  = fuzzify_confidence(0.70)
        severe_fuzzy = fuzzify_confidence(0.10)
        _, rules = evaluate_visual_severity(mild_fuzzy, severe_fuzzy, is_healthy=False)
        self._assert_tuple_format(rules, "vsi_rules - bệnh có mild cao")

    def test_vsi_rules_format_healthy(self):
        """evaluate_visual_severity (healthy=True) → format đúng?"""
        _, rules = evaluate_visual_severity({}, {}, is_healthy=True)
        self._assert_tuple_format(rules, "vsi_rules - healthy")

    def test_eri_rules_format_with_env(self):
        """evaluate_environmental_risk (có env) → eri_rules đúng format?"""
        temp_fuzzy = fuzzify_temp(28.0)
        hum_fuzzy  = fuzzify_humidity(85.0)
        _, rules = evaluate_environmental_risk(temp_fuzzy, hum_fuzzy, has_env=True)
        self._assert_tuple_format(rules, "eri_rules - nóng ẩm")

    def test_eri_rules_format_no_env(self):
        """evaluate_environmental_risk (không có env) → format mặc định đúng?"""
        _, rules = evaluate_environmental_risk({}, {}, has_env=False)
        self._assert_tuple_format(rules, "eri_rules - không có env")

    def test_alert_rules_format(self):
        """evaluate_final_alert → alert_rules đúng format?"""
        _, rules = evaluate_final_alert(vsi=60.0, eri=75.0)
        self._assert_tuple_format(rules, "alert_rules - vsi Moderate, eri High")

    def test_uncertainty_rules_format(self):
        """evaluate_uncertainty → fired_rules đã đổi sang format (w, z, txt) đồng bộ.
        
        Lưu ý: Sau khi sửa rules.py, uncertainty_rules cũng dùng (w, z, txt)
        nhưng uncertainty_rules KHÔNG được truyền vào generate_explanation.
        Kiểm tra format thực tế sau khi sửa.
        """
        margin_fuzzy = fuzzify_margin(0.05)  # margin nhỏ -> High uncertainty
        level, rules = evaluate_uncertainty(margin_fuzzy)
        for i, item in enumerate(rules):
            self.assertEqual(len(item), 3, f"uncertainty_rules tuple #{i} phải 3 phần tử")
            w, z, txt = item   # Format mới: (w, z, txt) - đồng bộ với các rules khác
            self.assertIsInstance(w,   (int, float), f"uncertainty item[0] phải là float (weight)")
            self.assertIsInstance(z,   (int, str),   f"uncertainty item[1] phải là singleton")
            self.assertIsInstance(txt, str,          f"uncertainty item[2] phải là str (rule text)")
            self.assertGreaterEqual(w, 0.0, f"w phải >= 0")
            self.assertLessEqual(w,   1.0,  f"w phải <= 1")

    def test_explanation_unpacking_does_not_crash(self):
        """End-to-end: generate_explanation không crash khi unpack vsi/eri/alert rules."""
        engine = FuzzyEngine()
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,
            top_class="Mild Blast",
            top_confidence=0.70,
            temperature=28.0,
            humidity=85.0,
            snail_density=0.0
        )
        try:
            out = engine.run(inp)
            self.assertIsInstance(out.explanation, str)
            self.assertGreater(len(out.explanation), 10)
        except (ValueError, TypeError) as e:
            self.fail(f"generate_explanation crash khi unpack tuple: {e}")

    def test_sum_computation_uses_correct_fields(self):
        """
        explanation.py dòng 66-68:
          total_w_vsi = sum(w for w, z, txt in vsi_rules)
        Kiểm tra sum(w) dùng đúng phần tử [0] = weight (float ∈ [0,1])
        và KHÔNG nhầm sang z (singleton, float ∈ [0,100]) hoặc txt (str).
        """
        mild_fuzzy   = fuzzify_confidence(0.70)
        severe_fuzzy = fuzzify_confidence(0.05)
        _, vsi_rules = evaluate_visual_severity(mild_fuzzy, severe_fuzzy, is_healthy=False)

        # Nếu thứ tự sai (txt, w, z), sum sẽ crash hoặc cộng chuỗi
        try:
            total_w = sum(w for w, z, txt in vsi_rules)
            self.assertGreaterEqual(total_w, 0.0)
            self.assertLessEqual(total_w, len(vsi_rules),  # tổng weight tối đa = số luật
                "Tổng weight vượt quá số luật, có thể đang cộng singleton (z) thay vì weight (w)")
        except TypeError as e:
            self.fail(f"sum(w) crash - thứ tự tuple SAI: {e}")


# ═════════════════════════════════════════════════════════════
# VẤN ĐỀ 2: EXPERT_GUIDED_MODE - kiểm tra edge cases
# ═════════════════════════════════════════════════════════════
class TestExpertGuidedMode(unittest.TestCase):
    """
    Kiểm tra toàn bộ logic Expert-Assisted Confidence Fusion trong engine.py.
    Ba nhánh chính:
      - AI_CONFIDENT:     max_score > 0.80 và Low uncertainty và expert_signal < 0.8
      - EXPERT_GUIDED_MODE: expert_signal >= 0.5 và (High uncertainty HOẶC max_score <= 0.65)
      - HYBRID_WARNING:   còn lại khi expert_signal > 0
      - Không có snail:   expert_signal = 0 -> luôn AI_CONFIDENT
    """

    def setUp(self):
        self.engine = FuzzyEngine()

    # ── NHÁNH: Không có ốc (snail_density = 0 hoặc None) ──────────────
    def test_no_snail_always_ai_confident(self):
        """snail_density=0 → inference_mode phải là AI_CONFIDENT."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,
            top_class="Mild Blast", top_confidence=0.70,
            temperature=28.0, humidity=85.0,
            snail_density=0.0
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "AI_CONFIDENT",
            "Không có ốc → phải AI_CONFIDENT")
        self.assertEqual(out.predicted_disease, "Blast",
            "Không có ốc → CNN quyết định, không bị ghi đè")

    def test_snail_density_none_ai_confident(self):
        """snail_density=None → không có expert_signal → AI_CONFIDENT."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,
            top_class="Mild Blast", top_confidence=0.70,
            snail_density=None
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "AI_CONFIDENT")
        self.assertEqual(out.predicted_disease, "Blast")

    # ── NHÁNH: AI_CONFIDENT (CNN mạnh, ốc chưa đủ) ────────────────────
    def test_ai_confident_high_cnn_low_snail(self):
        """CNN > 0.80, Low uncertainty, expert_signal < 0.8 → AI_CONFIDENT."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_HIGH_CONF_BLAST,
            top_class="Mild Blast", top_confidence=0.90,
            temperature=25.0, humidity=60.0,
            snail_density=5.0   # expert_signal = 0.5 < 0.8 → AI_CONFIDENT
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "AI_CONFIDENT",
            "CNN rất tự tin (0.90), Low uncertainty, snail=5 < 8 → AI_CONFIDENT")
        self.assertEqual(out.predicted_disease, "Blast",
            "Không được ghi đè khi AI_CONFIDENT")
        self.assertAlmostEqual(out.fused_confidence, 0.90,
            msg="fused_confidence = max_score khi AI_CONFIDENT")

    def test_ai_confident_boundary_expert_signal_just_below_08(self):
        """expert_signal = 0.79 (snail=7.9) + CNN cao → vẫn AI_CONFIDENT."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_HIGH_CONF_BLAST,
            top_class="Mild Blast", top_confidence=0.85,
            snail_density=7.9   # expert_signal = 0.79 < 0.8
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "AI_CONFIDENT")
        self.assertEqual(out.predicted_disease, "Blast")

    # ── NHÁNH: EXPERT_GUIDED_MODE ─────────────────────────────────────
    def test_expert_guided_high_uncertainty_high_snail(self):
        """High uncertainty + snail >= 5 → EXPERT_GUIDED_MODE, ghi đè thành Golden Apple Snail."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,     # margin nhỏ → High uncertainty
            top_class="Mild Blast", top_confidence=0.42,
            temperature=25.0, humidity=60.0,
            snail_density=5.0   # expert_signal = 0.5 >= 0.5, max_score 0.42 <= 0.65
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE",
            "High uncertainty + snail mạnh → EXPERT_GUIDED_MODE")
        self.assertEqual(out.predicted_disease, "Golden Apple Snail",
            "EXPERT_GUIDED_MODE phải ghi đè predicted_disease thành Golden Apple Snail")

    def test_expert_guided_low_cnn_high_snail(self):
        """max_score <= 0.65 + snail >= 5 → EXPERT_GUIDED_MODE (bất kể uncertainty)."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,    # top_confidence=0.70 > 0.65
            top_class="Mild Blast", top_confidence=0.60,  # ghi đè xuống 0.60
            snail_density=6.0   # expert_signal=0.6 >= 0.5
        )
        # Vì max_score sẽ được lấy từ inp.top_confidence = 0.60 <= 0.65
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE")
        self.assertEqual(out.predicted_disease, "Golden Apple Snail")

    def test_expert_guided_snail_density_above_3_red_alert(self):
        """EXPERT_GUIDED_MODE + snail > 3 → final_alert_level = Red Alert."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=5.0   # > 3 → Red Alert
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE")
        self.assertIn("Red Alert", out.final_alert_level,
            "snail > 3 trong EXPERT_GUIDED_MODE → phải là Red Alert")
        self.assertEqual(out.visual_severity_level, "Severe (Nghiêm trọng)")

    def test_expert_guided_always_red_alert_and_severe(self):
        """
        Sau khi xóa dead code: EXPERT_GUIDED_MODE luôn ra Red Alert + Severe.
        Lý do: điều kiện vào mode này là snail >= 5 > 3 → nhánh else (snail <= 3) là dead code,
        đã được xóa. Mọi trường hợp EXPERT_GUIDED đều gán Red Alert trực tiếp.
        """
        for snail in [5.0, 7.0, 10.0]:
            with self.subTest(snail=snail):
                inp = FuzzyInput(
                    cnn_scores=FULL_SCORES_UNCERTAIN,
                    top_class="Mild Blast", top_confidence=0.42,
                    snail_density=snail
                )
                out = self.engine.run(inp)
                self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE")
                self.assertIn("Red Alert", out.final_alert_level,
                    f"snail={snail} trong EXPERT_GUIDED → luôn Red Alert")
                self.assertEqual(out.visual_severity_level, "Severe (Nghiêm trọng)",
                    f"snail={snail} trong EXPERT_GUIDED → luôn Severe")

        # Xác nhận snail=3 KHÔNG vào EXPERT_GUIDED (expert_signal=0.3 < 0.5)
        inp_low = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=3.0
        )
        out_low = self.engine.run(inp_low)
        self.assertNotEqual(out_low.inference_mode, "EXPERT_GUIDED_MODE",
            "snail=3 không đủ kích hoạt EXPERT_GUIDED_MODE (expert_signal=0.3 < 0.5)")

    def test_expert_guided_fused_confidence_range(self):
        """EXPERT_GUIDED_MODE: fused_confidence = 0.60 + expert_signal * 0.39 ∈ [0.60, 0.99]."""
        for snail in [5.0, 8.0, 10.0, 15.0]:  # 15 → capped at 10 → expert_signal=1.0
            with self.subTest(snail=snail):
                inp = FuzzyInput(
                    cnn_scores=FULL_SCORES_UNCERTAIN,
                    top_class="Mild Blast", top_confidence=0.42,
                    snail_density=snail
                )
                out = self.engine.run(inp)
                if out.inference_mode == "EXPERT_GUIDED_MODE":
                    self.assertGreaterEqual(out.fused_confidence, 0.60,
                        f"fused_confidence >= 0.60 khi snail={snail}")
                    self.assertLessEqual(out.fused_confidence, 0.99,
                        f"fused_confidence <= 0.99 khi snail={snail}")

    def test_expert_guided_recommendation_contains_snail_info(self):
        """EXPERT_GUIDED_MODE: recommendation phải đề cập đến ốc bươu vàng."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=5.0
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE")
        self.assertIn("ốc", out.recommendation.lower(),
            "Khuyến nghị phải đề cập đến ốc bươu vàng")

    def test_expert_guided_explanation_mentions_mode(self):
        """EXPERT_GUIDED_MODE: explanation phải đề cập Expert-Guided Mode."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=5.0
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE")
        self.assertIn("Expert-Guided", out.explanation,
            "Giải thích phải đề cập chế độ Expert-Guided Mode")

    # ── NHÁNH: HYBRID_WARNING ──────────────────────────────────────────
    def test_hybrid_warning_mid_confidence_mid_snail(self):
        """
        Trường hợp trung gian: CNN vừa phải + ốc vừa phải 
        → không thỏa AI_CONFIDENT cũng không thỏa EXPERT_GUIDED → HYBRID_WARNING.
        Điều kiện: expert_signal > 0 VÀ KHÔNG thỏa 2 nhánh trên.
        CNN=0.70 (> 0.65), uncertainty=Low, expert_signal=0.3 (< 0.5) → HYBRID
        """
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_HIGH_CONF_BLAST,
            top_class="Mild Blast", top_confidence=0.70,
            temperature=22.0, humidity=55.0,
            snail_density=3.0   # expert_signal=0.3 < 0.5 → không EXPERT_GUIDED
            # max_score=0.70 > 0.65, nên không EXPERT_GUIDED dù expert_signal > 0
            # Nhưng AI_CONFIDENT cần max_score > 0.80 → không thỏa
            # → HYBRID_WARNING
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "HYBRID_WARNING",
            f"CNN=0.70, snail=3 → HYBRID_WARNING. Nhận được: {out.inference_mode}")
        self.assertEqual(out.predicted_disease, "Blast",
            "HYBRID_WARNING không ghi đè predicted_disease")

    def test_hybrid_warning_fused_confidence_formula(self):
        """HYBRID_WARNING: fused_confidence = 0.7 * max_score + 0.3 * expert_signal."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_HIGH_CONF_BLAST,
            top_class="Mild Blast", top_confidence=0.70,
            temperature=22.0, humidity=55.0,
            snail_density=3.0
        )
        out = self.engine.run(inp)
        if out.inference_mode == "HYBRID_WARNING":
            expected_fused = 0.7 * 0.70 + 0.3 * (3.0 / 10.0)
            self.assertAlmostEqual(out.fused_confidence, expected_fused, places=5,
                msg=f"fused_confidence phải = 0.7*0.70 + 0.3*0.3 = {expected_fused:.5f}")

    def test_hybrid_warning_normal_alert_bumped_to_attention(self):
        """HYBRID_WARNING: nếu final_alert_level=Normal thì phải nâng lên Attention."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_HEALTHY,
            top_class="Healthy", top_confidence=0.75,
            temperature=22.0, humidity=55.0,
            snail_density=3.0   # expert_signal=0.3, Healthy → alert=Normal
        )
        out = self.engine.run(inp)
        if out.inference_mode == "HYBRID_WARNING":
            self.assertNotIn("Normal", out.final_alert_level,
                "HYBRID_WARNING phải nâng Normal → Attention")

    # ── EDGE CASE: Boundary conditions ────────────────────────────────
    def test_boundary_snail_exactly_5_is_expert_guided(self):
        """snail=5.0 → expert_signal=0.5 (đúng bằng threshold 0.5) + uncertain CNN."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=5.0  # expert_signal = 0.5 >= 0.5 → EXPERT_GUIDED
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE",
            "snail=5 (expert_signal=0.5 đúng bằng threshold) phải là EXPERT_GUIDED_MODE")

    def test_boundary_snail_4_9_not_expert_guided(self):
        """snail=4.9 → expert_signal=0.49 < 0.5 → không phải EXPERT_GUIDED."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=4.9  # expert_signal=0.49 < 0.5
        )
        out = self.engine.run(inp)
        self.assertNotEqual(out.inference_mode, "EXPERT_GUIDED_MODE",
            "snail=4.9 (expert_signal<0.5) không được EXPERT_GUIDED_MODE")

    def test_boundary_cnn_exactly_065_triggers_expert_guided(self):
        """max_score=0.65 đúng bằng threshold + snail>=5 → EXPERT_GUIDED."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,
            top_class="Mild Blast", top_confidence=0.65,
            snail_density=5.0
        )
        out = self.engine.run(inp)
        self.assertEqual(out.inference_mode, "EXPERT_GUIDED_MODE",
            "max_score=0.65 (đúng bằng ngưỡng <= 0.65) + snail=5 → EXPERT_GUIDED_MODE")

    def test_boundary_cnn_066_high_snail_not_expert_guided(self):
        """max_score=0.66 (> 0.65) + snail>=5 + uncertainty không cao → không EXPERT_GUIDED."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,
            top_class="Mild Blast", top_confidence=0.66,
            temperature=22.0, humidity=50.0,
            snail_density=5.0
        )
        out = self.engine.run(inp)
        # Nếu uncertainty của 0.66 không phải High → không EXPERT_GUIDED
        if "High" not in out.uncertainty_level:
            self.assertNotEqual(out.inference_mode, "EXPERT_GUIDED_MODE",
                "max_score=0.66 > 0.65 và uncertainty không cao → không EXPERT_GUIDED_MODE")

    def test_snail_density_above_10_capped(self):
        """snail_density=100 → expert_signal bị cap tại 1.0 (không vượt quá)."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_UNCERTAIN,
            top_class="Mild Blast", top_confidence=0.42,
            snail_density=100.0  # min(100/10, 1.0) = 1.0
        )
        out = self.engine.run(inp)
        # fused_confidence không được vượt quá 0.99
        self.assertLessEqual(out.fused_confidence, 0.99,
            "fused_confidence không được vượt 0.99 dù snail rất cao")

    def test_output_has_new_fields(self):
        """FuzzyOutput phải có đủ 2 field mới: inference_mode và fused_confidence."""
        inp = FuzzyInput(
            cnn_scores=FULL_SCORES_MILD_BLAST,
            top_class="Mild Blast", top_confidence=0.70,
        )
        out = self.engine.run(inp)
        self.assertTrue(hasattr(out, "inference_mode"),
            "FuzzyOutput thiếu field 'inference_mode'")
        self.assertTrue(hasattr(out, "fused_confidence"),
            "FuzzyOutput thiếu field 'fused_confidence'")
        self.assertIsInstance(out.inference_mode, str)
        self.assertIsInstance(out.fused_confidence, float)

    def test_no_snail_disease_not_overridden(self):
        """Không có ốc → predicted_disease giữ nguyên từ CNN, không bao giờ = Golden Apple Snail."""
        diseases = [
            ("Healthy", 0.95, FULL_SCORES_HEALTHY),
            ("Mild Blast", 0.70, FULL_SCORES_MILD_BLAST),
            ("Mild Blast", 0.42, FULL_SCORES_UNCERTAIN),
        ]
        for top_class, top_conf, scores in diseases:
            with self.subTest(top_class=top_class):
                inp = FuzzyInput(
                    cnn_scores=scores, top_class=top_class, top_confidence=top_conf,
                    snail_density=0.0
                )
                out = self.engine.run(inp)
                self.assertNotEqual(out.predicted_disease, "Golden Apple Snail",
                    f"Không có ốc, không được ghi đè thành Golden Apple Snail (top_class={top_class})")


if __name__ == "__main__":
    # Chạy với verbosity cao để hiển thị chi tiết từng test
    unittest.main(verbosity=2)

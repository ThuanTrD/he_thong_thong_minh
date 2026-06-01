from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class FuzzyInput:
    """Đầu vào cho bộ suy diễn mờ (Fuzzy Logic)."""
    cnn_scores: Dict[str, float]
    top_class: str
    top_confidence: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    snail_density: Optional[float] = 0.0


@dataclass
class FuzzyOutput:
    """Đầu ra từ bộ suy diễn mờ và module giải thích XAI."""
    predicted_disease: str
    visual_severity_level: str
    diagnostic_confidence: float
    uncertainty_level: str
    environmental_risk_level: str
    final_alert_level: str
    inference_mode: str
    fused_confidence: float
    explanation: str
    recommendation: str
    normalized_entropy: float = 0.0
    best_group_disease: str = ""
    best_group_confidence: float = 0.0

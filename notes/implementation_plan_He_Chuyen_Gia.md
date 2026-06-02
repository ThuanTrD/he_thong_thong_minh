# Refactor: Expert Knowledge Base Separation

This plan outlines the extraction of expert explanation and confidence assessment logic from the Streamlit UI into a dedicated `expert_system` module. This emphasizes the Hybrid Expert Decision Support System architecture by separating knowledge rules from the presentation layer.

## User Review Required
> [!IMPORTANT]
> Please review the structure of the new `expert_system` module and the proposed helper functions. Confirm if the abstraction meets the academic requirements for your report.

## Open Questions
- Do you want to include dynamic recommendation rules (e.g., mapping specific diseases to agricultural advice) in this refactor, or should we stick to the confidence and uncertainty assessment messages for now? (The plan currently implements confidence and uncertainty messages based on your example).

## Proposed Changes

### `expert_system` (New Module)
This new package will act as the Knowledge Base for the Hybrid Expert Decision Support System.

#### [NEW] [__init__.py](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/5.BaitapNhom/code/expert_system/__init__.py)
Exposes the core functions from the knowledge base.

#### [NEW] [knowledge_base.py](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/5.BaitapNhom/code/expert_system/knowledge_base.py)
Contains the dictionaries and helper functions for expert logic:
- `get_uncertainty_assessment(entropy: float) -> dict`
  Evaluates Shannon entropy and returns a structured dictionary containing text (bilingual), UI color, and background style.
- `get_expert_guided_assessment(inference_mode: str, is_ood: bool) -> dict`
  Evaluates the fuzzy inference mode and OOD flag to return the appropriate expert advice and UI styling.
- `get_confidence_category(confidence: float) -> dict`
  Categorizes the final fused confidence into HIGH, MODERATE, or LOW ranges, returning the label, description, and color.

---

### UI Layer

#### [MODIFY] [app_streamlit.py](file:///t:/BeWaters2/1.%20248.218.749-32CNTT32/5.%20Hoc%20ky%20III/16.%20HTTM/5.BaitapNhom/code/app_streamlit.py)
- Import the new `expert_system` module.
- Refactor `show_confidence_analysis()` to remove hardcoded logic and if/else conditions for colors and text.
- Replace them with calls to `get_uncertainty_assessment()`, `get_expert_guided_assessment()`, and `get_confidence_category()`.
- The UI presentation will remain visually identical to the current implementation.

## Verification Plan

### Automated Tests
- N/A (UI and expert logic refactoring).

### Manual Verification
- Launch the Streamlit application.
- Upload a sample image and click `[ 📊 Analyze Confidence ]`.
- Verify that the dialog renders identically, with correct colors, bilingual text, and accurate dynamic content based on entropy and confidence values.
- Verify that no hardcoded explanation strings remain in `app_streamlit.py`'s `show_confidence_analysis` function.

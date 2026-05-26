import os
import sys
import json
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
import streamlit as st

# Setup Path to import from project root and rice/
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "rice"))

from inference_cnn import load_trained_model, get_inference_transform, predict_single
from rice_fuzzy_xai import FuzzyEngine, FuzzyInput, FuzzyOutput

# Set Page Config
st.set_page_config(
    page_title="Rice Disease Diagnosis - CNN & Fuzzy XAI",
    page_icon="🌾",
    layout="wide"
)

# Helpers for badges
def get_alert_badge(level: str):
    lvl = level.lower()
    if "danger" in lvl or "severe" in lvl or "nguy hiểm" in lvl:
        return f'<span class="badge badge-danger">{level}</span>'
    elif "warning" in lvl or "attention" in lvl or "chú ý" in lvl or "cảnh báo" in lvl:
        return f'<span class="badge badge-warning">{level}</span>'
    elif "healthy" in lvl or "khỏe mạnh" in lvl or "safe" in lvl:
        return f'<span class="badge badge-success">{level}</span>'
    else:
        return f'<span class="badge badge-info">{level}</span>'

def get_env_badge(level: str):
    lvl = level.lower()
    if "high" in lvl or "cao" in lvl:
        return f'<span class="badge badge-danger">{level}</span>'
    elif "medium" in lvl or "trung bình" in lvl:
        return f'<span class="badge badge-warning">{level}</span>'
    else:
        return f'<span class="badge badge-success">{level}</span>'

# Custom CSS for single page viewport layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
        color: #f1f5f9;
    }
    
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ade80 0%, #10b981 50%, #064e3b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -2rem;
        margin-bottom: 0.1rem;
        filter: drop-shadow(0px 4px 8px rgba(16, 185, 129, 0.15));
    }
    
    .subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #a7f3d0;
        border-bottom: 2px solid rgba(16, 185, 129, 0.2);
        padding-bottom: 0.2rem;
        margin-bottom: 0.6rem;
    }
    
    /* Custom Badge classes */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        text-align: center;
        border: 1px solid transparent;
    }
    
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fde047;
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #a7f3d0;
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .badge-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Adjust padding to fit in viewport */
    div.stSlider {
        margin-top: -10px !important;
        margin-bottom: -10px !important;
    }
    
    div.stFileUploader {
        margin-bottom: -15px !important;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 class='main-title'>🌾 HỆ THỐNG CHẨN ĐOÁN BỆNH HẠI LÚA LAI (CNN + FUZZY XAI)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Chẩn đoán hình ảnh qua học sâu CNN kết hợp đánh giá rủi ro mờ (Fuzzy Logic) dựa trên thời tiết</p>", unsafe_allow_html=True)

# Define CNN Checkpoint Path
CKPT_PATH = os.path.join(project_root, "rice_disease_cnn_outputs", "best_model.pt")

@st.cache_resource
def get_cached_model(ckpt_path, device_str):
    device = torch.device(device_str)
    model, classes, img_size = load_trained_model(ckpt_path, device)
    transform = get_inference_transform(img_size)
    return model, classes, transform

# Layout: 3 Columns
col1, col2, col3 = st.columns([1, 1.1, 1.1], gap="medium")

with col1:
    st.markdown("<div class='section-header'>📤 INPUT & THỜI TIẾT</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Tải ảnh lá lúa (.jpg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image with constrained height
        image = Image.open(uploaded_file)
        image_thumb = image.copy()
        image_thumb.thumbnail((300, 160))
        st.image(image_thumb, caption="Ảnh đầu vào (bản xem trước)", use_container_width=False)
        
        # Save original to temp location for inference
        temp_dir = os.path.join(project_root, "demo_assets")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, "temp_upload.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.info("💡 Vui lòng tải lên một ảnh lá lúa bị bệnh.")
        
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    temp = st.slider("Nhiệt độ (°C)", min_value=15.0, max_value=45.0, value=28.0, step=0.5)
    humidity = st.slider("Độ ẩm (%)", min_value=30.0, max_value=100.0, value=85.0, step=1.0)

with col2:
    st.markdown("<div class='section-header'>🎯 PHÂN TÍCH CNN</div>", unsafe_allow_html=True)
    if uploaded_file is not None and os.path.exists(CKPT_PATH):
        try:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            model, classes, transform = get_cached_model(CKPT_PATH, device_str)
            
            # CNN prediction
            cnn_result = predict_single(temp_image_path, model, classes, transform, torch.device(device_str))
            
            # Execute Fuzzy Logic Engine (shared across cols)
            inp = FuzzyInput(
                cnn_scores=cnn_result["scores"],
                top_class=cnn_result["pred_class"],
                top_confidence=cnn_result["pred_confidence"],
                temperature=temp,
                humidity=humidity
            )
            
            engine = FuzzyEngine()
            out = engine.run(inp)
            
            class_mapping = {
                "Healthy": "Khỏe mạnh",
                "Mild Bacterial blight": "Bạc lá nhẹ (Bacterial blight)",
                "Mild Blast": "Đạo ôn nhẹ (Blast)",
                "Mild Brownspot": "Đốm nâu nhẹ (Brownspot)",
                "Mild Tungro": "Tungro nhẹ",
                "Severe Bacterial blight": "Bạc lá nặng (Bacterial blight)",
                "Severe Blast": "Đạo ôn nặng (Blast)",
                "Severe Brownspot": "Đốm nâu nặng (Brownspot)",
                "Severe Tungro": "Tungro nặng"
            }
            pred_class_vi = class_mapping.get(cnn_result["pred_class"], cnn_result["pred_class"])
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.45); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Lớp bệnh nhận diện (CNN)</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #34d399; margin-top: 0.1rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{pred_class_vi}</div>
                <div style="font-size: 0.75rem; color: #cbd5e1; margin-top: 0.2rem;">Độ tự tin: <strong style="color:#60a5fa;">{cnn_result['pred_confidence']*100:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.2rem;'>Phân phối xác suất (Softmax Scores):</p>", unsafe_allow_html=True)
            scores_df = pd.DataFrame({
                "Độ tự tin": list(cnn_result["scores"].values())
            }, index=[class_mapping.get(c, c) for c in cnn_result["scores"].keys()])
            st.bar_chart(scores_df, height=130)
            
        except Exception as e:
            st.error(f"❌ Lỗi CNN: {e}")
    else:
        st.markdown("<div style='text-align: center; color: #64748b; padding-top: 4rem; font-size: 0.9rem;'>Chờ tải ảnh để chạy phân tích...</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='section-header'>🧠 SUY DIỄN MỜ</div>", unsafe_allow_html=True)
    if uploaded_file is not None and 'out' in locals():
        alert_badge = get_alert_badge(out.final_alert_level)
        env_badge = get_env_badge(out.environmental_risk_level)
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Mức cảnh báo</div>
            <div>{alert_badge}</div>
        </div>
        
        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Nguy cơ thời tiết</div>
            <div>{env_badge}</div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Nghiêm trọng trực quan</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #f43f5e;">{out.visual_severity_level}</div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Độ tin cậy hệ thống</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">{out.diagnostic_confidence:.2f}%</div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.55rem 0.8rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Độ bất định chẩn đoán</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #cbd5e1;">{out.uncertainty_level}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center; color: #64748b; padding-top: 4rem; font-size: 0.9rem;'>Chờ chẩn đoán...</div>", unsafe_allow_html=True)

# Bottom Section: Tabs for details to fit in viewport
if uploaded_file is not None and 'out' in locals():
    st.markdown("<div style='margin: 0.3rem 0;'><hr style='margin: 0.3rem 0; opacity: 0.3;'></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔍 GIẢI THÍCH SUY DIỄN MỜ (XAI)", "🌾 KHUYẾN NGHỊ HÀNH ĐỘNG", "💾 XUẤT BÁO CÁO (JSON)"])
    
    with tab1:
        formatted_explanation = out.explanation.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 8px; padding: 0.8rem; color: #e2e8f0; font-size: 0.88rem; line-height: 1.5; max-height: 160px; overflow-y: auto;">
            {formatted_explanation}
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        formatted_recommendation = out.recommendation.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 8px; padding: 0.8rem; color: #e2e8f0; font-size: 0.88rem; line-height: 1.5; max-height: 160px; overflow-y: auto;">
            {formatted_recommendation}
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown("<p style='font-size: 0.85rem; margin-bottom: 0.4rem;'>Tải xuống báo cáo tích hợp chứa đầy đủ dữ liệu ảnh đầu vào, kết quả phân loại CNN và các tập mờ giải thích XAI:</p>", unsafe_allow_html=True)
        
        export_dict = {
            "input_image": uploaded_file.name,
            "environment": {
                "temperature_C": temp,
                "humidity_percent": humidity
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
        
        json_bytes = json.dumps(export_dict, ensure_ascii=False, indent=2).encode('utf-8')
        
        st.download_button(
            label="📥 Tải xuống báo cáo kết quả (JSON)",
            data=json_bytes,
            file_name=f"rice_diagnosis_{Path(uploaded_file.name).stem}.json",
            mime="application/json"
        )

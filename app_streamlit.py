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
from expert_system.knowledge_base import (
    get_uncertainty_assessment,
    get_expert_guided_assessment,
    get_confidence_category
)

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
        return f'<span class="badge badge-danger">🚨 {level}</span>'
    elif "warning" in lvl or "attention" in lvl or "chú ý" in lvl or "cảnh báo" in lvl:
        return f'<span class="badge badge-warning">⚠️ {level}</span>'
    elif "healthy" in lvl or "khỏe mạnh" in lvl or "safe" in lvl:
        return f'<span class="badge badge-success">✅ {level}</span>'
    else:
        return f'<span class="badge badge-info">ℹ️ {level}</span>'

def get_env_badge(level: str):
    lvl = level.lower()
    if "high" in lvl or "cao" in lvl:
        return f'<span class="badge badge-danger">🔥 {level}</span>'
    elif "medium" in lvl or "trung bình" in lvl:
        return f'<span class="badge badge-warning">⚡ {level}</span>'
    else:
        return f'<span class="badge badge-success">🌱 {level}</span>'

# Helper for parsing risk colors
def parse_risk_level(level: str):
    lvl = level.lower()
    if "high" in lvl or "severe" in lvl or "nguy hiểm" in lvl or "nặng" in lvl or "danger" in lvl:
        return "danger"
    elif "medium" in lvl or "attention" in lvl or "chú ý" in lvl or "trung bình" in lvl or "warning" in lvl:
        return "warning"
    elif "healthy" in lvl or "khỏe mạnh" in lvl or "low" in lvl or "thấp" in lvl or "safe" in lvl:
        return "success"
    else:
        return "info"

@st.dialog("🔍 Confidence Analysis")
def show_confidence_analysis(cnn_res, entropy, status, inference_mode, is_ood, class_map):
    # 1. Raw CNN Confidence
    top_class_name = class_map.get(cnn_res["pred_class"], cnn_res["pred_class"])
    
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 15px; border-radius: 8px; border: 1px solid rgba(34, 211, 238, 0.2); margin-bottom: 15px;">
        <h4 style="color: #34d399; margin-top: 0; margin-bottom: 5px;">Độ tin cậy gốc CNN (Raw CNN Confidence)</h4>
        <div style="font-size: 1.2rem; font-weight: bold; color: #f1f5f9;">{top_class_name}: <span style="color: #22d3ee;">{cnn_res['pred_confidence']*100:.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Top-5 Distribution
    st.markdown("<h4 style='color: #cbd5e1; margin-bottom: 10px; font-size: 1rem;'>Phân phối Softmax Top-5 (Top-5 Softmax Distribution)</h4>", unsafe_allow_html=True)
    top_5_scores = sorted(cnn_res["scores"].items(), key=lambda x: x[1], reverse=True)[:5]
    bars_html = ""
    for idx, (cls_name, score) in enumerate(top_5_scores):
        cls_vi = class_map.get(cls_name, cls_name)
        percent = score * 100
        if idx == 0:
            bar_color = "linear-gradient(90deg, #10b981, #34d399)"
            text_style = "color: #34d399; font-weight: 600;"
        else:
            bar_color = "linear-gradient(90deg, #22d3ee, #0d9488)"
            text_style = "color: #cbd5e1;"
            
        bars_html += f'<div style="margin-bottom: 0.6rem;"><div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;"><span style="{text_style}">{cls_vi}</span><span style="font-family: monospace; color: #94a3b8;">{percent:.2f}%</span></div><div style="background: rgba(255,255,255,0.05); height: 8px; border-radius: 4px; overflow: hidden;"><div style="background: {bar_color}; width: {percent}%; height: 100%; border-radius: 4px;"></div></div></div>'
    st.markdown(bars_html, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
    
    # 3. Uncertainty Assessment
    unc_info = get_uncertainty_assessment(entropy)
    unc_text = unc_info["text"]
    unc_color = unc_info["color"]
    unc_bg = unc_info["bg"]
        
    st.markdown(f"""
    <div style="background: {unc_bg}; border-left: 4px solid {unc_color}; padding: 10px 15px; border-radius: 4px; margin-bottom: 15px;">
        <div style="font-weight: bold; color: {unc_color}; margin-bottom: 4px;">Đánh giá Độ bất định (Uncertainty Assessment) - Entropy: {entropy:.2f}</div>
        <div style="color: #e2e8f0; font-size: 0.9rem;">{unc_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Expert-Guided Assessment
    exp_info = get_expert_guided_assessment(inference_mode, is_ood)
    exp_text = exp_info["text"]
    exp_color = exp_info["color"]
    exp_bg = exp_info["bg"]
        
    st.markdown(f"""
    <div style="background: {exp_bg}; border-left: 4px solid {exp_color}; padding: 10px 15px; border-radius: 4px; margin-bottom: 15px;">
        <div style="font-weight: bold; color: {exp_color}; margin-bottom: 4px;">Phân tích Hỗ trợ Chuyên gia (Expert-Guided Assessment)</div>
        <div style="color: #e2e8f0; font-size: 0.9rem;">{exp_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 5. Confidence Interpretation
    conf_info = get_confidence_category(cnn_res['pred_confidence'])
    conf_cat = conf_info["category"]
    conf_desc = conf_info["description"]
    conf_color = conf_info["color"]
        
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(30, 41, 59, 0.6); padding: 12px 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
        <div>
            <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Cấp độ Tự tin (Confidence Category)</div>
            <div style="color: {conf_color}; font-weight: bold; font-size: 1.1rem; line-height: 1.2;">{conf_cat}</div>
        </div>
        <div style="font-size: 1.2rem; font-weight: bold; color: #f1f5f9; background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 6px;">
            {conf_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Expandable
    with st.expander("🔬 Chỉ số Tự tin Nâng cao (Advanced Confidence Metrics)"):
        st.markdown(f"""
        <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            <li><b>Điểm Shannon Entropy (Shannon Entropy Score):</b> {entropy:.4f}</li>
            <li><b>Trạng thái nhận diện OOD (OOD Detection Status):</b> {'Có (Out-of-Distribution)' if is_ood else 'Không (In-Distribution)'}</li>
            <li><b>Độ tự tin CNN gốc (Original CNN Confidence):</b> {cnn_res['pred_confidence']*100:.2f}%</li>
            <li><b>Độ tự tin Hệ thống Tổng hợp (Fused System Confidence):</b> {cnn_res['pred_confidence']*100:.2f}% (Sau khi điều chỉnh)</li>
            <li><b>Chế độ Suy luận (Inference Mode):</b> {inference_mode}</li>
        </ul>
        """, unsafe_allow_html=True)

@st.dialog("📊 Chi tiết Logic Suy giải & Luật Mờ (Reasoning Breakdown)", width="large")
def show_reasoning_dialog(formatted_explanation, rules_fired):
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(34, 211, 238, 0.15); border-radius: 8px; padding: 1.2rem; color: #e2e8f0; font-size: 1rem; line-height: 1.6; margin-bottom: 1.5rem;">
        {formatted_explanation}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 1.05rem; font-weight: 600; color: #f59e0b; margin-bottom: 0.6rem;'>⚡ Các Luật mờ được kích hoạt (Triggered Rules):</p>", unsafe_allow_html=True)
    
    rules_html = '<div style="padding-right: 0.2rem;">'
    for r in rules_fired:
        rules_html += (
            '<div style="background: rgba(245, 158, 11, 0.06); border: 1px solid rgba(245, 158, 11, 0.22); '
            'border-radius: 6px; padding: 0.6rem 1rem; font-size: 0.95rem; color: #fef08a; display: flex; '
            'align-items: center; gap: 0.6rem; margin-bottom: 0.6rem;">'
            '<span style="background: #f59e0b; color: #020617; border-radius: 50%; width: 20px; height: 20px; '
            'display: inline-flex; justify-content: center; align-items: center; font-weight: bold; '
            'font-size: 0.8rem;">⚡</span>'
            f'<span>{r}</span>'
            '</div>'
        )
    rules_html += '</div>'
    st.markdown(rules_html, unsafe_allow_html=True)

# Custom CSS for single page viewport layout & high-tech dark theme overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* 1. Global App View Background Override */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #070b19 0%, #021a14 100%) !important;
        color: #f1f5f9 !important;
    }
    
    /* Dialog / Modal Overrides for Dark Mode */
    div[data-testid="stDialog"] > div, div[role="dialog"] > div, div[role="dialog"] {
        background: #0f172a !important;
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }
    div[role="dialog"] h2, div[role="dialog"] p, div[role="dialog"] span {
        color: #f1f5f9 !important;
    }
    div[role="dialog"] button[aria-label="Close"] {
        color: #f1f5f9 !important;
    }
    
    /* Remove white/gray header area of Streamlit */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Optimize main block padding */
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
    }
    
    /* 2. Left Sidebar Styling Overrides */
    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at 50% 50%, #060b13 0%, #021410 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        width: 260px !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] li, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #34d399 !important;
    }
    
    /* 3. Typography & Titles */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #34d399 0%, #22d3ee 50%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0px 0px 12px rgba(34, 211, 238, 0.25));
        letter-spacing: 0.02em;
    }
    
    .subtitle {
        font-size: 0.9rem;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #22d3ee;
        border-bottom: 1px solid rgba(34, 211, 238, 0.2);
        padding-bottom: 0.2rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* 4. Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.78rem;
        text-align: center;
        border: 1px solid transparent;
    }
    
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.12);
        color: #fca5a5;
        border-color: rgba(239, 68, 68, 0.25);
        box-shadow: 0 0 6px rgba(239, 68, 68, 0.1);
    }
    
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fde047;
        border-color: rgba(245, 158, 11, 0.25);
        box-shadow: 0 0 6px rgba(245, 158, 11, 0.1);
    }
    
    .badge-success {
        background-color: rgba(16, 185, 129, 0.12);
        color: #a7f3d0;
        border-color: rgba(16, 185, 129, 0.25);
        box-shadow: 0 0 6px rgba(16, 185, 129, 0.1);
    }
    
    .badge-info {
        background-color: rgba(59, 130, 246, 0.12);
        color: #93c5fd;
        border-color: rgba(59, 130, 246, 0.25);
    }
    
    /* Status Pills for Sidebar */
    .status-pills {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 6px #10b981;
        margin-right: 8px;
    }
    
    /* Button Styling Overrides for Dark Mode */
    button[kind="secondary"], 
    button[kind="primary"],
    div[data-testid="stButton"] button, 
    div[data-testid="stDownloadButton"] button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(34, 211, 238, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover, 
    button[kind="primary"]:hover,
    div[data-testid="stButton"] button:hover, 
    div[data-testid="stDownloadButton"] button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: rgba(34, 211, 238, 0.2) !important;
        border-color: #34d399 !important;
        color: #fff !important;
        box-shadow: 0 0 10px rgba(34, 211, 238, 0.3) !important;
    }
    button p, 
    div[data-testid="stButton"] button p, 
    div[data-testid="stDownloadButton"] button p,
    [data-testid="baseButton-secondary"] p,
    [data-testid="baseButton-primary"] p {
        color: inherit !important;
    }
    
    /* 5. Custom styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-bottom: none !important;
        border-radius: 6px 6px 0 0 !important;
        color: #94a3b8 !important;
        padding: 0.35rem 0.9rem !important;
        font-size: 0.85rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(34, 211, 238, 0.06) !important;
        border-color: rgba(34, 211, 238, 0.25) !important;
        color: #22d3ee !important;
        font-weight: 600 !important;
    }
    
    /* 6. Custom scrollbar for reports */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.01);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(34, 211, 238, 0.15);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(34, 211, 238, 0.3);
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

# Define CNN Checkpoint Path
CKPT_PATH = os.path.join(project_root, "rice_disease_cnn_outputs", "best_model.pt")

@st.cache_resource
def get_cached_model(ckpt_path, device_str):
    device = torch.device(device_str)
    model, classes, img_size = load_trained_model(ckpt_path, device)
    transform = get_inference_transform(img_size)
    return model, classes, transform

# Sidebar Layout (System Status)
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-top: -1.5rem; margin-bottom: 1rem;">
        <span style="font-size: 2.5rem;">🌾</span>
        <h4 style="color: #34d399; margin-top: 0.2rem; margin-bottom: 0.1rem; font-weight: 700;">RICE DIAGNOSIS</h4>
        <p style="font-size: 0.7rem; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase;">Decision Support System</p>
    </div>
    <hr style="margin: 0.6rem 0; opacity: 0.15;">
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #a7f3d0; margin-bottom: 0.4rem;'>🖥️ System Status</p>", unsafe_allow_html=True)
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_icon = "⚡ GPU (CUDA)" if device_str == "cuda" else "💻 CPU Mode"
    
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 0.6rem; font-size: 0.75rem;">
        <div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
            <div class="status-pills"></div>
            <span style="color: #94a3b8;">CNN Engine:</span>
            <strong style="color: #10b981; margin-left: auto;">Active</strong>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
            <div class="status-pills"></div>
            <span style="color: #94a3b8;">Fuzzy Engine:</span>
            <strong style="color: #10b981; margin-left: auto;">Active</strong>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
            <span style="color: #94a3b8;">Device Acceleration:</span>
            <strong style="color: #38bdf8; margin-left: auto;">{device_icon}</strong>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
            <span style="color: #94a3b8;">Checkpoint:</span>
            <strong style="color: #cbd5e1; margin-left: auto; font-family: monospace; font-size: 0.7rem;">best_model.pt</strong>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="color: #94a3b8;">Active Branch:</span>
            <strong style="color: #f59e0b; margin-left: auto; font-family: monospace;">khanhtrang</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 0.6rem 0; opacity: 0.15;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #a7f3d0; margin-bottom: 0.4rem;'>📖 Quick Operations</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.4;">
    1. <b>Upload image:</b> Add a leaf file (.jpg, .png).<br>
    2. <b>Weather controls:</b> Drag temperature & humidity.<br>
    3. <b>Diagnosis:</b> View deep learning CNN output integrated with environmental rules.<br>
    4. <b>Export:</b> Download the joint XAI report JSON.
    </div>
    """, unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 class='main-title'>🌾 BÁO CÁO MÔN HỌC HỆ THỐNG THÔNG MINH - NHÓM 14</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>CNN + Fuzzy Expert System + Explainable AI (XAI) for Smart Agriculture</p>", unsafe_allow_html=True)

# Mini Pipeline Diagram
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem; margin-top: -0.5rem; margin-bottom: 1rem; font-size: 0.75rem; background: rgba(30, 41, 59, 0.25); padding: 0.3rem 0.8rem; border-radius: 9999px; border: 1px solid rgba(255,255,255,0.03); width: fit-content; margin-left: auto; margin-right: auto;">
    <span style="color: #64748b;">Pipeline:</span>
    <span style="color: #34d399; font-weight: 500;">🖼️ Image Input</span>
    <span style="color: #475569;">➔</span>
    <span style="color: #22d3ee; font-weight: 500;">🧠 CNN Classifier</span>
    <span style="color: #475569;">➔</span>
    <span style="color: #818cf8; font-weight: 500;">📊 Softmax Distribution</span>
    <span style="color: #475569;">➔</span>
    <span style="color: #f59e0b; font-weight: 500;">⚡ Fuzzy Inference</span>
    <span style="color: #475569;">➔</span>
    <span style="color: #f43f5e; font-weight: 500;">📝 XAI Recommendation</span>
</div>
""", unsafe_allow_html=True)

# Layout: 3 Columns
col1, col2, col3 = st.columns([1, 1.1, 1.1], gap="medium")

with col1:
    st.markdown("<div class='section-header'>📤 INPUT & ENVIRONMENT</div>", unsafe_allow_html=True)
    
    # Initialize session state for file tracking and snail slider
    if 'last_uploaded_filename' not in st.session_state:
        st.session_state.last_uploaded_filename = None
    if 'snail_density_val' not in st.session_state:
        st.session_state.snail_density_val = 0.0
        
    uploaded_file = st.file_uploader("Tải ảnh lá lúa (.jpg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Check if this is a new image to reset the override states
        if st.session_state.last_uploaded_filename != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.snail_density_val = 0.0 # Reset snail slider
            
        # Display image with constrained height
        image = Image.open(uploaded_file)
        image_thumb = image.copy()
        image_thumb.thumbnail((300, 150))
        st.image(image_thumb, caption="Bản xem trước ảnh đầu vào", use_container_width=False)
        
        # Save original to temp location for inference
        temp_dir = os.path.join(project_root, "demo_assets")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, "temp_upload.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.info("💡 Vui lòng chọn và tải ảnh lá lúa lên.")
        
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
    temp = st.slider("Nhiệt độ thời tiết (°C)", min_value=15.0, max_value=45.0, value=28.0, step=0.5)
    humidity = st.slider("Độ ẩm không khí (%)", min_value=30.0, max_value=100.0, value=85.0, step=1.0)
    
    st.markdown("<div style='margin-top:10px; margin-bottom:5px; font-size: 0.8rem; color: #f59e0b; font-weight: bold;'>🧑‍🌾 NHẬP LIỆU TỪ NÔNG DÂN (Manual Input)</div>", unsafe_allow_html=True)
    snail_density = st.slider("Mật độ Ốc bươu quan sát được (con/m²)", min_value=0.0, max_value=10.0, step=0.5, key="snail_density_val", help="Kéo thanh trượt này nếu nông dân trực tiếp quan sát thấy ốc tại ruộng. Logic mờ sẽ ghi đè mạng CNN.")

with col2:
    st.markdown("<div class='section-header'>🎯 DEEP LEARNING CNN</div>", unsafe_allow_html=True)
    if uploaded_file is not None and os.path.exists(CKPT_PATH):
        try:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            model, classes, transform = get_cached_model(CKPT_PATH, device_str)
            
            import requests
            
            # Gọi API Backend thay vì chạy model cục bộ để tận dụng OOD Detection
            api_url = "http://localhost:8000/predict"
            with open(temp_image_path, "rb") as f:
                resp = requests.post(api_url, files={"file": f})
                
            if resp.status_code == 200:
                api_data = resp.json()
                status = api_data["status"]
                is_ood = api_data["is_ood"]
                
                # Xử lý hiển thị dựa trên trạng thái (KNOWN, UNCERTAIN, OOD)
                if status == "KNOWN":
                    pred_class_val = api_data["predicted_class"]
                    pred_conf_val = api_data["confidence"]
                    title_label = "Lớp bệnh nhận dạng (Top Class)"
                elif status == "UNCERTAIN":
                    pred_class_val = f"Nhóm bệnh: {api_data.get('top_group')}"
                    pred_conf_val = api_data.get("group_confidence", 0.0)
                    title_label = "Nhóm bệnh nghi ngờ (Group Confidence)"
                else:
                    pred_class_val = "Out-of-Distribution/Unknown"
                    pred_conf_val = api_data["debug_info"]["max_softmax_probability"]
                    title_label = "Lớp bệnh nhận dạng (Top Class)"
                    
                # Tái tạo cnn_result từ API response
                cnn_result = {
                    "scores": api_data["debug_info"]["cnn_scores"],
                    "pred_class": pred_class_val,
                    "pred_confidence": pred_conf_val
                }
                
                if status != "KNOWN":
                    # Hiển thị cảnh báo OOD trực tiếp trên UI
                    st.error(f"🚨 {api_data['message']}")
                    if snail_density == 0:
                        st.info(f"💡 Khuyến nghị: {api_data['recommendation']}")


                if status == "KNOWN" or snail_density > 0:
                    # Chỉ chạy Fuzzy Logic Engine nếu ảnh là KNOWN HOẶC người dùng kích hoạt Ngoại lệ ốc bươu vàng
                    inp = FuzzyInput(
                        cnn_scores=cnn_result["scores"],
                        top_class=cnn_result["pred_class"],
                        top_confidence=cnn_result["pred_confidence"],
                        temperature=temp,
                        humidity=humidity,
                        snail_density=snail_density
                    )
                    
                    engine = FuzzyEngine()
                    out = engine.run(inp)
            else:
                st.error("❌ Không kết nối được tới API Backend. Vui lòng kiểm tra server uvicorn.")
                cnn_result = {"scores": {}, "pred_class": "Error", "pred_confidence": 0}
            
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
            if 'out' in locals():
                if out.inference_mode == "EXPERT_GUIDED_MODE":
                    pred_class_vi = "Dấu hiệu nghi ngờ Ốc bươu vàng (Expert-Guided)"
                    title_label = "SUY LUẬN HỖ TRỢ CHUYÊN GIA (EXPERT-GUIDED)"
                    cnn_result['pred_confidence'] = out.fused_confidence
                    st.warning("⚠️ Hệ thống chuyển sang chế độ suy luận có hỗ trợ tín hiệu thực địa do dữ liệu hình ảnh có độ bất định cao.")
                    
                elif out.inference_mode == "HYBRID_WARNING":
                    title_label = "CHẨN ĐOÁN CẢNH BÁO LAI (HYBRID INFERENCE)"
                    cnn_result['pred_confidence'] = out.fused_confidence
                    st.warning("⚠️ Hệ thống ghi nhận tín hiệu ngoại lệ từ thực địa. Kích hoạt Cảnh báo Lai (Hybrid Warning).")
                    
                elif out.inference_mode == "AI_CONFIDENT" and snail_density > 0:
                    st.info("ℹ️ Tín hiệu ngoại lệ từ thực địa chưa đủ mạnh để thay đổi chẩn đoán chính của mô hình AI.")
                
                if snail_density > 0:
                    quick_rec = "\\n".join([line for line in out.recommendation.split('\\n') if "Hành động" in line or "1." in line or "CẢNH BÁO" in line][:2])
                    st.info(f"💡 Ý kiến chuyên gia nhanh:\\n{quick_rec}\\n\\n(Xem chi tiết tại Tab 'Khuyến nghị chuyên gia' bên dưới)")
            
            # Highlight top prediction beautifully
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 0.65rem 0.8rem; margin-bottom: 0.4rem; box-shadow: 0 0 10px rgba(16, 185, 129, 0.05);">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">{title_label}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #10b981; margin-top: 0.1rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{pred_class_vi}</div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.15rem;">Độ tự tin tổng hợp (Fused Confidence): <strong style="color:#22d3ee;">{cnn_result['pred_confidence']*100:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Analysis Button
            col_btn1, col_btn2 = st.columns([1, 1.5])
            with col_btn1:
                if st.button("📊 Analyze Confidence", use_container_width=True, help="Mở bảng phân tích XAI chi tiết"):
                    show_confidence_analysis(
                        cnn_res=cnn_result, 
                        entropy=api_data.get("debug_info", {}).get("shannon_entropy", 0.0) if 'api_data' in locals() else 0.0,
                        status=status if 'status' in locals() else 'KNOWN',
                        inference_mode=out.inference_mode if 'out' in locals() else "NORMAL",
                        is_ood=is_ood if 'is_ood' in locals() else False,
                        class_map=class_mapping
                    )
            
            # Custom Top 5 HTML bar chart
            st.markdown("<p style='font-size: 0.78rem; font-weight: 600; color: #cbd5e1; margin-top: 0.25rem; margin-bottom: 0.25rem;'>Top-5 Lớp tin cậy nhất (Softmax Distribution):</p>", unsafe_allow_html=True)
            
            top_5_scores = sorted(cnn_result["scores"].items(), key=lambda x: x[1], reverse=True)[:5]
            bars_html = ""
            for idx, (cls_name, score) in enumerate(top_5_scores):
                cls_vi = class_mapping.get(cls_name, cls_name)
                percent = score * 100
                if idx == 0:
                    bar_color = "linear-gradient(90deg, #10b981, #34d399)"
                    glow_style = "box-shadow: 0 0 6px rgba(16, 185, 129, 0.3);"
                    text_style = "color: #34d399; font-weight: 600;"
                else:
                    bar_color = "linear-gradient(90deg, #22d3ee, #0d9488)"
                    glow_style = "box-shadow: 0 0 4px rgba(34, 211, 238, 0.15);"
                    text_style = "color: #cbd5e1;"
                    
                bars_html += f'<div style="margin-bottom: 0.4rem;"><div style="display: flex; justify-content: space-between; font-size: 0.74rem; margin-bottom: 2px;"><span style="{text_style}">{cls_vi}</span><span style="font-family: monospace; color: #94a3b8;">{percent:.2f}%</span></div><div style="background: rgba(255,255,255,0.03); height: 6px; border-radius: 3px; overflow: hidden;"><div style="background: {bar_color}; width: {percent}%; height: 100%; border-radius: 3px; {glow_style}"></div></div></div>'
            st.markdown(bars_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Lỗi CNN: {e}")
    else:
        st.markdown("<div style='text-align: center; color: #64748b; padding-top: 3.5rem; font-size: 0.85rem;'>Chờ chẩn đoán hình ảnh...</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='section-header'>⚡ FUZZY EXPERT REASONING</div>", unsafe_allow_html=True)
    if uploaded_file is not None and 'out' in locals():
        alert_type = parse_risk_level(out.final_alert_level)
        alert_badge = f'<span class="badge badge-{alert_type}">{out.final_alert_level}</span>'
        
        env_type = parse_risk_level(out.environmental_risk_level)
        env_badge = f'<span class="badge badge-{env_type}">{out.environmental_risk_level}</span>'
        
        if "low" in out.uncertainty_level.lower() or "thấp" in out.uncertainty_level.lower():
            unc_badge = f'<span class="badge badge-success">{out.uncertainty_level}</span>'
        elif "high" in out.uncertainty_level.lower() or "cao" in out.uncertainty_level.lower():
            unc_badge = f'<span class="badge badge-danger">{out.uncertainty_level}</span>'
        else:
            unc_badge = f'<span class="badge badge-warning">{out.uncertainty_level}</span>'
            
        # Severity bar calculation
        sev_str = out.visual_severity_level.lower()
        if "severe" in sev_str or "nặng" in sev_str:
            sev_percent = 85
            sev_color = "linear-gradient(90deg, #f43f5e, #e11d48)"
            sev_shadow = "box-shadow: 0 0 6px rgba(244, 63, 94, 0.35);"
        elif "mild" in sev_str or "nhẹ" in sev_str:
            sev_percent = 35
            sev_color = "linear-gradient(90deg, #fbbf24, #d97706)"
            sev_shadow = "box-shadow: 0 0 4px rgba(251, 191, 36, 0.25);"
        else:
            sev_percent = 5
            sev_color = "linear-gradient(90deg, #10b981, #059669)"
            sev_shadow = "box-shadow: 0 0 4px rgba(16, 185, 129, 0.15);"
            
        # System confidence bar
        conf_val = out.diagnostic_confidence
        conf_color = "linear-gradient(90deg, #10b981, #34d399)"
        conf_shadow = "box-shadow: 0 0 6px rgba(16, 185, 129, 0.3);"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 0.45rem 0.75rem; margin-bottom: 0.35rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">🚨 Mức cảnh báo (Alert)</span>
            <div>{alert_badge}</div>
        </div>
        
        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 0.45rem 0.75rem; margin-bottom: 0.35rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">🌦️ Nguy cơ thời tiết</span>
            <div>{env_badge}</div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 0.45rem 0.75rem; margin-bottom: 0.35rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">🌫️ Độ bất định chẩn đoán</span>
            <div>{unc_badge}</div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 0.45rem 0.75rem; margin-bottom: 0.35rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                <span style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">🌡️ Chỉ số tổn thương trực quan (VSI)</span>
                <span style="font-size: 0.82rem; font-weight: 600; color: #f43f5e;">{out.visual_severity_level}</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); height: 5px; border-radius: 2px; overflow: hidden;">
                <div style="background: {sev_color}; width: {sev_percent}%; height: 100%; border-radius: 2px; {sev_shadow}"></div>
            </div>
        </div>

        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 0.45rem 0.75rem; margin-bottom: 0.35rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                <span style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase;">🛡️ Độ chắc chắn chẩn đoán</span>
                <span style="font-size: 0.82rem; font-weight: 700; color: #10b981;">{out.diagnostic_confidence:.2f}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.03); height: 5px; border-radius: 2px; overflow: hidden;">
                <div style="background: {conf_color}; width: {conf_val}%; height: 100%; border-radius: 2px; {conf_shadow}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif uploaded_file is not None and 'status' in locals() and status != "KNOWN" and snail_density == 0:
        st.markdown(f"""
        <div style='text-align: center; padding-top: 2.5rem;'>
            <div style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🛡️</div>
            <div style='color: #f43f5e; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;'>Hệ mờ tạm ngưng (OOD Safeguard)</div>
            <div style='color: #94a3b8; font-size: 0.85rem; padding: 0 1.5rem; line-height: 1.5;'>
                Hệ thống chuyên gia đã bị chặn để đảm bảo an toàn do CNN không nhận diện được ảnh này.<br><br>
                <span style='color: #f59e0b; font-weight: 600;'>Nếu đây là Ốc Bươu Vàng, vui lòng sử dụng Thanh trượt Ngoại lệ (bên trái) để ép hệ thống xử lý bằng tay!</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center; color: #64748b; padding-top: 3.5rem; font-size: 0.85rem;'>Chờ tính toán quy tắc mờ...</div>", unsafe_allow_html=True)

# Bottom Section: Academic Tabs for Explainable AI & Expert Advice
if uploaded_file is not None and 'out' in locals():
    st.markdown("<div style='margin: 0.1rem 0;'><hr style='margin: 0.2rem 0; opacity: 0.12;'></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔍 BÁO CÁO GIẢI THÍCH (XAI REASONING)", "🌾 KHUYẾN NGHỊ CHUYÊN GIA (AI EXPERT ADVICE)", "💻 TRẠNG THÁI HỆ THỐNG & EXPORT"])
    
    with tab1:
        # Layout inside Tab 1: Left is explanation report, Right is rules timeline
        xai_col_l, xai_col_r = st.columns([1.1, 0.9])
        
        with xai_col_l:
            # Parse only the analysis text (not the rule list)
            explanation_parts = out.explanation.split("4. Các Luật Mờ được kích hoạt")[0]
            formatted_explanation = explanation_parts.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
            
            # Parse rules from explanation
            rules_fired = []
            for line in out.explanation.split('\n'):
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')) and 'Luật' in line:
                    rules_fired.append(line.strip())
                    
            col_lbl, col_btn = st.columns([1.5, 1])
            with col_lbl:
                st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #22d3ee; margin-bottom: 0.4rem; padding-top: 0.3rem;'>📊 Logic suy giải (Reasoning Breakdown):</p>", unsafe_allow_html=True)
            with col_btn:
                if st.button("🔍 Đọc Full Màn Hình", key="btn_full_xai", use_container_width=True):
                    show_reasoning_dialog(formatted_explanation, rules_fired)
                    
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(34, 211, 238, 0.15); border-radius: 8px; padding: 0.8rem; color: #e2e8f0; font-size: 0.82rem; line-height: 1.5; max-height: 140px; overflow-y: auto;">
                {formatted_explanation}
            </div>
            """, unsafe_allow_html=True)
            
        with xai_col_r:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #f59e0b; margin-bottom: 0.4rem;'>⚡ Luật mờ được kích hoạt (Triggered Rules):</p>", unsafe_allow_html=True)
            
            rules_html = '<div style="max-height: 140px; overflow-y: auto; padding-right: 0.2rem;">'
            for r in rules_fired:
                rules_html += (
                    '<div style="background: rgba(245, 158, 11, 0.06); border: 1px solid rgba(245, 158, 11, 0.22); '
                    'border-radius: 6px; padding: 0.35rem 0.5rem; font-size: 0.76rem; color: #fef08a; display: flex; '
                    'align-items: center; gap: 0.4rem; margin-bottom: 0.3rem;">'
                    '<span style="background: #f59e0b; color: #020617; border-radius: 50%; width: 14px; height: 14px; '
                    'display: inline-flex; justify-content: center; align-items: center; font-weight: bold; '
                    'font-size: 0.65rem;">⚡</span>'
                    f'<span>{r}</span>'
                    '</div>'
                )
            rules_html += '</div>'
            st.markdown(rules_html, unsafe_allow_html=True)
            
    with tab2:
        # Header with Alert Icon based on severity
        alert_val = out.final_alert_level.lower()
        if "danger" in alert_val or "severe" in alert_val or "nguy hiểm" in alert_val or "red alert" in alert_val or "báo động đỏ" in alert_val:
            rec_icon = "🚨"
            rec_title = "CẢNH BÁO: RỦI RO BÙNG PHÁT CAO"
            border_color = "rgba(239, 68, 68, 0.2)"
        elif "warning" in alert_val or "attention" in alert_val or "chú ý" in alert_val:
            rec_icon = "⚠️"
            rec_title = "CHÚ Ý: GIÁM SÁT RUỘNG LÚA"
            border_color = "rgba(245, 158, 11, 0.2)"
        else:
            rec_icon = "🌱"
            rec_title = "AN TOÀN: THEO DÕI ĐỊNH KỲ"
            border_color = "rgba(16, 185, 129, 0.2)"
            
        st.markdown(f"<p style='font-size: 0.85rem; font-weight: 600; color: #a7f3d0; margin-bottom: 0.4rem;'>{rec_icon} {rec_title}:</p>", unsafe_allow_html=True)
        
        # Parse recommendation text to clean cards
        lines = out.recommendation.split('\n')
        rec_html = ""
        for line in lines:
            l_strip = line.strip()
            if not l_strip:
                continue
            if l_strip.startswith(('1.', '2.', '3.', '4.', '5.')):
                rec_html += (
                    '<div style="display: flex; gap: 0.5rem; align-items: flex-start; margin-bottom: 0.25rem;">'
                    '<span style="color: #22d3ee; font-weight: bold; font-size: 0.8rem;">✔</span>'
                    f'<span style="color: #cbd5e1; font-size: 0.8rem;">{l_strip[2:].strip()}</span>'
                    '</div>'
                )
            elif l_strip.startswith('*'):
                rec_html += (
                    '<div style="background: rgba(251, 191, 36, 0.04); border-left: 2px solid #fbbf24; '
                    'padding: 0.35rem 0.5rem; margin-top: 0.4rem; border-radius: 0 4px 4px 0; font-size: 0.78rem; '
                    f'color: #fde047; line-height:1.4;">💡 <b>Biện pháp đặc thù:</b> {l_strip[1:].strip()}</div>'
                )
            else:
                rec_html += f"<p style='color: #e2e8f0; font-size: 0.82rem; margin-bottom: 0.3rem;'>{l_strip}</p>"
                
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid {border_color}; border-radius: 8px; padding: 0.8rem; min-height: 120px; max-height: 140px; overflow-y: auto;">
            {rec_html}
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        xai_col_sub1, xai_col_sub2 = st.columns([1.2, 0.8])
        with xai_col_sub1:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #38bdf8; margin-bottom: 0.3rem;'>💾 Xuất dữ liệu báo cáo tích hợp (Export Output):</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.76rem; color: #94a3b8; margin-bottom: 0.5rem;'>Tải xuống báo cáo chứa điểm phân phối Softmax từ CNN, thông số thời tiết, và hệ suy diễn giải mờ XAI dạng file JSON.</p>", unsafe_allow_html=True)
            
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
                label="📥 Tải xuống báo cáo JSON",
                data=json_bytes,
                file_name=f"rice_diagnosis_{Path(uploaded_file.name).stem}.json",
                mime="application/json"
            )
            
        with xai_col_sub2:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.3rem;'>ℹ️ Chi tiết hệ thống:</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.2); border: 1px solid rgba(255,255,255,0.03); border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.72rem; color: #94a3b8; line-height: 1.4;">
            - <b>CNN Model:</b> EfficientNet-B0 (PyTorch)<br>
            - <b>Fuzzy Engine:</b> Sugeno (Zero-order)<br>
            - <b>Inference Device:</b> {device_str.upper()}<br>
            - <b>API Version:</b> Streamlit 1.57.0
            </div>
            """, unsafe_allow_html=True)

# Tiny footer for academic prototype feel
st.markdown("<div style='text-align: center; font-size: 0.68rem; color: #475569; margin-top: 0.5rem; margin-bottom: -1rem;'>🌾 Hybrid Intelligent Rice Disease Diagnosis & Decision Support System • Smart Farming Research Prototype • Branch: khanhtrang</div>", unsafe_allow_html=True)

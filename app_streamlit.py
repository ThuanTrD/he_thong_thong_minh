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

# Custom CSS
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
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ade80 0%, #10b981 50%, #064e3b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        padding-top: 0.5rem;
        filter: drop-shadow(0px 4px 8px rgba(16, 185, 129, 0.15));
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #a7f3d0;
        border-bottom: 2px solid rgba(16, 185, 129, 0.2);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    
    /* Custom Badge classes */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
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

# Layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("<div class='section-header'>📤 TẢI ẢNH & THAM SỐ MÔI TRƯỜNG</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Tải lên ảnh lá lúa bị bệnh (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh lá lúa đầu vào", use_container_width=True)
        
        # Save to temp location for inference
        temp_dir = os.path.join(project_root, "demo_assets")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, "temp_upload.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        st.info("💡 Vui lòng tải lên một ảnh lá lúa bị bệnh để bắt đầu chẩn đoán.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Yếu tố môi trường thời tiết (Tùy chọn):**")
    temp = st.slider("Nhiệt độ môi trường (°C)", min_value=15.0, max_value=45.0, value=28.0, step=0.5)
    humidity = st.slider("Độ ẩm không khí (%)", min_value=30.0, max_value=100.0, value=85.0, step=1.0)

with col2:
    if uploaded_file is not None and os.path.exists(CKPT_PATH):
        with st.spinner("Đang chạy phân tích CNN và suy diễn mờ XAI..."):
            try:
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
                model, classes, transform = get_cached_model(CKPT_PATH, device_str)
                
                # CNN prediction
                cnn_result = predict_single(temp_image_path, model, classes, transform, torch.device(device_str))
                
                # Execute Fuzzy Logic Engine
                inp = FuzzyInput(
                    cnn_scores=cnn_result["scores"],
                    top_class=cnn_result["pred_class"],
                    top_confidence=cnn_result["pred_confidence"],
                    temperature=temp,
                    humidity=humidity
                )
                
                engine = FuzzyEngine()
                out: FuzzyOutput = engine.run(inp)
                
                # Mapping classes to Vietnamese names for display
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
                
                # Display CNN results
                st.markdown("<div class='section-header'>🎯 KẾT QUẢ PHÂN TÍCH CNN</div>", unsafe_allow_html=True)
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.45); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; height: 130px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Lớp bệnh nhận diện</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #34d399; margin-top: 0.2rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{pred_class_vi}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">CNN Code: <code>{cnn_result['pred_class']}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_col2:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.45); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; height: 130px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Độ tự tin dự đoán</div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: #60a5fa; margin-top: 0.1rem;">{cnn_result['pred_confidence']*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<p style='font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.5rem;'>Phân phối xác suất (Softmax Scores):</p>", unsafe_allow_html=True)
                scores_df = pd.DataFrame({
                    "Độ tự tin": list(cnn_result["scores"].values())
                }, index=[class_mapping.get(c, c) for c in cnn_result["scores"].keys()])
                st.bar_chart(scores_df, height=200)
                
                # Display Fuzzy Logic results
                st.markdown("<div class='section-header'>🧠 KẾT QUẢ SUY DIỄN MỜ (FUZZY LOGIC)</div>", unsafe_allow_html=True)
                
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    alert_badge = get_alert_badge(out.final_alert_level)
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Mức cảnh báo (Alert)</div>
                        <div style="margin-top: 0.4rem;">{alert_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Nghiêm trọng trực quan</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #f43f5e; margin-top: 0.2rem;">{out.visual_severity_level}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Nhóm bệnh mờ</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #fbbf24; margin-top: 0.2rem;">{out.predicted_disease}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with f_col2:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Độ tin cậy hệ thống</div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #10b981; margin-top: 0.1rem;">{out.diagnostic_confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Độ bất định chẩn đoán</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #a8a29e; margin-top: 0.2rem;">{out.uncertainty_level}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    env_badge = get_env_badge(out.environmental_risk_level)
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Nguy cơ do thời tiết</div>
                        <div style="margin-top: 0.4rem;">{env_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra khi suy diễn mờ: {e}")
                
    elif uploaded_file is not None and not os.path.exists(CKPT_PATH):
        st.error(f"❌ Không tìm thấy tệp trọng số CNN tại đường dẫn: `{CKPT_PATH}`. Vui lòng đặt đúng tệp trọng số.")
    else:
        st.markdown("<div style='text-align: center; color: #64748b; padding: 6rem 0;'>Vui lòng tải ảnh lá lúa lên ở cột bên trái để hiển thị kết quả chẩn đoán.</div>", unsafe_allow_html=True)

# Bottom Full-width XAI Report
if uploaded_file is not None and 'out' in locals():
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 1.5rem; font-weight: 700; color: #34d399; margin-top: 1rem; margin-bottom: 1.5rem; text-shadow: 0 0 10px rgba(52, 211, 153, 0.2);'>📊 BÁO CÁO GIẢI THÍCH CHI TIẾT (XAI REPORT) & KHUYẾN NGHỊ HÀNH ĐỘNG</div>", unsafe_allow_html=True)
    
    xai_col1, xai_col2 = st.columns(2, gap="large")
    
    with xai_col1:
        st.markdown("### 🔍 Giải thích Suy diễn Mờ (Fuzzy Explanation)")
        formatted_explanation = out.explanation.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 12px; padding: 1.5rem; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6; min-height: 250px;">
            {formatted_explanation}
        </div>
        """, unsafe_allow_html=True)
        
    with xai_col2:
        st.markdown("### 🌾 Khuyến nghị Hành động Nông nghiệp")
        formatted_recommendation = out.recommendation.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 12px; padding: 1.5rem; color: #e2e8f0; font-size: 0.95rem; line-height: 1.6; min-height: 250px;">
            {formatted_recommendation}
        </div>
        """, unsafe_allow_html=True)

    # Export Section
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("### 💾 Xuất Dữ Liệu Kết Quả")
    
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
        label="📥 Tải xuống báo cáo tích hợp (JSON)",
        data=json_bytes,
        file_name=f"rice_diagnosis_{Path(uploaded_file.name).stem}.json",
        mime="application/json"
    )

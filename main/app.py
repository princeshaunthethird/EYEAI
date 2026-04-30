import base64
import io
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# -----------------------
# CONFIG
# -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

disease_info = {
    "Cataract": {
        "desc": "Clouding of the eye's natural lens located behind the iris and pupil.",
        "severity": "Moderate",
        "color": "#F59E0B",
        "icon": "🟡",
        "symptoms": ["Blurry or cloudy vision", "Faded colors", "Glare sensitivity", "Night vision difficulty"],
        "actions": ["Refer to ophthalmologist", "Schedule lens evaluation", "Consider surgical consultation"],
        "icd": "H26.9",
        "urgency": "Non-urgent — schedule within 4–6 weeks"
    },
    "Diabetic Retinopathy": {
        "desc": "Progressive damage to the blood vessels of the retina caused by chronic high blood sugar.",
        "severity": "High",
        "color": "#EF4444",
        "icon": "🔴",
        "symptoms": ["Spots/floaters", "Blurred vision", "Dark areas in vision", "Vision fluctuation"],
        "actions": ["Urgent specialist referral", "HbA1c blood test", "Immediate lifestyle intervention"],
        "icd": "E11.319",
        "urgency": "Urgent — see specialist within 1–2 weeks"
    },
    "Glaucoma": {
        "desc": "Group of eye conditions causing optic nerve damage, often from elevated intraocular pressure.",
        "severity": "High",
        "color": "#8B5CF6",
        "icon": "🟣",
        "symptoms": ["Peripheral vision loss", "Eye pain", "Headaches", "Halos around lights"],
        "actions": ["Immediate IOP measurement", "Visual field test", "Urgent ophthalmology referral"],
        "icd": "H40.9",
        "urgency": "Urgent — assessment within 1 week"
    },
    "Normal": {
        "desc": "No significant retinal abnormalities detected. Retinal structures appear healthy.",
        "severity": "None",
        "color": "#10B981",
        "icon": "🟢",
        "symptoms": ["No symptoms detected"],
        "actions": ["Routine annual screening", "Maintain regular check-ups"],
        "icd": "Z01.00",
        "urgency": "Routine — next screening in 12 months"
    }
}

# -----------------------
# PAGE CONFIG (must be first)
# -----------------------
st.set_page_config(
    page_title="EyeAI Diagnostic System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# GLOBAL CSS
# -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #E8EDF5;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1400px; }

/* ---- SCAN LINE BACKGROUND ---- */
body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,200,255,0.015) 2px,
        rgba(0,200,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ---- HEADER BAND ---- */
.eyeai-header {
    background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 50%, #0A1628 100%);
    border-bottom: 1px solid rgba(0,200,255,0.15);
    padding: 22px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -2rem 2.5rem -2rem;
    position: relative;
    overflow: hidden;
}
.eyeai-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00C8FF 40%, #4DFFC3 60%, transparent);
}
.eyeai-logo {
    display: flex; align-items: center; gap: 14px;
}
.eyeai-logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #00C8FF, #4DFFC3);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 20px rgba(0,200,255,0.4);
}
.eyeai-logo-text h1 {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 800;
    letter-spacing: -0.3px;
    background: linear-gradient(90deg, #fff 30%, #4DFFC3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.eyeai-logo-text span {
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: #4A6080;
    letter-spacing: 1.5px; text-transform: uppercase;
}
.header-meta {
    text-align: right;
}
.header-meta .status {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: #4DFFC3; letter-spacing: 0.5px;
}
.status-dot {
    width: 7px; height: 7px;
    background: #4DFFC3; border-radius: 50%;
    box-shadow: 0 0 8px #4DFFC3;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.header-meta .timestamp {
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: #3A5070;
    margin-top: 4px;
}

/* ---- SECTION TITLES ---- */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px; font-weight: 600;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #4A6A8A;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ''; flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.2), transparent);
}

/* ---- CARDS ---- */
.glass-card {
    background: linear-gradient(135deg, rgba(13,31,60,0.8), rgba(8,18,36,0.9));
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), transparent);
}

/* ---- UPLOAD ZONE ---- */
.upload-zone {
    background: linear-gradient(135deg, rgba(0,200,255,0.04), rgba(77,255,195,0.04));
    border: 2px dashed rgba(0,200,255,0.25);
    border-radius: 20px;
    padding: 50px 30px;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 20px;
}
.upload-zone h3 {
    font-family: 'Syne', sans-serif;
    font-size: 20px; font-weight: 700;
    color: #C8D8E8;
    margin-bottom: 8px;
}
.upload-zone p { color: #4A6080; font-size: 14px; }

/* ---- DIAGNOSIS BADGE ---- */
.diagnosis-badge {
    font-family: 'Syne', sans-serif;
    font-size: 32px; font-weight: 800;
    letter-spacing: -0.5px;
    padding: 16px 28px;
    border-radius: 14px;
    text-align: center;
    margin: 12px 0;
    position: relative;
    overflow: hidden;
}

/* ---- METRIC TILES ---- */
.metric-tile {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-tile .value {
    font-family: 'DM Mono', monospace;
    font-size: 28px; font-weight: 500;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-tile .label {
    font-size: 11px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #4A6080;
}

/* ---- PROBABILITY BARS ---- */
.prob-row {
    display: flex; align-items: center;
    gap: 12px; margin-bottom: 14px;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 12px; width: 150px;
    color: #8AAAC8; flex-shrink: 0;
    letter-spacing: 0.3px;
}
.prob-bar-track {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 6px; height: 10px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%; border-radius: 6px;
    transition: width 1.2s cubic-bezier(.4,0,.2,1);
    position: relative;
}
.prob-bar-fill::after {
    content: '';
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 40px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
    border-radius: 0 6px 6px 0;
}
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 13px; width: 52px;
    text-align: right; color: #C8D8E8;
    flex-shrink: 0;
}

/* ---- SYMPTOM TAGS ---- */
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.tag {
    font-family: 'DM Mono', monospace; font-size: 11px;
    letter-spacing: 0.5px;
    padding: 5px 12px; border-radius: 20px;
    border: 1px solid rgba(0,200,255,0.2);
    color: #8AAAC8;
    background: rgba(0,200,255,0.05);
}

/* ---- ACTION ITEMS ---- */
.action-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 14px; color: #A0B8D0;
}
.action-item:last-child { border-bottom: none; }
.action-num {
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: #00C8FF; min-width: 20px; margin-top: 2px;
}

/* ---- CONFIDENCE GAUGE ---- */
.gauge-wrap { text-align: center; padding: 10px 0; }
.gauge-pct {
    font-family: 'DM Mono', monospace;
    font-size: 56px; font-weight: 500;
    line-height: 1; letter-spacing: -2px;
}
.gauge-label { font-size: 12px; letter-spacing: 2px; text-transform: uppercase; color: #4A6080; margin-top: 4px; }

/* ---- REPORT HEADER ---- */
.report-header {
    background: linear-gradient(135deg, rgba(0,200,255,0.06), rgba(77,255,195,0.06));
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 14px;
    padding: 20px 24px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px;
}
.report-id {
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: #4A6080;
    letter-spacing: 1.5px; text-transform: uppercase;
}
.report-id span {
    font-size: 16px; color: #00C8FF;
    letter-spacing: 0.5px; display: block; margin-top: 2px;
}

/* ---- ICD CODE ---- */
.icd-badge {
    font-family: 'DM Mono', monospace; font-size: 12px;
    padding: 4px 10px; border-radius: 6px;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2);
    color: #5ADAFF; letter-spacing: 0.5px;
    display: inline-block; margin-top: 8px;
}

/* ---- URGENCY STRIP ---- */
.urgency-strip {
    font-size: 13px; font-weight: 500;
    padding: 10px 16px; border-radius: 8px;
    margin-top: 14px;
    display: flex; align-items: center; gap: 8px;
}

/* ---- HEATMAP LABEL ---- */
.cam-legend {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: #4A6080; letter-spacing: 1px;
    margin-top: 8px;
}
.cam-gradient {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000);
    margin-top: 6px;
}

/* ---- STREAMLIT OVERRIDES ---- */
.stFileUploader > div {
    background: transparent !important;
    border: none !important;
}
.stFileUploader label {
    display: none !important;
}
div[data-testid="stImage"] img {
    border-radius: 12px;
}
.stSpinner > div { border-top-color: #00C8FF !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# GRAD-CAM
# -----------------------
def generate_gradcam(model, image_tensor):
    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.features[-1]
    h_fw = target_layer.register_forward_hook(forward_hook)
    h_bw = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    h_fw.remove()
    h_bw.remove()
    return cam

# -----------------------
# HEADER
# -----------------------
now = datetime.now()
st.markdown(f"""
<div class="eyeai-header">
    <div class="eyeai-logo">
        <div class="eyeai-logo-icon">👁️</div>
        <div class="eyeai-logo-text">
            <h1>EyeAI Diagnostic</h1>
            <span>Retinal Disease Detection System · v2.0</span>
        </div>
    </div>
    <div class="header-meta">
        <div class="status"><span class="status-dot"></span> System Online · EfficientNet-B0</div>
        <div class="timestamp">{now.strftime("%d %b %Y  %H:%M:%S")} · Device: {DEVICE.upper()}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# SYSTEM STATS ROW
# -----------------------
cols = st.columns(4)
stats = [
    ("Model Architecture", "EfficientNet-B0", "#00C8FF"),
    ("Validation Accuracy", "~90%", "#4DFFC3"),
    ("Classes Detected", "4 Conditions", "#A78BFA"),
    ("Explainability", "Grad-CAM XAI", "#F59E0B"),
]
for col, (label, value, color) in zip(cols, stats):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="value" style="color:{color}">{value}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------
# UPLOAD
# -----------------------
st.markdown('<div class="section-title">01 — Image Input</div>', unsafe_allow_html=True)

upload_col, info_col = st.columns([3, 2])

with upload_col:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:48px; margin-bottom:12px;">🔬</div>
        <h3>Upload Retinal Fundus Image</h3>
        <p>Supports JPG, PNG, JPEG · Recommended resolution 224×224px or higher</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

with info_col:
    st.markdown("""
    <div class="glass-card" style="height:100%">
        <div style="font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:#C8D8E8; margin-bottom:16px;">
            📋 System Capabilities
        </div>
        <div style="font-size:13px; color:#6A8AA8; line-height:2;">
            ✦ &nbsp; <span style="color:#A0B8D0">Cataract detection</span><br>
            ✦ &nbsp; <span style="color:#A0B8D0">Diabetic Retinopathy screening</span><br>
            ✦ &nbsp; <span style="color:#A0B8D0">Glaucoma risk assessment</span><br>
            ✦ &nbsp; <span style="color:#A0B8D0">Healthy retina confirmation</span><br>
            ✦ &nbsp; <span style="color:#A0B8D0">Grad-CAM spatial explanation</span><br>
            ✦ &nbsp; <span style="color:#A0B8D0">ICD-10 diagnostic codes</span><br>
            ✦ &nbsp; <span style="color:#A0B8D0">Clinical action recommendations</span>
        </div>
        <div style="margin-top:20px; padding:12px; background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.15); border-radius:10px; font-size:12px; color:#F87171;">
            ⚠️ For screening assistance only. Not a substitute for clinical diagnosis.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# MAIN ANALYSIS
# -----------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    progress_bar = st.progress(0)
    status = st.empty()

    for i, msg in enumerate(["Loading retinal image...", "Running inference...", "Generating Grad-CAM...", "Compiling report..."]):
        status.markdown(f'<div style="font-family:DM Mono,monospace; font-size:12px; color:#4DFFC3; letter-spacing:1px;">⟳ &nbsp;{msg}</div>', unsafe_allow_html=True)
        progress_bar.progress((i + 1) * 25)
        time.sleep(0.3)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100
    probs_np = probs.cpu().numpy()[0]
    info = disease_info[predicted_class]

    cam = generate_gradcam(model, img_tensor)

    progress_bar.empty()
    status.empty()

    # Report ID
    report_id = f"EYE-{now.strftime('%Y%m%d')}-{hash(uploaded_file.name) % 9999:04d}"
    st.markdown(f"""
    <div class="report-header">
        <div>
            <div class="report-id">Diagnostic Report ID <span>{report_id}</span></div>
        </div>
        <div style="text-align:right">
            <div class="report-id">Generated <span>{now.strftime("%d %B %Y, %H:%M")}</span></div>
        </div>
        <div style="text-align:right">
            <div class="report-id">File <span>{uploaded_file.name}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================
    # ROW 1: IMAGE + RESULT
    # =====================
    st.markdown('<div class="section-title">02 — Primary Diagnosis</div>', unsafe_allow_html=True)

    col_img, col_diag, col_conf = st.columns([2, 2.5, 1.5])

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:12px;">Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f'<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; text-align:center; margin-top:8px;">{image.size[0]}×{image.size[1]}px · RGB</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_diag:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:16px;">Diagnosis</div>', unsafe_allow_html=True)

        severity_colors = {"None": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"}
        sev_color = severity_colors.get(info["severity"], "#888")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {info['color']}12, {info['color']}06);
             border: 2px solid {info['color']}40; border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;">
            <div style="font-size:13px; color:{info['color']}; font-family:DM Mono,monospace; letter-spacing:1px; margin-bottom:6px;">PRIMARY FINDING</div>
            <div style="font-family:Syne,sans-serif; font-size:30px; font-weight:800; color:#F0F6FF; line-height:1.1;">{predicted_class}</div>
            <div class="icd-badge">ICD-10: {info['icd']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:13.5px; color:#8AAAC8; line-height:1.7; margin-bottom:16px;">{info['desc']}</div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:12px; color:#4A6080; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Clinical Indicators</div>', unsafe_allow_html=True)
        tags_html = ''.join([f'<span class="tag">{s}</span>' for s in info["symptoms"]])
        st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

        urg_bg = {"None": "rgba(16,185,129,0.08)", "Moderate": "rgba(245,158,11,0.08)", "High": "rgba(239,68,68,0.08)"}
        urg_border = {"None": "rgba(16,185,129,0.25)", "Moderate": "rgba(245,158,11,0.25)", "High": "rgba(239,68,68,0.25)"}
        st.markdown(f"""
        <div class="urgency-strip" style="background:{urg_bg.get(info['severity'],'rgba(0,200,255,0.08)')}; border:1px solid {urg_border.get(info['severity'],'rgba(0,200,255,0.25)')}; color:{sev_color};">
            <span>⏱</span> {info['urgency']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_conf:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:12px;">Confidence</div>', unsafe_allow_html=True)

        conf_color = "#10B981" if confidence_score > 80 else "#F59E0B" if confidence_score > 50 else "#EF4444"
        conf_label = "High Confidence" if confidence_score > 80 else "Moderate" if confidence_score > 50 else "Low — Verify"

        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-pct" style="color:{conf_color}">{confidence_score:.1f}<span style="font-size:22px; opacity:0.5">%</span></div>
            <div class="gauge-label">{conf_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:20px; padding:12px; background:rgba(255,255,255,0.03); border-radius:10px;">
            <div style="font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Severity</div>
            <div style="font-family:Syne,sans-serif; font-size:18px; font-weight:700; color:{sev_color};">{info['severity']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Mini confidence ring via SVG
        radius = 48
        cx = cy = 60
        circ = 2 * 3.14159 * radius
        filled = circ * (confidence_score / 100)
        st.markdown(f"""
        <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" style="width:100%; margin-top:18px;">
          <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="rgba(0,200,255,0.08)" stroke-width="8"/>
          <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{conf_color}" stroke-width="8"
            stroke-dasharray="{filled:.1f} {circ:.1f}"
            stroke-linecap="round"
            transform="rotate(-90 {cx} {cy})"/>
          <text x="{cx}" y="{cy+5}" text-anchor="middle" fill="{conf_color}"
            style="font-family:DM Mono,monospace; font-size:18px;">{confidence_score:.0f}%</text>
        </svg>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =====================
    # ROW 2: PROBS + ACTIONS
    # =====================
    st.markdown('<div class="section-title">03 — Probability Breakdown & Clinical Actions</div>', unsafe_allow_html=True)

    col_prob, col_act = st.columns([3, 2])

    prob_colors = {
        "Cataract": "#F59E0B",
        "Diabetic Retinopathy": "#EF4444",
        "Glaucoma": "#8B5CF6",
        "Normal": "#10B981"
    }

    with col_prob:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:20px;">Class Probabilities</div>', unsafe_allow_html=True)

        sorted_indices = np.argsort(probs_np)[::-1]
        for idx in sorted_indices:
            cls = CLASS_NAMES[idx]
            prob = probs_np[idx] * 100
            color = prob_colors[cls]
            is_top = cls == predicted_class
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="{'color:#E8EDF5; font-weight:500;' if is_top else ''}">{cls}</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width:{prob:.1f}%; background: linear-gradient(90deg, {color}80, {color});"></div>
                </div>
                <div class="prob-pct" style="color:{color if is_top else '#4A6080'}">{prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Table
        st.markdown('<div style="margin-top:24px; border-top:1px solid rgba(255,255,255,0.05); padding-top:18px;">', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Condition": CLASS_NAMES,
            "Probability": [f"{p*100:.2f}%" for p in probs_np],
            "Logit Score": [f"{v:.4f}" for v in outputs.cpu().detach().numpy()[0]]
        })
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_act:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:18px;">Recommended Actions</div>', unsafe_allow_html=True)

        for i, action in enumerate(info["actions"], 1):
            st.markdown(f"""
            <div class="action-item">
                <div class="action-num">0{i}</div>
                <div>{action}</div>
            </div>
            """, unsafe_allow_html=True)

        # Differential summary
        second_idx = sorted_indices[1]
        second_cls = CLASS_NAMES[second_idx]
        second_prob = probs_np[second_idx] * 100

        st.markdown(f"""
        <div style="margin-top:20px; padding:14px; background:rgba(0,200,255,0.04); border:1px solid rgba(0,200,255,0.12); border-radius:10px;">
            <div style="font-size:11px; color:#4A6080; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px;">Differential Consideration</div>
            <div style="font-size:14px; color:#A0B8D0;">
                Secondary finding: <span style="color:#00C8FF; font-weight:500;">{second_cls}</span>
                <span style="font-family:DM Mono,monospace; font-size:12px; color:#4A6080;"> ({second_prob:.1f}%)</span>
            </div>
            <div style="font-size:12px; color:#4A6080; margin-top:6px; line-height:1.5;">
                {disease_info[second_cls]['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =====================
    # ROW 3: GRAD-CAM
    # =====================
    st.markdown('<div class="section-title">04 — Explainability · Grad-CAM Attention</div>', unsafe_allow_html=True)

    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
    pure_heat = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Original Scan</div>', unsafe_allow_html=True)
        st.image(image.resize((224, 224)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cam_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Activation Heatmap</div>', unsafe_allow_html=True)
        st.image(pure_heat, use_container_width=True)
        st.markdown("""
        <div class="cam-gradient"></div>
        <div class="cam-legend"><span>Low</span><span>Activation Intensity</span><span>High</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cam_col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:DM Mono,monospace; font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Overlay (AI Focus)</div>', unsafe_allow_html=True)
        st.image(overlay_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Grad-CAM explanation
    st.markdown(f"""
    <div class="glass-card" style="background: rgba(0,200,255,0.03);">
        <div style="display:flex; gap:40px; flex-wrap:wrap;">
            <div>
                <div style="font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">Method</div>
                <div style="font-size:14px; color:#A0B8D0;">Gradient-weighted Class Activation Mapping (Grad-CAM)</div>
            </div>
            <div>
                <div style="font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">Target Layer</div>
                <div style="font-family:DM Mono,monospace; font-size:13px; color:#00C8FF;">model.features[-1]</div>
            </div>
            <div>
                <div style="font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">Activation Peak</div>
                <div style="font-family:DM Mono,monospace; font-size:13px; color:#4DFFC3;">{cam.max():.4f}</div>
            </div>
            <div>
                <div style="font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">Interpretation</div>
                <div style="font-size:13px; color:#8AAAC8;">Red/warm regions = high diagnostic relevance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================
    # ROW 4: MODEL INFO
    # =====================
    st.markdown('<div class="section-title">05 — Model & Technical Details</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    model_metrics = [
        ("Architecture", "EfficientNet-B0", "#00C8FF"),
        ("Input Shape", "224 × 224 × 3", "#4DFFC3"),
        ("Parameters", "~5.3M", "#A78BFA"),
        ("Training Epochs", "15", "#F59E0B"),
        ("Optimizer", "Adam + StepLR", "#F87171"),
    ]
    for col, (label, val, color) in zip([m1, m2, m3, m4, m5], model_metrics):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="value" style="color:{color}; font-size:18px;">{val}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture pipeline
    pipeline_steps = ["Input 224×224", "EfficientNet Features", "Global Avg Pool", "FC 1280→4", "Softmax", f"→ {predicted_class}"]
    pipe_html = ""
    for i, step in enumerate(pipeline_steps):
        is_last = i == len(pipeline_steps) - 1
        bg = f"linear-gradient(135deg, {info['color']}20, {info['color']}10)" if is_last else "rgba(0,200,255,0.06)"
        border = f"{info['color']}50" if is_last else "rgba(0,200,255,0.15)"
        color = info["color"] if is_last else "#8AAAC8"
        pipe_html += f"""
        <div style="display:flex; align-items:center; gap:8px; flex:1;">
            <div style="background:{bg}; border:1px solid {border}; border-radius:10px; padding:10px 12px; text-align:center; flex:1; font-family:DM Mono,monospace; font-size:11px; color:{color}; letter-spacing:0.5px;">{step}</div>
            {'<div style="color:#2A4060; font-size:18px;">→</div>' if not is_last else ''}
        </div>
        """

    st.markdown(f"""
    <div class="glass-card">
        <div style="font-size:11px; color:#3A5070; letter-spacing:1px; text-transform:uppercase; margin-bottom:14px;">Inference Pipeline</div>
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">{pipe_html}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:72px; margin-bottom:20px; opacity:0.4;">👁️</div>
        <div style="font-family:Syne,sans-serif; font-size:28px; font-weight:700; color:#2A4060; margin-bottom:10px;">
            Awaiting Retinal Image
        </div>
        <div style="font-size:14px; color:#2A3A50;">
            Upload a fundus photograph above to begin AI-powered diagnostic analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# FOOTER
# -----------------------
st.markdown("""
<div style="margin-top:48px; padding:20px 0; border-top:1px solid rgba(0,200,255,0.08); display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div style="font-family:DM Mono,monospace; font-size:11px; color:#2A3A50; letter-spacing:1px;">
        EYEAI DIAGNOSTIC SYSTEM · EfficientNet-B0 · Grad-CAM XAI
    </div>
    <div style="font-size:12px; color:#2A3A50;">
        ⚠️ For research and educational screening assistance only — not a substitute for clinical medical diagnosis.
    </div>
</div>
""", unsafe_allow_html=True)
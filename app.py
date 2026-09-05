import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime
from io import BytesIO
from gtts import gTTS
from fpdf import FPDF
import plotly.express as px

# Ensure you have your local modules in the same folder
from database import init_db, save_lot_record
from vision_engine import VisionEngine

# --- 1. PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="VisionGrade - Mandi Portal", page_icon="🧅", layout="wide")

# --- 2. GLOBAL CSS (Cyberpunk & Glassmorphism Theme) ---
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global App Background & Typography */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #070b14 60%, #020617 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0;
    }

    /* Keyframe Animations */
    @keyframes subtleGlow {
        0% { border-color: rgba(56, 189, 248, 0.25); box-shadow: 0 0 10px rgba(56, 189, 248, 0.05); }
        50% { border-color: rgba(46, 196, 182, 0.55); box-shadow: 0 0 20px rgba(46, 196, 182, 0.2); }
        100% { border-color: rgba(56, 189, 248, 0.25); box-shadow: 0 0 10px rgba(56, 189, 248, 0.05); }
    }

    @keyframes pulseDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        animation: subtleGlow 6s infinite ease-in-out !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 12px 28px rgba(56, 189, 248, 0.25) !important;
    }

    /* Metric Label and Value Typography */
    div[data-testid="stMetricLabel"] p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }

    /* Glass Containers & Cards */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stFileUploader"]),
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stCameraInput"]) {
        background: rgba(15, 23, 42, 0.45);
        border: 1px dashed rgba(56, 189, 248, 0.35);
        border-radius: 16px;
        padding: 18px;
        backdrop-filter: blur(10px);
    }

    /* Cyberpunk Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(46, 196, 182, 0.15) 0%, rgba(14, 159, 110, 0.25) 100%) !important;
        border: 1.5px solid #2ec4b6 !important;
        color: #2ec4b6 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 0 12px rgba(46, 196, 182, 0.15) !important;
        transition: all 0.25s ease-in-out !important;
    }

    .stButton > button:hover {
        background: #2ec4b6 !important;
        color: #041014 !important;
        box-shadow: 0 0 24px rgba(46, 196, 182, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Sleek Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.18) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Sidebar Reskin */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 29, 0.85) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
        backdrop-filter: blur(16px) !important;
    }

    /* Status Badges */
    .telemetry-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .pulse-indicator {
        width: 8px;
        height: 8px;
        background-color: #2ec4b6;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #2ec4b6;
        animation: pulseDot 1.8s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. INIT LOGIC & MODELS ---
init_db()

st.title("🧅 VisionGrade: Edge AI Mandi Portal")
st.markdown("Automated Agmark Classification • e-NAM API Ready • SQLite Auditing")

# --- 4. SIDEBAR ---
st.sidebar.header("⚙️ Hardware & Edge Config")
model_source = st.sidebar.text_input("YOLO Model Weight", "best.pt")
pixels_per_mm = st.sidebar.slider("Calibration (Pixels/mm)", 1.0, 5.0, 2.5, 0.1)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.9, 0.15, 0.05)
lang_pref = st.sidebar.radio("Audio Alert Language", ["hi", "en"], format_func=lambda x: "Hindi" if x == "hi" else "English")

st.sidebar.header("📋 Batch Details")
mandi_id = st.sidebar.selectbox("Mandi Location", ["Sonipat APMC", "Azadpur Mandi", "Karnal Mandi"])
inspector_name = st.sidebar.text_input("Inspector Name", "R. Singh")
farmer_id = st.sidebar.text_input("Farmer ID", "RK-0981")

@st.cache_resource
def load_engine(path, ppm, conf):
    return VisionEngine(model_path=path, pixels_per_mm=ppm, conf_thresh=conf)

vision = load_engine(model_source, pixels_per_mm, conf_thresh)

# --- 5. HELPER FUNCTIONS ---
def generate_audio(weight, payout, lang='hi'):
    if lang == 'en':
        text = f"Grading complete. Net weight is {weight} kilograms. Total payout is {payout} rupees."
    else:
        text = f"Grading poori hui. Kul vajan {weight} kilo. Payout {payout} rupaye."
    
    sound_file = BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(sound_file)
    return sound_file.getvalue()

def generate_pdf_receipt(lot_id, weight, payout, grades):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="VISIONGRADE - APMC ASSAYING SLIP", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Mandi Location: {mandi_id}", ln=True)
    pdf.cell(200, 10, txt=f"Farmer Token: {farmer_id}", ln=True)
    pdf.cell(200, 10, txt=f"Lot ID: {lot_id}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Net Weight: {weight} kg", ln=True)
    pdf.cell(200, 10, txt=f"Total Digital Payout: INR {payout}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Quality Breakdown (Agmark Standards):", ln=True)
    pdf.cell(200, 10, txt=f" - Grade A (>55mm): {grades['Grade A']} units", ln=True)
    pdf.cell(200, 10, txt=f" - Grade B (45-55mm): {grades['Grade B']} units", ln=True)
    pdf.cell(200, 10, txt=f" - Grade C (<45mm): {grades['Grade C']} units", ln=True)
    pdf.cell(200, 10, txt=f" - Reject/Spoilage: {grades['Reject']} units", ln=True)
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tfile.name)
    with open(tfile.name, "rb") as f:
        return f.read()

# --- 6. MAIN UI TABS ---
tab1, tab2, tab3 = st.tabs(["🎥 Edge Inference Engine", "📊 Cloud Analytics Dashboard", "📋 e-NAM Audit & Receipts"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📡 Live Feed Acquisition")
        input_mode = st.radio("Source Type", ["Live Camera Capture", "Upload Image"], horizontal=True)
        
        if 'latency' not in st.session_state:
            st.session_state.latency = 0.0
            
        if input_mode == "Live Camera Capture":
            camera_image = st.camera_input("Take a snapshot from the Conveyor Rig")
            if camera_image:
                file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                start_t = time.time()
                annotated_frame, summary = vision.process_frame(frame)
                st.session_state.latency = round((time.time() - start_t) * 1000, 2)
                
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                st.session_state.current_summary = summary
                
        elif input_mode == "Upload Image":
            uploaded_image = st.file_uploader("Upload Static Batch (.jpg, .png)", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                start_t = time.time()
                annotated_frame, summary = vision.process_frame(frame)
                st.session_state.latency = round((time.time() - start_t) * 1000, 2)
                
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                st.session_state.current_summary = summary

    with col2:
        st.markdown("### ⚙️ Telemetry & Sizing")
        
        # Animated Telemetry Badge
        st.markdown(f"""
        <div class="telemetry-badge">
            <span class="pulse-indicator"></span>
            EDGE INFERENCE: {st.session_state.latency} ms | MODEL: YOLOv8
        </div>
        """, unsafe_allow_html=True)
        
        current_sum = st.session_state.get('current_summary', {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Reject": 0})
        
        st.metric("Grade A (>55mm) 🟢", current_sum["Grade A"])
        st.metric("Grade B (45-55mm) 🟠", current_sum["Grade B"])
        st.metric("Grade C (<45mm) 🟡", current_sum["Grade C"])
        st.metric("Reject (Rot) 🔴", current_sum["Reject"])

        if st.button("✅ Finalize Batch & Sync to DB", use_container_width=True):
            total_cnt = sum(current_sum.values())
            weight = round(total_cnt * 0.35, 2)
            payout = round(weight * 24.5, 2)
            lot_id = f"LOT-{datetime.now().strftime('%H%M%S')}"
            
            save_lot_record(lot_id, inspector_name, farmer_id, mandi_id, total_cnt, 
                            current_sum["Grade A"], current_sum["Grade B"], 
                            current_sum["Grade C"], current_sum["Reject"], weight, payout)
            
            st.success(f"Batch {lot_id} synced to SQLite ledger!")
            
            # Generate Audio and PDF
            st.session_state.latest_audio = generate_audio(weight, payout, lang_pref)
            st.session_state.latest_pdf = generate_pdf_receipt(lot_id, weight, payout, current_sum)
            
        # Autoplay Audio Alert
        if 'latest_audio' in st.session_state:
            st.audio(st.session_state.latest_audio, format="audio/mp3", autoplay=True)

with tab2:
    st.markdown("### 📈 Real-Time Mandi Quality Trends")
    try:
        conn = sqlite3.connect('mandi_logs.db')
        df = pd.read_sql_query("SELECT * FROM lots", conn)
        
        if not df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Tonnage Processed", f"{round(df['weight_kg'].sum(), 2)} kg")
            c2.metric("Total Economic Value", f"₹ {round(df['payout'].sum(), 2)}")
            c3.metric("Total Rejected Units", df['reject'].sum())
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Lot-by-Lot Yield Analytics")
            
            # Interactive Plotly Area Chart
            fig = px.area(
                df,
                x="id", 
                y=["grade_a", "grade_b", "grade_c", "reject"],
                color_discrete_map={
                    "grade_a": "#2ec4b6",  
                    "grade_b": "#f97316",  
                    "grade_c": "#eab308",  
                    "reject": "#ef4444"    
                },
                labels={"value": "Unit Count", "variable": "Agmark Grade", "id": "Lot Token"}
            )
            
            # Transparent Styling to match CSS
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",  
                paper_bgcolor="rgba(0,0,0,0)", 
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
                legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=40, b=0),
                hovermode="x unified"
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(56, 189, 248, 0.15)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(56, 189, 248, 0.15)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No data available in the local SQLite database yet. Process a batch to visualize trends.")
    except Exception as e:
        st.warning("Database not initialized yet. Run your first inference to generate the ledger.")

with tab3:
    st.markdown("### 🗄️ Secure Audit Trail & Exports")
    try:
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Full Audit Log to CSV",
                data=csv,
                file_name=f"Mandi_Audit_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            if 'latest_pdf' in st.session_state:
                st.download_button(
                    label="🖨️ Download Latest Assaying Slip (PDF)",
                    data=st.session_state.latest_pdf,
                    file_name="Assaying_Slip.pdf",
                    mime="application/pdf"
                )
                
            st.dataframe(df.sort_values(by="id", ascending=False), use_container_width=True)
    except Exception as e:
         st.info("Process a batch to generate audit trails.")
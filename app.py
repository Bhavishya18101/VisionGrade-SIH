import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
from io import BytesIO
from gtts import gTTS
from fpdf import FPDF
import plotly.express as px
from streamlit_lottie import st_lottie

# 🚨 IMPORTING YOUR DB ENGINEERS' OFFICIAL FILE
from database import get_connection, setup_database, add_new_farmer, insert_lot_data, seed_sample_data
from vision_engine import VisionEngine

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="VisionGrade Pro", page_icon="🧅", layout="wide", initial_sidebar_state="expanded")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = ""
    st.session_state.auth_page = 'selection'

# --- 2. ENTERPRISE UI/UX CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Base App Styling */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Hide Streamlit Footer and Top Right Menu, but keep Sidebar Toggle */
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    header {background-color: transparent !important;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Clean Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }

    /* Professional Buttons */
    .stButton > button {
        background-color: #3B82F6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #2563EB !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        font-weight: 500;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom-color: #3B82F6 !important;
    }

    /* Camera Box */
    [data-testid="stCameraInput"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
    }
    
    /* Login Card Container */
    .login-container {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin-top: 5vh;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOTTIE ANIMATIONS (Safely Cached) ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None

lottie_ai_scan = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_6n9m9l_01.json")
lottie_success = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_pqnfmone.json")
lottie_farmer = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_3rwasyjy.json") 

# --- 4. BACKEND LOGIC ---
setup_database()
seed_sample_data() 

def auth_farmer(token, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM Farmers WHERE farmer_token=? AND password=?", (token, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_farmer_mobile(token):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT phone_number FROM Farmers WHERE farmer_token=?", (token,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

@st.cache_resource
def load_engine(path, ppm, conf, defect_sensitivity):
    return VisionEngine(model_path=path, pixels_per_mm=ppm, conf_thresh=conf, defect_sensitivity=defect_sensitivity)

def send_free_whatsapp(destination_number, message_body):
    """
    Mock function for cloud deployment. 
    In production, this would hit an API like Twilio or WhatsApp Business.
    """
    if not destination_number.startswith("+91"):
        destination_number = "+91" + str(destination_number)
    print(f"[CLOUD DEMO] Simulated WhatsApp sent to {destination_number}: {message_body}")
    return True

def generate_audio(weight, payout, lang='hi'):
    if lang == 'en':
        text = f"Grading complete. Net weight is {weight} kilograms. Total payout is {payout} rupees."
    else:
        text = f"Grading poori hui. Kul vajan {weight} kilo. Payout {payout} rupaye."
    sound_file = BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(sound_file)
    return sound_file.getvalue()

def generate_pdf_receipt(lot_id, weight, payout, grades, mandi_id, farmer_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="VISIONGRADE - APMC ASSAYING SLIP", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Mandi: {mandi_id}", ln=True)
    pdf.cell(200, 10, txt=f"Farmer ID: {farmer_id}", ln=True)
    pdf.cell(200, 10, txt=f"Batch ID: {lot_id}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Total Quantity: {weight} units", ln=True)
    pdf.cell(200, 10, txt=f"Total Payout: INR {payout}", ln=True)
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tfile.name)
    with open(tfile.name, "rb") as f:
        return f.read()

# --- 5. UI SCREENS ---
def login_screen():
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            try:
                st.image("logo.png", use_container_width=True)
            except:
                if lottie_ai_scan:
                    st_lottie(lottie_ai_scan, height=150, key="login_anim")
                
        st.markdown("<h2 style='text-align: center; color: white;'>VisionGrade Pro</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94A3B8;'>Enterprise Mandi Authentication</p><br>", unsafe_allow_html=True)
        
        if st.session_state.auth_page == 'selection':
            if st.button("🌾 Access Farmer Portal"):
                st.session_state.auth_page = 'farmer_login'
                st.rerun()
            st.write("")
            if st.button("⚙️ Access Manager Portal"):
                st.session_state.auth_page = 'manager_login'
                st.rerun()

        elif st.session_state.auth_page == 'farmer_login':
            st.markdown("#### Farmer Login")
            f_token = st.text_input("Farmer ID (Token)", placeholder="VG-FARM-1001")
            f_pass = st.text_input("Password", type="password")
            
            st.write("")
            if st.button("Secure Login"):
                user_name = auth_farmer(f_token, f_pass)
                if user_name:
                    st.session_state.logged_in = True
                    st.session_state.role = "User (Farmer)"
                    st.session_state.farmer_token = f_token 
                    st.session_state.username = user_name
                    st.rerun()
                else:
                    st.error("Authentication Failed. Please check your credentials.")
                    
            st.write("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Back"):
                    st.session_state.auth_page = 'selection'
                    st.rerun()
            with col_b:
                if st.button("Register"):
                    st.session_state.auth_page = 'farmer_signup'
                    st.rerun()

        elif st.session_state.auth_page == 'farmer_signup':
            st.markdown("#### Farmer Registration")
            reg_name = st.text_input("Full Legal Name")
            reg_mobile = st.text_input("Mobile Number")
            reg_address = st.text_area("Registered Village/Address")
            reg_pass = st.text_input("Create Password", type="password")
            
            st.write("")
            if st.button("Register Account"):
                if reg_name and reg_mobile and reg_pass:
                    new_token = f"VG-FARM-{np.random.randint(1000, 9999)}"
                    if add_new_farmer(new_token, reg_name, reg_pass, reg_mobile, reg_address):
                        st.success("Registration Successful.")
                        st.info(f"Your Farmer ID is: **{new_token}**. Please save this.")
                    else:
                        st.error("Registration Error.")
                else:
                    st.warning("All fields are required.")
            
            st.write("---")
            if st.button("Back to Login"):
                st.session_state.auth_page = 'farmer_login'
                st.rerun()

        elif st.session_state.auth_page == 'manager_login':
            st.markdown("#### Manager Authentication")
            secret_code = st.text_input("System Administrator Key", type="password")
            
            st.write("")
            if st.button("Verify & Login"):
                if secret_code == "ADMIN123":
                    st.session_state.logged_in = True
                    st.session_state.role = "Mandi Manager"
                    st.session_state.username = "Administrator"
                    st.rerun()
                else:
                    st.error("Access Denied.")
                    
            st.write("---")
            if st.button("Back"):
                st.session_state.auth_page = 'selection'
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

def farmer_dashboard():
    with st.sidebar:
        try:
            st.image("logo.png", width=80)
        except: pass
        st.title("Farmer Portal")
        st.write(f"**Profile:** {st.session_state.username}")
        st.write(f"**ID:** {st.session_state.farmer_token}")
        st.divider()
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.auth_page = 'selection'
            st.rerun()
        
    st.markdown(f"<h2>Overview: {st.session_state.username}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Manage procurement slots and view certified e-NAM payouts.</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🗓️ Slot Allocation", "💳 Payout Ledger"])
    
    with tab1:
        st.markdown("### Schedule Produce Drop-off")
        col1, col2 = st.columns(2)
        with col1:
            mandi_choice = st.selectbox("Destination APMC", ["Sonipat APMC", "Azadpur Mandi", "Karnal Mandi"])
            if st.button("Request Grading Slot"):
                arrival_time = (datetime.now() + timedelta(days=1)).strftime('%I:%M %p')
                st.success(f"Slot Confirmed at {mandi_choice} for Tomorrow, {arrival_time}.")
                
                phone = get_farmer_mobile(st.session_state.farmer_token)
                if phone:
                    msg = f"VisionGrade: {st.session_state.username}, slot at {mandi_choice} confirmed for {arrival_time}. ID: {st.session_state.farmer_token}."
                    with st.spinner("Dispatching secure WhatsApp confirmation..."):
                        if send_free_whatsapp(phone, msg):
                            st.toast("WhatsApp confirmation sent successfully.", icon="✅")
                        else:
                            st.warning("WhatsApp dispatch failed. Ensure browser automation is allowed.")
            
    with tab2:
        st.markdown("### Certified Ledger History")
        conn = get_connection()
        df = pd.read_sql_query(f"SELECT date, batch_id, quantity, grade_a, grade_b, grade_c, payout FROM Lots WHERE farmer_token = '{st.session_state.farmer_token}' ORDER BY date DESC", conn)
        conn.close()
        
        if not df.empty:
            df['Agmark Distribution (A/B/C)'] = df.apply(lambda r: f"{r['grade_a']} / {r['grade_b']} / {r['grade_c']}", axis=1)
            df['Payout'] = df['payout'].apply(lambda x: f"₹ {x:,.2f}")
            display_df = df[['date', 'batch_id', 'quantity', 'Agmark Distribution (A/B/C)', 'Payout']]
            display_df.columns = ["Timestamp", "Batch ID", "Total Units", "Agmark Distribution (A/B/C)", "Net Payout"]
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No records found. Submit your first batch to the Mandi Manager.")

def manager_dashboard():
    with st.sidebar:
        try:
            st.image("logo.png", width=80)
        except: pass
        st.title("Control Center")
        st.divider()
        st.markdown("**Hardware Calibration**")
        model_source = st.text_input("Model Weights", "best.pt")
        # INCREASING DEFAULT CONFIDENCE TO 0.55 TO ELIMINATE BACKGROUND NOISE
        pixels_per_mm = st.slider("Scale (px/mm)", 1.0, 5.0, 2.5, 0.1)
        conf_thresh = st.slider("AI Confidence", 0.05, 0.9, 0.55, 0.05)
        defect_sensitivity = st.slider(
            "Defect Sensitivity", 0.0, 1.0, 0.35, 0.05,
            help="Lower = more tolerant of natural blemishes on real onions (fewer false rejects). "
                 "Higher = stricter, closer to how the model behaves on clean/synthetic images."
        )
        lang_pref = st.radio("Voice Alert", ["hi", "en"], format_func=lambda x: "Hindi" if x == "hi" else "English")
        st.divider()
        mandi_id = st.selectbox("Active Location", ["Sonipat APMC", "Azadpur Mandi", "Karnal Mandi"])
        farmer_token_input = st.text_input("Active Farmer ID", placeholder="VG-FARM-1001")
        
        if st.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.auth_page = 'selection'
            st.rerun()

    vision = load_engine(model_source, pixels_per_mm, conf_thresh, defect_sensitivity)

    st.markdown("<h2>Mandi Operations Console</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Real-time AI Vision Engine & e-NAM Sync</p>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Edge Vision", "📊 Market Analytics", "📑 Audit Trail"])

    with tab1:
        col_cam, col_data = st.columns([2, 1])
        
        with col_cam:
            st.markdown("#### Live Sensor Feed")
            st.info("💡 **Pro-Tip:** Place the onion on a dark, matte surface with clear lighting. Avoid shadows to prevent false rejections.")
            input_mode = st.radio("Input Source", ["Camera", "Upload File"], horizontal=True, label_visibility="collapsed")
            
            if 'latency' not in st.session_state: st.session_state.latency = 0.0
                
            if input_mode == "Camera":
                camera_image = st.camera_input("Conveyor Camera")
                if camera_image:
                    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)
                    start_t = time.time()
                    annotated_frame, summary = vision.process_frame(frame)
                    st.session_state.latency = round((time.time() - start_t) * 1000, 2)
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.current_summary = summary
                    
            elif input_mode == "Upload File":
                uploaded_image = st.file_uploader("Upload Batch Image (.jpg/.png)", type=["jpg", "jpeg", "png"])
                if uploaded_image:
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)
                    start_t = time.time()
                    annotated_frame, summary = vision.process_frame(frame)
                    st.session_state.latency = round((time.time() - start_t) * 1000, 2)
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.current_summary = summary

        with col_data:
            st.markdown("#### Telemetry")
            if lottie_ai_scan: st_lottie(lottie_ai_scan, height=120, key="ai_scanning")
            st.caption(f"Inference Latency: **{st.session_state.latency} ms**")
            
            current_sum = st.session_state.get('current_summary', {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Reject": 0})
            st.metric("Grade A (Export)", current_sum.get("Grade A", 0))
            st.metric("Grade B (Local)", current_sum.get("Grade B", 0))
            st.metric("Grade C / Reject", current_sum.get("Grade C", 0) + current_sum.get("Reject", 0))

            st.write("")
            if st.button("Finalize & Sync Batch"):
                if not farmer_token_input:
                    st.error("Input Farmer ID in sidebar.")
                else:
                    total_qty = sum(current_sum.values())
                    if total_qty > 0:
                        payout = float((current_sum.get("Grade A", 0) * 40) + (current_sum.get("Grade B", 0) * 25) + (current_sum.get("Grade C", 0) * 10))
                        batch_id = f"BATCH-{datetime.now().strftime('%H%M%S')}"
                        today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if insert_lot_data(batch_id, farmer_token_input, current_sum.get("Grade A", 0), current_sum.get("Grade B", 0), current_sum.get("Grade C", 0), current_sum.get("Reject", 0), float(total_qty), payout, today_date):
                            st.success("Batch Synced to Ledger.")
                            if lottie_success: st_lottie(lottie_success, height=80)
                            st.session_state.latest_pdf = generate_pdf_receipt(batch_id, total_qty, payout, current_sum, mandi_id, farmer_token_input)
                            audio_bytes = generate_audio(total_qty, payout, lang_pref)
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        else:
                            st.error("Farmer ID not found in database.")
                    else:
                        st.warning("No produce detected.")

    with tab2:
        st.markdown("#### Today's Quality Distribution")
        conn = get_connection()
        df = pd.read_sql_query("SELECT * FROM Lots", conn)
        conn.close()
        
        if not df.empty:
            df['date_only'] = pd.to_datetime(df['date']).dt.date
            df_today = df[df['date_only'] == datetime.now().date()]

            if not df_today.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", f"{int(df_today['quantity'].sum())} units")
                c2.metric("Total Payout", f"₹ {round(df_today['payout'].sum(), 2)}")
                c3.metric("Rejected", int(df_today['reject'].sum()))
                
                totals = {
                    'Grade A': df_today['grade_a'].sum(),
                    'Grade B': df_today['grade_b'].sum(),
                    'Grade C': df_today['grade_c'].sum(),
                    'Reject': df_today['reject'].sum()
                }
                fig_df = pd.DataFrame(list(totals.items()), columns=['Category', 'Units'])
                
                fig = px.bar(fig_df, x='Category', y='Units', color='Category',
                             color_discrete_map={"Grade A": "#3B82F6", "Grade B": "#6366F1", "Grade C": "#94A3B8", "Reject": "#EF4444"},
                             text_auto=True)
                fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor='#1E293B')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No batches processed today.")
        else:
            st.info("Database is empty.")

    with tab3:
        st.markdown("#### e-NAM System Audit")
        conn = get_connection()
        df_audit = pd.read_sql_query("SELECT Lots.date, Lots.batch_id, Farmers.name, Lots.farmer_token, Lots.quantity, Lots.payout FROM Lots JOIN Farmers ON Lots.farmer_token = Farmers.farmer_token ORDER BY Lots.date DESC", conn)
        conn.close()
        
        if not df_audit.empty:
            df_audit.columns = ["Timestamp", "Batch ID", "Farmer Name", "Farmer Token", "Units", "Payout (INR)"]
            st.dataframe(df_audit, hide_index=True, use_container_width=True)
            
            c_a, c_b = st.columns(2)
            with c_a:
                csv = df_audit.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export CSV Log", data=csv, file_name="Mandi_Audit.csv", mime="text/csv", use_container_width=True)
            with c_b:
                if 'latest_pdf' in st.session_state:
                    st.download_button("🖨️ Download Last Receipt", data=st.session_state.latest_pdf, file_name="Receipt.pdf", mime="application/pdf", use_container_width=True)

# --- 6. ROUTER ---
if not st.session_state.logged_in:
    login_screen()
else:
    if st.session_state.role == "User (Farmer)":
        farmer_dashboard()
    elif st.session_state.role == "Mandi Manager":
        manager_dashboard()

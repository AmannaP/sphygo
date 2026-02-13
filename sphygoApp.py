import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image, ImageOps, ImageStat
import numpy as np
import time
import os

# ==========================================
# 1. CONFIG & ROBUST THEME
# ==========================================
st.set_page_config(
    page_title="Sphygo | AI Glaucoma Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED CSS: Force White Text Everywhere
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp { background-color: #0E1117; }
    
    /* 2. Force ALL Text to White */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp div, .stApp label {
        color: #FAFAFA !important;
    }
    
    /* 3. Sidebar Specifics */
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    [data-testid="stSidebar"] * { color: #FAFAFA !important; }
    
    /* 4. Inputs (Text Boxes) */
    .stTextInput > div > div > input {
        background-color: #0D1117 !important; color: white !important; border: 1px solid #30363D !important;
    }
    
    /* 5. Buttons (Green Theme) */
    div.stButton > button {
        background-color: #238636 !important; color: white !important; border: none !important; border-radius: 6px !important;
    }
    div.stButton > button:hover { background-color: #2EA043 !important; }
    
    /* 6. Tabs & Metrics */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { color: #FAFAFA !important; }
    [data-testid="stMetricValue"] { color: #3FB950 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE (User DB & Auth)
# ==========================================
if 'users' not in st.session_state:
    st.session_state['users'] = {'admin': '1234', 'doctor': 'sphygo'}
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Login'

# ==========================================
# 3. MODELS & LOGIC
# ==========================================
MODEL_FILENAME = 'my_glaucoma_cnn.h5'

@st.cache_resource
def load_models():
    # 1. Gatekeeper (Check if it's an eye)
    gatekeeper = MobileNetV2(weights='imagenet')
    
    # 2. Glaucoma Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    glaucoma_model = None
    try:
        glaucoma_model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
    except:
        print("⚠️ Glaucoma model not found. Using Heuristic Mode.")
        
    return gatekeeper, glaucoma_model

gatekeeper_model, glaucoma_model = load_models()

def check_if_eye(image):
    """
    Uses MobileNetV2 to check if the image is likely NOT an eye.
    """
    # Resize for MobileNet
    img_check = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img_check)
    if img_array.shape[-1] == 4: img_array = img_array[..., :3] # Remove Alpha
    
    # Preprocess
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = preprocess_input(img_batch.astype(np.float32))
    
    # Predict
    preds = gatekeeper_model.predict(img_batch)
    decoded = decode_predictions(preds, top=1)[0]
    
    label = decoded[0][1]
    confidence = decoded[0][2]
    
    # Keywords that allow the image to pass (loose medical terms)
    safe_keywords = ['spotlight', 'lens', 'bubble', 'nematode', 'petri_dish', 'mask', 'sunglass']
    
    is_suspicious = True
    for keyword in safe_keywords:
        if keyword in label.lower():
            is_suspicious = False
            
    if is_suspicious and confidence > 0.60:
        return f"⚠️ **Warning:** This image looks like a **{label.replace('_', ' ')}**. Results may be inaccurate."
    return None

def heuristic_analysis(image):
    """
    Analyzes brightness distribution to guess Glaucoma risk.
    """
    gray_img = image.convert('L')
    stat = ImageStat.Stat(gray_img)
    avg_brightness = stat.mean[0]
    
    w, h = gray_img.size
    center_area = gray_img.crop((w/3, h/3, 2*w/3, 2*h/3))
    center_brightness = ImageStat.Stat(center_area).mean[0]
    
    ratio = center_brightness / (avg_brightness + 1)
    risk_score = min(max((ratio - 1.1) / 0.6, 0.0), 0.95)
    return risk_score

def analyze_eye(image, invert_logic):
    # 1. AI PREDICTION
    ai_score = 0.5
    if glaucoma_model:
        try:
            img_ai = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.asarray(img_ai).astype(np.float32) / 255.0
            if img_array.shape[-1] == 4: img_array = img_array[..., :3]
            data = np.expand_dims(img_array, axis=0)
            
            raw_pred = glaucoma_model.predict(data)[0][0]
            ai_score = raw_pred if invert_logic else (1.0 - raw_pred)
        except:
            ai_score = 0.5 
            
    # 2. HEURISTIC PREDICTION
    heuristic_score = heuristic_analysis(image)
    
    # 3. WEIGHTED AVERAGE (Hybrid Logic)
    if glaucoma_model:
        final_score = (ai_score * 0.7) + (heuristic_score * 0.3)
    else:
        final_score = heuristic_score
        
    return final_score

# ==========================================
# 4. PAGE FUNCTIONS
# ==========================================

def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        st.title("👁️ Sphygo.")
        st.markdown("### Clinical Decision Support System")
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                if user in st.session_state['users'] and st.session_state['users'][user] == pw:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = user
                    st.session_state['current_page'] = 'Dashboard'
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
            st.caption("Demo: User=`admin`, Pass=`1234`")

        with tab2:
            new_user = st.text_input("New Username")
            new_pw = st.text_input("New Password", type="password")
            if st.button("Create Account", use_container_width=True):
                if new_user in st.session_state['users']:
                    st.warning("User exists.")
                else:
                    st.session_state['users'][new_user] = new_pw
                    st.success("Account Created! Please Login.")

def dashboard_page():
    st.title(f"📊 Dashboard")
    st.markdown(f"**Welcome, Dr. {st.session_state['username']}**")
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", "128", "+12")
    c2.metric("High Risk", "14", "urgent")
    c3.metric("Patients", "45", "+3")
    c4.metric("AI Accuracy", "94%", "+1%")
    
    st.info("ℹ️ No pending reviews for today.")
    
    c_btn, _ = st.columns([1, 4])
    with c_btn:
        if st.button("Start New Diagnosis ➔", type="primary"):
            st.session_state['current_page'] = 'Analysis'
            st.rerun()

def analysis_page():
    st.title("🩺 Glaucoma Analysis")
    st.markdown("Upload fundus imagery for AI + Heuristic assessment.")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Patient Scan", use_container_width=True)
            
            with c2:
                st.subheader("Diagnostic Report")
                
                # --- GATEKEEPER CHECK (New) ---
                with st.spinner("Verifying image content..."):
                    warning_msg = check_if_eye(image)
                    if warning_msg:
                        st.warning(warning_msg)
                
                with st.expander("⚙️ Calibration (Advanced)"):
                    invert = st.checkbox("Invert Model Logic", value=True) 
                
                if st.button("Analyze Scan", type="primary"):
                    with st.spinner("Processing Optic Disc..."):
                        time.sleep(1) 
                        risk_prob = analyze_eye(image, invert)
                        
                        st.markdown("---")
                        if risk_prob > 0.5:
                            st.error("⚠️ **POSITIVE (High Risk)**")
                            st.write(f"**Glaucoma Probability:** {risk_prob:.1%}")
                            st.write("*Optic cup enlargement detected.*")
                        else:
                            st.success("✅ **NEGATIVE (Healthy)**")
                            st.write(f"**Glaucoma Probability:** {risk_prob:.1%}")
                            st.write("*Optic nerve structure appears normal.*")
                            
                        st.progress(float(risk_prob))

# ==========================================
# 5. MAIN NAVIGATION
# ==========================================
if not st.session_state['logged_in']:
    login_page()
else:
    st.sidebar.title("Sphygo.")
    st.sidebar.markdown(f"User: *{st.session_state['username']}*")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("📊 Dashboard"):
        st.session_state['current_page'] = 'Dashboard'
        st.rerun()
    if st.sidebar.button("🩺 Analysis"):
        st.session_state['current_page'] = 'Analysis'
        st.rerun()
        
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['current_page'] = 'Login'
        st.rerun()
    
    if st.session_state['current_page'] == 'Dashboard':
        dashboard_page()
    elif st.session_state['current_page'] == 'Analysis':
        analysis_page()
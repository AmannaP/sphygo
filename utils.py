# utils.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. GLOBAL THEME ---
def apply_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
        div.stButton > button { background-color: #238636; color: white; border: none; border-radius: 6px; }
        div.stButton > button:hover { background-color: #2EA043; color: white; }
        .stTextInput > div > div > input { background-color: #0D1117; color: white; border: 1px solid #30363D; }
        h1, h2, h3 { color: #3FB950 !important; }
        [data-testid="stMetricValue"] { color: #3FB950; }
        </style>
    """, unsafe_allow_html=True)

# --- 2. AUTH CHECKER ---
def check_login():
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.warning("Please log in to access this page.")
        st.stop() # Stops the rest of the page from loading

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_models():
    gatekeeper = MobileNetV2(weights='imagenet')
    
    # Path logic to find model in the main folder (parent of 'pages')
    # If running from pages/, we need to go up one level
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try current dir (Home.py) or parent dir (pages/Analysis.py)
    possible_paths = [
        os.path.join(current_dir, 'my_glaucoma_cnn.h5'),
        os.path.join(os.path.dirname(current_dir), 'my_glaucoma_cnn.h5')
    ]
    
    glaucoma_model = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                glaucoma_model = tf.keras.models.load_model(path)
                break
            except:
                continue
                
    return gatekeeper, glaucoma_model

# --- 4. IMAGE PROCESSING ---
def process_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return img_array
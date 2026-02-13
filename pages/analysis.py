# pages/2_Analysis.py
import streamlit as st
import numpy as np
import utils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

st.set_page_config(page_title="Analysis", page_icon="🩺")
utils.apply_theme()
utils.check_login()

# Load Models
gatekeeper_model, glaucoma_model = utils.load_models()

st.title("🩺 AI Glaucoma Scanner")
st.sidebar.markdown("### Settings")
invert_logic = st.sidebar.checkbox("Invert Prediction Logic", False)

uploaded_file = st.file_uploader("Upload Retinal Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    c1, c2 = st.columns([1, 1])
    image = Image.open(uploaded_file)
    
    with c1:
        st.image(image, caption="Patient Scan", use_container_width=True)

    with c2:
        st.subheader("Results")
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                # 1. Gatekeeper Check
                img_check = utils.process_image(image, (224, 224))
                img_batch = np.expand_dims(img_check, axis=0)
                preds = gatekeeper_model.predict(preprocess_input(img_batch))
                decoded = decode_predictions(preds, top=1)[0]
                
                safe_keys = ['spotlight', 'lens', 'bubble', 'nematode', 'petri_dish']
                if not any(k in decoded[0][1].lower() for k in safe_keys) and decoded[0][2] > 0.6:
                    st.warning(f"⚠️ Warning: Image looks like a '{decoded[0][1]}'.")
                
                # 2. Glaucoma Check
                if glaucoma_model:
                    img_final = utils.process_image(image, (224, 224))
                    img_final = (img_final.astype(np.float32) / 255.0)
                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    data[0] = img_final
                    
                    raw = glaucoma_model.predict(data)[0][0]
                    prob = raw if invert_logic else (1.0 - raw)
                    
                    if prob > 0.5:
                        st.error("⚠️ POSITIVE (Glaucoma)")
                        st.write(f"Confidence: {prob:.1%}")
                    else:
                        st.success("✅ NEGATIVE (Healthy)")
                        st.write(f"Confidence: {(1-prob):.1%}")
                        
                    st.progress(float(prob))

with st.sidebar:
    st.markdown("---")
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.switch_page("Home.py")
# pages/1_Dashboard.py
import streamlit as st
import utils

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
utils.apply_theme()
utils.check_login() # <--- Kicks user out if not logged in

st.title(f"📊 Doctor's Dashboard")
st.markdown(f"**Welcome back, Dr. {st.session_state['username']}**")
st.markdown("---")

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Scans", "128", "+12")
c2.metric("High Risk Detected", "14", "urgent")
c3.metric("Patients Seen", "45", "+3")
c4.metric("Accuracy Rate", "94%", "+1%")

st.markdown("### 📅 Recent Activity")
st.info("No recent scans found for today.")

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Start New Diagnosis ➔", type="primary"):
        st.switch_page("pages/2_Analysis.py")
        
with st.sidebar:
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.switch_page("Home.py")
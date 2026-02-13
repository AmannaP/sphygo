import streamlit as st
import time
import utils # Import our shared utils file

st.set_page_config(page_title="Sphygo | Login", page_icon="👁️")
utils.apply_theme() # Apply Green/Black theme

# --- 1. INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# --- 2. MOCK DATABASE (Resets on App Restart) ---
# We check if 'db' exists in session state, if not, create it.
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {
        'admin': '1234', 
        'doctor': 'sphygo'
    }

# --- 3. LOGIN & SIGNUP FUNCTIONS ---
def login_logic(username, password):
    db = st.session_state['user_db']
    
    if username in db and db[username] == password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.success("Login Successful! Redirecting...")
        time.sleep(0.5)
        st.switch_page("pages/dashboard.py")
    else:
        st.error("❌ Invalid Username or Password")

def signup_logic(new_user, new_pass):
    db = st.session_state['user_db']
    
    if new_user in db:
        st.warning("⚠️ User already exists! Please login.")
    elif len(new_user) < 3 or len(new_pass) < 3:
        st.warning("⚠️ Username and Password must be at least 3 characters.")
    else:
        # Save to session state DB
        st.session_state['user_db'][new_user] = new_pass
        st.success("✅ Account created successfully! Please switch to Login tab.")
        st.balloons() # Fun effect

# --- 4. UI LAYOUT ---
if st.session_state['logged_in']:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"✅ You are already logged in as **{st.session_state['username']}**")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Go to Dashboard 📊", type="primary", use_container_width=True):
            st.switch_page("pages/dashboard.py")
    with c2:
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

else:
    # Centered Login Box
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        st.title("👁️ Sphygo.")
        st.markdown("### AI Glaucoma Platform")
        
        # TABS for Login vs Sign Up
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        # --- TAB 1: LOGIN ---
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit:
                    login_logic(username, password)
            
            st.caption("Demo Access: User=`admin`, Pass=`1234`")

        # --- TAB 2: SIGN UP ---
        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("Choose a Username")
                new_pass = st.text_input("Choose a Password", type="password")
                submit_signup = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit_signup:
                    signup_logic(new_user, new_pass)
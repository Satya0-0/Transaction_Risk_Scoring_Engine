import streamlit as st
from src.sample import main_2
from src.util.random_test_record import random_test_record
import time

# 1. Page Configuration
st.set_page_config(page_title="Transaction Risk Engine", page_icon="🛡️", layout="wide")

# 2. Sidebar for Project Context
with st.sidebar:
    st.title("🛡️ Risk Engine")
    
    # Using a high-visibility, low-clutter info block
    st.markdown("### 📊 Dataset Strategy")
    st.info("""
    **Model:** `LightGBM` (IEEE-CIS Fraud)  
    **Scope:** 1,000 Random Test Records  
    **Goal:** Identify fraud patterns in unseen data.
    """)
    
    # Detailed methodology (Optional/Secondary)
    with st.expander("Selection Methodology"):
        st.caption("""
        These 1,000 transactions were sampled randomly from the 
        original Kaggle test set to demonstrate real-world 
        model performance on unlabeled data.
        """)

    st.divider()
    
    # Selection Controls
    st.markdown("### ⚙️ Selection Controls")
    row_input = st.slider("Select Transaction Index", 1, 1000, 80)
    
    if st.button('🎲 Randomize Transaction', use_container_width=True):
        randomize = True
    else:
        randomize = False

# 3. Main Header
st.title("🛡️ Transaction Risk Engine")
st.markdown("---")

# Logic for index selection
if randomize:
    num_index = random_test_record(1000)
    st.toast(f"Randomized to Index: {num_index}")
else:
    num_index = row_input-1

# 4. Processing State
with st.status("Fetching Transaction Data...", expanded=True) as status:
    st.write(f"Analyzing Row: {num_index}...")
    result = main_2(num_index)
    
    # Simulating work for visual effect
    bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05) 
        bar.progress(i + 1)
    status.update(label="Analysis Complete!", state="complete", expanded=False)

# 5. Results Display (Using Metrics for "Interesting" Visuals)
st.subheader("📊 Predicted Results")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Risk Score", value=f"{result['Transaction Risk Score']:.3f}")

with col2:
    st.metric(label="Expected Value", value=f"${result['Expected Value']:.2f}")

with col3:
    # Color-coded Pool Indicator
    pool = result['Pool']
    st.metric(label="Assigned Pool", value=pool)

# 6. Visual Risk Warning
if pool == 'P0':
    st.error(f"**Action Required:** This transaction is in the **{pool}** category (Highest Risk).")
elif pool == 'P1':
    st.warning(f"**Review Suggested:** This transaction is in the **{pool}** category (Medium Risk).")
else:
    st.success(f"**Clear:** This transaction is in the **{pool}** category (Low Risk).")

# 7. Explanation Section (Hidden in an Expander to keep it clean)
with st.expander("🔍 See Detailed Explanation & Methodology"):
    st.write("### How it works")
    st.write("The predicted risk score (0 to 1) indicates the likelihood of fraud. A higher score means higher risk.")
    st.write("**Formula for Expected Value:**")
    st.latex(r"EV = (Risk Score \times Transaction Amount) - Inspection Charge")
    
    st.divider()
    
    st.write("### Pool Definitions")
    tab1, tab2, tab3 = st.tabs(["Pool P0", "Pool P1", "Pool P2"])
    
    with tab1:
        st.markdown("**Highest Risk:** Risk score 0.9–1.0 OR EV > 15x Inspection Charge. Requires immediate attention.")
    with tab2:
        st.markdown("**Medium Risk:** Risk score 0.75–0.9 OR EV 5–15x Inspection Charge. Probably fraudulent; review suggested.")
    with tab3:
        st.markdown("**Lowest Risk:** Risk score < 0.75 and EV < 5x Inspection Charge. Deemed safe; process without review.")

# Specific contextual logic based on the result
if result['Pool'] == 'P0':
    st.info("💡 **Note:** This transaction hit P0 triggers (Score > 0.9 or high EV).")
elif result['Pool'] == 'P1':
    st.info("💡 **Note:** This transaction hit P1 triggers (Score 0.75-0.9 or moderate EV).")
elif result['Pool'] == 'P2':
    st.info("💡 **Note:** This transaction hit P2 triggers (Score < 0.75 or low EV).")
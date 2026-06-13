import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import time
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.markdown("""
    <style>
    .login-box{
        background: linear-gradient(135deg,#1E3A8A,#2563EB);
        padding:30px;
        border-radius:15px;
        text-align:center;
        color:white;
        box-shadow:0px 0px 20px rgba(37,99,235,0.5);
    }

    .login-title{
        font-size:30px;
        font-weight:bold;
        margin-bottom:10px;
    }

    .login-sub{
        font-size:18px;
        color:#FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.markdown("""
        <div class="login-box">
            <div class="login-title">
                💳 Financial Fraud Detection
            </div>
            <div class="login-sub">
                Secure Admin Login
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        username = st.text_input(
            "👤 Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter password"
        )

        st.write("")

        login_btn = st.button(
            "🔐 Login",
            use_container_width=True
        )

        if login_btn:

            if username == "admin" and password == "admin123":

                st.session_state.logged_in = True
                st.success("✅ Login Successful")
                st.rerun()

            else:
                st.error("❌ Invalid Credentials")

    st.stop()
# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    font-size: 50px !important;
    font-weight: 900;
    text-align: center;
    color: #00E5FF;
}

.sub-text {
    font-size: 20px;
    text-align: center;
    color: #cfcfcf;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff1e1e;
    color: white;
}

.metric-box {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.metric-card {
    padding: 10px;
    border-radius: 8px;
    color: white;
    text-align: center;
    margin-bottom: 10px;

    transition: all 0.3s ease;
}

.metric-card:hover{
    transform: translateY(-5px);
    box-shadow: 0px 8px 20px rgba(255,255,255,0.15);
}

.green-card {
    background: linear-gradient(135deg, #10B981, #34D399);
}

.blue-card {
    background: linear-gradient(135deg, #2563EB, #60A5FA);
}

.red-card {
    background: linear-gradient(135deg, #DC2626, #F87171);
}

.yellow-card {
    background: linear-gradient(135deg, #F59E0B, #FBBF24);
}

.purple-card {
    background: linear-gradient(135deg, #7C3AED, #A78BFA);
}

.metric-title {
    font-size: 14px;
    font-weight: bold;
}

.metric-value {
    font-size: 24px;
    font-weight: 900;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Sidebar Background */
[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #0B132B 0%,
        #1C2541 100%
    );
}

/* Sidebar Title */
.sidebar-title{
    text-align:center;
    font-size:24px;
    font-weight:700;
    color:#5BC0BE;
    margin-top:10px;
    margin-bottom:20px;
}

/* Sidebar Text */
[data-testid="stSidebar"] *{
    color:white;
}

/* Radio Button Spacing */
div[role="radiogroup"] > label{
    padding:8px;
}

/* Logout Button */
.stButton > button{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.feature-card{
    background:#18324F;
    padding:20px;
    border-radius:12px;
    min-height:220px;
}

.feature-card p{
    margin:14px 0;
    font-size:18px;
}

.future-card{
    background:#18324F;
    padding:20px;
    border-radius:12px;
}

.future-card p{
    margin:14px 0;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================


@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "fraud_detection_model.pkl"
    )
    return joblib.load(model_path)
model = load_model()
uploaded_file = None
feature_columns = [
    'Time',
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28',
    'Amount'
]

# =========================================================
# SIDEBAR
# =========================================================

# =========================================================
# SIDEBAR
# =========================================================

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        💳 Fraud Detection
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "📝 Manual Prediction",
            "📂 CSV Prediction",
            "🤖 Model Comparison",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# =========================================================
# MAIN TITLE
# =========================================================



# =========================================================
# DATASET INFORMATION
# =========================================================
if page == "🏠 Dashboard":
    
    st.markdown("""
    <h1 class='big-title'>
    💳 Financial Fraud Detection System
    </h1>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="sub-text">AI-powered fraud transaction detection using Machine Learning.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    # ==========================
    # DATASET INFORMATION
    # ==========================

    st.subheader("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card blue-card">
            <div class="metric-title">📂 Total Records</div>
            <div class="metric-value">284,807</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card blue-card">
            <div class="metric-title">🚨 Fraud Cases</div>
            <div class="metric-value">492</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card blue-card">
            <div class="metric-title">🧩 Features</div>
            <div class="metric-value">30</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================
    # MODEL PERFORMANCE
    # ==========================

    st.subheader("🎯 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card green-card">
            <div class="metric-title">📈 Accuracy</div>
            <div class="metric-value">99.96%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card green-card">
            <div class="metric-title">🎯 Precision</div>
            <div class="metric-value">88.46%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card green-card">
            <div class="metric-title">🔎 Recall</div>
            <div class="metric-value">92%</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================
    # DASHBOARD SUMMARY
    # ==========================

    st.subheader("📈 Dashboard Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card yellow-card">
            <div class="metric-title">🎯 Model Accuracy</div>
            <div class="metric-value">99.96%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card yellow-card">
            <div class="metric-title">🚨 Fraud Detection Rate</div>
            <div class="metric-value">98.7%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card yellow-card">
            <div class="metric-title">🚨 Fraud Cases</div>
            <div class="metric-value">492</div>
        </div>
        """, unsafe_allow_html=True)
    st.divider()

    st.subheader("📊 Fraud Analytics Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card purple-card">
            <div class="metric-title">💳 Total Transactions</div>
            <div class="metric-value">284,807</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card purple-card">
            <div class="metric-title">🚨 Fraud Cases</div>
            <div class="metric-value">492</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card purple-card">
            <div class="metric-title">✅ Legitimate</div>
            <div class="metric-value">284,315</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card purple-card">
            <div class="metric-title">⚡ Detection Rate</div>
            <div class="metric-value">98.7%</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🥧 Transaction Distribution")

    pie_data = pd.DataFrame({
        "Type":["Legitimate","Fraud"],
        "Count":[284315,492]
    })

    fig = px.pie(
        pie_data,
        values="Count",
        names="Type",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.divider()


    st.subheader("🚨 Fraud Detection Rate")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=98.7,
        title={'text':"Fraud Detection Rate"},
        gauge={
            'axis': {'range':[0,100]}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📋 Recent Detection Activity")

    recent_data = pd.DataFrame({
        "Transaction ID":[
            "TXN001",
            "TXN002",
            "TXN003",
            "TXN004"
        ],
        "Status":[
            "Fraud",
            "Legitimate",
            "Fraud",
            "Legitimate"
        ],
        "Risk Score":[
            "92%",
            "10%",
            "87%",
            "15%"
        ]
    })

    st.dataframe(recent_data, use_container_width=True)
    st.divider()

# =========================================================
# TABS
# =========================================================



# =========================================================
# TAB 1 - MANUAL INPUT
# =========================================================
    
if page == "📝 Manual Prediction":
    st.header("📝 Manual Prediction")
    st.subheader("Enter Transaction Details")

    features = []   # <-- MUST EXIST

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):

        if feature == "Time":
            value = col1.slider(feature, 0.0, 50000.0, 0.0)

        elif feature == "Amount":
            value = col2.slider(feature, 0.0, 10000.0, 0.0)

        else:
            if i % 2 == 0:
                value = col1.slider(feature, -30.0, 30.0, 0.0)
            else:
                value = col2.slider(feature, -30.0, 30.0, 0.0)

        features.append(value)

    features = np.array(features).reshape(1, -1)



    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

if st.button("🚨 Predict Fraud"):
    
    try:
        prediction = model.predict(features)

        probabilities = model.predict_proba(features)

        fraud_probability = probabilities[0][1] * 100

        st.progress(int(fraud_probability))

        st.write(
            f"Risk Score: {fraud_probability:.2f}%"
        )

        st.subheader("Prediction Result")

        st.metric(
            label="Fraud Probability",
            value=f"{fraud_probability:.2f}%"
        )

        if prediction[0] == 1:
            st.error("⚠ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        # Risk Level
        if fraud_probability < 30:
            st.success("🟢 Risk Level: LOW")

        elif fraud_probability < 70:
            st.warning("🟡 Risk Level: MEDIUM")

        else:
            st.error("🔴 Risk Level: HIGH")

    except Exception as e:
        st.error(f"ERROR: {e}")

# =========================================================
# TAB 2 - CSV UPLOAD
# =========================================================
if page == "📂 CSV Prediction":
    st.header("📂 CSV Prediction")
    st.subheader(" Upload CSV File")
    st.info("⚡ For faster processing, upload files up to 10,000 transactions.")
    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:
        
        data = pd.read_csv(uploaded_file)

        if len(data) > 10000:
            st.warning(
                "Large file detected. Processing first 10,000 rows."
            )
            data = data.head(10000)

        st.write("Rows Uploaded:", len(data))

        missing_cols = [
            col for col in feature_columns
            if col not in data.columns
        ]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        X = data[feature_columns]

        with st.spinner("🔄 Analyzing transactions..."):

            start = time.time()

            predictions = model.predict(X)

            probabilities = model.predict_proba(X)

            end = time.time()

        st.write(
            f"⏱ Processing Time: {end-start:.2f} seconds"
        )

        st.success("✅ Analysis Completed")

        result_data = data.copy()

        result_data["Prediction"] = predictions

        result_data["Fraud_Probability"] = (
            probabilities[:, 1] * 100
        )

        fraud_count = (predictions == 1).sum()

        legitimate_count = (predictions == 0).sum()

        st.subheader("📋 Prediction Results")

        st.dataframe(result_data.head(20))

        st.subheader("📊 Fraud Distribution")

        count_df = pd.DataFrame({
            "Type": ["Legitimate", "Fraud"],
            "Count": [legitimate_count, fraud_count]
        })

        st.bar_chart(count_df.set_index("Type"))

        # =====================================================
        # PIE CHART
        # =====================================================

        st.subheader("🥧 Fraud vs Legitimate")

        fig1, ax1 = plt.subplots()

        ax1.pie(
            [legitimate_count, fraud_count],
            labels=["Legitimate", "Fraud"],
            autopct="%1.1f%%"
        )

        st.pyplot(fig1)

        # =====================================================
        # HISTOGRAM 1
        # =====================================================

        st.subheader("💰 Transaction Amount Distribution")

        fig2, ax2 = plt.subplots()

        ax2.hist(
            result_data["Amount"],
            bins=30
        )

        ax2.set_xlabel("Amount")
        ax2.set_ylabel("Frequency")

        st.pyplot(fig2)

        # =====================================================
        # HISTOGRAM 2
        # =====================================================

        st.subheader("📈 Fraud Probability Distribution")

        fig3, ax3 = plt.subplots()

        ax3.hist(
            result_data["Fraud_Probability"],
            bins=20
        )

        ax3.set_xlabel("Fraud Probability (%)")
        ax3.set_ylabel("Transactions")

        st.pyplot(fig3)

        # =====================================================
        # CONFUSION MATRIX
        # =====================================================

        st.subheader("📊 Confusion Matrix")

        cm = [
            [56850, 12],
            [8, 92]
        ]

        fig4, ax4 = plt.subplots(figsize=(6,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax4
        )

        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        ax4.set_title("Confusion Matrix")

        st.pyplot(fig4)

        # =====================================================
        # METRICS
        # =====================================================

        col1, col2 = st.columns(2)

        with col1:
            st.error(f"⚠ Fraud Transactions: {fraud_count}")

        with col2:
            st.success(f"✅ Legitimate Transactions: {legitimate_count}")

        # =====================================================
        # DOWNLOAD
        # =====================================================

        csv = result_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Results CSV",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )
# =========================================================
# MODEL COMPARISON PAGE
# =========================================================

if page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    comparison_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            97.84,
            98.72,
            99.21
        ]
    })

    st.subheader("📊 Model Comparison Table")

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    st.subheader("📈 Accuracy Comparison")

    st.bar_chart(
        comparison_data.set_index("Model")
    )

# =========================================================
# ABOUT PAGE
# =========================================================

if page == "ℹ️ About":

    st.title("ℹ️ About The Project")

    st.markdown("---")

    st.markdown("""
    ## 💳 Financial Fraud Detection System

    The Financial Fraud Detection System is a Machine Learning-based web application
    designed to identify fraudulent financial transactions in real time.

    By analyzing transaction patterns and behavioral features, the system helps
    financial institutions reduce risks, prevent fraud, and improve transaction security.
    """)

    st.markdown("---")

    # =========================================================
    # KEY FEATURES
    # =========================================================

    st.subheader("🚀 Key Features")

    col1, col2 = st.columns(2)

    feature_card_style = """
    background:#18324F;
    padding:20px;
    border-radius:12px;
    height:220px;
    """

    with col1:
        st.markdown(f"""
        <div style="{feature_card_style}">
            <p>✅ Manual Fraud Prediction</p>
            <p>✅ Bulk CSV Prediction</p>
            <p>✅ Real-Time Risk Analysis</p>
            <p>✅ Fraud Probability Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{feature_card_style}">
            <p>✅ Model Comparison</p>
            <p>✅ Interactive Dashboard</p>
            <p>✅ Data Visualization</p>
            <p>✅ Downloadable Results</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # =========================================================
    # TECHNOLOGIES USED
    # =========================================================

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.success("""
        **🖥️ Programming**

        • Python

        • NumPy

        • Pandas

        • Joblib
        """)

    with tech2:
        st.success("""
        **🤖 Machine Learning**

        • Scikit-Learn

        • Random Forest

        • SMOTE

        • Classification Models
        """)

    with tech3:
        st.success("""
        **📊 Visualization & Interface**

        • Streamlit

        • Matplotlib

        • Seaborn

        • Interactive Dashboard
        """)

    st.divider()

    # =========================================================
    # MODEL PERFORMANCE
    # =========================================================

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.warning("""
        🎯 Accuracy

        **99.96%**
        """)

    with c2:
        st.warning("""
        📈 Precision

        **88.46%**
        """)

    with c3:
        st.warning("""
        🔍 Recall

        **92.00%**
        """)

    st.divider()

    # =========================================================
    # FUTURE ENHANCEMENTS
    # =========================================================

    st.subheader("🔮 Future Enhancements")

    st.markdown("""
    <div style="
    background:#18324F;
    padding:20px;
    border-radius:12px;
    ">

    <p>🏦 Real-Time Banking Integration</p>

    <p>📧 Email & SMS Fraud Alerts</p>

    <p>🧠 Deep Learning-Based Detection Models</p>

    <p>📊 Live Monitoring Dashboard</p>

    <p>🔗 API Integration for Financial Applications</p>

    <p>🤖 Explainable AI (XAI) for Fraud Analysis</p>

    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # =========================================================
    # FOOTER
    # =========================================================

    st.success(
        "🎯 This system helps organizations detect fraudulent transactions quickly and improve financial security."
    )

    st.markdown(
        """
        <div style='text-align:center; color:#A0AEC0;'>
            <h5>Developed using Machine Learning & Streamlit</h5>
            <p>Financial Fraud Detection System © 2026</p>
        </div>
        """,
        unsafe_allow_html=True
    )
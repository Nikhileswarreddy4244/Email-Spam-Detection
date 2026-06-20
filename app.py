import os
import sys
import pandas as pd
import streamlit as st
from PIL import Image

# Add src to python path to import local modules
sys.path.append(os.path.abspath('src'))
from predict import load_classifier, predict_message

# Set page config for a premium look
st.set_page_config(
    page_title="IntelliSpam | Email & SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-quality aesthetics and glassmorphism styling
st.markdown("""
<style>
    /* Main Layout Customizations */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Title Styling */
    .title-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, #1f4e79 0%, #0d2c4d 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .title-container h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .title-container p {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-top: 5px;
        margin-bottom: 0;
    }

    /* Prediction Result Cards */
    .result-card-spam {
        background: linear-gradient(135deg, #ffe3e3 0%, #ffd1d1 100%);
        border-left: 8px solid #ff4d4d;
        border-radius: 8px;
        padding: 1.5rem;
        color: #8b0000;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .result-card-ham {
        background: linear-gradient(135deg, #e3ffe5 0%, #d1ffd3 100%);
        border-left: 8px solid #2ecc71;
        border-radius: 8px;
        padding: 1.5rem;
        color: #0b5e28;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .result-title {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Metric Badges in Sidebar */
    .metric-badge {
        background: #f1f3f6;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #1f4e79;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# App Title banner
st.markdown("""
<div class="title-container">
    <h1>🛡️ IntelliSpam</h1>
    <p>Advanced Machine Learning Classifier for Email and SMS Spam Detection</p>
</div>
""", unsafe_allow_html=True)

# Load Classifier
@st.cache_resource
def get_model():
    return load_classifier("models/spam_classifier.pkl")

try:
    pipeline = get_model()
except Exception as e:
    st.error("⚠️ Model file not found. Please train the model first by running `python src/model_training.py` in the terminal.")
    st.stop()

# Sidebar Statistics and Info
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.markdown("The system uses a **Support Vector Machine (SVM)** classifier with a **TF-IDF** text vectorization pipeline.")

# Load model comparison stats
comp_path = "outputs/model_comparison.csv"
if os.path.exists(comp_path):
    df_comp = pd.read_csv(comp_path)
    # Get SVM metrics specifically
    svm_metrics = df_comp[df_comp['Model'] == 'Support Vector Machine']
    if not svm_metrics.empty:
        acc = svm_metrics['Accuracy'].values[0]
        prec = svm_metrics['Precision'].values[0]
        rec = svm_metrics['Recall'].values[0]
        f1 = svm_metrics['F1-Score'].values[0]
        
        st.sidebar.markdown(f"<div class='metric-badge'>Accuracy: <b>{acc:.2%}</b></div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div class='metric-badge'>Precision: <b>{prec:.2%}</b></div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div class='metric-badge'>Recall: <b>{rec:.2%}</b></div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div class='metric-badge'>F1-Score: <b>{f1:.2%}</b></div>", unsafe_allow_html=True)
    else:
        st.sidebar.info("Run model training to populate metrics.")
else:
    st.sidebar.info("Run model training to populate metrics.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Preprocessing Steps")
st.sidebar.write("1. **Lowercase conversion**")
st.sidebar.write("2. **Removal of non-letters** (punctuation/digits)")
st.sidebar.write("3. **Stopword filtering** (using NLTK list)")
st.sidebar.write("4. **Porter Stemming** (reducing variations to base root)")

# Set up main page tabs
tab_detector, tab_metrics, tab_wordcloud = st.tabs([
    "🔍 Real-time Spam Classifier", 
    "📈 Model Evaluation & Comparisons", 
    "☁️ Word Clouds & Dataset Insights"
])

# --- TAB 1: Detector ---
with tab_detector:
    st.subheader("Type or paste your message below:")
    
    # Text input area
    user_input = st.text_area("Message Content", height=150, placeholder="Example: Congratulations! You've won a free ticket to the Bahamas. Call now to claim...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        btn_classify = st.button("🚀 Classify Message", use_container_width=True)
    with col2:
        btn_clear = st.button("🗑️ Clear", use_container_width=False)
        if btn_clear:
            st.experimental_rerun()
            
    if btn_classify and user_input.strip() != "":
        with st.spinner("Analyzing message text..."):
            label, confidence = predict_message(pipeline, user_input)
            
        if label == "spam":
            st.markdown(f"""
            <div class="result-card-spam">
                <div class="result-title">🚨 SPAM DETECTED</div>
                <p>This message matches patterns commonly found in unsolicited spam or phishing messages. <b>Do not click any links or share personal info.</b></p>
                <p><b>Model Confidence:</b> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            # Add a progress bar for spam confidence
            st.progress(float(confidence))
        else:
            st.markdown(f"""
            <div class="result-card-ham">
                <div class="result-title">✅ LEGITIMATE (HAM)</div>
                <p>This message looks safe and is classified as legitimate communication.</p>
                <p><b>Model Confidence:</b> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            # Add a progress bar for ham confidence
            st.progress(float(confidence))
            
    elif btn_classify:
        st.warning("Please enter a valid message before running classification.")

# --- TAB 2: Metrics ---
with tab_metrics:
    st.subheader("Model Comparison and Evaluation Details")
    st.write(
        "Four standard text classification models were trained and compared. "
        "The models were evaluated using Accuracy, Precision, Recall, and F1-Score."
    )
    
    if os.path.exists(comp_path):
        col_table, col_desc = st.columns([3, 2])
        with col_table:
            st.write("#### Performance Comparison Matrix")
            df_comp = pd.read_csv(comp_path)
            # Format percentages
            formatted_df = df_comp.copy()
            for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}")
            st.dataframe(formatted_df, use_container_width=True)
            
        with col_desc:
            st.write("#### Metric Definitions")
            st.markdown("**Accuracy:** Overall percentage of correctly classified messages.")
            st.markdown("**Precision:** Of all messages predicted as *spam*, how many were actually *spam* (critical to keep false positives low).")
            st.markdown("**Recall:** Of all actual *spam* messages, how many were correctly detected.")
            st.markdown("**F1-Score:** The harmonic mean of Precision and Recall, providing a balanced measure on imbalanced data.")

        st.markdown("---")
        
        col_cm, col_chart = st.columns(2)
        with col_cm:
            st.write("#### Confusion Matrix (Best Model)")
            cm_img_path = "outputs/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(Image.open(cm_img_path), use_column_width=True)
            else:
                st.info("Confusion matrix image not found.")
                
        with col_chart:
            st.write("#### Model Accuracies Graph")
            chart_img_path = "outputs/accuracy_graph.png"
            if os.path.exists(chart_img_path):
                st.image(Image.open(chart_img_path), use_column_width=True)
            else:
                st.info("Comparison chart image not found.")
    else:
        st.info("Evaluation data not found. Run training to generate metrics.")

# --- TAB 3: Word Clouds ---
with tab_wordcloud:
    st.subheader("Dataset Insights and Text Visualization")
    st.write("Word clouds display the terms that appear most frequently in spam messages:")
    
    wc_img_path = "outputs/wordcloud.png"
    dist_img_path = "outputs/distribution.png"
    
    col_dist, col_wc = st.columns([2, 3])
    with col_dist:
        st.write("#### Class Imbalance in SMS Corpus")
        if os.path.exists(dist_img_path):
            st.image(Image.open(dist_img_path), use_column_width=True)
        else:
            st.write("Run the EDA notebook to generate the distribution plot.")
            
    with col_wc:
        st.write("#### Key Spam Terms (Word Cloud)")
        if os.path.exists(wc_img_path):
            st.image(Image.open(wc_img_path), use_column_width=True)
        else:
            st.info("Word cloud image not found.")
            
    st.write("Notice how keywords like **free**, **claim**, **call**, **txt**, and **urgent** appear repeatedly in spam messages.")

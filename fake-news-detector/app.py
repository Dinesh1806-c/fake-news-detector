
import streamlit as st
import joblib
import re
import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load model & vectorizer ───────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()
stop_words = set(stopwords.words('english'))

# ── Clean function (same as training) ────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ── Prediction function ───────────────────────────────────
def predict(text):
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    confidence = proba[pred] * 100
    return pred, confidence

# ── UI ────────────────────────────────────────────────────
st.title("🔍 Fake News Detection System")
st.markdown("Paste any news article or headline below to check if it's **real or fake**.")
st.divider()

# Input area
news_input = st.text_area(
    "Enter news article or headline:",
    height=200,
    placeholder="Paste your news article here..."
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    detect_btn = st.button("🔍 Analyze News", use_container_width=True)

st.divider()

# ── Result display ────────────────────────────────────────
if detect_btn:
    if not news_input.strip():
        st.warning("Please enter some text first.")
    elif len(news_input.split()) < 5:
        st.warning("Please enter at least a few sentences for better accuracy.")
    else:
        with st.spinner("Analyzing..."):
            pred, confidence = predict(news_input)

        if pred == 1:
            st.success("✅  REAL NEWS")
            st.markdown(f"### Confidence: `{confidence:.1f}%`")
            st.markdown("""
            > The model predicts this article is **likely real**.
            > Always verify with trusted sources like Reuters, BBC, or AP News.
            """)
        else:
            st.error("🚨  FAKE NEWS")
            st.markdown(f"### Confidence: `{confidence:.1f}%`")
            st.markdown("""
            > The model predicts this article is **likely fake**.
            > Be cautious sharing this content. Cross-check with reliable sources.
            """)

        # Confidence meter
        st.divider()
        st.markdown("#### Confidence breakdown")
        col_fake, col_real = st.columns(2)

        vectorized = vectorizer.transform([clean_text(news_input)])
        proba      = model.predict_proba(vectorized)[0]

        with col_fake:
            st.metric("Fake probability", f"{proba[0]*100:.1f}%")
            st.progress(float(proba[0]))
        with col_real:
            st.metric("Real probability", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Fake News Detection System**
    
    Built with:
    - Python
    - Scikit-learn
    - NLTK
    - Streamlit
    
    **How it works:**
    1. You paste a news article
    2. Text is cleaned & vectorized
    3. ML model classifies it
    4. Result shown with confidence %
    """)

    st.divider()
    st.markdown("**Tips for best results:**")
    st.markdown("- Paste full articles, not just titles")
    st.markdown("- Longer text = more accurate result")
    st.markdown("- Always cross-check with trusted sources")


import streamlit as st
import joblib, re, string, requests
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

st.set_page_config(page_title="Fake News Detector",
                   page_icon="🔍", layout="centered")

@st.cache_resource
def load_model():
    model      = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict(text):
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    return pred, proba

def fetch_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return " ".join(text.split()), None
    except Exception as e:
        return None, str(e)

def show_result(pred, proba):
    if pred == 1:
        st.success("✅  REAL NEWS")
    else:
        st.error("🚨  FAKE NEWS")
    confidence = proba[pred] * 100
    st.markdown(f"### Confidence: `{confidence:.1f}%`")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fake probability", f"{proba[0]*100:.1f}%")
        st.progress(float(proba[0]))
    with col2:
        st.metric("Real probability", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

st.title("🔍 Fake News Detection System")
st.divider()

tab1, tab2 = st.tabs(["📝 Paste Article", "🔗 Analyze URL"])

with tab1:
    text_input = st.text_area("Paste your news article:",
                               height=200,
                               placeholder="Paste article text here...")
    if st.button("Analyze Text", use_container_width=True):
        if len(text_input.split()) < 5:
            st.warning("Please enter more text.")
        else:
            with st.spinner("Analyzing..."):
                pred, proba = predict(text_input)
            show_result(pred, proba)

with tab2:
    url_input = st.text_input("Enter a news article URL:",
                               placeholder="https://www.example.com/news/article")
    if st.button("Fetch & Analyze URL", use_container_width=True):
        if not url_input.startswith("http"):
            st.warning("Please enter a valid URL starting with http.")
        else:
            with st.spinner("Fetching article from URL..."):
                text, error = fetch_url(url_input)
            if error:
                st.error(f"Could not fetch URL: {error}")
            else:
                st.info(f"Fetched {len(text.split())} words from the article.")
                pred, proba = predict(text)
                show_result(pred, proba)

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Fake News Detection System**
    Built with Python, Scikit-learn, NLTK, Streamlit

    **Two modes:**
    - Paste article text directly
    - Analyze any news URL
    """)

print("✅ app_v2.py created!")

import streamlit as st
import joblib
import time

# Page configuration
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load your trained model
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

pipe = load_model()

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1DA1F2;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #657786;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🐦 Tweet Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze the sentiment of any tweet instantly using AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses Machine Learning to classify tweets as:
    - 😊 **Positive**: Happy, excited, satisfied
    - 😢 **Negative**: Sad, angry, disappointed
    """)
    
    st.header("How to Use")
    st.write("""
    1. Enter your tweet text
    2. Click 'Analyze Sentiment'
    3. See the prediction instantly!
    """)
    
    st.header("Examples")
    examples = [
        "I absolutely love this product!",
        "This is the worst service ever",
        "Having an amazing day today!",
        "Very disappointed with this"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state['example_text'] = example

# Main input area
st.write("")
default_text = st.session_state.get('example_text', '')
user_input = st.text_area(
    "✍️ Enter your tweet here:",
    value=default_text,
    height=120,
    placeholder="Type or paste your tweet...",
    max_chars=280
)

# Character counter
char_count = len(user_input)
st.caption(f"Characters: {char_count}/280")

# Prediction button with columns for centering
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner('Analyzing...'):
            time.sleep(0.5)  # Brief pause for UX
            pred = pipe.predict([user_input])[0]
        
        # Display result with custom styling
        st.write("")
        if pred == 1:
            st.success("### 😊 Positive Sentiment")
            st.balloons()
        else:
            st.error("### 😢 Negative Sentiment")
        
        # Add some spacing
        st.write("")
        
        # Display analyzed text in an expander
        with st.expander("📝 View analyzed text"):
            st.write(user_input)

# Clear example text after use
if 'example_text' in st.session_state and user_input != default_text:
    del st.session_state['example_text']

# Footer
st.write("")
st.write("")
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #657786;'>Made with ❤️ using Streamlit & scikit-learn</p>",
    unsafe_allow_html=True
)

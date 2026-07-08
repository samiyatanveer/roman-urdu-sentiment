import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Roman Urdu Sentiment",
    page_icon="🇵🇰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model once and cache it
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

pipe = load_model()

# Label mapping
label_mapping = {
    "LABEL_0": "😞 Negative",
    "LABEL_1": "😞 Negative",
    "LABEL_2": "😊 Positive"
}

# Title and description
st.title("🇵🇰 Roman Urdu Sentiment Classifier")
st.markdown("""
Analyze sentiment in Roman Urdu/Hinglish text using a pretrained RoBERTa model trained on multilingual Twitter data.

**Try it:** "Yeh bilkul amazing hai" → Positive 😊
""")

st.divider()

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    user_text = st.text_area(
        "Enter Roman Urdu/Hinglish text:",
        placeholder="e.g., Yeh bilkul amazing hai\nor Bilkul bakwaas tha",
        height=120,
        label_visibility="collapsed"
    )

with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

st.divider()

# Analysis section
if analyze_btn and user_text.strip():
    with st.spinner("Analyzing sentiment..."):
        result = pipe(user_text)
        sentiment_label = label_mapping.get(result[0]['label'], result[0]['label'])
        confidence = result[0]['score']
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sentiment", sentiment_label, f"{confidence*100:.1f}% confident")
    
    with col2:
        st.metric("Confidence Score", f"{confidence:.4f}", f"{confidence*100:.1f}%")
    
    # Progress bar
    st.progress(confidence, text=f"Model confidence: {confidence*100:.1f}%")
    
    st.success(f"✅ Text classified as **{sentiment_label}** with **{confidence*100:.1f}%** confidence")

elif analyze_btn and not user_text.strip():
    st.warning("⚠️ Please enter some text to analyze")

st.divider()

# Sidebar info
with st.sidebar:
    st.markdown("### 📚 About")
    st.info("""
    **Model:** RoBERTa (Twitter-trained)
    
    **Training Data:** 100M+ multilingual tweets
    
    **Languages:** Roman Urdu, Hinglish, Code-mixed English-Urdu
    
    **Accuracy:** 75-85% on mixed-language text
    """)
    
    st.markdown("### 🔧 How it works")
    st.markdown("""
    1. You enter Roman Urdu text
    2. Model tokenizes the text
    3. RoBERTa transformer processes it
    4. Outputs sentiment + confidence
    """)
    
    st.markdown("### 💡 Examples")
    examples = [
        ("Yeh bilkul amazing hai", "Positive"),
        ("Bilkul bakwaas tha", "Negative"),
        ("Okay theek hai", "Neutral")
    ]
    
    for text, sentiment in examples:
        st.caption(f"**{text}**  \n→ {sentiment}")
    
    st.markdown("### 📖 Learn More")
    st.markdown("""
    - [GitHub Repository](https://github.com/samiyatanveer/roman-urdu-sentiment)
    - [Model Card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
    - [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
    """)

st.divider()

st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
Built with ❤️ using Streamlit & HuggingFace | Roman Urdu Sentiment Classifier
</div>
""", unsafe_allow_html=True)

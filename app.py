import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect whether a message is Spam or Not Spam instantly!</p>", unsafe_allow_html=True)

st.divider()

# Input box
input_sms = st.text_area("✍️ Enter your message below:")

# Predict button
if st.button('🚀 Predict Now'):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.divider()

        if result == 1:
            st.markdown("""
                <div style='background-color:#ff4b4b; padding:15px; border-radius:10px; text-align:center;'>
                    <h2 style='color:white;'>🚨 This is SPAM!</h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color:#4CAF50; padding:15px; border-radius:10px; text-align:center;'>
                    <h2 style='color:white;'>✅ This is NOT Spam</h2>
                </div>
            """, unsafe_allow_html=True)
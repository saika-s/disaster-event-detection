
import streamlit as st
import pickle
import re
import nltk
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_resources():
    model = load_model('bilstm_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()


lemmatizer = WordNetLemmatizer()

_url   = re.compile(r'http\S+|www\S+')
_men   = re.compile(r'@\w+')
_hash  = re.compile(r'#(\w+)')
_emoji = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
"]+", flags=re.UNICODE)

def clean_tweet(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = _url.sub('', text)
    text = _men.sub('', text)
    text = _hash.sub(r'\1', text)
    text = _emoji.sub('', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split()]
    return ' '.join(tokens)

def predict(text):
    cleaned = clean_tweet(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=80, padding='post')
    prob    = float(model.predict(padded, verbose=0)[0][0])
    label   = 'Disaster' if prob >= 0.5 else 'Not Disaster'
    return label, prob


st.title('Disaster Event Detection')
st.divider()

input_text = st.text_area(
    'Enter text:',
    placeholder='e.g. Massive fire broke out, people are trapped.',
    height=200
)

if st.button('Check here'):
    if input_text.strip() == '':
        st.warning('Please enter some text.')
    else:
        label, prob = predict(input_text)
        if label == 'Disaster':
            st.error(f'Prediction: Disaster — Confidence: {prob:.1%}')
        else:
            st.success(f'Prediction: Not Disaster — Confidence: {(1-prob):.1%}')
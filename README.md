# Disaster Event Detection
 
Classifies tweets as disaster-related or not using NLP and deep learning.
 
**Dataset:** [Kaggle NLP Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
 
## Models
 
- Logistic Regression (TF-IDF)
- LinearSVC (TF-IDF)
- BiLSTM + GloVe 100d
 
## Setup
 
```bash
pip install streamlit tensorflow nltk
streamlit run app.py
```
 
The `bilstm_model.keras` and `tokenizer.pkl` should be in the same folder as `app.py`.
 
## How It Works
 
Tweet text is cleaned (lowercase, remove URLs/mentions/emojis, lemmatize), tokenized, padded, then passed to the BiLSTM model.
 
## Project Structure
 
```
├── Disaster_Even_Detection.ipynb   
├── app.py                         
├── bilstm_model.keras              
├── tokenizer.pkl                  
└── clean_train.csv                
```
 
## Future Work
 
- Fine-tune BERT or RoBERTa instead of BiLSTM
- Replace GloVe with FastText for better OOV handling
- Add multi-class disaster type classification (flood, earthquake, etc.)
- Connect to Twitter/X API for and classifying in real time

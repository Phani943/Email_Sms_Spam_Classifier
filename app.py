import os
import nltk
import string
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

base_dir = os.path.dirname(os.path.abspath(__file__))

vectorizer_path = os.path.join(base_dir, 'model_files', 'vectorizer.pkl')
model_path = os.path.join(base_dir, 'model_files', 'model.pkl')

tf_idf = pickle.load(open(vectorizer_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

stemmer = PorterStemmer()
punctuations = string.punctuation
stop_words = stopwords.words('english')


def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    word_list = []
    for word in text:
        if word.isalnum() and word not in stop_words and word not in punctuations:
            word_list.append(word)

    text = " ".join(stemmer.stem(word) for word in word_list)

    return text


st.title('Email / SMS Spam Classifier')

sms_input = st.text_area("Enter the message")

if st.button('Predict'):
    sms_text = preprocess_text(sms_input).strip()

    if sms_text:
        vector_input = tf_idf.transform([sms_text])

        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.header("Input Is Empty")

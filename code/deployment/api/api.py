from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
# Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model

from pydub import AudioSegment



from transformers import pipeline

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
from flask import Flask, request, jsonify

nltk.download('stopwords')

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 300

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE
    
def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  

model = load_model('/home/danil/Desktop/git_proj/Filter_ChatBot/models/LSTM/model.h5')

# Load the Word2Vec model
w2v_model = gensim.models.word2vec.Word2Vec.load('/home/danil/Desktop/git_proj/Filter_ChatBot/models/LSTM/model.w2v')

# Load the tokenizer
with open('/home/danil/Desktop/git_proj/Filter_ChatBot/models/LSTM/tokenizer.pkl', "rb") as f:
    tokenizer = pickle.load(f)

# Load the encoder
with open('/home/danil/Desktop/git_proj/Filter_ChatBot/models/LSTM/encoder.pkl', "rb") as f:
    encoder = pickle.load(f)


app = Flask(__name__)

@app.route('/predict_label', methods=['POST'])
def predict_method():
    try:
        text = request.json.get('text')
        preprocessed = preprocess(text, stem=True)
        result = predict(preprocessed)
        print('----------------------------------------------------------------------------')
        print(text)
        print(preprocessed)
        print(result)
        print('----------------------------------------------------------------------------')
        return result
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    
def convert_ogg_to_mp3(ogg_file, mp3_file):
    ogg_audio = AudioSegment.from_file(ogg_file, format="ogg")
    ogg_audio.export(mp3_file, format="mp3")
    

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    if request.method == 'POST':
        try:
            f_ogg = request.files['the_file']
            name_ogg = f'{time.time_ns()}'
            f_ogg.save(f'code/deployment/api/files/{name_ogg}.ogg')
            f_ogg.close()
            convert_ogg_to_mp3(f'code/deployment/api/files/{name_ogg}.ogg', f'code/deployment/api/files/{name_ogg}_result.mp3')
            with open(f'code/deployment/api/files/{name_ogg}_result.mp3') as f:
                text = transcriber(f'code/deployment/api/files/{name_ogg}_result.mp3')['text']
                preprocessed = preprocess(text, stem=True)
                result = predict(preprocessed)
                print('----------------------------------------------------------------------------')
                print(text)
                print(preprocessed)
                print(result)
                print('----------------------------------------------------------------------------')
                os.remove(f'code/deployment/api/files/{name_ogg}.ogg')
                os.remove(f'code/deployment/api/files/{name_ogg}_result.mp3')
                return result
        except Exception as e:
            return jsonify({'error': str(e)}), 400



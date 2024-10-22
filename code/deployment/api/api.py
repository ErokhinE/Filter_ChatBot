# sklearn
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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from pydub import AudioSegment
import joblib

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

# paths
SENTIMENT_MODEL_PATH = os.environ['SENTIMENT_MODEL_PATH']
SPAM_MODEL_PATH = os.environ['SPAM_MODEL_PATH']
TEMP_VOICE_PATH = os.environ['TEMP_VOICE_PATH']

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 300

# models and preprocessing
nltk.download('stopwords')

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")

spam_detector = joblib.load(f'{SPAM_MODEL_PATH}/spam_detector.pkl')

model = load_model(f'{SENTIMENT_MODEL_PATH}/model.h5')

# Load the Word2Vec model
w2v_model = gensim.models.word2vec.Word2Vec.load(f'{SENTIMENT_MODEL_PATH}/model.w2v')

# Load the tokenizer
with open(f'{SENTIMENT_MODEL_PATH}/tokenizer.pkl', "rb") as f:
    tokenizer = pickle.load(f)

# Load the encoder
with open(f'{SENTIMENT_MODEL_PATH}/encoder.pkl', "rb") as f:
    encoder = pickle.load(f)
    
# crate falsk app instance
app = Flask(__name__)

def preprocess(text: str, stem=False)->str:
    '''
    Preprocesses the raw text.
    
    Parameters
    ----------
    **text**: str
        Raw text to preprocess
    
    **stem**: bool
        To use or not the stemmer
    
    Returns
    ----------
    preprocessed text as a string 
    '''
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


def decode_sentiment(score, include_neutral=True):
    '''
    Decodes the santiment based on the score.
    
    Parameters
    ----------
    **score**: float
        A float number between 0 and 1
    
    **include_neutral**: bool
        To use or not the neutral sentiment label
    
    Returns
    ----------
    Sentiment label as a string in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    '''
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
    '''
    Predicts the category of the text.
    
    Parameters
    ----------
    **text**: str
        Text to which predict its label.
    
    **include_neutral**: bool
        To use or not the neutral sentiment label
    
    Returns
    ----------
    dict with keys: 'label', 'score', and 'elapsed_time'
    
    Sentiment label as a string in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    
    Score is a float number withib [0, 1] range
    
    Elapsed_time is the time passed to predict the label for the text
    '''
    start_at = time.time()
    spam_score = spam_detector.predict_proba([text])[:, 1][0]
    is_spam = spam_score > 0.5
    print(spam_score, is_spam)
    if not is_spam:
        preprocessed = preprocess(text, stem=True)
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences([preprocessed]), maxlen=SEQUENCE_LENGTH)
        # Predict
        score = model.predict([x_test])[0]
        # Decode sentiment
        label = decode_sentiment(score, include_neutral=include_neutral)
    else:
        score = spam_score
        label = 'spam'

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  


@app.route('/predict_label', methods=['POST'])
def predict_method():
    '''
    Function to predict the lable of the text.
    
    Returns
    ----------
    json with keys: 'label', 'score', and 'elapsed_time'
    
    Sentiment label as a string in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    
    Score is a float number withib [0, 1] range
    
    Elapsed_time is the time passed to predict the label for the text
    '''
    try:
        text = request.json.get('text')
        
        result = predict(text)
        print('----------------------------------------------------------------------------')
        print(text)
        print(preprocess(text, stem=True))
        print(result)
        print('----------------------------------------------------------------------------')
        return result
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    
def convert_ogg_to_mp3(ogg_file, mp3_file):
    '''
    Function to convert voice message from ogg to mp3.
    '''
    ogg_audio = AudioSegment.from_file(ogg_file, format="ogg")
    ogg_audio.export(mp3_file, format="mp3")
    

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    '''
    Function to predict the lable of the voice message.
    
    Returns
    ----------
    json with keys: 'label', 'score', and 'elapsed_time'
    
    Sentiment label as a string in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    
    Score is a float number withib [0, 1] range
    
    Elapsed_time is the time passed to predict the label for the text
    '''
    if request.method == 'POST':
        try:
            f_ogg = request.files['the_file']
            name_ogg = f'{time.time_ns()}'
            f_ogg.save(f'{TEMP_VOICE_PATH}/{name_ogg}.ogg')
            f_ogg.close()
            convert_ogg_to_mp3(f'{TEMP_VOICE_PATH}/{name_ogg}.ogg', f'{TEMP_VOICE_PATH}/{name_ogg}_result.mp3')
            with open(f'{TEMP_VOICE_PATH}/{name_ogg}_result.mp3') as f:
                text = transcriber(f'{TEMP_VOICE_PATH}/{name_ogg}_result.mp3')['text']
                
                result = predict(text)
                print('----------------------------------------------------------------------------')
                print(text)
                print(preprocess(text, stem=True))
                print(result)
                print('----------------------------------------------------------------------------')
                os.remove(f'{TEMP_VOICE_PATH}/{name_ogg}.ogg')
                os.remove(f'{TEMP_VOICE_PATH}/{name_ogg}_result.mp3')
                return result
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


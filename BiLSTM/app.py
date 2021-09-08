from flask import Flask,render_template,url_for,request
from google_play_scraper import app as gs
import nltk
import re
import string

import tensorflow as tf
print(tf.__version__)
import keras
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,Bidirectional,TimeDistributed
from keras.layers.embeddings import Embedding


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report

import re
import numpy as np



from keras.models import load_model


app = Flask(__name__)

def preprocess(text):  
    text = text.translate(string.punctuation)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    score=0

    
    
    if request.method == 'POST':
        url = request.form['url']
        file = open("cc.txt", "w",encoding='utf-8')
        link=url
        findId=link.find('id=')

        url=link[findId+3:]
        file.write(str(gs(
            url,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
        )))
        file.close()

        myfile=[]
        with open("cc.txt",encoding='utf8') as mydata:
            for data in mydata:
                myfile.append(data)

        start=myfile[0].find('comments')
        end=myfile[0].find('editorsChoice')
        c=data[start:end]

        c= c.lower() 
        c =  re.sub('[^a-zA-z0-9\s]','',c) 
        c= c.replace('rt','')
        nltk.download('stopwords')
        c = preprocess(c)
        tokenizer = Tokenizer(num_words= 200)
        tokenizer.fit_on_texts(c)
        score=0
        sequences = tokenizer.texts_to_sequences(c)
        data = pad_sequences(sequences, maxlen=29)
        new_model = tf.keras.models.load_model('bilstm_final.h5')
        result=np.argmax(new_model.predict(data), axis=-1)
        score=result.sum()/len(result)
        print(score)
        # result=new_model.predict_classes(data)
        # print(len(result))
       
    
    return render_template('result.html',score = score)



if __name__ == '__main__':
    app.debug = True
    app.run()
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:22:33 2020

@author: shalo
"""

from flask import Flask, request, render_template
from keras.preprocessing.text import one_hot
from keras.models import load_model
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)#just a module in python(starting point of the api)

#loading the saved model
model = load_model('tweets_TFL_96Per.h5')

@app.route('/')#will route URL with the function
def home():
    return render_template('tweets.html')#will redirect to the template,#check2 and index1

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tweet=request.form['tw']
    str(tweet)
    tweet=[tweet]

    voc_size=5000
    
    tweet_onehot_repr=[one_hot(words,voc_size)for words in tweet] 
    
    sent_length=20
    tweet_embedded_docs=pad_sequences(tweet_onehot_repr,padding='pre',maxlen=sent_length)
    tweet_final=np.array(tweet_embedded_docs)
    
    pred=model.predict_classes(tweet_final)

    if pred==1:
        return render_template('tweets.html', prediction='This tweet may belong to some disaster.')
    else:
        return render_template('tweets.html', prediction='This tweet may seems to be normal one.')

if __name__ == "__main__":#if this code is running other than python then this command will come into existence
    app.run(debug=True)#means it will show the realtime changes done by the user without stopping the command prompt
#when above written condition is executed then app will run

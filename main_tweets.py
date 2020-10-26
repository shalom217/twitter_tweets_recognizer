# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:00:56 2020

@author: shalo
"""

#importing all necessary libraries
import pandas as pd
import seaborn  as sns
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import time
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from nltk.stem.porter import PorterStemmer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from imblearn.over_sampling import RandomOverSampler#used for oversampling
from collections import Counter

#starting with loading the dataset
df=pd.read_csv("tweets.csv")

##Drop Nan Values
df.isnull().sum()
#since null values is only in "location" feature and dropping using dropna will delete many usefull data from "text" feature also which is an important feature. so only we will drop that feature
df=df[["text","target"]]

#checking the counts and also data is balanced or imbalanced ?
df['target'].value_counts()
#lets plot the same
LABELS = ["Normal", "Disaster"]
#0=normal tweet,1=disaster tweet
sns.countplot(data=df,x='target')
plt.title("Tweets Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Tweets")
plt.ylabel("Count")

##hence data is highly imbalanced

#we will make the data balanced in later part of the code after preprocessing the text

## Get the Independent Features
X=df.drop('target',axis=1)

## Get the Dependent features
y=df['target']

#copying to new variable so that further nlp task can be done 
tweets=X.copy()
tweets.reset_index(inplace=True)

### Dataset/text Preprocessing(step 1: making the sentences ready)
ps = PorterStemmer()
corpus = []
for i in range(0, len(tweets)):
    
    review = re.sub('[^a-zA-Z]', ' ', tweets['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
### Vocabulary size(since we are using LSTM along with word Embedding)
voc_size=5000#step 2

#step 3 one hot:giving index(a number) to each word in corpus by voc size using one hot encoding technique
onehot_repr=[one_hot(words,voc_size)for words in corpus] 

#Embedding Representation
#step 4:preprocessing the sentenses means equalising size of each sentense
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

## Creating model
embedding_vector_features=40 #each word/index have 40 diffrents values
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

X_final=np.array(embedded_docs)
y_final_old=np.array(y)
y_final=np.array(y)

#making the data balanced now
os =  RandomOverSampler()
X_final, y_final = os.fit_sample(X_final, y_final)

#checking/counting the values 
print('Old dataset shape {}'.format(Counter(y_final_old)))
print('Resampled dataset shape {}'.format(Counter(y_final)))

#spliting the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


checkpoint = ModelCheckpoint("tweets_TFL.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]



### Finally Training
start=time.time()
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=64, callbacks = callbacks)
model.save("tweets.h5")
stop=time.time()
print(stop-start,"secs")

#ploting accuracy and loss grapgh
# from IPython.display import Inline
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Performance Metrics And Accuracy

y_pred=model.predict_classes(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#testing new data

#text preprocessing
tweet="World will be crashed very soon"
tweet=[tweet]

voc_size=5000

tweet_onehot_repr=[one_hot(words,voc_size)for words in tweet] 

sent_length=20
tweet_embedded_docs=pad_sequences(tweet_onehot_repr,padding='pre',maxlen=sent_length)
tweet_final=np.array(tweet_embedded_docs)

model = load_model('tweets_TFL_96Per.h5')
pred=model.predict_classes(tweet_final)
print(pred)

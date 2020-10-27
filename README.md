# twitter_tweets_recognizer

# Embedded Layer
This is how text is converted to numbers with the help of Word Embedding.
In word embedding a word is converted into vectors of some dimension.
There are four steps involve in this process: 
Step1: Making the sentense ready(preprocessing the text, removing stop words, numbers,special Characters).
Step2: Taking the Vocublary size.
Step3: One Hot Encoding(Means assigning a number to each word)
Step4: Embedding Layer(Feature Representing) means each word which is now converted into number will now have 40 diffrent numbers too creating an array which is a vector and when we plot this vector in vector space then cosine similarity between this vector and other vectors is checked and hence we find the similar words.

![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/Embedded%20layer.png)

# Count of labels
Since count of normal tweets is high, so i used randomover sampler to balance the imbalanced dataset.
![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/Figure_1.png)

# Overview of the model

![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/model.png)

# Here is the Accuracy log and loss/accuracy Vs Epoch Curves
In the project i used Callbacks method(Early Stopping, ModelCheckpoint), So two models were created having 96% accuracy with modelcheckpoint and 99% accuracy with early stopping.
Although model was supposed to train on 50 epochs but earlystopping executed at 7 epochs only.
![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/acc_log.png)
![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/lossVSepoch.png)
![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/accVSepoch.png)

# Technology/Library Used:
Python, Sklearn, Natural Language Processing, Tensorflow, Flask, Heroku, Bidirectional LSTM, Keras, Pandas, Numpy and Seaborn.

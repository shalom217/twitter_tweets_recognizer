# twitter_tweets_recognizer
![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/model_image2.jpg)
We see many tweets on twitter with having information regarding some disaster such as earthquake, cyclone, landslides etc. However many tweets may appear as real or true but a large number of tweets just being spread by multiple groups just to make fake rumours, spread negitivity, fear in between innocent people. In some cases these groups traps people and ask for money. All in all this being done on a large scale in proabably every country and it must be supervised by any government body to take care of people of nation.
AI along with its amazing tools such as Natural Language Processing, LSTM etc allow us to easily identify all these fraudlent type of tweets and can take actions against such accounts.
In this method a huge data set is used for the training of model with having 11k approx tweets of both real and fake. So whenever a new tweet is posted containing some hazardous information in it, the tweet will be analyzed by the deep learning model itself and if it is found to be real then its ok but in case of fake the action must be taken against the account accociated with the tweet and also tweet must be deleted.
The tweets have some common words, that  may used repeatedly.
![alt text](https://github.com/shalom217/twitter_tweets_recognizer/blob/main/images/model_image1.jpg)

# Dataset:
The dataset is takken from the kaggle.


# Embedded Layer
This is how text is converted to numbers with the help of Word Embedding.
In word embedding a word is converted into vectors of some dimension.
There are four steps involve in this process: 
Step1: Making the sentense ready(preprocessing the text, removing stop words, numbers,special Characters as well).
Step2: Taking the Vocublary size.
Step3: One Hot Encoding(Means assigning a number to each word)
Step4: Embedding Layer(Feature Representing) means each word which is now converted into some number will now have its 40 diffrent numbers too, hence creating an array which is a vector and when we plot this vector in vector space then cosine similarity between this vector and other vectors is checked and hence we find the similar words.
Words having less cosine distance(or more cosine similarity) will be called as similar words.

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
Python, Sklearn, Natural Language Processing, Tensorflow, Bidirectional LSTM, Keras, Pandas, Numpy and Seaborn for model creation.
Flask for web frame work with HTML and CSS for styling and Heroku cloud for the model deployment.

# Deployment:
See how it is working
https://tweet-recognizer.herokuapp.com/


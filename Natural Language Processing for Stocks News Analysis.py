#!/usr/bin/env python
# coding: utf-8

# ## Natural Language Processing for Stocks News Analysis:

# Introduction
# 
# I. Conceptualization of the modeling task
# 
# II. Data Collection: Import Libraries/Datasets and Perform Exploratory Data Analysis
# 
# III. Data Preparation & Wrangling
# 
# IV. Data Exploration
# 
# V. Model Training

# ### Introduction:

# Organizations are now dealing with structured, semi-structured, and unstructured data from within and outside the entreprise.
# 
# Unstructured data are generated from social media (eg. posts, tweets, blogs), email, and text communications, web traffic, online news sites, electronic images and other electronic information sources.
# 
# Unlike structured data that can be readily organized into data tables to be read and analyzed by computers, unstructured data require specific methods of preparation and refinement before being usable by computers and useful to investment professional.
# 
# Natural language processing (NLP) works by converting words (text) into numbers, these numbers are then used to train an AI/ML model to make predictions.
# 
# The main steps in building ML model include:
# 1. Conceptualization of the modeling task
# 2. Data collection
# 3. Data preparation and wrangling
# 4. Data exploration
# 5. Model training 

# ### I. Conceptualization of the modeling task

# Conceptualization of the modeling task requires to define the problem, how the output of the model will be specified, how the model will be used and for whom, and whether the model will be embedded in existing business process.
# 
# In this project, we will build a machine learning model to analyze thousands of Twitter tweets to predict people’s sentiment towards a particular company or stock. AI/ML based sentiment analysis models can be used automatically understand the sentiment from public tweets, which could be used as a factor while making buy/sell decision of securities.
# 
# In this hands-on project, we will train a Long Short Term Memory (LSTM) deep learning model to perform stocks sentiment analysis. 
# 
# Now we will build a machine learning model to perform news sentiment analysis. In this hands-on project, we will complete the following tasks:
# 1. Apply python libraries to import and visualize datasets
# 2. Perform exploratory data analysis and plot word-cloud
# 3. Perform text data cleaning such as removing punctuation and stop words
# 4. Understand the concept of tokenizer
# 5. Perform tokenizing and padding on text corpus to feed the deep learning model
# 6. Understand the theory and intuition behind Recurrent Neural Networks and LSTM
# 7. Build and train the deep learning model
# 8. Assess the performance of the trained model

# ### II. Data Collection: Import Libraries/Datasets and Perform Exploratory Data Analysis:

# First, we are going to install Wordcloud to perform powerful visualization of words and then we are going to install two key libraries:
# - "gensim" which is an open-source library for unsupervised topic modeling and natural language processing and it is implemented in Python and Cython.
# - "NLTK" which is Natural Language Tool Kit.

# In[12]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install gensim')
get_ipython().system('pip install nltk')


# In[13]:


# import key libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# We are going to set the style of the notebook to be monokai theme. This line of code is important to ensure that we are able to see the x and y axes clearly.

# In[14]:


from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False) 


# In[15]:


# load the stock news data
stock_df = pd.read_csv('C:\\Users\\Pattu\\OneDrive\\Documents\\02. Machine Learning\\01. Projects\\Natural Language Processing for Stocks News Analysis\\Stock_sentiment.csv')
stock_df.head()


# In[16]:


# dataframe information
stock_df.info()


# In[17]:


# check for null values
stock_df.isnull()


# In[18]:


# Find the number of unique values in a particular column
stock_df['Sentiment'].nunique()


# Or we can use Seaborn to plot the count plot for our data to find the the number of unique values in a particular column.

# In[19]:


sns.countplot(stock_df['Sentiment'])


# There are 2 classes in our dataframe 0 and 1. For the 0 sentiments, we have around 2000 tweets while for the 1 sentiments we have 3700 tweets.

# in the next, we'll go ahead and perform data cleaning by removing punctuation from text.

# ### III. Data Preparation & Wrangling:

# #### 1. Remove Punctuations from Text:

# In[20]:


import string
string.punctuation


# We are going to create a for loop that goes through every single character in each text for the entire 5791 tweets in order to get rid of every punctuation in the text.

# In[21]:


# define a function to remove punctuations
def remove_punc(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)

    return Test_punc_removed_join


# In[22]:


# remove punctuations from our dataset
stock_df['Text Without Punctuation'] = stock_df['Text'].apply(remove_punc)


# In[23]:


stock_df.head()


# In[24]:


stock_df['Text'][2]


# In[25]:


stock_df['Text Without Punctuation'][2]


# #### 2. Remove Stopwords:

# In[26]:


# download stopwords
nltk.download("stopwords")
stopwords.words('english')


# In[27]:


# Obtain additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['https','from', 'subject', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'])


# In[28]:


# Remove stopwords and remove short words (less than 2 characters)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) >= 3 and token not in stop_words:
            result.append(token)
            
    return result


# In[29]:


# apply pre-processing to the text column
stock_df['Text Without Punc & Stopwords'] = stock_df['Text Without Punctuation'].apply(preprocess)


# In[30]:


stock_df['Text'][0]


# In[31]:


stock_df['Text Without Punc & Stopwords'][0]


# In[32]:


# join the words into a string
# stock_df['Processed Text 2'] = stock_df['Processed Text 2'].apply(lambda x: " ".join(x))


# In[33]:


stock_df.head()


# ### IV. Data Exploration:

# #### 1. Plot Wordcloud:

# A word cloud is a visual representation of all the words in a BOW (Bag of Words) such that words with higher frequency have a larger font size. The most commonly occuring words in the dataset can be shown by varying font size, and color is used to add more dimensions, such as frequency and length of words. This allows the analyst to determine which words are contextually more important.

# In[34]:


# join the words into a string
stock_df['Text Without Punc & Stopwords Joined'] = stock_df['Text Without Punc & Stopwords'].apply(lambda x: " ".join(x))


# In[35]:


# plot the word cloud for text with positive sentiment
plt.figure(figsize = (20, 20)) 
wc = WordCloud(max_words = 1000, width = 1600, height = 800).generate("".join(stock_df[stock_df['Sentiment']==1]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc)


# In[36]:



# Visualize the wordcloud for tweets that have negative sentiment
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 1000, width = 1600, height = 800 ).generate(" ".join(stock_df[stock_df['Sentiment'] == 0]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation = 'bilinear');


# #### 2. Visualize Cleaned Datasets:

# In[37]:


stock_df.head()


# In[38]:


nltk.download('punkt')


# In[39]:


# word_tokenize is used to break up a string into words
print(stock_df['Text Without Punc & Stopwords Joined'][0])
print(nltk.word_tokenize(stock_df['Text Without Punc & Stopwords Joined'][0]))


# In[40]:


tweets_length = [ len(nltk.word_tokenize(x)) for x in stock_df['Text Without Punc & Stopwords Joined'] ]
tweets_length


# In[41]:


plt.hist(tweets_length, bins=50)
plt.show()


# The distribution of the number of words that we have in the text is a bell curve which shows that most of tweets range between 5 and 11 number of words and then decay afterwards. 
# 
# We are going to use Seaborn Countplot to visually indicate how many samples belong to the positive and negative sentiments class.

# In[42]:


# plot the word count
sns.countplot(stock_df['Sentiment'])


# #### 3. Tokenization:

# Tokenization is the process of splitting a given text into separate tokens. This step takes place after cleansing the raw text data. The tokens are then normalized to create the bag of words (BOW) which is a collection of distinct set of tokens from all the texts in a sample dataset.

# In[43]:


stock_df.head()


# In[44]:


# Obtain the total words present in the dataset
list_of_words = []

for i in stock_df['Text Without Punc & Stopwords']:
    for j in i:
        list_of_words.append(j)


# In[45]:


list_of_words


# In[46]:


# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))
total_words


# ### V. Model Training:

# Model training involves selecting the appropriate ML algorithm, evaluating performance of the trained model, and tuning the model accordingly. These steps are iterative because model building is an iterative process.

# #### 1. Prepare the Data by Tokenizing & Padding:

# In[47]:


# split the data into test and train 
X = stock_df['Text Without Punc & Stopwords']
y = stock_df['Sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[48]:


X_train.shape


# In[49]:


X_test.shape


# In[50]:


X_train


# In[51]:


# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(X_train)

# Training data
train_sequences = tokenizer.texts_to_sequences(X_train)

# Testing data
test_sequences = tokenizer.texts_to_sequences(X_test)


# In[52]:


train_sequences


# In[53]:


test_sequences


# In[54]:


print("The encoding for document\n", X_train[1:2],"\n is: ", train_sequences[1])


# In[55]:


# Add padding to training and testing
padded_train = pad_sequences(train_sequences, maxlen = 15, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences, maxlen = 15, truncating = 'post')


# In[56]:


for i, doc in enumerate(padded_train[:3]):
     print("The padded encoding for document:", i+1," is:", doc)


# In[57]:


# Convert the data to categorical 2D representation
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# In[58]:


y_train_cat.shape


# In[59]:


y_test_cat.shape


# In[60]:


y_train_cat


# #### 2. Understand the Theory behind Recurrent Neural Networks & Long Short Term Memory Networks (LSTM):

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# #### 3. Build a Custom-Based Deep Neural Network (LSTM Network) to perform Sentiment Analysis:

# Embedding layers learn low-dimensional continuous representation of discrete input variables.

# In[61]:


# Sequential Model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim = 512))

# Bi-Directional RNN and LSTM
model.add(LSTM(256))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(2,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()


# In[62]:


# train the model
model.fit(padded_train, y_train_cat, batch_size = 32, validation_split = 0.2, epochs = 2)


# #### 4. Assess Trained Model Performance:

# In order to validate a ML model, we must measure its training performance. We will next consider several techniques to measure model performance or goodness of fit of a ML algorithm that are well suited specifically for binary classification models.
# 
# Model performance can be evaluated by using error analysis. For a classification model, a confusion matrix for error analysis (TPs, TNs, FPs and FNs are determined) is created, and evaluation metrics such as Precision, Recall, Accuracy score and F1 score are claculated. The higher the accuracy and F1 score, the better the model performance.

# ![image.png](attachment:image.png)

# In[63]:


# make prediction
pred = model.predict(padded_test)


# In[64]:


pred


# In[65]:


# make prediction
prediction = []
for i in pred:
  prediction.append(np.argmax(i))


# In[66]:


prediction


# In[67]:


y_test_cat


# In[68]:


# list containing original values
original = []
for i in y_test_cat:
  original.append(np.argmax(i))


# In[69]:


original


# In[70]:


# acuracy score on text data
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(original, prediction)
accuracy


# In[71]:


# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot = True)


# In[ ]:





# In[ ]:





# In[ ]:





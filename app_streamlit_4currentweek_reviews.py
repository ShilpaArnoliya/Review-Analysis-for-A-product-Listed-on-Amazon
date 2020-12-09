# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:30:43 2020

@author: DELL
"""
import warnings
warnings.filterwarnings("ignore")

import os
os.getcwd()
#os.chdir("C:\\Users\\hp\\Downloads\\Group_5")



############################################################################################################################
############################################################################################################################

############################################################################################################################

###   define current week
import pandas as pd
import numpy as np
import re
import datetime
today = datetime.date.today()
print(today)
weekdays = ([today + datetime.timedelta(days=i) for i in range(0 - today.weekday(), 7 - today.weekday())])
from datetime import datetime
############################################################################################################################
##                    filter only current week reviews

#data_cw = pd.read_csv("C:\\Users\\hp\\Downloads\\Group_5\\CleanData_Samsung_SDcardReviews_weekly_update.csv")
import subprocess
import sys

# Some code here

pid = subprocess.Popen([sys.executable, "AmazonReview1.py"]) # Call subprocess

data_cw = pd.read_csv("C:\\Users\\hp\\Downloads\\Group_5\\CleanData_Samsung_SDcardReviews_weekly_update.csv")
data_cw.head(10)
data_cw = data_cw.drop_duplicates()

data_cw = data_cw.rename(columns ={"comment": "text"})

data_cw['review_date'] = data_cw['review_date'].apply(lambda x: str.replace(x, 'Reviewed in India on ', ''))
data_cw['review_date'] = data_cw['review_date'].apply(lambda x: datetime.strptime(x, '%d %B %Y'))
    
data_cw = data_cw[data_cw['review_date'].apply(lambda x: x in weekdays)] 
data_cw.info()
data_cw.shape
############################################################################################################################

####   import old amazon reviews

data = pd.read_csv("C:\\Users\\hp\\Downloads\\Group_5\\CleanData_Samsung_SDcardReviews_weekly_update.csv")
#data = pd.read_csv("Samsung_SDcardReviews_latest.csv")
data = data.drop_duplicates()
data = data.rename(columns ={"comment": "text"})
#data['review_date'] = data['review_date'].apply(lambda x: str.replace(x, 'Reviewed in India on ', ''))
#data['review_date'] = data['review_date'].apply(lambda x: datetime.strptime(x, '%d %B %Y'))
    
data.info()
data.shape

############################################################################################################################
 
##          data cleaning and preprocessing using spacy

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 


# In[

import codecs
import unidecode
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_cleaner(text):
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    apostrophe_handled = re.sub("’", "'", decoded)
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


# In[ ]:


data['clean_text'] = [spacy_cleaner(t) for t in data.text]
data_cw['clean_text'] = [spacy_cleaner(t) for t in data_cw.text]

############################################################################################################################
#  define sentiment class and build a model
############################################################################################################################
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    if(score['compound']>=0.1):
        return 'pos'
    elif(score['compound']<=-0.1):
        return 'neg'
    else:
        return 'neu'

data['sentiment'] = data['clean_text'].apply(lambda x: sentiment_analyzer_scores(x))   
#  model building
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000, min_df=5, max_df=0.7, stop_words=None, decode_error="replace")
X = vectorizer.fit_transform(data.clean_text).toarray()
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X_tfidf = tfidfconverter.fit_transform(X).toarray()
#Save vectorizer.vocabulary_
import pickle
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data.sentiment, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


################                   dealing with imbalanced data             ###################
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train,np.array(y_train).ravel())
print(X_train_res.shape, y_train.shape)

# Classifier - Algorithm - SVM
from sklearn import svm
from sklearn.svm import SVC
# fit the training dataset on the classifier
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
################                   dealing with imbalanced data             ###################
###  building svm model after applying SMOTE 
smote_svm = SVM.fit(X_train_res,y_train_res).predict(X_test)

###########################################################################################################################
############################################################################################################################
#  predict the sentiment of the current reviews
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
from sklearn.feature_extraction.text import CountVectorizer
#Load it later
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
X_TEST = loaded_vec.fit_transform(data_cw.clean_text).toarray()
X_TEST_tfidf = tfidfconverter.fit_transform(X_TEST).toarray()

# In[ ]:
#Load it later
#file_name = pickle.load(open("nlp_model.pkl", "rb"))
# In[ ]:
#df_weekly['sentiment'] = SVM.predict(X_TEST_tfidf, file_name)
data_cw['sentiment'] = SVM.predict(X_TEST_tfidf)

############################################################################################################################
############################################################################################################################

############################################################################################################################
#  streamlit code
############################################################################################################################
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud ,STOPWORDS
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Sentiment Analysis For Samsung SD Card Amazon Reviews")

st.sidebar.title("User Info charts")

st.markdown("DASHBOARD TO ANALYSE REVIEWS ")
st.sidebar.markdown("DASHBOARD TO ANALYSE REVIEWS ")

st.sidebar.subheader(" wordclouds")
word_sentiment=st.sidebar.radio('sentiment',('pos','neg','neu'))
if not st.sidebar.checkbox("Close",True,key='3'):
    st.header('Word cloud for %s sentiment' % (word_sentiment))
    df=data[data['sentiment']==word_sentiment]
    words=str(' '.join(df['clean_text'].astype('str').tolist()))
    #processed_words=' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'] )
    wordcloud=WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()



st.sidebar.markdown(" number of reviews by sentiment")
select= st.sidebar.selectbox('Visualization type',['Histogram', 'Pie chart'] ,key='1')

sentiment_count= data['sentiment'].value_counts()
sentiment_count=pd.DataFrame({'sentiment':sentiment_count.index,'reviews':sentiment_count.values})

if not st.sidebar.checkbox('Hide',True):
    st.markdown("number of reviews by sentiment")
    if select == "Histogram":
        fig=px.bar(sentiment_count,x='sentiment',y='reviews',color='sentiment',height=500)
        st.plotly_chart(fig)
    else :
        fig=px.pie(sentiment_count,values='reviews', names='sentiment')
        st.plotly_chart(fig)
        
        
#############################################################################################################################
#                   current week reviews analysis
#############################################################################################################################
        
st.sidebar.subheader(" wordcloud for current week reviews")
#word_sentiment=st.sidebar.selectbox('wordcloud')
if not st.sidebar.checkbox("Close_wordclod",True,key='3'):
    st.header('Word cloud for current week reviews')
   # df=data_cw[data_cw['sentiment']==word_sentiment]
    words=str(' '.join(data_cw['clean_text'].astype('str').tolist()))
    #processed_words=' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'] )
    wordcloud=WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()



st.sidebar.markdown(" Number of reviews by sentiment for current week")
select= st.sidebar.selectbox('Visualization_type',['Histo_gram', 'Pie_chart'] ,key='1')

sentiment_count_cw= data_cw['sentiment'].value_counts()
sentiment_count_cw=pd.DataFrame({'sentiment':sentiment_count_cw.index,'reviews':sentiment_count_cw.values})

if not st.sidebar.checkbox('Hide_plot',True):
    st.markdown("number of reviews by sentiment for current week")
    if select == "Histo_gram":
        fig=px.bar(sentiment_count_cw,x='sentiment',y='reviews',color='sentiment',height=500)
        st.plotly_chart(fig)
    else :
        fig=px.pie(sentiment_count_cw,values='reviews', names='sentiment')
        st.plotly_chart(fig)
        
        
##############################################################################################################################
        
###             append current week reviews in old file
#df_weekly_updated = pd.read_csv("C:\\Users\\DELL\\P30_Group5_ExcelR_Project\\Samsung_SDcardReviews_weekly_update.csv")     

data.append(data_cw, ignore_index = True)
#data = data.drop_duplicates()
data.shape
#os.remove('Samsung_SDcardReviews_weekly_update.csv')
# saving the DataFrame as a CSV file 
data.to_csv('Samsung_SDcardReviews_weekly_update.csv', index = True, encoding = 'utf-8') 


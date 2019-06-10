#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:44:58 2018

@author: msarjun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF 
from sklearn.decomposition import FastICA
from nltk.tokenize import word_tokenize 
import re
from imblearn.over_sampling import SMOTE
#%%%%%%%%%%%%%%%%%
address=pd.read_csv("res.csv",names=[0])
#%%%%%%%%%%%%%%%%5
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet  
from nltk.stem import WordNetLemmatizer

ps = WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
stop_words=list(stop_words)
stop_words.append('n')
stop_words=set(stop_words)
#%%%%%%%%%%%%%%%%5555
patentData=pd.DataFrame()
for i in range(len(address)):
    data=pd.read_json(address.iloc[i,0])   
    patents=data["content"].iloc[0]
#    print(address.iloc[i,0])
#    print(patents)
    patents=pd.DataFrame(patents)
    d=patents.docs
    p=[d[i] for i in range(len(d))]
    p=pd.DataFrame(p)
    patentData=patentData.append(p)
p=patentData
p=p.dropna()
patentData=patentData.dropna()
#%%%%%%%%%%%%%%%%%%%%5
filtered_sentence=""
p["filtered_sentence"]=""

for i in range(len(p)):
    #print(i)
    stem_text=""
    text1=re.sub('\n', '', str(p.ab_en.iloc[i]))
    text1=re.sub('<.*?>', '', text1)
    text="".join([w for w in text1 ])
    word_tokens =  re.split('\W+',text)
    #print(text)
    filtered_sentence =" ".join([w for w in word_tokens if not w in stop_words]) 
    text="".join([w for w in str(filtered_sentence) if w not in string.punctuation])
    word_tokens =  re.split('\W+',text)
   
    for w in word_tokens:  
        #print(ps.stem(w))
       #stem_text=stem_text.join([ps.stem(w)])
       stem_text=stem_text+ps.lemmatize(w)+" "
         
    p["filtered_sentence"].iloc[i]=stem_text
    word2vec_tokenize = word_tokenize(p["filtered_sentence"].iloc[i])
#%%%%%%%%%%%%%%%%%5
mystring=p.iloc[i,4]
msystring=mystring.split(" ")
list(nltk.bigrams(msystring))
#%%%%%%%%%%%55
megastring=""
for i in range(len(p)):
    megastring=megastring+str(p.iloc[i,4])+""
#%%%%%%%%%%%%5
from nltk.collocations import BigramCollocationFinder
def bi(text):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder=BigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(5)
    finder.nbest(bigram_measures.pmi, 5) 
    return finder.ngram_fd.items()
#bigramlist=[]
#for i in range(len(p)):
#    bigramdictlist=bi(str(p.iloc[i,4]))
#    bigramlist.append(list(bigramdictlist))
mybigram=bi(megastring)
#%%%%%555
mybigram
mybigram=list(mybigram)
#%%%%%%%55
mybigramlist=[]
for j in range(len(mybigram)):
    word=mybigram[j][0]
    mybigramlist.append(word[0]+" "+word[1])
#%%%%%%%%%%%%%%%%
#bigramlist[3][2][0]
#mybigramlist=[]
##for i in range(len(bigramlist)):
#    for j in range(len(bigramlist[i])):
#        word=bigramlist[i][j][0]
#        mybigramlist.append(word[0]+" "+word[1])
#%%55
len(mybigramlist)
mybigramlist=set(mybigramlist)
mybigramlist=list(mybigramlist)
#%%%%%%%%%%%555555
vectorizer = TfidfVectorizer(vocabulary=mybigramlist,ngram_range=(2,2))
dtm2 = vectorizer.fit_transform(str(p.iloc[i,4]) for i in range(len(p)))
cols2=pd.DataFrame(dtm2.toarray(),columns=vectorizer.get_feature_names())
vectorizer.get_feature_names()
#%%%%%%%555
cols2.head(10)
#%%%%%%%%%%5555
cols=cols2
#%%%%%%55
#vectorizer = TfidfVectorizer(ngram_range=(1,1))
#dtm = vectorizer.fit_transform(str(p.iloc[i,4]) for i in range(len(p)))
#cols=pd.DataFrame(dtm.toarray(),columns=vectorizer.get_feature_names())
#vectorizer.get_feature_names()
#%%%%%%%%%%%%%%%%%%%
publication=[str(p.pd.iloc[i]) for i in range(len(p))]
publication_year=[]
publication_month=[]
#%%%%%%%%%55
for i in range(len(p)):
    newstring=publication[i]
    publication_year.append(int(newstring[:4]))
    publication_month.append(int(newstring[4:6]))
#%%%%%%%%%%%%%%%
subs={1:"January",
      2:"February",
      3:"March",
      4:"April",
      5:"May",
      6:"June",
      7:"July",
      8:"August",
      9:"September",
      10:"October",
      11:"November",
      12:"December"}
publicationmonths=[subs.get(item,item)  for item in publication_month]
#%%%%%%%%%
publication_year=pd.Series(publication_year)
cols["publication_year"]=publication_year
publication_month=pd.Series(publicationmonths)
cols["publication_month"]=publication_month
#%%%%%%%%%%%%%%%%%%%%%%%%%%%5
year=pd.Series()
month=pd.Series()
polarity=pd.read_csv("polaritydataframe.csv")
mymonths=['January','February','March','April','May','June','July','August','September','October','November','December']
polarity["month"]=pd.Categorical(polarity['month'], categories=mymonths, ordered=True)
polarity=polarity.sort_values(["year","month"])
vals=polarity["values"].shift(-1)
vals=vals.fillna(0)
polarity2=polarity
polarity2["values"]=vals
#%%%%%%%%%%55
vals=polarity["values"].shift(-2)
vals=vals.fillna(0)
polarity3=polarity
polarity3["values"]=vals
#%%%%%%%%%%%
vals=polarity["values"].shift(-3)
vals=vals.fillna(0)
polarity4=polarity
polarity4["values"]=vals
#%%%%%%%%%
val=polarity["values"].shift(-4)
vals=vals.fillna(0)
polarity5=polarity
polarity5["values"]=vals
#%%%%%%%%5555
vals=polarity["values"].shift(-5)
vals=vals.fillna(0)
polarity6=polarity
polarity6["values"]=vals
#%%%%%%%%%%%%%5
vals=polarity["values"].shift(-6)
vals=vals.fillna(0)
polarity7=polarity
polarity["values"]=vals
#%%%%%%%%5555
polarity_dict={}
for i in range(len(polarity)):
    polarity_dict[polarity7["year"].iloc[i], polarity7["month"].iloc[i]]=polarity7["values"].iloc[i]
polarity=polarity_dict
#%%%%%%%%55555
sentiments=[]
for i in range(len(p)):
    #print(publication_year[i])
    #print(publication_month[i])
    print(polarity[publication_year[i],publication_month[i]])
    sentiments.append(polarity[(publication_year[i], publication_month[i])])
#%%%%%%%%%%%
sentiments=pd.Series(sentiments)
sentiments=sentiments.fillna(0)
cols["sentiments"]=sentiments
#%%%%%%%%%%%%%%%%'
#for i in range(1567,1572):
#    print(cols.publication_year.iloc[i])
#    print(cols.publication_month.iloc[i])
#    print(cols.sentiments.iloc[i])

#%%%%%%%%%%%%%%%%%%%%%%%55
poscols=cols
#%%%%%%%%%%%%%%%%%%5
pos=poscols.sentiments>=0
pos=[int(i) for i in pos]
pos=pd.Series(pos)
#%%%%%%%%%%%%%%55
x=poscols.drop(["publication_year","publication_month","sentiments"],axis=1)
y=pos

#%%%%%%%%%%%%%%%%%55555
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain ,ytest= train_test_split(x,y,test_size=0.2)
#%%%%%%%%%%55
sm = SMOTE(random_state=12)
xtrain_res, ytrain_res = sm.fit_sample(xtrain, ytrain)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5,algorithm="auto",)
classifier.fit(xtrain,ytrain)
ypred=classifier.predict(xtest)
#%%%%%%%%%%%%5
from sklearn.metrics import confusion_matrix
confusion_matrix(ytest,ypred)
from sklearn.metrics import accuracy_score
accuracy_score(ytest, ypred)
#%%%%%%%%%%%%%%%%%%%55
from sklearn.metrics import roc_curve,auc
y_predict_probabilities = classifier.predict_proba(xtest)[:,1]
fpr, tpr, _ = roc_curve(ytest, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
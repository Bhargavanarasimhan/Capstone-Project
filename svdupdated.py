#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:46:05 2018

@author: msarjun
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%Import all the be necessary packages
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
#%%%%%%read in the address csv file
address=pd.read_csv("res.csv",names=[0])
#%%%%importing NLTK and setting the stop words for the removal
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words=set(stopwords.words('english'))
#%%%%using the address dataframe , reading all the patent files and dumping them into the pandas dataframe
patentData=pd.DataFrame()
for i in range(len(address)):
    data=pd.read_json(address.iloc[i,0])   
    patents=data["content"].iloc[0]
    patents=pd.DataFrame(patents)
    d=patents.docs
    p=[d[i] for i in range(len(d))]
    p=pd.DataFrame(p)
    patentData=patentData.append(p)
p=patentData
p=p.dropna()
patentData=patentData.dropna()
#%%removing the stop words and text cleansing
p["filtered_sentence"]=""
filtered_sentence=""
for i in range(len(p)):
    #print(i)
    text="".join([w for w in str(p.ab_en.iloc[i]) if w not in string.punctuation])
    word_tokens =  re.split('\W+',text)
    #print(text)
    filtered_sentence =" ".join([w for w in word_tokens if not w in stop_words]) 
    p["filtered_sentence"].iloc[i]=filtered_sentence
#%%%%%%%%Implementing the count vectorizer for the patent dataframe 
#vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
#dtm = vectorizer.fit_transform(str(p.iloc[i,0]) for i in range(len(p)))
#cols=pd.DataFrame(dtm.toarray(),columns=vectorizer.get_feature_names())
#vectorizer.get_feature_names()
#lsa = TruncatedSVD(5, algorithm = 'arpack')
#dtm=dtm.asfptype()
#dtm_lsa = lsa.fit_transform(dtm)
#dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
#%%%%%%%%%%%%Implementing TfIDF vectorizer
vectorizer = TfidfVectorizer()
dtm = vectorizer.fit_transform(str(p.iloc[i,3]) for i in range(len(p)))
cols=pd.DataFrame(dtm.toarray(),columns=vectorizer.get_feature_names())
vectorizer.get_feature_names()
#%%%%%%%%%%%%%Implementing SVD
lsa = TruncatedSVD(100, algorithm = 'arpack')
dtm=dtm.asfptype()
dtm_lsa = lsa.fit_transform(dtm)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
#%%%%%%%%%%%%Screeplot for the SVD
var1=np.cumsum(np.round(lsa.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.xlabel("n_components")
plt.ylabel("variance explained")
#%%%%%%%%%%%%%%%%Implementing Non negative matrix featurization
nmf=NMF(n_components =100)
dtm_nmf=nmf.fit_transform(dtm)
dtm_nmf = Normalizer(copy=False).fit_transform(dtm_nmf)
#%%%%%%%%%%%%%%%%%Implementing PCA
pca= PCA(n_components=100)
dtm_pca=pca.fit_transform(dtm.toarray())
dtm_pca = Normalizer(copy=False).fit_transform(dtm_pca)
#%%%%%%%%%Screeplot for the PCA
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.xlabel("n_components")
plt.ylabel("variance explained")
#%%%%%%%%%%%%Implementing FAST ICA
fica=FastICA(n_components=100)
dtm_fica=fica.fit_transform(dtm.toarray())
dtm_fica=Normalizer(copy=False).fit_transform(dtm_fica)
#%%%%%%%%%%%%55implementing MDS
from sklearn.manifold import MDS
mds=MDS(n_components=100)
dtm_mds=mds.fit_transform(dtm.toarray())
dtm_mds=Normalizer(copy=False).fit_transform(dtm_mds)
#%%%%implementing ISOMAP
from sklearn.manifold import Isomap
ism = Isomap(n_components=100)
dtm_ism=ism.fit_transform(dtm.toarray())
dtm_ism=Normalizer(copy=False).fit_transform(dtm_ism)
#%%%%%%%%%%%%%%5 implementing Laplacian EigenMap
from sklearn.manifold import SpectralEmbedding
lle= SpectralEmbedding(n_components=100)
dtm_lle=lle.fit_transform(dtm.toarray())
dtm_lle=Normalizer(copy=False).fit_transform(dtm_lle)
#%%%%%%%%%%%%%%%%555 implementing TSNE
from sklearn.manifold import TSNE
tsne=TSNE(n_components=100,method='exact')
dtm_tsne=tsne.fit_transform(dtm.toarray())
dtm_tsne=Normalizer(copy=False).fit_transform(dtm_tsne)
#%%%%%Elbow plot for the kmeans to find the optimum K value
from sklearn.cluster import KMeans
val=[]
for i in range(1,11):
    clust= KMeans(n_clusters=i)
    f=clust.fit(cols.sample(250))
    val.append(f.inertia_)
plt.plot(range(1,11),val)
plt.title("ELbow plot for the KMeans")
plt.show()
#%%%Implementing the Kmeans clustering
cluster=KMeans(n_clusters=5)
ypred=cluster.fit_predict(cols)
#%%%Validating the Kmeans clustering
from sklearn import metrics
print(metrics.silhouette_score(cols,ypred))
print(metrics.davies_bouldin_score(cols,ypred))
#%%%%%%%Implementing Dendrograms for the Agglomerative clustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
z=linkage(cols,'ward')
dendrogram(z)
#%%%%%%%Implementing the agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=13)
ypred=cluster.fit_predict(cols)
#%%%%%Validating the agglomerative clustering
from sklearn import metrics
print(metrics.silhouette_score(cols,ypred))
print(metrics.davies_bouldin_score(cols,ypred))
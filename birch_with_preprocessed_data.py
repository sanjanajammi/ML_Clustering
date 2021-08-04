from sklearn.cluster import Birch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
import pickle

data_set= [line.rstrip('\n') for line in open("f1.txt")]
v = TfidfVectorizer(stop_words= 'english')
X= v.fit_transform(data_set[:50000])
cl=AgglomerativeClustering(n_clusters=None, distance_threshold=1.68)
brc = Birch(n_clusters=cl)
i=1000
while i<50000:
    brc.partial_fit(X[i-1000:i])
    print(i)
    i+=1000
    
pickle.dump(brc, open("model.pkl","wb"))
labels = brc.predict(X)
clusters = {}
n = 0
for item in labels:
    if item in clusters:
        clusters[item].append(data_set[n] + str(n))
    else:
            clusters[item] = [data_set[n] + str(n)]
    n +=1

max=0
min=600000
for item in clusters:
    print("Cluster ", item)
    print(len(clusters[item]))
    if(len(clusters[item])>max):
        max= len(clusters[item])
    if(len(clusters[item])<min):
        min= len(clusters[item])
print(len(clusters))
print(len(labels))
print("Maximum no. of items in a cluster : " + str(max))
print("Minimum no. of items in a cluster : " + str(min))
print("Average no. of items in a cluster : " + str(len(labels)/len(clusters)))

sample= ["system crash"]
ss = v.transform(sample)
c = brc.predict(ss)
print(c)

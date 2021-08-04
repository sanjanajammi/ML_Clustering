#import tensorflow as tf
# from tensorflow.keras import layers
# import keras
# from keras.datasets import reuters
from sklearn.model_selection import train_test_split
#from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
from scipy import stats
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
import pickle
import sys
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words
   return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        n_word = re.sub(r'[(){}<>""\',=`;:\[\]\?\\/|]', ' ', word)
        #new_word = re.sub(r'[_]',' ',n_word)
        if n_word != '':
            new_words.append(n_word)
    return new_words
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words
#
# def stem_words(words):
#     """Stem words in list of tokenized words"""
#     stemmer = LancasterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems
#
# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas
#
def normalize(words):
    words = remove_non_ascii(words)
#     words = to_lowercase(words)
    words = remove_punctuation(words)
#     words = replace_numbers(words)
    words = remove_stopwords(words)
    return words
#
# def stemming(words):
#     stems = stem_words(words)
#     return stems
#
# def lemmatize(words):
#     lemmas = lemmatize_verbs(words)
#     return lemmas
#
# ''' End of Functions '''

sys.setrecursionlimit(3000)
bigdata=pd.read_csv("C:/Users/ishishar/Documents/Text Clustering/TextClustering/webapp/NewQuery.csv")
# print(bigdata.head())
data_s=list(bigdata['Title'])
data_set = data_s[7150:7170]
tokens=[]
for line in data_set:
    linetoken=nltk.word_tokenize(line)
    tokens.append(linetoken)
clean=[]
i=0
for title in tokens:
    nice=normalize(title)
    clean.append(nice)
    i=i+1
    print(i)
clean = list(clean)
# stems=[]
# lemmas=[]
# for title in clean:
#     stem=stemming(title)
#     lemma=lemmatize(title)
#     stems.append(stem)
#     lemmas.append(lemma)
# data_set_final=[]
# for title in stems:
#     string=" "
#     string=string.join(title)
#     data_set_final.append(string)
data_set_final=[]
for title in clean:
    s=" "
    s=s.join(title)
    data_set_final.append(s)
print(data_set_final)
with open('f1.txt', 'w') as f:
    for item in data_set_final:
        f.write("%s\n" % item)
f.close()


# v=TfidfVectorizer(stop_words='english')
# X=v.fit_transform(data_set_final)
# print(v.get_feature_names()[:10000])
# print(X.shape)
# print(len(v.get_feature_names()))
# cl = AgglomerativeClustering(n_clusters=None, distance_threshold=1.68)
# brc  = Birch(n_clusters=  cl)
# brc.fit(X)
# pickle.dump(brc, open("model.pkl","wb"))
# labels = brc.predict(X)

# clusters = {}
# n = 0
# for item in labels:
#     if item in clusters:
#         clusters[item].append(data_set[n] + str(n))
#     else:
#             clusters[item] = [data_set[n] + str(n)]
#     n +=1
# max=0
# min=2600000
# for item in clusters:
#     print("Cluster ", item)
#     print(len(clusters[item]))
#     if(len(clusters[item])>max):
#         max= len(clusters[item])
#     if(len(clusters[item])<min):
#         min= len(clusters[item])
# print(len(clusters))
# print(len(labels))
# print("Maximum no. of items in a cluster : " + str(max))
# print("Minimum no. of items in a cluster : " + str(min))
# print("Average no. of items in a cluster : " + str(len(labels)/len(clusters)))

# sample= ["system crash"]
# ss = v.transform(sample)
# c = brc.predict(ss)
# print(c)



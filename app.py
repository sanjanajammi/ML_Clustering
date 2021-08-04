import os
import numpy as np
import flask
import joblib
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import linalg as LA
from scipy.spatial import distance
import operator
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
import joblib
import sys
from sklearn.decomposition import PCA
sys.setrecursionlimit(3000)

'''Pre-Processing Functions'''
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
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

def normalize(words):
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words
'''End of Pre-Processing Functions'''

'''Predictor Function'''
def clusterpredictor(title):
    titlelist=[]
    titlelist.append(title)
    '''Preprocessing the entered text'''
    # tokens=[]
    # for line in titlelist:
    #     linetoken=nltk.word_tokenize(line)
    #     tokens.append(linetoken)
    # clean=[]
    # i=0
    # for title in tokens:
    #     nice=normalize(title)
    #     clean.append(nice)
    #     i=i+1
    #     print(i)
    # clean = list(clean)
    # data_set_final=[]
    # for title in clean:
    #     s=" "
    #     s=s.join(title)
    #     data_set_final.append(s)
    #v = TfidfVectorizer(stop_words= 'english')
    #v= joblib.load(open("vectorizer.pkl","rb"))
    '''Predicting'''
    v= joblib.load('vectorizer.joblib')
    old_X= v.transform(titlelist)
    pca = joblib.load('PCA.joblib')
    X= pca.transform(old_X.toarray())

    #model=joblib.load(open("model1.pkl","rb"))
    model=joblib.load('model1.joblib')
    cluster=model.predict(X)
    distances = {}
    #clusters = joblib.load(open("clusters.pkl","rb"))
    clusters = joblib.load('clusters.joblib')
    print(type(clusters[cluster[0]]))
    for c in clusters[cluster[0]]:
        print(c)
        somelist=[]
        somelist.append(c)
        Y= v.transform(somelist)
        Y= pca.transform(Y.toarray())
        distances[c] = distance.euclidean(X,Y)

    sorted_distances = sorted(distances.items(), key = operator.itemgetter(1))
    return sorted_distances[:10]

'''creating instance of the class'''
app=Flask(__name__)

'''to tell flask what url shoud trigger the function index()'''
@app.route('/')
@app.route('/index',  methods=["GET", "POST"])
def index():
    return flask.render_template('index.html')

@app.route('/submit', methods=["POST"])
def submit():
    title=request.form["CR_title"]
    sentences=clusterpredictor(title)
    return flask.render_template('submit.html', sentences = sentences)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

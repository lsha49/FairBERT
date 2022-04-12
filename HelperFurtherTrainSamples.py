import pandas as pd
import numpy as np
import time
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from nltk import tokenize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import json
import nltk
import textstat

# MonashOrigin_text_lang
# casenote_demo_nodup
FileName = 'data/pretrain/forum_10000_filtered.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

# remove invalid encoding
# validContentCol = Corpus['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
# Corpus['contentCol'] = validContentCol

# translate label col
# labelCol = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1
# Corpus['labelCol'] = labelCol


# native = list()
# nonNative = list()
for index,entry in enumerate(Corpus['contentCol']):
    sents = sent_tokenize(entry)
    
    if index > 0:
        with open('data/pretrain/forum_10000_filtered_MLM.txt', 'a') as f:
                f.write("\n")
    
    for sent in sents: 
        sent = sent.replace("   ", "")
        with open('data/pretrain/forum_10000_filtered_MLM.txt', 'a') as f:
            if index > 0:
                f.write("\n")
            f.write(sent)
    



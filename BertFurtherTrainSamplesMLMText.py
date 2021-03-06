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
import re
from random import randrange

# forum_2021_demo_final
# forum_2021_gender_equal
# forum_2021_lang_equal
FileName = 'data/forum_2021_lang_equal.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')



for index,entry in enumerate(Corpus['forum_message']):
    sents = sent_tokenize(entry)
    
    # if index > 0:
        # with open('data/pretrain/forum_2021_lang_equal_MLM.txt', 'a') as f:
                # f.write("\n")
    
    for sent in sents: 
        sent = sent.replace("   ", "")
        sent = re.sub("[\(\[].*?[\)\]]", "", sent)
        words = word_tokenize(sent)
        # print(words);exit()
        # masked = False
        if ':' in words:
            continue; 

        masked = False 
        for word in words:
            if word.isalpha() and len(word) > 3 and randrange(10) == 1 and masked == False:
                toWrite = ' [MASK]'  
                toWriteOriginal = ' ' + word
                masked = True 
            elif len(word) > 20: 
                 toWrite = ''
                 toWriteOriginal = ''
            else:
                toWrite = ' ' + word
                toWriteOriginal = ' ' + word
            # sent = sent[:256]
            with open('data/pretrain/forum_2021_lang_equal_MLM.txt', 'a') as f:
                f.write(toWrite)
            with open('data/pretrain/forum_2021_lang_equal_MLM_original.txt', 'a') as f:
                f.write(toWriteOriginal)
        with open('data/pretrain/forum_2021_lang_equal_MLM.txt', 'a') as f:
            f.write("\n")
        with open('data/pretrain/forum_2021_lang_equal_MLM_original.txt', 'a') as f:
            f.write("\n")
    



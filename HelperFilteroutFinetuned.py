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
FileName = 'data/forum_units_users_2019_init5.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

# Monash_fine_tune is the fine-tuning labelled dataset
CorpusLabelled = pd.read_csv('data/Monash_fine_tune.csv', encoding='latin-1')
shouldNotContainIndex = CorpusLabelled['index'].to_numpy()


filteredCorpus = pd.DataFrame()
for index,entry in enumerate(Corpus['forum_message']):
    studentId = Corpus.loc[index,'person_id']
    if studentId in shouldNotContainIndex:
        Corpus.loc[index, 'forum_message'] = ''
        

Corpus['forum_message'].replace('', np.nan, inplace=True)
Corpus.dropna(subset=['forum_message'], inplace=True)
Corpus.to_csv('data/forum_2019_demo_final.csv',index=False)


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
FileName = 'data/forum_demographics.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

# MonashOrigin_text_demographic is the fine-tuning labelled dataset
CorpusLabelled = pd.read_csv('data/MonashOrigin_text_demographic.csv', encoding='latin-1')
shouldNotContainIndex = CorpusLabelled['index'].to_numpy()


# remove invalid encoding
validContentCol = Corpus['forum_post_clean'].str.encode('ascii', 'ignore').str.decode('ascii')
Corpus['forum_post_clean'] = validContentCol

filteredCorpus = pd.DataFrame()
for index,entry in enumerate(Corpus['forum_post_clean']):
    studentId = Corpus.loc[index,'V1']
    if studentId in shouldNotContainIndex:
        continue 
    filteredCorpus.loc[index, 'unit_owning_org_primary'] = Corpus.loc[index, 'unit_owning_org_primary']
    filteredCorpus.loc[index, 'V1'] = Corpus.loc[index, 'V1']
    filteredCorpus.loc[index, 'forum_post_clean'] = Corpus.loc[index, 'forum_post_clean']
    filteredCorpus.loc[index, 'sex'] = Corpus.loc[index, 'sex']
    filteredCorpus.loc[index, 'lang'] = Corpus.loc[index, 'lang']
        
filteredCorpus.to_csv('data/forum_demographics_filtered.csv',index=False)


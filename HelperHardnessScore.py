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
from deslib.util.instance_hardness import kdn_score
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


# MonashOrigin_fairness_bert_embed  zCasenote_demo_bert_courtR
filename = 'data/zCasenote_demo_bert_courtR.csv'
Corpus = pd.read_csv(filename, encoding='latin-1')  # CSV file containing posts
useLabel = 'label'

if filename == 'data/zCasenote_demo_bert_courtR.csv' or filename == 'data/zCasenote_demo_bert_tuned.csv':
    Corpus.drop('Casenote ID', inplace=True, axis=1)
    Corpus.drop('Content', inplace=True, axis=1)

    # single label
    labelCol = Corpus['CourtR']

    Corpus.drop('Material', inplace=True, axis=1)
    Corpus.drop('Procedural', inplace=True, axis=1)
    Corpus.drop('CourtR', inplace=True, axis=1)
    Corpus.drop('Title', inplace=True, axis=1)
    Corpus.drop('CourtD', inplace=True, axis=1)
    Corpus.drop('Footnotes', inplace=True, axis=1)
    Corpus[useLabel] = labelCol
    labels = Corpus[useLabel].to_numpy()
            
    Corpus['gender'].replace(' ', np.nan, inplace=True)
    Corpus = Corpus.dropna(subset=['gender'])

    # genders = np.where(Corpus['gender']=='F', 2, 1) # use gender
    genders = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 2) # use language 1 male is native English

    Corpus.drop('home_language', inplace=True, axis=1)
    Corpus.drop('birth_country', inplace=True, axis=1)
    Corpus.drop('gender', inplace=True, axis=1)
    Corpus.drop(useLabel, inplace=True, axis=1)


if filename == 'data/MonashOrigin_fairness_bert_embed.csv':
    Corpus[useLabel] = np.where(pd.isnull(Corpus[useLabel]), 0, 1)
    Corpus.drop('facaulty', inplace=True, axis=1)

    # genders = np.where(Corpus['gender'] == 'F', 2, 1)  # if using gender 
    genders = np.where(Corpus['lang'].str.contains('english', case=False), 1, 2) # language 1 male is native English

    labels = Corpus[useLabel].to_numpy()

    Corpus.drop(useLabel, inplace=True, axis=1)
    Corpus.drop('protected', inplace=True, axis=1)
    Corpus.drop('lang', inplace=True, axis=1)
    Corpus.drop('gender', inplace=True, axis=1)



X = Corpus
X = X.replace(np.nan, 0)
features = X.to_numpy()   

features = pd.DataFrame(features)
features = features.replace(np.nan, 0)

maleIndexs  = np.where(genders==1)[0]
femaleIndexs  = np.where(genders==2)[0]
inde0 = np.where(labels==0)[0]
inde1 = np.where(labels==1)[0]

male0 = np.intersect1d(maleIndexs, inde0)
female0 = np.intersect1d(femaleIndexs, inde0)
male1 = np.intersect1d(maleIndexs, inde1)
female1 = np.intersect1d(femaleIndexs, inde1)

        
kdnResult = kdn_score(features, labels, 2)

maleKDNlist0 = kdnResult[0][male0]
femaleKDNlist0 = kdnResult[0][female0]
maleKDNlist1 = kdnResult[0][male1]
femaleKDNlist1 = kdnResult[0][female1]

print('native num: ', len(male0), len(male1))
print('non-native num: ', len(female0), len(female1))


print('native: ', sum(maleKDNlist0)/len(maleKDNlist0), sum(maleKDNlist1)/len(maleKDNlist1))
print('non-native: ', sum(femaleKDNlist0)/len(femaleKDNlist0), sum(femaleKDNlist1)/len(femaleKDNlist1))


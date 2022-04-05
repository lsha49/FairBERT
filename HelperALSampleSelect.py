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
from alipy import ToolBox
from collections.abc import Iterable
from alipy.query_strategy import QueryInstanceLAL,QueryInstanceQUIRE,QueryInstanceSPAL

# forum_demographics_filtered
# MonashOrigin_fairness_bert_embed
FileName = 'data/MonashOrigin_fairness_bert_embed.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

label = np.where(pd.isnull(Corpus['label']), 0, 1)
labelL = np.where(Corpus['lang'].str.contains('english', case=False), 1, 0) # native is 1
labelG = np.where(Corpus['gender'] == 'F', 0, 1) 


###### AL strategies select representative samples ######
allIndList = []
firstIndList = []
secondIndList = []
for i in range(0, len(label)):
    allIndList = allIndList + [i]

for i in range(0, 1000):
    firstIndList = firstIndList + [i]

for i in range(1000, 2000):
    secondIndList = secondIndList + [i]

# only leave embedding to corpus
Corpus.drop('facaulty', inplace=True, axis=1)
Corpus.drop('label', inplace=True, axis=1)
Corpus.drop('gender', inplace=True, axis=1)
Corpus.drop('lang', inplace=True, axis=1)
features = Corpus.replace(np.nan, 0)


# original sampling baseline
# random selection baseline

# group fairness, equal sampling: random
# group fairness, equal sampling: representativeness, informativeness
# group fairness, equal sampling: hardness




### Random select baseline




######### Representativeness
### QueryInstanceBMDR, Representative and informative
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceBMDR')
# select_ind = Strategy.select(firstIndList, secondIndList, batch_size=100)

### QueryInstanceGraphDensity: representativeness
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceGraphDensity', train_idx=allIndList)
# select_ind = Strategy.select(firstIndList, allIndList, batch_size=100)





######### Uncertainty
### QueryInstanceQBC: Heterogeneity-based, uncertainty-based, query-by-committee
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)


### QueryInstanceUncertainty: Heterogeneity-based, uncertainty-based, fast
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)





######### ERROR REDUCTION
### QueryExpectedErrorReduction: Expected Error reduction
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryExpectedErrorReduction')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)

### QueryInstanceLAL: Expected Error Reduction on a trained regressor
# alibox = ToolBox(X=features, y=label, query_type='AllLabels', saving_path='')
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceLAL')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)




#########Hardness 
# may need to be replaced by instance hardness
# alibox = ToolBox(X=Corpus, y=labelColLang, query_type='AllLabels', saving_path='')
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceSPAL')
# select_ind = Strategy.select(firstIndList, allIndList, model=None, batch_size=100)





print(select_ind)


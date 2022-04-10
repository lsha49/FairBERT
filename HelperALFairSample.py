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
from imblearn.under_sampling import RandomUnderSampler
from deslib.util.instance_hardness import kdn_score
from scipy.spatial import distance

### Baseline
# no further pre-training  
# further pre-training original sampling 
# further pre-training random under-sampling 

### Fair sampling (worst demo predictability), w/ or w/o demo label
# unrepresentative, uncertain

### group fairness (best task predictability) => either improve general accuracy or group fairness
# representative, certain


# XXXXX
# Monash_fine_tune, MonashOrigin_fairness_bert_embed
Corpus = pd.read_csv('data/XXXXX.csv', encoding='latin-1')
FineTuneCorpus = pd.read_csv('data/MonashOrigin_fairness_bert_embed.csv', encoding='latin-1')


labelFineY = np.where(pd.isnull(FineTuneCorpus['label']), 0, 1)
labelFineG = np.where(FineTuneCorpus['gender'] == 'F', 0, 1) 
# labelFineG = np.where(FineTuneCorpus['lang'].str.contains('english', case=False), 1, 0) # native is 1

labelG = np.where(Corpus['gender'] == 'F', 0, 1) 
# labelG = np.where(Corpus['lang'].str.contains('english', case=False), 1, 0) # native is 1



# get BERT embeddings fine tune samples
FineTuneCorpus.drop('facaulty', inplace=True, axis=1)
FineTuneCorpus.drop('label', inplace=True, axis=1)
FineTuneCorpus.drop('gender', inplace=True, axis=1)
FineTuneCorpus.drop('lang', inplace=True, axis=1)
featuresFine = FineTuneCorpus.replace(np.nan, 0)

# get BERT embeddings pre-train samples
features = Corpus.replace(np.nan, 0)


###### select a small fair task-labelled set for AL-based selection ######
labelFineYG = np.char.add(labelFineY.astype(str), labelFineG.astype(str))
taskInd00 = np.where(labelFineYG=='00')[0]
taskInd01 = np.where(labelFineYG=='01')[0]
taskInd10 = np.where(labelFineYG=='10')[0]
taskInd11 = np.where(labelFineYG=='11')[0]
taskInd00_ran = np.random.choice(taskInd00, size=10, replace=False)
taskInd01_ran = np.random.choice(taskInd01, size=10, replace=False)
taskInd10_ran = np.random.choice(taskInd10, size=10, replace=False)
taskInd11_ran = np.random.choice(taskInd11, size=10, replace=False)
taskAll = np.concatenate([taskInd00,taskInd01,taskInd10,taskInd11])
tasklabelledIndList = np.concatenate([taskInd00_ran,taskInd01_ran,taskInd10_ran,taskInd11_ran])
# taskunlabelledIndList = np.setxor1d(taskAll, tasklabelledIndList)

# h-bias
# kdnResult = kdn_score(featuresFine.iloc[tasklabelledIndList], labelFineY[tasklabelledIndList], 5)
kdnResult = kdn_score(featuresFine, labelFineY, 5)
KDNlist00 = kdnResult[0][taskInd00_ran]
KDNlist01 = kdnResult[0][taskInd01_ran]
KDNlist10 = kdnResult[0][taskInd10_ran]
KDNlist11 = kdnResult[0][taskInd11_ran]

kl_pq0 = distance.jensenshannon(KDNlist00, KDNlist01)
kl_pq1 = distance.jensenshannon(KDNlist10, KDNlist11)

print('H-bias:', (kl_pq0 + kl_pq1)/2)

savedTasklabelledIndList = pd.DataFrame(tasklabelledIndList)
savedTasklabelledIndList.to_csv('savedTasklabelledIndList.csv',index=False)


exit()




###### AL select samples ######

### Random select baseline
# @todo


### QueryInstanceQBC: query-by-committee, fast
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceQBC')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)

### QueryInstanceUncertainty: uncertainity, fast
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceUncertainty')
# select_ind = Strategy.select(firstIndList, secondIndList, model=None, batch_size=100)

### QueryInstanceGraphDensity: representativeness
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceGraphDensity', train_idx=allIndList)
# select_ind = Strategy.select(firstIndList, allIndList, batch_size=100)

### QueryInstanceBMDR, representative and informative
# alibox = ToolBox(X=features, y=label)
# Strategy = alibox.get_query_strategy(strategy_name='QueryInstanceBMDR')
# select_ind = Strategy.select(firstIndList, secondIndList, batch_size=100)






print(select_ind)


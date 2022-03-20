from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch
import tensorflow as tf
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# MonashOrigin_fairness_bert_embed

filename = 'data/MonashOrigin_fairness_bert_embed.csv'
Corpus = pd.read_csv(filename, encoding='latin-1')  # CSV file containing posts
useLabel = 'label'

if filename == 'data/MonashOrigin_fairness_bert_embed.csv':
    Corpus[useLabel] = np.where(pd.isnull(Corpus[useLabel]), 0, 1)
    Corpus.drop('facaulty', inplace=True, axis=1)

    # Corpus['protected'] = np.where(Corpus['gender'] == 'F', 2, 1)  # if using gender 
    Corpus['protected'] = np.where(Corpus['lang'].str.contains('english', case=False), 1, 2) # language 1 male is native English

Y = Corpus[useLabel]

Corpus.drop(useLabel, inplace=True, axis=1)
Corpus.drop('protected', inplace=True, axis=1)
Corpus.drop('lang', inplace=True, axis=1)
Corpus.drop('gender', inplace=True, axis=1)

X = Corpus
X = X.replace(np.nan, 0)
X = X.to_numpy()   


# scl = LogisticRegression()
# scl.fit(self.Train_X,self.Train_Y)
# predicted = scl.predict(X)
# preditedProb = scl.predict_proba(X)
# preditedProb1 = preditedProb[:, 1]


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X, Y, test_size=0.2, random_state=11)

# hidden_layer_sizes=(100,100,100) represent three layers, each layer has 100 nerons
clf = MLPClassifier(hidden_layer_sizes=(100,), random_state=1, max_iter=300).fit(Train_X, Train_Y)

# print(clf.n_layers_);exit()

predicted = clf.predict_proba(Test_X)
preditedProb = clf.predict_proba(Test_X)
preditedProb1 = preditedProb[:, 1]

# print(predicted);exit()
        
# ABROCA computation
# abrocaDf = self.getDfToComputeAbroca(predicted, preditedProb)
# self.computeAbroca(abrocaDf)

# AUC computation 
print("AUC Score -> ", roc_auc_score(Test_Y, preditedProb1))




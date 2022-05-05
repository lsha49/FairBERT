from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch
from sklearn import model_selection, naive_bayes, svm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from abroca import *

# Monash_fine_tune_clean_further_embed
Corpus = pd.read_csv('../../uq67_scratch/bfiledata/Monash_fine_tune_test_embed.csv', encoding='latin-1')


Corpus['gender'] = np.where(Corpus['gender']=='F', 0, 1)
Corpus['home_language'] = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1
labelCol = Corpus['label']

Corpus.drop('forum_message', inplace=True, axis=1)
Corpus.drop('label', inplace=True, axis=1)
Corpus = Corpus.replace(np.nan, 0)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus, labelCol, test_size=0.2, random_state=11)

Train_X.drop('gender', inplace=True, axis=1)
Train_X.drop('home_language', inplace=True, axis=1)
Test_G = Test_X['gender']
Test_L = Test_X['home_language']
Test_X.drop('gender', inplace=True, axis=1)
Test_X.drop('home_language', inplace=True, axis=1)

### Logistic regression
lr_clf=LogisticRegression()
lr_clf.fit(Train_X,Train_Y)
predicted = lr_clf.predict(Test_X)
preditedProb = lr_clf.predict_proba(Test_X)
preditedProb1 = preditedProb[:, 1]


print(predicted)

print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
print("AUC Score -> ", roc_auc_score(Test_Y,predicted))
# print("AUC Score -> ", roc_auc_score(Test_Y,preditedProb1))
print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))


print("dem1 AUC Score -> ", roc_auc_score(Test_Y,predicted))


# ABROCA computation
abrocaDf = pd.DataFrame(predicted, columns = ['predicted'])
abrocaDf['prob_1'] = pd.DataFrame(preditedProb)[1]
abrocaDf['label'] = Test_Y
abrocaDf['demo'] = Test_L.astype(str)

slice = compute_abroca(abrocaDf, 
                        pred_col = 'prob_1' , 
                        label_col = 'label', 
                        protected_attr_col = 'demo',
                        majority_protected_attr_val = '1',
                        compare_type = 'binary', # binary, overall, etc...
                        n_grid = 10000,
                        plot_slices = False)    
print(slice)


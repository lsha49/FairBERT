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


Corpus = pd.read_csv('data/forum_units_users_2019_init2.csv', encoding='latin-1')
mapp = Corpus['person_id'].value_counts(dropna=False)
total = 0

for index,val in mapp.items():
    total = val + total
    if val > 20:
        Corpus['person_id'].replace(index, np.nan, inplace=True)
        
        
Corpus.dropna(subset=['person_id'], inplace=True)

# print(len(mapp))

# print(total/len(mapp))

# print(Corpus['person_id'].value_counts(dropna=False))

Corpus.to_csv('data/forum_units_users_2019_init3.csv',index=False)

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


Corpus = pd.read_csv('data/forum_units_users_2019_init.csv', encoding='latin-1')

ind1 = 0
ind2 = 0
ind3 = 0

for index,entry in enumerate(Corpus['person_id']):
    if len(str(Corpus.loc[index, 'forum_message']).strip()) < 100:  
        Corpus.loc[index, 'forum_message'] = ''
        ind1 = ind1 + 1
    if 'img' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '^' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '*' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '=' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '+' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1


print(ind1)
print(ind2)
print(ind3)

Corpus['forum_message'].replace(r'http\S+', '', regex=True, inplace=True)
Corpus['forum_message'].replace(r'www\S+', '', regex=True, inplace=True)
Corpus['forum_message'].replace('', np.nan, inplace=True)
Corpus.dropna(subset=['forum_message'], inplace=True)

Corpus.to_csv('data/forum_units_users_2019_init2.csv',index=False)

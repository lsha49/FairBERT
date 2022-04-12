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


Corpus = pd.read_csv('data/student_demographics.csv', encoding='latin-1')

for index,entry in enumerate(Corpus['update_rank']):
    if entry == 2:  
        Corpus.loc[index, 'update_rank'] = ''
        

Corpus['update_rank'].replace('', np.nan, inplace=True)
Corpus.dropna(subset=['update_rank'], inplace=True)

Corpus.to_csv('data/student_demographics_no_uprank2.csv',index=False)

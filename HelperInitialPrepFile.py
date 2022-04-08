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

# forum_10000_filtered

Corpus = pd.read_csv('data/forum_units_users_2021_6.csv', encoding='latin-1')
Corpus['indexx'] = Corpus['Unnamed: 0']
Corpus['forum_message'] = Corpus['forum_message'].str.encode('ascii', 'ignore').str.decode('ascii')

Corpus['person_id'].replace('', np.nan, inplace=True)
# Corpus[Corpus.person_id.apply(lambda x: x.isnumeric())]
Corpus[pd.to_numeric(Corpus['person_id'], errors='coerce').notnull()]

Corpus['forum_message'].replace('', np.nan, inplace=True)
Corpus['forum_message'].replace(r'<[^<>]*>', '', regex=True, inplace=True)

Corpus.dropna(subset=['person_id'], inplace=True)
Corpus.dropna(subset=['forum_message'], inplace=True)

Corpus = Corpus[['indexx','person_id','forum_message']]

Corpus.to_csv('data/forum_units_users_2021_6_init.csv',index=False)

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


FileName = 'data/Monash_fine_tune.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

# remove invalid encoding
validContentCol = Corpus['Content'].str.encode('ascii', 'ignore').str.decode('ascii')
Corpus['Content'] = validContentCol

Corpus.to_csv('data/Monash_fine_tune_clean.csv',index=False)

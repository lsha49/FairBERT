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

# load base bert
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# forum_units_users_2021_1_init_demo
FileName = 'forum_units_users_2021_1_init_demo.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

for index,entry in enumerate(Corpus['Content']):
    input_ids = torch.tensor(tokenizer.encode(entry)).unsqueeze(0)  
    outputs = model(input_ids)

    hidden_states = outputs[1]
    embedding_output = hidden_states[12].detach().numpy()[0]
    finalEmb = embedding_output[len(embedding_output)-1]

    for iindex,ientry in enumerate(finalEmb):
        Corpus.loc[index, iindex] = str(ientry)

Corpus.to_csv('forum_units_users_2021_1_init_demo_embed.csv',index=False)



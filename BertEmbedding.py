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

# First fine-tune the model with 80% data 

# Then, use the fine-tuned model to generate embedding of the 20% test data

# last pass the test data embedding to classify


# legal bert model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", output_hidden_states=True)
# model = AutoModelForSequenceClassification.from_pretrained("test_trainer_CourtR", local_files_only=True, output_hidden_states=True)

FileName = 'zcleanedAllSenten_liwc_demo.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')

for index,entry in enumerate(Corpus['Content']):
    input_ids = torch.tensor(tokenizer.encode(entry)).unsqueeze(0)  
    outputs = model(input_ids)

    hidden_states = outputs[1]
    embedding_output = hidden_states[12].detach().numpy()[0]
    finalEmb = embedding_output[len(embedding_output)-1]

    for iindex,ientry in enumerate(finalEmb):
        Corpus.loc[index, iindex] = str(ientry)

Corpus.to_csv('zCasenote_demo_bert_courtR.csv',index=False)



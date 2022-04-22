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
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",model_max_length=512)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", output_hidden_states=True)
# model = AutoModelForSequenceClassification.from_pretrained("saved_model/further_2021_original", output_hidden_states=True, local_files_only=True)

# forum_2021_demo_final
# forum_2021_gender_test
# forum_2021_lang_train
Corpus = pd.read_csv('data/forum_2021_lang_train.csv', encoding='latin-1')

excep = 0
for index,entry in enumerate(Corpus['forum_message']):
    # try:
        input_ids = torch.tensor(tokenizer.encode(entry,truncation=True)).unsqueeze(0)  
        outputs = model(input_ids)

        hidden_states = outputs[1]
        embedding_output = hidden_states[12].detach().numpy()[0]
        finalEmb = embedding_output[len(embedding_output)-1]

        for iindex,ientry in enumerate(finalEmb):
            Corpus.loc[index, iindex] = str(ientry)
    # except:
        # excep = excep + 1
        # continue

print(excep)

Corpus.to_csv('data/forum_2021_lang_train_embed_bert_base.csv',index=False)



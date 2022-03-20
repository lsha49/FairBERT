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


# legal bert tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# @todo change to the current and new model directory
model = AutoModelForSequenceClassification.from_pretrained("test_trainer_multisix", local_files_only=True, num_labels=6)
training_args = TrainingArguments(
    output_dir='test_trainer_multisix_retrained',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10
)

# @todo depending on the previous user marked error labelling saved in DB, maybe retrieve as csv containing text and label column
useLabel = 'Multi6'
FileName = 'data/cleanedAllSenten_liwc.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')
TextArr = Corpus['Content']
if useLabel == 'Multi6':
    labelCol = Corpus['Material'].astype(str) + Corpus['Procedural'].astype(str) + Corpus['CourtR'].astype(str) + Corpus['Title'].astype(str) + Corpus['CourtD'].astype(str) + Corpus['Footnotes'].astype(str)
    labelCol = np.where(labelCol == '100000', 0, labelCol)
    labelCol = np.where(labelCol == '010000', 1, labelCol)
    labelCol = np.where(labelCol == '001000', 2, labelCol)
    labelCol = np.where(labelCol == '000100', 3, labelCol)
    labelCol = np.where(labelCol == '000010', 4, labelCol)
    labelCol = np.where(labelCol == '000001', 5, labelCol)

x_train, x_val, y_train, y_val = model_selection.train_test_split(TextArr, labelCol, test_size = 0.05, random_state=12)

# change all to list
x_train = x_train.tolist();x_val = x_val.tolist();y_train = y_train.tolist();y_val = y_val.tolist()

# tokenise train and validation data for training
train_encodings = tokenizer(x_train, truncation=False, padding=True)
val_encodings = tokenizer(x_val, truncation=False, padding=True)

class EncodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EncodeDataset(train_encodings, y_train)
eval_dataset = EncodeDataset(val_encodings, y_val)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()
trainer.save_model()

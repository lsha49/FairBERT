from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_metric
import torch
import tensorflow as tf
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForMaskedLM, BertForPreTraining, BertConfig
from transformers import TextDatasetForNextSentencePrediction
from transformers import DataCollatorForLanguageModeling


Corpus = pd.read_csv('data/forum_2021_lang_equal_mask_manual.csv', encoding='latin-1')

### perform BertForMaskedLM only
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
# labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
inputs = tokenizer(Corpus['masked'].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=256)
inputs['labels'] = tokenizer(Corpus['original'].tolist(), return_tensors="pt",  truncation=True, padding=True, max_length=256)["input_ids"]
# labels = tokenizer(Corpus['original'].tolist(), return_tensors="pt",  truncation=True, padding=True, max_length=256)["input_ids"]
# outputs = model(**inputs, labels=labels)

args = TrainingArguments(
    output_dir='saved_model/further_2021_lang_equal_mlm_manual',
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
)

class EncodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = EncodeDataset(inputs)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model()

# loss = outputs.loss
# logits = outputs.logits
# model.save_pretrained("saved_model/further_2021_lang_equal_mlm")

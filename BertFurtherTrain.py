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

config = BertConfig()
bert_cased_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining(config)

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=bert_cased_tokenizer,
    file_path="data/sample_further_train2.txt",
    block_size = 256
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_cased_tokenizer, 
    mlm=True,
    mlm_probability= 0.15
)



training_args = TrainingArguments(
    output_dir= "saved_model/further_3",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_gpu_train_batch_size= 16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model()
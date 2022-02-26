from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import TextClassificationPipeline
import pandas as pd
import numpy as np
from transformers import TrainingArguments
import torch
import tensorflow as tf
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from nltk.tokenize import sent_tokenize, word_tokenize
from bertPredict import test_bert
import nltk


# input text for testing purposes
TestCorpus = '\
this was an action commenced in the original jurisdiction of the high court of australia before windeyer j in mchale v watson (1964). \
the claim against watson is based on trespass to the person and in negligence; the action was to recover damages for personal injuries suffered by mchale. \
the principle argument by the appellant counsel was a question of law; the common law prescribes a minimum standard of care to be observed by every person. \
therefore, infancy is no defence to negligence. \
the defence counsel countered this with a question of fact; the standard of care should be measured against a reasonable boy of the same age rather than the standard of care expected of a reasonable man. \
his honour found watson was not at fault nor negligent in the legal sense, holding that childhood is not an idiosyncrasy.  \
this established the principle of law that child tortfeasors face a lower standard of care expected of a reasonable child. \
judgement of the high court of australia the appeal was argued on two grounds. \
firstly that windeyer j had errored in finding a variance between an adult and child standard of care, and, secondly that his honour should have found negligence, regardless of the measured standard. \
the majority judgement consisted of individual judicial reasoning but agreed that age, specifically childhood, is not a subjective characteristic but an objective stage of human development.'


# load legal bert model tokenizer, model and piplines
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("test_trainer_multisix", local_files_only=True)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# break paragraph into sentences
sentences = sent_tokenize(TestCorpus)

# generate predictions on the sentences array
preds = pipe(sentences)

print(pred)
# the predicted labels are as follows:
# LABEL_0: 'Material'
# LABEL_1: 'Procedural'
# LABEL_2: 'CourtR'
# LABEL_3: 'Title'
# LABEL_4: 'CourtD'
# LABEL_5: 'Footnotes'

    


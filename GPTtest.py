from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments, BertTokenizer
from transformers import Trainer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch
from sklearn import model_selection, naive_bayes, svm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from abroca import *
from imblearn.under_sampling import RandomUnderSampler
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Hi Jao,Hope you're doing well! I just had a quick question regarding the mid-semester exam and whether research methods are examinable?Hope you have a lovely rest of your weekend!Thankyou kindly, Ella Thomas
# While I do agree that the presence and prevalence of weapons of mass destruction do ensure peace, the form of peace is more akin to order through fear than mutual agreement through serenity - no violation of sovereignty out of fear of mutually assured destruction as opposed to consensual agreement. This form of peace is, in my opinion, not something that should be clung to. This change in the nature of the peace is achievable due to the more impactful role of interstate organisations and the greater availability of channels of communication available to states in the modern world, allowing grievances to be aired and reasonable compromise to be made.I agree that the use and procurement of chemical weapons should be banned, but this is due to a belief that all weapons with a higher likely hood of collateral damage (drone strikes, mines, biological weapons) should be blanket banned. Small arms however should not be banned as these are more precise instruments of war - thus being the lesser of the necessary evils.

prompt = 'While I do agree that the presence and prevalence of weapons of mass destruction do ensure peace,'
generated = generator(prompt, max_length=100, num_return_sequences=5)



print(generated)
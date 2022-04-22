from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments
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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression


# forum_2021_demo_final
Corpus = pd.read_csv('data/forum_2021_lang_train.csv', encoding='latin-1')

# femaleRow_ind = np.where(Corpus['gender']=='F')[0]
femaleRow_ind = np.where(Corpus['home_language'].str.contains('english', case=False))[0]
femaleRow_eq_ind = np.random.choice(femaleRow_ind, size=11260, replace=False)
femaleEqCorpus = Corpus.loc[femaleRow_eq_ind]

# maleRow_ind = np.where(Corpus['gender']=='M')[0]
maleRow_ind = np.where(Corpus['home_language'].str.contains('english', case=False) == False)[0]
maleRow_eq_ind = np.random.choice(maleRow_ind, size=11260, replace=False)
maleEqCorpus = Corpus.loc[maleRow_eq_ind]

allTrain = np.concatenate([femaleEqCorpus,maleEqCorpus])
allTrainDf = pd.DataFrame(allTrain, columns = ['gender','home_language','birth_country','indexx','person_id','forum_message'])
allTrainDf.to_csv('data/forum_2021_lang_equal.csv',index=False)


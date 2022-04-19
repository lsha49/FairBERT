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
Corpus = pd.read_csv('data/forum_2021_demo_final.csv', encoding='latin-1')

# femaleRow_ind = np.where(Corpus['gender']=='F')[0]
femaleRow_ind = np.where(Corpus['home_language'].str.contains('english', case=False))[0]
femaleRow_test_ind = np.random.choice(femaleRow_ind, size=5630, replace=False)
femaleRow_train_ind = np.setxor1d(femaleRow_ind, femaleRow_test_ind)
femaleTestCorpus = Corpus.loc[femaleRow_test_ind]
femaleTrainCorpus = Corpus.loc[femaleRow_train_ind]

# maleRow_ind = np.where(Corpus['gender']=='M')[0]
maleRow_ind = np.where(Corpus['home_language'].str.contains('english', case=False) == False)[0]
maleRow_test_ind = np.random.choice(maleRow_ind, size=5630, replace=False)
maleRow_train_ind = np.setxor1d(maleRow_ind, maleRow_test_ind)
maleTestCorpus = Corpus.loc[maleRow_test_ind]
maleTrainCorpus = Corpus.loc[maleRow_train_ind]

allTrain = np.concatenate([femaleTrainCorpus,maleTrainCorpus])
allTest = np.concatenate([femaleTestCorpus,maleTestCorpus])

allTrainDf = pd.DataFrame(allTrain, columns = ['gender','home_language','birth_country','indexx','person_id','forum_message'])
allTestDf = pd.DataFrame(allTest, columns = ['gender','home_language','birth_country','indexx','person_id','forum_message'])


allTrainDf.to_csv('data/forum_2021_lang_train.csv',index=False)
allTestDf.to_csv('data/forum_2021_lang_test.csv',index=False)

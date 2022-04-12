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

Corpus = pd.read_csv('data/forum_2021_demo_final_embed1.csv', encoding='latin-1')
# Corpus['1'].replace('', np.nan, inplace=True)
# Corpus.dropna(subset=['forum_message'], inplace=True)
# Corpus.to_csv('data/forum_2021_demo_final_embed1.csv',index=False)
# exit()

# using gender language  
labelCol = np.where(Corpus['gender']=='F', 0, 1)
# labelCol = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1

Corpus.drop('gender', inplace=True, axis=1)
Corpus.drop('home_language', inplace=True, axis=1)
Corpus.drop('birth_country', inplace=True, axis=1)
Corpus.drop('indexx', inplace=True, axis=1)
Corpus.drop('person_id', inplace=True, axis=1)
Corpus.drop('forum_message', inplace=True, axis=1)
Corpus = Corpus.replace(np.nan, 0)

# print(Corpus);exit()

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus, labelCol, test_size=0.05, random_state=11)

lr_clf=LogisticRegression()
lr_clf.fit(Train_X,Train_Y)
predicted = lr_clf.predict(Test_X)
preditedProb = lr_clf.predict_proba(Test_X)
preditedProb1 = preditedProb[:, 1]


print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
print("AUC Score -> ", roc_auc_score(Test_Y,preditedProb1))
print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))


resdf = pd.DataFrame(predicted, columns=['predicted'])
resdf['testy'] = Test_Y
resdf.to_csv('data/resdf_demo.csv',index=False)


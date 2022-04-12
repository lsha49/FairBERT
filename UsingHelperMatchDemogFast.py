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


Corpus = pd.read_csv('data/forum_units_users_2017_init.csv', encoding='latin-1')
Demo = pd.read_csv('data/student_demographics_no_uprank2.csv', encoding='latin-1')


# Match demo
genderDict = pd.Series(Demo.sex.values,index=Demo.person_id).to_dict()
lanDict = pd.Series(Demo.home_language.values,index=Demo.person_id).to_dict()
birthDict = pd.Series(Demo.birth_country.values,index=Demo.person_id).to_dict()

savedDf = pd.DataFrame()
for index,entry in enumerate(Corpus['person_id']):
    intPersonId = int(float(entry))
    if intPersonId in genderDict:
        savedDf.loc[index, 'gender'] = genderDict[intPersonId]
        savedDf.loc[index, 'home_language'] = lanDict[intPersonId]
        savedDf.loc[index, 'birth_country'] = birthDict[intPersonId]
        savedDf.loc[index, 'indexx'] = Corpus.loc[index, 'indexx']
        savedDf.loc[index, 'person_id'] = Corpus.loc[index, 'person_id']
        savedDf.loc[index, 'forum_message'] = Corpus.loc[index, 'forum_message']
                
savedDf.to_csv('data/forum_2017_demo_matched_demo.csv',index=False)
savedDf.reset_index(drop=True, inplace=True)


# remove invalid lang
ind4 = 0
for index,entry in enumerate(savedDf['person_id']):
    if 'NON-ENGLISH (NO INFORMATION)' in str(savedDf.loc[index, 'home_language']):  
        savedDf.loc[index, 'forum_message'] = ''
        ind4 = ind4 + 1
    

print(ind4)

savedDf['forum_message'].replace('', np.nan, inplace=True)
savedDf.dropna(subset=['forum_message'], inplace=True)

savedDf.to_csv('data/forum_2017_demo_final.csv',index=False)

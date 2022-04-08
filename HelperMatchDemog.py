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


Corpus = pd.read_csv('data/forum_units_users_2021_3_init.csv', encoding='latin-1')
Demo = pd.read_csv('data/student_demographics.csv', encoding='latin-1')

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

savedDf = pd.DataFrame()
for index,entry in enumerate(Corpus['person_id']):
    skipOuter = False
    for iindex,ientry in enumerate(Demo['person_id']):
        if isfloat(entry):
            if int(float(ientry)) == int(float(entry)) and Demo.loc[iindex, 'update_rank'] == 1:
                postText = Corpus.loc[index, 'forum_message']
                if len(str(postText).strip()) < 10: 
                    skipOuter = True
                    continue

                savedDf.loc[index, 'gender'] = Demo.loc[iindex, 'sex']
                savedDf.loc[index, 'home_language'] = Demo.loc[iindex, 'home_language']
                savedDf.loc[index, 'birth_country'] = Demo.loc[iindex, 'birth_country']
                
                savedDf.loc[index, 'indexx'] = Corpus.loc[index, 'indexx']
                savedDf.loc[index, 'person_id'] = Corpus.loc[index, 'person_id']
                savedDf.loc[index, 'forum_message'] = Corpus.loc[index, 'forum_message']
                
                skipOuter = True
    if skipOuter:
        continue


savedDf.to_csv('data/forum_units_users_2021_3_init_demo.csv',index=False)

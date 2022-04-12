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

# forum_10000_filtered

Corpus = pd.read_csv('data/forum_units_users_2017.csv', encoding='latin-1')
Corpus['indexx'] = Corpus['Unnamed: 0']
Corpus['forum_message'] = Corpus['forum_message'].str.encode('ascii', 'ignore').str.decode('ascii')

Corpus['person_id'].replace('', np.nan, inplace=True)
Corpus[pd.to_numeric(Corpus['person_id'], errors='coerce').notnull()]

Corpus['forum_message'].replace('', np.nan, inplace=True)
Corpus['forum_message'].replace(r'<[^<>]*>', '', regex=True, inplace=True)

Corpus.dropna(subset=['person_id'], inplace=True)
Corpus.dropna(subset=['forum_message'], inplace=True)

Corpus = Corpus[['indexx','person_id','forum_message']]

Corpus.reset_index(drop=True, inplace=True)

# remove invalid posts
ind1 = 0
ind2 = 0


# print(type(Corpus));exit()

for index,entry in enumerate(Corpus['person_id']):
    if len(str(Corpus.loc[index, 'forum_message']).strip()) < 100:  
        Corpus.loc[index, 'forum_message'] = ''
        ind1 = ind1 + 1
    if 'img' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '^' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '*' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '=' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1
    if '+' in str(Corpus.loc[index, 'forum_message']):  
        Corpus.loc[index, 'forum_message'] = ''
        ind2 = ind2 + 1


print(ind1)
print(ind2)


# remove url
Corpus['forum_message'].replace(r'http\S+', '', regex=True, inplace=True)
Corpus['forum_message'].replace(r'www\S+', '', regex=True, inplace=True)
Corpus['forum_message'].replace('', np.nan, inplace=True)
Corpus.dropna(subset=['forum_message'], inplace=True)
Corpus.reset_index(drop=True, inplace=True)

# remove posts with more than 20 posts, usually tutor
mapp = Corpus['person_id'].value_counts(dropna=False)
total = 0

for index,val in mapp.items():
    total = val + total
    if val > 20:
        Corpus['person_id'].replace(index, np.nan, inplace=True)
        
        
Corpus.dropna(subset=['person_id'], inplace=True)
Corpus.reset_index(drop=True, inplace=True)

# remove rows duplicated fine-tuned row
CorpusLabelled = pd.read_csv('data/Monash_fine_tune.csv', encoding='latin-1')
shouldNotContainIndex = CorpusLabelled['index'].to_numpy()
ind5 = 0
filteredCorpus = pd.DataFrame()
for index,entry in enumerate(Corpus['forum_message']):
    studentId = Corpus.loc[index,'person_id']
    if studentId in shouldNotContainIndex:
        Corpus.loc[index, 'forum_message'] = ''
        ind5 = ind5 + 1
        
Corpus['forum_message'].replace('', np.nan, inplace=True)
Corpus.dropna(subset=['forum_message'], inplace=True)
Corpus.reset_index(drop=True, inplace=True)
print(ind5)

Corpus.to_csv('data/forum_units_users_2017_init.csv',index=False)

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
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import tokenize
from nltk.corpus import words
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer

Corpus = pd.read_csv('data/pretrain/forum_2021_lang_confi_mlm.csv', encoding='latin-1')

st = LancasterStemmer()

for index,entry in enumerate(Corpus['masked']):
    if '[MASK]' not in str(entry):  
        Corpus.loc[index, 'masked'] = ''
    if 'George' in str(entry):  
        Corpus.loc[index, 'masked'] = ''
    if 'george' in str(entry):  
        Corpus.loc[index, 'masked'] = ''
    if entry.find('[MASK]') == 0:
        Corpus.loc[index, 'masked'] = ''
    if entry.find('[MASK]') == 1:
        Corpus.loc[index, 'masked'] = ''
    if entry.find('[MASK]') == 2:
        Corpus.loc[index, 'masked'] = ''

    # tag_map = defaultdict(lambda : wn.NOUN)
    # tag_map['J'] = wn.ADJ
    # tag_map['V'] = wn.VERB
    # tag_map['R'] = wn.ADV
    # wordList = tokenize.word_tokenize(entry)
    # for word, tag in pos_tag(wordList):
    #     word_Lemmatized = WordNetLemmatizer()
    #     word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
    #     word_Final = word_Final.lower()
    #     word_Final = st.stem(word_Final)
    #     if word_Final not in words.words() and word != 'MASK' and word.isalpha():
    #         Corpus.loc[index, 'masked'] = ''


Corpus['masked'].replace('', np.nan, inplace=True)
Corpus.dropna(subset=['masked'], inplace=True)

Corpus.to_csv('data/pretrain/forum_2021_lang_confi_mlm_1.csv',index=False)

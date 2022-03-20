
import pandas as pd
import numpy as np
import time
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from nltk import tokenize
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import math
import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import OneSidedSelection
from sklearn import metrics
from deslib.util.instance_hardness import kdn_score
from numpy import where
import random
from math import log2, log, sqrt
from abroca import *
from alipy import ToolBox
from scipy.spatial import distance
from sklearn.utils import _safe_indexing
from statistics import stdev, mean

# restructure_feature_med # duolingo

class test_cbt(object):
    def preprocessing(self): 
        # MonashOrigin_fairness_bert_embed
        # train_feat # KDD 
        # X_train_view # OU 
        # zCasenote_demo_bert_courtR # Casenote untuned
        # matched_ADM_dktp # ADM

        filename = 'data/X_train_view.csv'
        Corpus = pd.read_csv(filename, encoding='latin-1') # CSV file containing posts
        useLabel = 'label'

        # Monash label from text to binary, Gender value text to 1/0
        if filename == 'data/MonashOrigin_fairness_bert_embed.csv': 
            Corpus[useLabel] = np.where(pd.isnull(Corpus[useLabel]), 0, 1) 
            Corpus.drop('facaulty', inplace=True, axis=1)

            Corpus['gender'] = np.where(Corpus['gender']=='F', 2, 1) # if using gender
            Corpus.drop('lang', inplace=True, axis=1)

            # Corpus['gender'] = np.where(Corpus['lang'].str.contains('english', case=False), 1, 2) # language 1 male is native English
            # Corpus.drop('lang', inplace=True, axis=1)

        if filename == 'data/train_feat.csv':
            Corpus.drop('facaulty', inplace=True, axis=1)

        if filename == 'data/X_train_view.csv':
            Corpus['gender'] = np.where(Corpus['gender']==0, 2, 1) 

        if filename == 'data/zCasenote_demo_bert_courtR.csv' or filename == 'data/zCasenote_demo_bert_tuned.csv':
            Corpus.drop('Casenote ID', inplace=True, axis=1)
            Corpus.drop('Content', inplace=True, axis=1)

            # single label
            labelCol = Corpus['CourtR']

            Corpus.drop('Material', inplace=True, axis=1)
            Corpus.drop('Procedural', inplace=True, axis=1)
            Corpus.drop('CourtR', inplace=True, axis=1)
            Corpus.drop('Title', inplace=True, axis=1)
            Corpus.drop('CourtD', inplace=True, axis=1)
            Corpus.drop('Footnotes', inplace=True, axis=1)
            Corpus[useLabel] = labelCol
            
            Corpus['gender'].replace(' ', np.nan, inplace=True)
            Corpus = Corpus.dropna(subset=['gender'])



            Corpus['gender'] = np.where(Corpus['gender']=='F', 2, 1) # use gender
            Corpus.drop('home_language', inplace=True, axis=1)
            Corpus.drop('birth_country', inplace=True, axis=1)

            # Corpus['gender'] = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 2) # use language 1 male is native English
            # Corpus.drop('home_language', inplace=True, axis=1)
            # Corpus.drop('birth_country', inplace=True, axis=1)



            # reorder column label first gender second 
            first_column = Corpus.pop('label')
            Corpus.insert(1, 'label', first_column)

            # count rows belongs to male and female different classes
            # male0 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 0)];print(len(male0))
            # male1 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 1)];print(len(male1))
            # male2 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 2)];print(len(male2))
            # male3 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 3)];print(len(male3))
            # male4 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 4)];print(len(male4))
            # male5 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 5)];print(len(male5))
            # female0 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 0)];print(len(female0))
            # female1 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 1)];print(len(female1))
            # female2 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 2)];print(len(female2))
            # female3 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 3)];print(len(female3))
            # female4 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 4)];print(len(female4))
            # female5 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 5)];print(len(female5))

            # male0 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 0)];print(len(male0))
            # male1 = Corpus[(Corpus['gender'] == 1) & (Corpus['label'] == 1)];print(len(male1))
            # female0 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 0)];print(len(female0))
            # female1 = Corpus[(Corpus['gender'] == 2) & (Corpus['label'] == 1)];print(len(female1))

        if filename == 'data/matched_ADM_dktp.csv':
            Corpus['gender'] = np.where(Corpus['gender']=='Female', 2, 1) 
            Corpus.drop('ITEST_id', inplace=True, axis=1)


        maleC = Corpus[Corpus['gender'] == 1];print('male', len(maleC))
        femaleC = Corpus[Corpus['gender'] == 2];print('female', len(femaleC))
            
        # Labels
        corpusY = Corpus[useLabel]
        corpusX = Corpus
        
        corpusX = corpusX.replace(np.nan, 0)
        corpusX = corpusX.to_numpy()                                                                                # 0.85 0.2 0.3
        self.Train_X, self.Test_X, self.Train_Y, self.Test_Y = model_selection.train_test_split(corpusX, corpusY, test_size=0.2, random_state=11)
        
        self.Train_Y = self.Train_Y.astype(int)
        self.Test_Y = self.Test_Y.astype(int)

        # class balancing extensions 
        self.cbt()
        
        # calculate KDN post CBT (with or without CB) on train split
        self.calKDN()

        # firstHbias = self.calKDN()
        # try second iteration
        # priorTrain_X = np.insert(self.Train_X, 0, self.Train_Y, axis=1) # insert label at col 2
        # priorTrain_X = np.insert(priorTrain_X, 0, self.Train_G, axis=1) # insert gender at col 1
        # self.Train_Y = pd.Series(self.Train_Y)
        # priorTrain_Y = self.Train_Y
        # priorTrain_G = self.Train_G
        # self.Train_X = priorTrain_X
        # self.cbt()
        # secondHbias = self.calKDN()
        # print(firstHbias, secondHbias)
        # if firstHbias < secondHbias: 
        #     self.Train_X = priorTrain_X
        #     self.Train_Y = priorTrain_Y
        #     self.Train_Y = pd.Series(self.Train_Y)
        #     self.Train_G = priorTrain_G
        # self.cbt()
        # thirdHbias = self.calKDN()
        # print(thirdHbias)

                

    def cbt(self):            
        # C  D  OV     
        # CU DU UNOSTR  (update code in Tomek Link class)
        balanceMode = 'OV' # change mode
        
###### class
        if balanceMode == '':
            self.Train_G = self.Train_X[:, 0].astype(int)

            # countMale0 = self.Train_X[(self.Train_X[:, 0] == 1) & (self.Train_X[:, 1] == 0)]
            # countMale1 = self.Train_X[(self.Train_X[:, 0] == 1) & (self.Train_X[:, 1] == 1)]
            # countFemale0 = self.Train_X[(self.Train_X[:, 0] == 2) & (self.Train_X[:, 1] == 0)]
            # countFemale1 = self.Train_X[(self.Train_X[:, 0] == 2) & (self.Train_X[:, 1] == 1)]
            # malecount = len(countMale0) + len(countMale1)
            # femalecount = len(countFemale0) + len(countFemale1)
            # totalCount = malecount + femalecount
            # male1prob = len(countMale1) / totalCount
            # female1prob = len(countFemale1) / totalCount
            # male0prob = len(countMale0) / totalCount
            # female0prob = len(countFemale0) / totalCount
            # print('D-bias: ', (abs(male1prob - female1prob) + abs(male0prob - female0prob))/2)

            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_Y = self.Train_Y.to_numpy()
            
        if balanceMode == 'C':
            self.Train_X, self.Train_Y = SMOTE().fit_resample(self.Train_X, self.Train_Y) 
            
            
            # countMale0 = self.Train_X[(self.Train_X[:, 0] == 1) & (self.Train_X[:, 1] == 0)]
            # countMale1 = self.Train_X[(self.Train_X[:, 0] == 1) & (self.Train_X[:, 1] == 1)]
            # countFemale0 = self.Train_X[(self.Train_X[:, 0] == 2) & (self.Train_X[:, 1] == 0)]
            # countFemale1 = self.Train_X[(self.Train_X[:, 0] == 2) & (self.Train_X[:, 1] == 1)]
            # malecount = len(countMale0) + len(countMale1)
            # femalecount = len(countFemale0) + len(countFemale1)
            # totalCount = malecount + femalecount
            # male1prob = len(countMale1) / totalCount
            # female1prob = len(countFemale1) / totalCount
            # male0prob = len(countMale0) / totalCount
            # female0prob = len(countFemale0) / totalCount
            # print('D-bias: ', (abs(male1prob - female1prob) + abs(male0prob - female0prob))/2)
            # exit()


            self.Train_G = self.Train_X[:, 0].astype(int)
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_Y = self.Train_Y.to_numpy()


        if balanceMode == 'D':
            self.Train_G = self.Train_X[:, 0].astype(int)
            self.Train_X = self.Train_X[:,1:] # remove gender 
            self.Train_X, self.Train_G = SMOTE().fit_resample(self.Train_X, self.Train_G) 
            self.Train_Y = self.Train_X[:, 0].astype(int)
            self.Train_X = self.Train_X[:,1:] # remove label


        if balanceMode == 'CU': # class undersample
            self.Train_X, self.Train_Y = NearMiss().fit_resample(self.Train_X, self.Train_Y) 
            

            # countMale0 = self.Train_X[(self.Train_X[:, 0] == 1) & (self.Train_X[:, 1] == 0)]
            # countMale1 = self.Train_X[(self.Train_X[:, 0] == 1) & (self.Train_X[:, 1] == 1)]
            # countFemale0 = self.Train_X[(self.Train_X[:, 0] == 2) & (self.Train_X[:, 1] == 0)]
            # countFemale1 = self.Train_X[(self.Train_X[:, 0] == 2) & (self.Train_X[:, 1] == 1)]
            # malecount = len(countMale0) + len(countMale1)
            # femalecount = len(countFemale0) + len(countFemale1)
            # totalCount = malecount + femalecount
            # male1prob = len(countMale1) / totalCount
            # female1prob = len(countFemale1) / totalCount
            # male0prob = len(countMale0) / totalCount
            # female0prob = len(countFemale0) / totalCount
            # print(abs(male1prob - female1prob) + abs(male0prob - female0prob))
            # exit()


            self.Train_G = self.Train_X[:, 0].astype(int)
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_Y = self.Train_Y.to_numpy()

        
        if balanceMode == 'DU':
            self.Train_G = self.Train_X[:, 0].astype(int)
            self.Train_X = self.Train_X[:,1:] # remove gender 
            self.Train_X, self.Train_G = NearMiss().fit_resample(self.Train_X, self.Train_G) 
            self.Train_Y = self.Train_X[:, 0].astype(int)
            self.Train_X = self.Train_X[:,1:] # remove label

# Class and Demographic     
        if balanceMode == 'UNO': # over sample with instance strategy 
            self.Train_G = self.Train_X[:, 0].astype(int) # gender is the first column, label is the second column
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_GY = self.Train_G.astype(str) + self.Train_Y.astype(str)

            self.Train_X, self.Train_GY = SMOTE().fit_resample(self.Train_X, self.Train_GY)  # KMeansSMOTE BorderlineSMOTE SMOTE
            
            self.Train_G = self.Train_GY.str[0].astype(int)
            self.Train_Y = self.Train_GY.str[1].astype(int)

            # get KDN list of Train_X
            kdnScores = kdn_score(self.Train_X, self.Train_Y.to_numpy(), 5)[0]
            
            self.Train_X = np.insert(self.Train_X, 0, self.Train_G, axis=1) # insert gender at col 2 back for tomeklink
            self.Train_X = np.insert(self.Train_X, 0, kdnScores, axis=1) # insert KDN at col 1 for tomeklink

            print(len(self.Train_X))
            self.Train_X, self.Train_GY = TomekLinks().fit_resample(self.Train_X, self.Train_GY)  # sampling_strategy="all"
            print(len(self.Train_X))

            self.Train_G = self.Train_GY.str[0].astype(int)
            self.Train_Y = self.Train_GY.str[1].astype(int).to_numpy()
            
        if balanceMode == 'OV': # over sample with or without h-bias strategy
            self.Train_G = self.Train_X[:, 0].astype(int) # gender is the first column, label is the second column
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_GY = self.Train_G.astype(str) + self.Train_Y.astype(str)

            originalLen = len(self.Train_X)
            print('original len:', originalLen)
            self.Train_X, self.Train_GY = SMOTE().fit_resample(self.Train_X, self.Train_GY)  # KMeansSMOTE BorderlineSMOTE SMOTE            
            self.Train_G = self.Train_GY.str[0].astype(int)
            self.Train_Y = self.Train_GY.str[1].astype(int).to_numpy()

            postLen = len(self.Train_X)
            print('post len:', postLen)
            # print((postLen - originalLen) / originalLen)
            # exit()

            
            # recomposedGY = np.column_stack((self.Train_G.to_numpy(), self.Train_Y))
            # countMale0 = recomposedGY[(recomposedGY[:, 0] == 1) & (recomposedGY[:, 1] == 0)]
            # countMale1 = recomposedGY[(recomposedGY[:, 0] == 1) & (recomposedGY[:, 1] == 1)]
            # countFemale0 = recomposedGY[(recomposedGY[:, 0] == 2) & (recomposedGY[:, 1] == 0)]
            # countFemale1 = recomposedGY[(recomposedGY[:, 0] == 2) & (recomposedGY[:, 1] == 1)]
            # print(len(countMale0), len(countMale1),len(countFemale0),len(countFemale1))
            # lenArr = [len(countMale0), len(countMale1),len(countFemale0),len(countFemale1)]
            # print(stdev(lenArr)/mean(lenArr))
            # malecount = len(countMale0) + len(countMale1)
            # femalecount = len(countFemale0) + len(countFemale1)
            # totalCount = malecount + femalecount
            # male1prob = len(countMale1) / totalCount
            # female1prob = len(countFemale1) / totalCount
            # male0prob = len(countMale0) / totalCount
            # female0prob = len(countFemale0) / totalCount
            # print(abs(male1prob - female1prob) + abs(male0prob - female0prob))
            # exit()

        if balanceMode == 'UNOSTR': # under sample without strategy
            self.Train_G = self.Train_X[:, 0].astype(int) # gender is the first column, label is the second column
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_GY = self.Train_G.astype(str) + self.Train_Y.astype(str)

            # get KDN list of Train_X
            # kdnScores = kdn_score(self.Train_X, self.Train_Y.to_numpy(), 5)[0]
            
            # self.Train_X = np.insert(self.Train_X, 0, self.Train_G, axis=1) # insert gender at col 2 back for tomeklink
            # self.Train_X = np.insert(self.Train_X, 0, kdnScores, axis=1) # insert KDN at col 1 for tomeklink

            # sampling_strategy='not minority' TomekLinks NearMiss
            self.Train_X, self.Train_GY = NearMiss().fit_resample(self.Train_X, self.Train_GY)  
            
            self.Train_G = self.Train_GY.str[0].astype(int)
            self.Train_Y = self.Train_GY.str[1].astype(int).to_numpy()

            # recomposedGY = np.column_stack((self.Train_G.to_numpy(), self.Train_Y))
            # countMale0 = recomposedGY[(recomposedGY[:, 0] == 1) & (recomposedGY[:, 1] == 0)]
            # countMale1 = recomposedGY[(recomposedGY[:, 0] == 1) & (recomposedGY[:, 1] == 1)]
            # countFemale0 = recomposedGY[(recomposedGY[:, 0] == 2) & (recomposedGY[:, 1] == 0)]
            # countFemale1 = recomposedGY[(recomposedGY[:, 0] == 2) & (recomposedGY[:, 1] == 1)]
            # print(len(countMale0), len(countMale1),len(countFemale0),len(countFemale1))
            # malecount = len(countMale0) + len(countMale1)
            # femalecount = len(countFemale0) + len(countFemale1)
            # totalCount = malecount + femalecount
            # male1prob = len(countMale1) / totalCount
            # female1prob = len(countFemale1) / totalCount
            # male0prob = len(countMale0) / totalCount
            # female0prob = len(countFemale0) / totalCount
            # print(abs(male1prob - female1prob) + abs(male0prob - female0prob))
            # exit()

        if balanceMode == 'UNU': # under sample with instance strategy
            self.Train_G = self.Train_X[:, 0].astype(int) # gender is the first column, label is the second column
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_GY = self.Train_G.astype(str) + self.Train_Y.astype(str)

            # get KDN list of Train_X
            kdnScores = kdn_score(self.Train_X, self.Train_Y.to_numpy(), 5)[0]
            
            self.Train_X = np.insert(self.Train_X, 0, self.Train_G, axis=1) # insert gender at col 2 back for tomeklink
            self.Train_X = np.insert(self.Train_X, 0, kdnScores, axis=1) # insert KDN at col 1 for tomeklink

            print(len(self.Train_X))
            self.Train_X, self.Train_GY = TomekLinks(sampling_strategy="majority").fit_resample(self.Train_X, self.Train_GY)  # sampling_strategy="all"
            print(len(self.Train_X))

            self.Train_G = self.Train_GY.str[0].astype(int)
            self.Train_Y = self.Train_GY.str[1].astype(int).to_numpy()
        

        if balanceMode == 'SPL':
            self.Train_G = self.Train_X[:, 0].astype(int) # gender is the first column, label is the second column
            self.Train_X = self.Train_X[:,2:] # remove gender and label
            self.Train_GY = self.Train_G.astype(str) + self.Train_Y.astype(str)

            originalLen = len(self.Train_X)

            self.Train_X, self.Train_GY = SMOTE().fit_resample(self.Train_X, self.Train_GY)  # KMeansSMOTE BorderlineSMOTE SMOTE
            
            self.Train_G = self.Train_GY.str[0].astype(int)
            self.Train_Y = self.Train_GY.str[1].astype(int).to_numpy()

            ###### AL strategies select representative samples 
            afterCBLen = len(self.Train_X)
            originalIndList = []
            for i in range(0, originalLen):
                originalIndList = originalIndList + [i]
            generatedIndlist = []
            for i in range(originalLen+1, afterCBLen):
                generatedIndlist = generatedIndlist + [i]

            alibox = ToolBox(X=self.Train_X, y=self.Train_Y, query_type='AllLabels', saving_path='')
            QBCStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceSPAL') # QueryInstanceSPAL QueryInstanceQBC QueryInstanceLAL
            select_ind = QBCStrategy.select(originalIndList, generatedIndlist, model=None, batch_size=100)
            self.Train_X = np.concatenate((self.Train_X[0:originalLen], self.Train_X[select_ind]), axis=0)
            self.Train_Y = np.concatenate((self.Train_Y[0:originalLen], self.Train_Y[select_ind]), axis=0)
            self.Train_G = np.concatenate((self.Train_G[0:originalLen], self.Train_G[select_ind]), axis=0)


        # reconstruct test data
        self.Test_G = pd.Series(self.Test_X[:, 0].astype(int))
        self.Test_Y = pd.Series(self.Test_X[:, 1].astype(int))
        self.Test_X = self.Test_X[:,2:]
        
        # random under-sample Test data to have equal male, female, true, false label 
        self.Test_GY = self.Test_G.astype(str) + self.Test_Y.astype(str)
        self.Test_X, self.Test_GY = RandomUnderSampler().fit_resample(self.Test_X, self.Test_GY) 
        self.Test_GY = pd.Series(self.Test_GY)
        self.Test_G = self.Test_GY.str[0].astype(int)
        self.Test_Y = self.Test_GY.str[1].astype(int)


        # KDN variables
        self.genders = self.Train_G
        self.labels = self.Train_Y
        self.features = pd.DataFrame(self.Train_X)
        


    def calKDN(self): 
        self.features = self.features.replace(np.nan, 0)

        self.maleIndexs  = np.where(self.genders==1)[0]
        self.femaleIndexs  = np.where(self.genders==2)[0]
        self.inde0 = np.where(self.labels==0)[0]
        self.inde1 = np.where(self.labels==1)[0]

        male0 = np.intersect1d(self.maleIndexs, self.inde0)
        female0 = np.intersect1d(self.femaleIndexs, self.inde0)
        male1 = np.intersect1d(self.maleIndexs, self.inde1)
        female1 = np.intersect1d(self.femaleIndexs, self.inde1)

        
        kdnResult = kdn_score(self.features, self.labels, 5)

        maleKDNlist0 = kdnResult[0][male0]
        femaleKDNlist0 = kdnResult[0][female0]
        maleKDNlist1 = kdnResult[0][male1]
        femaleKDNlist1 = kdnResult[0][female1]

        if len(maleKDNlist0) > len(femaleKDNlist0):
            smallerList0 = femaleKDNlist0
            largerList0 = maleKDNlist0
        else: 
            smallerList0 = maleKDNlist0
            largerList0 = femaleKDNlist0

        if len(maleKDNlist1) > len(femaleKDNlist1):
            smallerList1 = femaleKDNlist1
            largerList1 = maleKDNlist1
        else: 
            smallerList1 = maleKDNlist1
            largerList1 = femaleKDNlist1

        # smaller padding 
        largerList0 = np.pad(largerList0, (0, 1000), 'mean')
        largerList1 = np.pad(largerList1, (0, 1000), 'mean')
        
        smallerList0 = np.pad(smallerList0, (0, len(largerList0) - len(smallerList0)), 'mean')
        smallerList1 = np.pad(smallerList1, (0, len(largerList1) - len(smallerList1)), 'mean')

        # larger padding
        # smallerList0 = np.pad(smallerList0, (0, len(largerList0) - len(smallerList0)), 'linear_ramp', end_values=(0, 100000))
        # smallerList1 = np.pad(smallerList1, (0, len(largerList1) - len(smallerList1)), 'linear_ramp', end_values=(0, 100000))

    
        kl_pq0 = distance.jensenshannon(smallerList0, largerList0)
        kl_pq1 = distance.jensenshannon(smallerList1, largerList1)

        print('H-bias:', (kl_pq0 + kl_pq1)/2)
        # print('MaleKDN:', sum(maleKDNlist)/len(maleKDNlist)) 
        # print('FemaleKDN:', sum(femaleKDNlist)/len(femaleKDNlist))

        return (kl_pq0 + kl_pq1)/2



    def eval(self):
        rfc=RandomForestClassifier(n_estimators=100)
        rfc.fit(self.Train_X,self.Train_Y)
        predicted = rfc.predict(self.Test_X)
        preditedProb = rfc.predict_proba(self.Test_X)
        preditedProb1 = preditedProb[:, 1]
        
        # ABROCA computation
        abrocaDf = self.getDfToComputeAbroca(predicted, preditedProb)
        self.computeAbroca(abrocaDf)

        # AUC computation 
        print("AUC Score -> ", roc_auc_score(self.Test_Y, preditedProb1))
        # print("Multi AUC Score -> ", roc_auc_score(self.Test_Y, preditedProb, average='weighted', multi_class='ovo'))

        maleTestIndexs  = np.where(self.Test_G==1)
        femaleTestIndexs  = np.where(self.Test_G==2)
    
        self.Test_Y = self.Test_Y.to_numpy()
        print("Male AUC Score -> ", roc_auc_score(self.Test_Y[maleTestIndexs], preditedProb1[maleTestIndexs]))
        print("Female AUC Score -> ", roc_auc_score(self.Test_Y[femaleTestIndexs], preditedProb1[femaleTestIndexs]))



    # ABROCA related calculations
    def getDfToComputeAbroca(self, predicted, predictionProb):
        predictionDataframe = pd.DataFrame(predicted, columns = ['predicted'])
        predictionDataframe['prob_1'] = pd.DataFrame(predictionProb)[1]
        predictionDataframe['label'] = self.Test_Y.tolist()
        predictionDataframe['gender'] = self.Test_G.astype(str)
        return predictionDataframe

    # ABROCA calculation
    def computeAbroca(self, abrocaDf):
        slice = compute_abroca(abrocaDf, 
                            pred_col = 'prob_1' , 
                            label_col = 'label', 
                            protected_attr_col = 'gender',
                            majority_protected_attr_val = '2',
                            compare_type = 'binary', # binary, overall, etc...
                            n_grid = 10000,
                            plot_slices = False)
        print(slice)

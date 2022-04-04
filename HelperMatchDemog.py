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

Corpus = pd.read_csv('data/forum_10000_filtered.csv', encoding='latin-1')
Demo = pd.read_csv('data/forum_demographics.csv', encoding='latin-1')


for index,entry in enumerate(Corpus['index']):
    for iindex,ientry in enumerate(Demo['person_id']):
        if int(ientry) == int(entry) and Demo.loc[iindex, 'update_rank'] == 1:
            Corpus.loc[index, 'gender'] = Demo.loc[iindex, 'sex']
            Corpus.loc[index, 'home_language'] = Demo.loc[iindex, 'home_language']
            Corpus.loc[index, 'birth_country'] = Demo.loc[iindex, 'birth_country']

Corpus.to_csv('data/forum_10000_filtered_demo.csv',index=False)








exit()

test_encodings = tokenizer(Test_X, truncation=False, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = IMDbDataset(test_encodings, Test_Y)


trainer = Trainer(model=model)

predicted = trainer.predict(test_dataset)

prediction_logits = predicted[0]
print(prediction_logits)
prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()[:,1]
print(f'The prediction probs are: {prediction_probs}')

predicted = np.where(prediction_probs > 0.5, 1, 0)

predictionDataframe = pd.DataFrame(prediction_probs, columns = ['prediction_probs'])
predictionDataframe['predicted'] = predicted
predictionDataframe['label'] = Test_Y

predictionDataframe.to_csv('predicted_clf.csv',index=False)



print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
print("ROC AUC Score -> ",roc_auc_score(predicted, Test_Y))
# print("ROC AUC Score -> ", roc_auc_score(Test_Y.astype(str), predictions_rfc_multi, average='weighted', multi_class='ovo'))
print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))
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

# load legal bert model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("test_trainer_multisix", local_files_only=True)

FileName = 'data/cleanedAllSenten.csv'
Corpus = pd.read_csv(FileName, encoding='latin-1')
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text'], Corpus['label'], test_size=0.2, random_state=11)

Test_X = Test_X.tolist()
Test_Y = Test_Y.tolist()

test_encodings = tokenizer(Test_X, truncation=False, padding=True)

class EncodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = EncodeDataset(test_encodings, Test_Y)

trainer = Trainer(model=model)

predicted = trainer.predict(test_dataset)
prediction_logits = predicted[0]
prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()[:,1]
print(f'The prediction probs are: {prediction_probs}')

predicted = np.where(prediction_probs > 0.5, 1, 0)
predictionDataframe = pd.DataFrame(prediction_probs, columns = ['prediction_probs'])
predictionDataframe['predicted'] = predicted
predictionDataframe['label'] = Test_Y
predictionDataframe.to_csv('predicted_clf.csv',index=False)

print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
print("ROC AUC Score -> ", roc_auc_score(Test_Y.astype(str), predictions_rfc_multi, average='weighted', multi_class='ovo'))
print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))
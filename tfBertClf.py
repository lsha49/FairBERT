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

# labels to use for prediction, multi6 is the label which can predict all 6 labels together
useLabel = 'Multi6' # Multi6

# input is sentence by sentence, therefore there needs to be pre-tokenized into sentences 
Corpus = pd.read_csv('data/cleanedAllSenten.csv', encoding='latin-1')

if useLabel == 'Multi6':
    labelCol = Corpus['Material'].astype(str) + Corpus['Procedural'].astype(str) + \
        Corpus['CourtR'].astype(str) + Corpus['Title'].astype(str) + \
        Corpus['CourtD'].astype(str) + Corpus['Footnotes'].astype(str)
    labelCol = np.where(labelCol == '100000', 0, labelCol)
    labelCol = np.where(labelCol == '010000', 1, labelCol)
    labelCol = np.where(labelCol == '001000', 2, labelCol)
    labelCol = np.where(labelCol == '000100', 3, labelCol)
    labelCol = np.where(labelCol == '000010', 4, labelCol)
    labelCol = np.where(labelCol == '000001', 5, labelCol)
else: 
    labelCol = Corpus[useLabel]

# legal bert model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased",num_labels=6)

# training parameter, using 1 epoch for quick testing
training_args = TrainingArguments(
    output_dir='saved_model/test_trainer',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10
)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Content'], labelCol, test_size=0.2, random_state=11)

x_train, x_val, y_train, y_val = model_selection.train_test_split(Train_X, Train_Y, test_size = 0.2, random_state=12)


x_train = x_train.tolist()
x_val = x_val.tolist()
y_train = y_train.tolist()
y_val = y_val.tolist()

Test_X = Test_X.tolist()
Test_Y = Test_Y.tolist()

train_encodings = tokenizer(x_train, truncation=False, padding=True)
val_encodings = tokenizer(x_val, truncation=False, padding=True)
test_encodings = tokenizer(Test_X, truncation=False, padding=True)

# this is used for encoding BERT embedding for model training
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

train_dataset = EncodeDataset(train_encodings, y_train)
eval_dataset = EncodeDataset(val_encodings, y_val)
test_dataset = EncodeDataset(test_encodings, Test_Y)


trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)

# model training
trainer.train()

# save the model
trainer.save_model()


# using model to make a prediction
predicted = trainer.predict(test_dataset)
prediction_logits = predicted[0]
prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()
predicted = np.argmax(prediction_logits, axis=-1)

# output metrics
print("Accuracy Score -> ",accuracy_score(predicted, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predicted, Test_Y))
print("ROC AUC Score -> ", roc_auc_score(Test_Y, prediction_probs, average='weighted', multi_class='ovo'))
print("F1 Score -> ",f1_score(predicted, Test_Y, average='weighted'))
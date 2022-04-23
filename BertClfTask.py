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

# forum_2021_gender_test
# forum_2021_lang_test
# Monash_fine_tune
Corpus = pd.read_csv('data/Monash_fine_tune.csv', encoding='latin-1')

# using gender language  
# labelCol = np.where(Corpus['gender']=='F', 0, 1)
# labelCol = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1

# load further pre-trained bert
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("saved_model/further_2021_equal_test", num_labels=2, local_files_only=True)
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# training parameter, using 1 epoch for quick testing
training_args = TrainingArguments(
    output_dir='saved_model/test_finetune',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1000
)

# default train args
# training_args = TrainingArguments("saved_model/further_2021_original_demo_fine_tune", evaluation_strategy="epoch", num_train_epochs=3)

# balance dataset to avoid overfitting on the majority class
# textCol = Corpus['forum_message'].to_numpy()
# textCol = np.reshape(textCol,(-1, 1)) 
# textCol, labelCol = RandomUnderSampler(random_state=11).fit_resample(textCol, labelCol) 
# textCol = textCol.flatten()

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Content'], Corpus['label'], test_size=0.2, random_state=111)

x_train, x_val, y_train, y_val = model_selection.train_test_split(Train_X, Train_Y, test_size = 0.2, random_state=12)


x_train = x_train.tolist()
x_val = x_val.tolist()
y_train = y_train.tolist()
y_val = y_val.tolist()
Test_X = Test_X.tolist()
Test_Y = Test_Y.tolist()

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=500)
val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=500)
test_encodings = tokenizer(Test_X, truncation=True, padding=True, max_length=500)

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
preditedProb1 = prediction_probs[:, 1]
predictedLabel = np.argmax(prediction_logits, axis=-1)

print(predictedLabel)

print("Accuracy Score -> ",accuracy_score(predictedLabel, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predictedLabel, Test_Y))
print("AUC Score -> ", roc_auc_score(Test_Y,preditedProb1))
print("F1 Score -> ",f1_score(predictedLabel, Test_Y, average='weighted'))

# output metrics
# print("Accuracy Score -> ",accuracy_score(preditedProb1, Test_Y))
# print("Kappa Score -> ",cohen_kappa_score(preditedProb1, Test_Y))
# print("ROC AUC Score -> ",roc_auc_score(preditedProb1, Test_Y))
# # print("ROC AUC Score -> ", roc_auc_score(Test_Y, prediction_probs, average='weighted', multi_class='ovo'))
# print("F1 Score -> ",f1_score(preditedProb1, Test_Y, average='weighted'))
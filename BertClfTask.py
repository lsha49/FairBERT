from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
from transformers import TrainingArguments, BertTokenizer
from transformers import Trainer, AutoModelForSequenceClassification, BertForSequenceClassification
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
from abroca import *
from imblearn.under_sampling import RandomUnderSampler

# forum_2021_gender_test
# forum_2021_lang_test
# Monash_fine_tune
# Monash_fine_tune_clean
Corpus = pd.read_csv('data/Monash_fine_tune_clean.csv', encoding='latin-1')

# bert_base_no_further_train
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("saved_model/further_1", num_labels=2)
# model = AutoModelForSequenceClassification.from_pretrained("saved_model/further_2021_lang_equal_mlm_manual", num_labels=2, local_files_only=True)
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) 
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# training parameter, using 1 epoch for quick testing
training_args = TrainingArguments(
    output_dir='saved_model/test_finetune',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    # logging_dir='./logs',            # directory for storing logs
    # logging_steps=1000
)

# default train args
# training_args = TrainingArguments("saved_model/further_2021_original_demo_fine_tune", evaluation_strategy="epoch", num_train_epochs=3)

Corpus['gender'] = np.where(Corpus['gender']=='F', 0, 1)
Corpus['home_language'] = np.where(Corpus['home_language'].str.contains('english', case=False), 1, 0) # native is 1

Train, Test, Train_Y, Test_Y = model_selection.train_test_split(Corpus, Corpus['label'], test_size=0.2, random_state=11) # 111

Train_X = Train['Content']
Test_X = Test['Content']
Test_G = Test['gender']
Test_L = Test['home_language']

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
# trainer.save_model()


# using model to make a prediction
predicted = trainer.predict(test_dataset)
prediction_logits = predicted[0]

prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()
preditedProb1 = prediction_probs[:, 1]
predictedLabel = np.argmax(prediction_logits, axis=-1)

print(predictedLabel)

print("Accuracy Score -> ",accuracy_score(predictedLabel, Test_Y))
print("Kappa Score -> ",cohen_kappa_score(predictedLabel, Test_Y))
print("AUC Score -> ", roc_auc_score(Test_Y,predictedLabel)) # preditedProb1
print("F1 Score -> ",f1_score(predictedLabel, Test_Y, average='weighted'))

# ABROCA computation
abrocaDf = pd.DataFrame(predictedLabel, columns = ['predicted'])
abrocaDf['prob_1'] = pd.DataFrame(prediction_probs)[1]
abrocaDf['label'] = Test_Y
abrocaDf['demo'] = Test_L.astype(str)

slice = compute_abroca(abrocaDf, 
                        pred_col = 'prob_1' , 
                        label_col = 'label', 
                        protected_attr_col = 'demo',
                        majority_protected_attr_val = '1',
                        compare_type = 'binary', # binary, overall, etc...
                        n_grid = 10000,
                        plot_slices = False)    
print(slice)

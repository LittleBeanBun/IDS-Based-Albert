from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from transformers import AlbertTokenizer,TFAlbertForSequenceClassification
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import time

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def plot_confusion_matrix(y_true, y_pred, title = "Confusion matrix",
                          cmap = plt.cm.Blues, save_flg = False):
    classes = [str(i) for i in range(5)]
    labels = range(5)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    if save_flg:
        plt.savefig("./confusion_matrix_val.png")
    plt.show()


def split_dataset(df):
    train_set, x = train_test_split(df, \
        stratify=df['label'],
        test_size=0.1, 
        random_state=42)
    val_set, test_set = train_test_split(x, \
        stratify=x['label'],
        test_size=0.5, 
        random_state=43)

    return train_set,val_set,test_set


def convert_example_to_feature(review):
    return tokenizer.encode_plus(review, 
                                 add_special_tokens = True, # add [CLS], [SEP]
                                 max_length = max_length, # max length of the text that can go to BERT
                                 pad_to_max_length = True, # add [PAD] tokens
                                 return_attention_mask = True, # add attention mask to not focus on pad tokens
                                )

# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
    
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)
    i=0
    for index, row in ds.iterrows():
        
        review = row["data"]
        label = row["label"]
        data_time = time.time()
        bert_input = convert_example_to_feature(review)
        print(time.time()-data_time)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
        i+=1
        if i==1:
            break
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict),label_list

if __name__ =='__main__':

    max_length = 256
    batch_size = 128
    learning_rate = 2e-5
    number_of_epochs = 10
    num_classes = 5 
    model_load_path = './save_weights/albert_weights_2'

    df_raw = pd.read_csv("./can_classify_data.txt",sep=",")    
    df_raw = shuffle(df_raw)
    _,val_data,test_data = split_dataset(df_raw)


    # test dataset
    ds_val_encoded,label_list = encode_examples(val_data)
    ds_val_encoded = ds_val_encoded.batch(1)
    
    #model
    model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_classes)
    #reload model
    try:
        model.load_weights(model_load_path)
        print('load model succeed')
    except:
        print('no saved model')
    
    classify_time = time.time()
    predict_classes = model.predict(ds_val_encoded)
    print(time.time()-classify_time)
    true_classes = np.argmax(predict_classes.logits,1)
    predect_label = []
    for lable in true_classes:
        
        predect_label.append([lable])

    plot_confusion_matrix(predect_label, label_list, save_flg = True)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer,TFBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

    for index, row in ds.iterrows():
        review = row["data"]
        label = row["label"]
        bert_input = convert_example_to_feature(review)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

if __name__ == '__main__':


    max_length = 256
    batch_size = 128
    learning_rate = 2e-5
    number_of_epochs = 10
    num_classes = 5 

    df_raw = pd.read_csv("./can_classify_data.txt",sep=",")    
    df_raw = shuffle(df_raw)
    train_data,val_data,test_data = split_dataset(df_raw)
    # train dataset
    ds_train_encoded = encode_examples(train_data).shuffle(len(train_data)).batch(batch_size)
    # val dataset
    ds_val_encoded = encode_examples(val_data).batch(batch_size)
    # test dataset
    ds_test_encoded = encode_examples(test_data).batch(batch_size)
    #model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, verbose=1, validation_data=ds_val_encoded)

    print("# evaluate test_set:",model.evaluate(ds_test_encoded))
from transformers import BertTokenizer, glue_convert_examples_to_features,TFBertForSequenceClassification
import tensorflow as tf
import numpy as np 
import os
import pandas as pd
import glob
import time 
import cv2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

EPOCHS = 50
BATCH_SIZE = 1
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)



def save_data(data_name,str_64,all_data,all_type):

    if data_name=='normal_run_data.txt':
        all_data.append(str_64)
        all_type.append('0')
    elif data_name=='DoS_dataset.txt':
        all_data.append(str_64)
        all_type.append('1')
    elif data_name=='Fuzzy_dataset.txt':
        all_data.append(str_64)
        all_type.append('2')
    elif data_name=='gear_dataset.txt':
        all_data.append(str_64)
        all_type.append('3')
    elif data_name=='RPM_dataset.txt':
        all_data.append(str_64)
        all_type.append('4')        

def read_data(data_path):
    
    all_data = []
    all_type = []
    data_list = os.listdir(data_path)
    for data_name in data_list:       
        if data_name.endswith('.txt'):
            print('Start {}'.format(data_name))
            data_id = []
            data_type = []
            if data_name == 'normal_run_data.txt':
                with open(os.path.join(data_path,data_name), 'r') as f:
                    while True:
                        lines = f.readline() 
                        if not lines:
                            break
                            pass
                        try:
                            can_id = lines.split()[3]
                            data_id.append(can_id)
                            data_type.append('R\n')
                        except:
                            continue
                
            else:
                with open(os.path.join(data_path,data_name), 'r') as f:
                    while True:
                        lines = f.readline() 
                        if not lines:
                            break
                        try:
                            can_id = lines.split(',')[1]
                            data_id.append(can_id)
                            can_type = lines.split(',')[-1]
                            data_type.append(can_type)
                        except:
                            continue
            str_64 = data_id[0][1:4]
            iter_type = []
            for indx,ids in enumerate(data_id[1:]):
                
                end_str = '100%'
                process_bar(indx/len(data_id), start_str='', end_str=end_str, total_length=15)
                ids = ids[1:4]
                if str_64 == '':
                    str_64 = str_64+ids
                else:
                    str_64 = str_64+' '+ids
                iter_type.append(data_type[indx])
                if (indx+1)%64==0:
                    for types in iter_type:
                        if types=='T\n':
                            save_data(data_name,str_64,all_data,all_type)
                            break
                    if data_name == 'normal_run_data.txt':
                        save_data(data_name,str_64,all_data,all_type)
                    iter_type = []
                    str_64 = ''
            print('\nComplete {}'.format(data_name))
    
    
    save_df = {'data':all_data,"label":all_type}
    
    df = pd.DataFrame(save_df)
    df.to_csv('./can_classify_data.txt')
    
    return all_data,all_type

if __name__ == '__main__':
    
    data_path = './car_hacking'

    data_set,data_label = read_data(data_path) 
        

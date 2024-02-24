import cv2
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.preprocessing import OneHotEncoder
import time

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)

enc = OneHotEncoder()
enc.fit([['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8'],['9'],['10'],['a'],['b'],['c'],['d'],['e'],['f']])

def save_image(data,save_path,i):
    image = np.zeros((16*3,64))
    for indx,can_data in enumerate(data):
        can_data = np.reshape(can_data,(16*3))
        image[:,indx]=can_data
    image = image.T
    cv2.imwrite(save_path+'/{}.png'.format(i),image*255)


def one_hot_vector(ids):
    annon = np.zeros((16,3))
    for indx,iter_id in enumerate(ids):
        if iter_id=='0':
            iter_id='10'
        ans = enc.transform([[iter_id]]).toarray()
        ans = ans[0]
        annon[:,indx] = ans
    return np.reshape(annon,(16,3))   

def main():
    data_dir = r'./car_hacking'
    hacking_dataname = os.listdir(data_dir)
    for indx,data_name in enumerate(hacking_dataname):
        if data_name.endswith('.txt'):     
            image_save_path = os.path.join(data_dir,data_name[:-4])
            if not os.path.exists(image_save_path):
                os.mkdir(image_save_path)
            print('Start {}.'.format(data_name))
            i = 1
            all_data = []
            image_data = []
            iter_type = []
            data_id = []
            data_type = []
            if data_name == 'normal_run_data.txt':
                with open(os.path.join(data_dir,data_name), 'r') as f:
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
                with open(os.path.join(data_dir,data_name), 'r') as f:
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
            for indx,ids in enumerate(data_id):
                
                end_str = '100%'
                process_bar(indx/len(data_id), start_str='', end_str=end_str, total_length=15)
                ids = ids[1:4]
                one_hot_form = one_hot_vector(ids)  #16*3
                image_data.append(one_hot_form)
                iter_type.append(data_type[indx])
                if (indx+1)%64==0:
                    for types in iter_type:
                        if types=='T\n':
                            save_image(image_data,image_save_path,i)
                            i = i+1
                            break
                    if data_name == 'normal_run_data.txt':
                        save_image(image_data,image_save_path,i)
                        i = i+1
                    image_data = []
                    iter_type = []
            print('Complete {}'.format(data_name))
            # unqine = np.unqine(all_data)
        
main()
import tensorflow as tf  #导入tensorflow
import tensorflow.keras as keras #导入keras
import matplotlib.pyplot as plt #plt绘图工具，可视化结果
import numpy as np  #导入numpy
import os
import PIL
import glob # glob 功能:文件操作模块,可查找目标文件,返回所有匹配的文件路径列表
import time #导入时间库
import cv2 #CV2指的是OpenCV2

# BUFFER_SIZE = 60000  #缓冲区容量tensorflow中的数据集类Dataset有一个shuffle方法，
# 用来打乱数据集中数据顺序，训练时非常常用。其中shuffle方法有一个参数buffer_size
BATCH_SIZE = 64 #表示单次传递给程序用以训练的数据(样本)个数
EPOCHS = 100 #训练轮次
z_dim = 6144 #噪声维数 #生成器从随机噪声开始生成图片,深度为100,与生成器模型输入层一致
num_examples_to_generate = 1 #生成1张图片显示
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True) #定义交叉熵损失函数
thred = 0.1 #阈值
rate = 10
model_indx = 'ckpt-10'   #如果模型保存为.ckpt的文件,则使用该文件就可以查看.ckpt文件里的变量
# generator
def make_generator():
    generator = keras.Sequential([
        keras.layers.Reshape((3,4,512)),
        keras.layers.Conv2DTranspose(256,(3,3),strides = (2,2),padding ='same',use_bias = False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2DTranspose(128,(3,3),strides = (2,2),padding ='same',use_bias = False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2DTranspose(64,(3,3),strides = (2,2),padding ='same',use_bias = False),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2DTranspose(1,(3,3),strides = (2,2),padding ='same',use_bias = False,activation = 'tanh'),
    ])

    return generator

def make_discriminator():
    discriminator = keras.Sequential([
        keras.layers.Reshape((16,3,64)),
        keras.layers.Conv2D(32,(3,3),strides = (1,1),padding ='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2D(16,(3,3),strides = (1,1),padding ='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Flatten(),
        keras.layers.Dense(1,activation='sigmoid'),
    ])

    return discriminator


# loss function
#定义生成器loss函数
def generator_loss(fake_pred):
    return cross_entropy(tf.ones_like(fake_pred),fake_pred)

#定义判别器损失函数
def discriminator_loss(fake_pred,real_pred):
    real_loss = cross_entropy(tf.ones_like(real_pred),real_pred)
    fake_loss = cross_entropy(tf.zeros_like(fake_pred),fake_pred)
    return real_loss + fake_loss


# traing
@tf.function
def train_d1_step(nor_image_batch,abnor_image_batch,d_1_optimizer,d_1):
    
    with tf.GradientTape() as d_tape:
        
        real_image = nor_image_batch
        fake_images = abnor_image_batch
        real_pred = d_1(real_image,training = True) #判别器提取测试集特征
        fake_pred = d_1(fake_images,training = True) #判别器判断生成器返回的图片
        # 计算D的loss值
        d_loss = discriminator_loss(fake_pred,real_pred)
        
    d_gradients = d_tape.gradient(d_loss,d_1.trainable_variables)
    
    d_1_optimizer.apply_gradients(zip(d_gradients,d_1.trainable_variables))
    
    return d_loss
    
def train_d2_step(nor_image_batch,g,g_optimizer,d_2,d_2_optimizer):

    z = tf.random.normal([BATCH_SIZE,z_dim])

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    
        fake_images = g(z,training = True)
        real_image = nor_image_batch
        real_pred = d_2(real_image,training = True)
        fake_pred = d_2(fake_images,training = True)
        
        d_loss = discriminator_loss(fake_pred,real_pred)
        g_loss = generator_loss(fake_pred)
        
    d_gradients = d_tape.gradient(d_loss,d_2.trainable_variables)    
    g_gradients = g_tape.gradient(g_loss,g.trainable_variables)
    
    d_2_optimizer.apply_gradients(zip(d_gradients,d_2.trainable_variables))
    g_optimizer.apply_gradients(zip(g_gradients,g.trainable_variables))
     
    return g_loss,d_loss


def generate_and_save_images(model, epoch, test_input):
    
    predictions = model(test_input, training=False)
    predictions = np.reshape(predictions,(48,64))
    predictions = (predictions*127.5+127.5).astype(np.uint8)
    save_path = './result_image'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path,'image_at_epoch_{:04d}.png'.format(epoch)),predictions.T)


def eval_model(d_1,d_2,test_data):

    for keys in test_data.keys():
        attack_data = test_data[keys]
        i = 0
        for indx,iter_data in enumerate(attack_data):
            iter_data = np.reshape(iter_data,(1,16,3,64))
            pred = d_1(iter_data,training = False)
            if pred < thred:
                i+=1
        print('{} Eval_result:'.format(keys),i/len(attack_data))


def train(normal_dataset,abnormal_dataset,epochs):

    train_abnormal_data = []
    test_abnormal_data = {}
    for keys in abnormal_dataset.keys():
        abnor_data = abnormal_dataset[keys]
        train_abnormal_data.extend(abnor_data[:int(len(abnor_data)*0.8)]) 
        test_abnormal_data[keys] = abnor_data[int(len(abnor_data)*0.8):]
    train_nor_dataset = tf.data.Dataset.from_tensor_slices(normal_dataset).shuffle(len(normal_dataset)).batch(BATCH_SIZE)
    train_abnor_dataset = tf.data.Dataset.from_tensor_slices(train_abnormal_data).shuffle(len(train_abnormal_data)).batch(BATCH_SIZE)
    #net
    g = make_generator()
    d_1 = make_discriminator()
    d_2 = make_discriminator()
    # optimizers
    g_optimizer = keras.optimizers.Adam(1e-4,beta_1=0.5)
    d_1_optimizer = keras.optimizers.Adam(1e-4,beta_1=0.5)
    d_2_optimizer = keras.optimizers.Adam(1e-4,beta_1=0.5)
    # checkpoint
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
    checkpoint = tf.train.Checkpoint(g_optimizer = g_optimizer,
                                    d_1_optimizer = d_1_optimizer,
                                    d_2_optimizer = d_2_optimizer,
                                    g = g,
                                    d_1 = d_1,
                                    d_2 = d_2)
    
    #load model from ckpt
    try:
        checkpoint_prefix_indx = os.path.join(checkpoint_dir,model_indx)
        checkpoint.restore(checkpoint_prefix_indx)
        print('Model restore success')     
    except:
        print("No model ckpt, initialize model")

    #
    normal_iter_num = int(len(normal_dataset)/64)   
    train_nor_data = []
    for indx,nor_image_batch in enumerate(train_nor_dataset):
        train_nor_data.append(nor_image_batch)

    #start training
    for epoch in range(epochs):
        print('Epoch:',epoch)
        start = time.time()        
        # train d_1
        print('Start Traing D_1:')
        for indx,abnor_image_batch in enumerate(train_abnor_dataset):
            normal_num = int(indx%normal_iter_num)
            nor_image_batch = train_nor_data[normal_num]
            
            d_1_loss = train_d1_step(nor_image_batch,abnor_image_batch,d_1_optimizer,d_1)
            if indx%100==0:
                print("iter_num:{},D_1 loss is {}".format(indx,d_1_loss.numpy())) 
            
        # train d_2
        print('Start Traing D_2:')
        for i in range(rate):
            for indx,nor_image_batch in enumerate(train_nor_dataset):
                
                g_loss,d_2_loss = train_d2_step(nor_image_batch,g,g_optimizer,d_2,d_2_optimizer)

                if indx%100==0:
                    print("iter_num:{}, D_2 loss is {}, G loss is {}".format(indx,d_2_loss.numpy(),g_loss.numpy()))

        # display.clear_output(wait = True)
        seed = tf.random.normal([num_examples_to_generate,z_dim])
        generate_and_save_images(g,epoch + 1,seed)
        
        #eval
        if (epoch+1)%100==0:
            eval_model(d_1,d_2,test_abnormal_data)

        #save_ckpt and print time
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1,time.time() - start),'CKPT file is saved')
        
    # diplay_clear_output(wait = True)
    generate_and_save_images(g,epochs,seed)


            
if __name__ == '__main__':

    normal_path = './car_hacking/normal_run_data'
    abnormal_path = ['./car_hacking/Dos_dataset','./car_hacking/Fuzzy_dataset','./car_hacking/gear_dataset','./car_hacking/RPM_dataset']    
    normal_list = os.listdir(normal_path)
    normal_data = []
    abnormal_data = {}
    for indx,data in enumerate(normal_list):
        image = cv2.imread(os.path.join(normal_path,data),cv2.IMREAD_GRAYSCALE)
        image = image.T.astype(np.float32)
        image = np.reshape(image,(16,3,64))/127.5-1
        normal_data.append(image)
    
    for abnormal in abnormal_path:
        types = os.path.basename(abnormal)
        data_set = []
        attack_list = os.listdir(abnormal)
        for indx,data in enumerate(attack_list):
            image = cv2.imread(os.path.join(abnormal,data),cv2.IMREAD_GRAYSCALE)
            image = image.T.astype(np.float32)
            image = np.reshape(image,(16,3,64))/127.5-1
            data_set.append(image)
        abnormal_data[types] = data_set
        
    train(normal_data,abnormal_data,EPOCHS)


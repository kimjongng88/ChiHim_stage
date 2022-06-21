"""
Main script for de EMG classification model. This models is trained to classify needle artefacts, 
voluntary contraction and rest during needle EMG.

This file trains with looking at the amount of reviewers. This is part of project of Chi Him
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import glob
import os
from pathlib import Path
import random
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
#from plot_keras_history import plot_history
from preprocessing.soft_label_class import segment2
from sklearn.preprocessing import LabelBinarizer
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from dollo.data_augmentation import Skew, Distort
from dollo.readnames import readnames
from test_algo.learning_rate_schedulers import PolynomialDecay
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from test_algo.minigooglenet import MiniGoogLeNet
import librosa
import warnings
import glob
import datetime

import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

# save numpy array as npy file
from numpy import asarray, full
from numpy import save
from numpy import load
from numpy import array, hstack, vstack
from numpy import ones
import dask.array as da


NUM_EPOCHS = 20
#INIT_LR = 5e-3
#BATCH_SIZE = 8

# initialize the label names
labelNames = ["rest","contraction","needle"]


class practice:
    def __init__(self, test_files):
        self.test_files = test_files


def parse_npy(file_path_n):#, file_path_c, file_path_r):
    '''
    print('parsing')
    numpyArray = make_array(file_path_n)
    numpyArray_c = make_array(file_path_c)
    numpyArray_r = make_array(file_path_r)

    test_n,train_n,val_n = split_rand(numpyArray)
    test_c,train_c,val_c = split_rand(numpyArray_c)
    test_r,train_r,val_r = split_rand(numpyArray_r)

    testX = [*test_n, *test_c,*test_r]
    trainX = [*train_n, *train_c,*train_r]
    valX = [*val_n, *val_c,*val_r]
    return testX, trainX, valX
    '''
    print('parsing')
    numpyArray = make_array(file_path_n)
    print('1e gemaakt')

    #train_n,val_n = split_rand(numpyArray)
    #print('1e split')

    #testX = [*test_n, *test_c,*test_r]
    #trainX = [*train_n]
    #valX = [*val_n]
    #trainX = [numpyArray,numpyArray_c,numpyArray_r]
    return numpyArray#trainX#, valX

def make_array(file_path):
    print('array aan het vormen')
    x = 0
    numpy_vars = { }
    for np_name in glob.glob(file_path):
        path_name = Path(np_name).stem
        if path_name.find("CV")!=-1:
            path_name = path_name.replace("CV","") 
        elif path_name.find("DH")!=-1:
            path_name = path_name.replace("DH","") 
        elif path_name.find("LW")!=-1:
            path_name = path_name.replace("LW","") 
        else:
            path_name = path_name.replace("WVP","") 
        # Saves filename and array to dictionary
        numpy_vars[x] = {'name':path_name,'array':np.load(np_name, mmap_mode='r',allow_pickle=True)}
        x += 1
    # pairs in the dictionary
    result = numpy_vars.items()
    data = list(result)
    numpyArray = np.array(data,dtype=object)
    return numpyArray

def split_rand(numpyArray):
    random.shuffle(numpyArray)
    #twenty = len(numpyArray)//5
    ten = len(numpyArray)//10
    thir = 3*ten
    twen = 2*ten
    #portion_array = numpyArray[:twenty]
    #b = portion_array[ten:]
    #c = portion_array[:ten]
    portion_array = numpyArray[:thir]
    #b = portion_array[twen:]
    c = portion_array[:thir]
    return portion_array, c

def get_list(file_path):
    print('lijst aan het maken')
    no_dubs = {}
    fh = open(file_path).readlines()
    sum = 0
    for line in fh:               
        row = line.split(',')
        seg_number, needle, contraction, rest = [i.strip() for i in row]
        seg = segment2(sum, seg_number, needle, contraction, rest)
        no_dubs[sum] = seg
        sum += 1 
    return no_dubs

def dense_net_training_generator(train_files,full_label_list,augmentation_flag=True):
        """
        Training generator. Reads a .npy file and converts this into a mel-spectrogram. The spectrogram is normalised
        and, depending on augmentation_flag data augmentation is applied. The label is parsed from the filename.
        :param augmentation_flag: Boolean flag to indicate whether or not data augmentation should be applied.
        Default is set to True.
        """
        X = [ ]
        Y = [ ]     
        A = [ ]
        print(len(train_files),len(full_label_list))
        for f, element in train_files:
            a = 0
            if element['name'] not in A:
                # Load data and convert to float (required for melspectrogram)
                mel_spect = element['array']
                #data = np.asarray([np.float(i) for i in data])
                # Create mel spectrogram, cast to np.array and transform to dB.
                mel_spect = librosa.feature.melspectrogram(y=mel_spect, sr=128, n_mels=173, fmax=10000)
                mel_spect = np.asarray(mel_spect).reshape(128,173)
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

                # Normalise input
                max_mel_spect = np.max(mel_spect)
                min_mel_spect = np.min(mel_spect)
                mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)

                # sanity check
                if np.max(mel_spect) > 1:
                    print("ERROROROROR")
                '''
                # Apply image augmentation to the generated mel spectograms.
                if augmentation_flag:
                    #print('Augmentation')
                    # Load as PIL Image for augmentation
                    im = Image.fromarray(np.uint8(mel_spect*255))
                    augmentation_option = np.random.randint(3)
                    if augmentation_option == 0:
                        # Do nothing, pass image as is
                        pass
                    elif augmentation_option == 1:
                        # Apply skew
                        #im = Skew.perform_operation(Image.fromarray(np.uint8(mel_spect*255)))
                        pass
                    elif augmentation_option == 2:
                        # Apply distortion
                        #im = Distort.perform_operation(Image.fromarray(np.uint8(mel_spect*255)))
                        pass
                    # Convert back to numpy and normalise.
                    mel_spect = np.array(im) / 255.0
                    '''

                # Stack for model input
                mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
                mel_spect = mel_spect.reshape([128, 173, 3])#verandert
                x = 0
                while x<len(full_label_list):
                    if element['name'] in full_label_list[x].seg_number:
                        label = np.array([float(full_label_list[x].needle),float(full_label_list[x].contraction),float(full_label_list[x].rest)])
                        A.append(element['name'])
                        X.append(mel_spect)
                        Y.append(label)
                    else:
                        pass
                    x+=1
            else:
                pass
            a += 1
        X = np.array(X,dtype=np.float32)
        Y = np.array(Y,dtype=np.float32)
        return X,Y

def build_model(layers,base_layers_trainable_flag):
    print('aan het bouwen')
    m = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights='inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
    input_shape=(128,173,3),
    pooling='avg')


    for layer in m.layers:
        layer.trainable = base_layers_trainable_flag

    x = m.output
    if layers ==4:
        #Build model
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
    elif layers ==3:
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)

    elif layers ==2:
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)

    elif layers ==1:
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
            
    outputs = Dense(3, activation='softmax')(x)
    model_transfer = tf.keras.Model(inputs=m.input, outputs=outputs)
    model = model_transfer
    print(type(model))
    print('returned model')
    return model

def train_model(model,d,c,batch_size,INIT_LR):
    a = 'b'
    #X= trainX
    #y = trainY

    print(len(d),len(c))


    skf = KFold(n_splits=5, shuffle=False)
    skf.get_n_splits(d,c)
    fold_no = 0
    print(skf)

    for train_index, test_index in skf.split(d,c):
        #N-fold cross validation
        fold_no += 1
        # print("TRAIN:", train_index, "TEST:", test_index)
        trainX, testX = d[train_index], d[test_index]
        trainY, testY = c[train_index], c[test_index]

        print(len(trainX),len(trainY),len(testX),len(testY))
        print(type(trainX),'type')
        
        print('begin training')
        aug = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")
        # construct the learning rate scheduler callback
        schedule = PolynomialDecay(maxEpochs=NUM_EPOCHS,
                                initAlpha=INIT_LR,
                                power=1.0)
        callbacks = [LearningRateScheduler(schedule)]

        # initialize the optimizer and model
        print("[INFO] compiling model...")
        opt = SGD(lr=INIT_LR, momentum=0.7)
        print('model gecompiled')
        #model = MiniGoogLeNet.build(width=173, height=128, depth=3, classes=3)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer='Adam',
                metrics=["accuracy"])

        print('model gecompiled, model aan het fitten',trainX.shape,trainY.shape)
        #fit model
        H = model.fit(
            x = trainX,
            y = trainY,
            batch_size = 256,
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // batch_size,
            validation_steps=len(trainY) // batch_size,
            epochs=20,
            callbacks=callbacks,
            verbose=1)

        #save model
        model.save('model'+ str(fold_no)+ '.hdf5')
        print('model gesaved')
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=batch_size)
        print(classification_report(testY.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                   target_names=labelNames))

        a = str(INIT_LR).replace(".","")

        #plot_history(H, path = "loss_value"+"_"+str(fold_no)+"_"+str(batch_size))

        plot_history_metrics(H.history["loss"],
                            H.history["val_loss"],
                            "train_loss",
                            "val_loss",
                            "Training Loss vs Validation Loss",
                            "Loss",
                            "loss_value"+"_"+str(fold_no)+"_"+str(batch_size))

        plot_history_metrics(H.history["accuracy"],
                            H.history["val_accuracy"],
                            "train_accuracy",
                            "val_accuracy",
                            "Training Accuracy vs Validation Accuracy",
                            "Accuracy",
                            "Accuracy_value"+"_"+str(fold_no)+"_"+str(batch_size))

        # initialize the Optimizer and Loss\
        opt = SGD(lr=INIT_LR, momentum=0.9)


        validation_loss=np.amin(H.history['val_loss'])

        return validation_loss, model

def plot_history_metrics(metric, val_metric, lbl_metric, lbl_val_metric, title, ylabel, plt_file_name):
    print('aan het plotten')
    N = np.arange(0, 20)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, metric, label=lbl_metric)
    #plt.plot(N, val_metric, label=lbl_val_metric)
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(ylabel)
    plt.legend(loc="lower left")
    plt.savefig(plt_file_name)

def results(experiment_name,number_of_epochs,log_file_path,experiment_directory,path_to_original_data,dataset_generation_override_flag,path_to_annotated_data,sample_time,
sliding_window_step_time,sample_rate,n_mels,fmax,batch_size,learning_rate,base_layers_trainable_flag):
    time = str(datetime.datetime.now())
    time_stamp = time[8:10] + time[5:7] + time[0:4] + "_" + time[11:13] + time[14:16]
    save_name = str(time_stamp) + "_" + str(experiment_name) +\
                              "_" + str(number_of_epochs) + "epochs"
        
    with open(log_file_path + str(save_name) + ".txt", "w") as a:
        #Write data etc.
        a.write("Date: {} \n".format(time_stamp))
        a.write("Logfile for experiment: {} \n".format(experiment_name))
        a.write(" \n")
        a.write("Dataset creation options:\n")
        a.write("-------------------------\n")
        a.write('Experiment directory \t :{}\n'.format(experiment_directory))
        a.write('Path to original data \t: {}\n'.format(path_to_original_data))
        a.write('Path to annotated data \t: {}\n'.format(path_to_annotated_data))
        a.write('Override flag is set to \t: {}\n'.format(dataset_generation_override_flag))
        a.write("\n")
        a.write("Input data options: \n")
        a.write('Sample time \t\t: {}\n'.format(sample_time))
        a.write('Sliding window step time \t : {}\n'.format(sliding_window_step_time))
        #a.write("Sample rate \t\t: {}\n".format(sample_rate))
        a.write("Number of mels \t\t; {}\n".format(n_mels))
        a.write("Maximum frequency for mel-spect \t: {}\n".format(fmax))
        #a.write("Input dimension width \t: {}\n".format(input_dimension_width))
        a.write('\n')
        a.write("Experiment options:\n")
        a.write("------------------- \n")
        a.write("Batch size \t\t: {} \n".format(batch_size))
        a.write("Learning rate \t\t: {}\n".format(learning_rate))
        a.write("Number of epochs \t: {}\n".format(number_of_epochs))
        a.write('Base layer trainable flag is set to \t: {}\n'.format(base_layers_trainable_flag))



if __name__ == '__main__':
    # Begin, adjust parameters here
    working_directory = "H:/Downloads/Stage/"
    experiment_name = 'validations'
    number_of_epochs = NUM_EPOCHS
    log_file_path = "Experimentation/Experiment_log_files/"
    experiment_directory = working_directory +  "Experimentation/Experiments/"
    path_to_original_data = 'path verwijdert wegens opslag, dus geen path'
    dataset_generation_override_flag = True
    path_to_annotated_data = working_directory +'a_subset/'
    sample_time = 2
    sliding_window_step_time = 0.1
    sample_rate = 44100
    n_mels = 173
    fmax = 10000
    batch_size = 256
    learning_rate = 0.003
    base_layers_trainable_flag = True
    layers = 2
#Load dataset here
'''
    trainX= np.load('data02.npy',mmap_mode='r',allow_pickle=True)
    trainX_sub = np.load('data03.npy',mmap_mode='r',allow_pickle=True)
    print(len(trainX),len(trainX_sub))
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    del trainX_sub
    trainX_sub = np.load('data04.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data05.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data06.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data07.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data08.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    trainX_sub = np.load('data085.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    trainX_sub = np.load('data09.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data0_c1.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data0_r.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub))
    del trainX_sub
    trainX_sub = np.load('data0_n.npy',mmap_mode='r',allow_pickle=True)
    trainX = da.concatenate([trainX,trainX_sub],axis=0)
    print(len(trainX),len(trainX_sub),'laaastasgaf')
    del trainX_sub
    print(len(trainX))



    trainY= np.load('datay02.npy',mmap_mode='r',allow_pickle=True)
    trainY_sub = np.load('datay03.npy',mmap_mode='r',allow_pickle=True)
    print(len(trainY),len(trainY_sub))
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    del trainY_sub
    trainY_sub = np.load('datay04.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay05.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay06.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay07.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay08.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay085.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay09.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay0_c1.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay0_r.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    trainY_sub = np.load('datay0_n.npy',mmap_mode='r',allow_pickle=True)
    trainY = da.concatenate([trainY,trainY_sub],axis=0)
    print(len(trainY),len(trainY_sub))
    del trainY_sub
    print(len(trainY))
    print(trainX.shape,trainY.shape,'shaaapppeeee')
    print(len(trainX),len(trainY))
'''
    model = build_model(layers,True)
    validation_loss, model = train_model(model,trainX,trainY,batch_size,learning_rate)

    
    results(experiment_name,number_of_epochs,log_file_path,experiment_directory,path_to_original_data,dataset_generation_override_flag,path_to_annotated_data,sample_time,
sliding_window_step_time,sample_rate,n_mels,fmax,batch_size,learning_rate,base_layers_trainable_flag)

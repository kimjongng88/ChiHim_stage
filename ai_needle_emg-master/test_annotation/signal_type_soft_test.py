import numpy as np
import glob
import os
from pathlib import Path
import random
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
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
from numpy import asarray
from numpy import save
from numpy import load


NUM_EPOCHS = 1
INIT_LR = 5e-3
#BATCH_SIZE = 8

# initialize the label names
labelNames = ["needle","contraction","rest"]


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
        numpy_vars[x] = {'name':path_name,'array':np.load(np_name)}
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
    b = portion_array[twen:]
    c = portion_array[:twen]
    return b, numpyArray[thir:], c

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
        print('nu hier')
        print(type(train_files))
        for f, element in train_files:
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
                    X.append(mel_spect)
                    Y.append(label)
                else:
                    pass
                x+=1
        Y = np.array(Y)
        X = np.array(X)
        return X,Y

def build_model(layers,base_layers_trainable_flag):
    print('aan het bouwen')
    m = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(128,173,3),
    pooling='avg')

    for layer in m.layers:
        layer.trainable = base_layers_trainable_flag

    x = m.output
    if layers ==4:
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
    print('returned model')
    return model

def train_model(model,trainX,trainY,batch_size,INIT_LR,param):
    a = 'b'
    X= trainX
    y=trainY

    print(len(X),len(y))

    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(X, y)
    fold_no = 0
    print(skf)

    for train_index, test_index in skf.split(X, y):
        fold_no += 1
        # print("TRAIN:", train_index, "TEST:", test_index)
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = y[train_index], y[test_index]

        print(len(trainX),len(trainY),len(testX),len(testY))
                

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
        #model = MiniGoogLeNet.build(width=87, height=128, depth=3, classes=3)

        print('model aan het compilen')
        model.compile(loss="binary_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])

        print('model gecompiled, model aan het fitten',trainX.shape,trainY.shape)
        H = model.fit(
            aug.flow(trainX, trainY, batch_size=batch_size),
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // batch_size,
            epochs=NUM_EPOCHS,
            callbacks=callbacks,
            verbose=1)

        model.save('test'+str(fold_no)+'_'+str(batch_size)+"_"+str(round(INIT_LR,3))+'_'+str(H.history["loss"])+'_'+str(H.history["accuracy"])+'.hdf5')
        print('model gesaved')
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=batch_size)
        print(classification_report(testY.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                    target_names=labelNames))

        a = str(INIT_LR).replace(".","")

        plot_history_metrics(H.history["loss"],
                            H.history["val_loss"],
                            "train_loss",
                            "val_loss",
                            "Training Loss vs Validation Loss",
                            "Loss",
                            "loss_value"+"_"+str(fold_no)+"_"+str(batch_size)+"_"+a)

        plot_history_metrics(H.history["accuracy"],
                            H.history["val_accuracy"],
                            "train_accuracy",
                            "val_accuracy",
                            "Training Accuracy vs Validation Accuracy",
                            "Accuracy",
                            "Accuracy_value"+"_"+str(fold_no)+"_"+str(batch_size)+"_"+a)

        # initialize the Optimizer and Loss\
        opt = SGD(lr=INIT_LR, momentum=0.9)

        print("[INFO] compiling model...")
        #model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
        #model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

        validation_loss=np.amin(H.history['val_loss'])

        return validation_loss, model

def plot_history_metrics(metric, val_metric, lbl_metric, lbl_val_metric, title, ylabel, plt_file_name):
    print('aan het plotten')
    N = np.arange(0, NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, metric, label=lbl_metric)
    plt.plot(N, val_metric, label=lbl_val_metric)
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

def begin(param):
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
    batch_size = 32
    learning_rate = INIT_LR
    base_layers_trainable_flag = False


    
    trainX = parse_npy('ai_needle_emg-master/test_annotation/test_npy/0.5/*/*.npy')
    trainX2 = save('data.npy', trainX)

    trainX2 = load('data.npy', allow_pickle=True)

    warnings.filterwarnings('ignore') 
    #testX, trainX, valX = parse_npy('ai_needle_emg-master/test_annotation/test_npy/0.5/needle/*.npy','ai_needle_emg-master/test_annotation/test_npy/0.5/contraction/*.npy', 'ai_needle_emg-master/test_annotation/test_npy/0.5/rest/*.npy')
    full_label_list = get_list('ai_needle_emg-master/test_annotation/test_list/test_model_list.txt')
    #testX, trainX, valX = parse_npy('Experimentation/Experiments/*/Needle/*.npy','Experimentation/Experiments/*/Contraction/*.npy', 'Experimentation/Experiments/*/Rest/*.npy')
    #full_label_list = get_list('ai_needle_emg-master/analyse/smoothed_labels/all_smoothed.txt')
    #testX, testY = dense_net_training_generator(testX,full_label_list)
    trainX, trainY = dense_net_training_generator(trainX2,full_label_list)
    #print(testX.shape,testY.shape,'shaaapppeeee')

    model = build_model(1,True)
    H = train_model(model,trainX,trainY,batch_size,learning_rate,param)

    

    results(experiment_name,number_of_epochs,log_file_path,experiment_directory,path_to_original_data,dataset_generation_override_flag,path_to_annotated_data,sample_time,
sliding_window_step_time,sample_rate,n_mels,fmax,batch_size,learning_rate,base_layers_trainable_flag)



if __name__ == '__main__':
    a='b'
    if 'b'in a:
        print('we beginnen')
        space = {'batch_size': hp.choice('batch_size',[32,64,128]),
                'learning_rate': hp.uniform('learning_rate', 0.001, 0.005),
                'layers': hp.choice('layers', [1,2,3,4])} # ammount of trainable layers
                #'sample_time':hp.choice('sample_time', [0.5,1,2])} #TODO make this in the preprocessing (dus 3x opslaan en dan kiezen in het script)
                #'config': hp.choice('config', [config])}
                #'dropout_prob': hp.uniform('dropout_prob', 0., 0.5),}
        trials = Trials()
        best = fmin(fn=begin,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=2,
                    trials=trials)
        print(trials.results,'bbbbb')
        best_model=trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        worst_model=trials.results[np.argmax([r['loss'] for r in trials.results])]['model']
        best_params=trials.results[np.argmin([r['loss'] for r in trials.results])]['param']
        worst_params=trials.results[np.argmax([r['loss'] for r in trials.results])]['param']
        print(best)
        #print("Best model: " + str(best_model))
        #print("Worst model: " + str(worst_model))
        print("Best parameters: " + 'batch_size: ' + str(best_params['batch_size']) + 'learning rate: ' + 
              str(best_params['learning_rate'])+ 'layers: ' + str(best_params['layers']))
        print("Worst parameters: " + 'batch_size: ' + str(worst_params['batch_size']) + 'learning rate: ' + 
              str(worst_params['learning_rate'])+ 'layers: ' + str(worst_params['layers']))

        #make k-fold crossvalidation possible, the return function (line 222) prevents it and needs to be only used in hyperparameter optimalisation
        #best_params['config']['hyperoptimization_flag']=False  
        begin(best_params)
        #train_initial_emg_classification_model(best_params)


    warnings.filterwarnings('ignore') 
'''
    warnings.filterwarnings('ignore') 
    testX, trainX, valX = parse_npy('ai_needle_emg-master/test_annotation/test_npy/0.5/needle/*.npy','ai_needle_emg-master/test_annotation/test_npy/0.5/contraction/*.npy', 'ai_needle_emg-master/test_annotation/test_npy/0.5/rest/*.npy')
    full_label_list = get_list('ai_needle_emg-master/test_annotation/test_list/test_model_list.txt')
    #testX, trainX, valX = parse_npy('Experimentation/Experiments/*/Needle/*.npy','Experimentation/Experiments/*/Contraction/*.npy', 'Experimentation/Experiments/*/Rest/*.npy')
    #full_label_list = get_list('ai_needle_emg-master/analyse/smoothed_labels/all_smoothed.txt')
    testX, testY = dense_net_training_generator(testX,full_label_list)
    trainX, trainY = dense_net_training_generator(trainX,full_label_list)
    print(testX.shape,testY.shape,'shaaapppeeee')

    model = build_model(1,True)
    H = train_model(model,trainX,trainY,testX,testY)

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
    batch_size = 32
    learning_rate = INIT_LR
    base_layers_trainable_flag = False


    results(experiment_name,number_of_epochs,log_file_path,experiment_directory,path_to_original_data,dataset_generation_override_flag,path_to_annotated_data,sample_time,
sliding_window_step_time,sample_rate,n_mels,fmax,batch_size,learning_rate,base_layers_trainable_flag)
'''

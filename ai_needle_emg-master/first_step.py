import random
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import librosa
from PIL import Image
#from utils.data_augmentation import Skew, Distort
#from utils.readnames import readnames
import glob


class b:

    def __init__(self,a):
        print(a)
        self.test()
        
    def test(self):
        print('test')
        self.dense_net_training_generator()

    def dense_net_training_generator(self):
        augmentation_flag=True
        print('hier')
        label = ' '
        aaa = glob.glob('Experimentation/Experiments/0/*/*.npy')
        #aaa = glob.glob('annoted_labels/0900+ initial/*/*.npy')
        test = np.array(aaa)
        label_list=[]
        #a = 'Experimentation/Experiments/validations/0.5/npy_data1(cont)/0002_R_vas_med_1_1DH_374850.npy'
        #test = []
        #test = np.append(test,a)
        for x in test:
            soort = ' '
            base=os.path.basename(x)
            filename = os.path.splitext(base)[0]
            string_zonder = filename
            if string_zonder.find("CV")!=-1:
                string_zonder = string_zonder.replace("CV","") 
                initialen = "CV"
            elif string_zonder.find("DH")!=-1:
                string_zonder = string_zonder.replace("DH","") 
                initialen = "DH"
            elif string_zonder.find("LW")!=-1:
                string_zonder = string_zonder.replace("LW","") 
                initialen = "LW"
            else:
                string_zonder = string_zonder.replace("WVP","") 
                initialen = "WVP"
            mel_spect = np.load(x)
            #print(mel_spect)
                #data = np.asarray([np.float(i) for i in data])
            if augmentation_flag:
                    #print('Augmentation')
                    # Load as PIL Image for augmentation
                im = Image.fromarray(np.uint8(mel_spect*255))
                augmentation_option = np.random.randint(3)
                #print(augmentation_option)
                if augmentation_option == 0:
                        # Do nothing, pass image as is
                    pass
                elif augmentation_option == 1:
                        # Apply skew
                    #print('1')
                    pass
                elif augmentation_option == 2:
                        # Apply distortion
                    #print('2')
                    pass
                    # Convert back to numpy and normalise.
                mel_spect = np.array(im) / 255.0
                #print('tot 60 priem')
                # Stack for model input
            mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
            #print(mel_spect,'stack is goed')
            if 'Rest' in x:
                #print('rest')
                label = to_categorical(0, num_classes=3)
                soort = 'rest'
            elif 'Contraction' in x:
                #print('contraction')
                label = to_categorical(1, num_classes=3)
                soort = 'contraction'
            elif 'Needle' in x:
                #print('needle')
                label = to_categorical(2, num_classes=3)
                soort = 'needle'
            else:
                soort = 'empty'
            #yield mel_spect, label
            testen = string_zonder, initialen, soort
            #print(label)
            label_list.append(testen)
        #print(label_list)

        with open('ai_needle_emg-master/analyse/Correct_0-0299/0_list.txt', 'a') as file_handler:
            for item in label_list:
                file_handler.write("{}\n".format(item))

if __name__ == '__main__':
    print('ok')
    a='k'
    b(a)
    #b.dense_net_training_generator(a)


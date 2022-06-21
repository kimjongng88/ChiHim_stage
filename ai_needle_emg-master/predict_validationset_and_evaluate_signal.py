import tensorflow
#from models.signal_type_modelDH import InitialClassificationModel
import random
import numpy as np
import librosa
from options.model_training_options import ModelTrainingOptions
from dataset_generation.dataset_generation_signal_type_model import DatasetGenerationInitialModel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy 
from plot_keras_history import plot_history
from sklearn.model_selection import KFold
from sklearn import metrics
import os
import matplotlib.pyplot as plt
from dollo.readnames import readnames
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
#from hyperopt import hp, tpe, fmin

# limit Tensorflow spam
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# # For reproducibility
# random.seed(42)

# Example usage
# --experiment_name EMG_classification_test
# --sample_time 0.5
# --sliding_window_step_time 0.2
# --dataset_generation_override_flag False

# config = tensorflow.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tensorflow.compat.v1.Session(config=config)


class Validation:
    def __init__(self, config):
        
        self.config = config
        '''
        dataset_generation_input_parameters = {
            'base_working_path': config.working_directory,
            'experiment_directory': config.working_directory + str('/')+ 'validationpreprocessed'+str(config.sample_time) + str('/'),
            'path_to_annotated_data': config.path_to_annotated_data,
            'path_to_original_data': config.path_to_original_data,
            'dataset_generation_override_flag': config.dataset_generation_override_flag,
            'sample_time': config.sample_time,
            'sliding_window_step_time': config.sliding_window_step_time,
            'border_flag': True,
            'hop_length': config.hop_length,
            'n_mels': config.n_mels,
            'input_dimension_width': config.input_dimension_width,
            'fmax': config.fmax
            }
            '''
        working_dir = config.working_directory + 'validationpreprocessed'+str(config.sample_time)+ str('/')
        #dataset_generation = DatasetGenerationInitialModel(**dataset_generation_input_parameters)
        #dataset_generation.create_dataset(needle_flag=False)
        vali_files = [f.split('.wav')[0] for f in os.listdir(config.path_to_annotated_data) if ".wav" in f]
        rest_train, contraction_train, needle_train = readnames(vali_files, working_dir)
        validation_files = rest_train + contraction_train + needle_train
        self.val_files = validation_files
        self.num_val_files = len(validation_files)
        self.val_file_indices = np.random.randint(len(self.val_files), size=self.num_val_files)

        model_weights_name = 'model_smooth.hdf5'
        self.model_weights_path = config.working_directory + "/Experimentation/Model_weights/" + model_weights_name

        # Model
        self.model = load_model(self.model_weights_path)
        
        self.validation()

    def validation(self):
        labels = []
        predictions = []
        for i in tqdm(self.val_file_indices):
            data, label = self.test_generator_based_on_filename(self.val_files[i])
            prediction = self.model.predict(data)
            # if prediction[0][0]>0.99: #rest
            #     prediction = 0
            # elif prediction[0][1] > 0.99: #contraction
            #     prediction = 1
            # elif prediction[0][2] > 0.99: #needle
            #     prediction = 2
            # else:
            #     prediction = 3
            # if prediction == 3:
            #     pass # all unsure predictions are excluded
            # else:
            labelnew = np.argmax(label)
            predictionnew = np.argmax(prediction)
            labels = np.append(labels, labelnew)
            predictions = np.append(predictions,predictionnew)         
        print("evaluationmetric: ", metrics.classification_report(labels, predictions))
        confusion = confusion_matrix(labels,predictions)
        print(confusion)
        print('Done')

    def test_generator_based_on_filename(self, f):
        mel_spect = np.load(f)

        # sanity check
        if np.max(mel_spect) > 1:
            print("ERROROROROR")

        mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
        mel_spect = mel_spect.reshape([1, self.config.n_mels, self.config.input_dimension_width, 3])
        if 'Rest' in f:
            label = to_categorical(0, num_classes=3)
        elif 'Contraction' in f:
            label = to_categorical(1, num_classes=3)
        elif 'Needle' in f:
            label = to_categorical(2, num_classes=3)
        label = label.reshape([1, 3])
        return mel_spect, label

    
if __name__ == '__main__':

    config = ModelTrainingOptions().parse()
    Validation(config)

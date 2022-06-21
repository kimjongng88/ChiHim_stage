"""
Main script for de EMG classification model. This models is trained to classify needle artefacts, 
voluntary contraction and rest during needle EMG.
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard, CSVLogger)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from plot_keras_history import plot_history
from sklearn.model_selection import KFold
from sklearn import metrics
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from models.signal_type_model import InitialClassificationModel
from dataset_generation.dataset_generation_signal_type_model import DatasetGenerationInitialModel
from options.model_training_options import ModelTrainingOptions

# limit Tensorflow spam
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# For reproducibility
random.seed(42)

# Example usage
# --experiment_name EMG_classification_test
# --sample_time 0.5
# --sliding_window_step_time 0.2
# --dataset_generation_override_flag False

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)

class SanityCheckCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, test_files, fold_no, config, param, num_test_files):
        super(SanityCheckCallback, self).__init__()
        self.parameters=param
        self.test_files = test_files
        self.config = config
        self.fold_no = fold_no
        self.num_test_files = num_test_files
        self.test_file_indices = np.random.randint(len(self.test_files), size=self.num_test_files)
'''
    def on_epoch_end(self,epoch, logs=None):
        print("Sanity Check callback")
        labels=[]
        predictions=[]
        for i in self.test_file_indices:
            data, label = self.test_generator_based_on_filename(self.test_files[i])
            prediction = self.model.predict(data)
            # print("Evaluating: ", self.test_files[i])
            # print("True label: ", label)
            # print("Prediction: ", prediction)
            labelnew = np.argmax(label)
            predictionnew = np.argmax(prediction)
            labels = np.append(labels, labelnew)
            predictions = np.append(predictions,predictionnew)
        report = metrics.classification_report(labels, predictions, output_dict=True)            
        print("evaluationmetric: ", metrics.classification_report(labels, predictions))
        df = pd.DataFrame(report).transpose()
        df.to_csv((self.config.model_weights_path + self.config.experiment_name + 'fold'+ 
                  str(self.fold_no) +'batch_size' + str(self.parameters['batch_size']) +'layers' +
                  str(self.parameters['layers']) +'sample_time' +str(self.parameters['sample_time']) +".csv"), 
                  index = False)

       
    def test_generator_based_on_filename(self, f):
        mel_spect = np.load(f)

        # sanity check
        if np.max(mel_spect) > 1:
            print("ERROROROROR")
        mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
        mel_spect = mel_spect.reshape([1, self.config.n_mels, self.config.input_dimension_width, 3])#TODO verandert
        if 'Rest' in f:
            label = to_categorical(0, num_classes=4)
        elif 'Contraction' in f:
            label = to_categorical(1, num_classes=4)
        elif 'Needle' in f:
            label = to_categorical(2, num_classes=4)
        label = label.reshape([1, 3])#TODO verandert
        return mel_spect, label
'''

def train_initial_emg_classification_model(param):
    # input_dimension_width too small for model when sample_time is 0.5s, this changes hop_length to 256 as a fix
    config.input_dimension_width=np.int(np.ceil(param['sample_time'] * config.sample_rate / config.hop_length))
    if config.input_dimension_width<75:
        hop_length = 256
        config.input_dimension_width=np.int(np.ceil(param['sample_time'] * config.sample_rate / hop_length))
        print(config.input_dimension_width)
    else:
        hop_length=config.hop_length
        print(config.input_dimension_width)
    dataset_generation_input_parameters = {
        'base_working_path': config.working_directory,
        'experiment_directory': config.experiment_directory + str('/')+ str(param['sample_time']) + str('/'),
        'path_to_annotated_data': config.path_to_annotated_data,
        'path_to_original_data': config.path_to_original_data,
        'dataset_generation_override_flag': config.dataset_generation_override_flag,
        'sample_time': param['sample_time'],
        'sliding_window_step_time': config.sliding_window_step_time,
        'border_flag': True,
        'hop_length': hop_length,
        'n_mels': config.n_mels,
        'input_dimension_width':config.input_dimension_width,
        'fmax': config.fmax        
        }

    print("Attempting to create dataset in: {}".format(config.experiment_directory))
    dataset_generation = DatasetGenerationInitialModel(**dataset_generation_input_parameters)
    dataset_generation.create_dataset(needle_flag=False)
    
    initial_classification_model_input_parameters = {
        'experiment_directory': config.experiment_directory + str('/')+ str(param['sample_time']) + str('/') ,
        'base_layers_trainable_flag': config.base_layers_trainable_flag,
        'pkl_file_path': config.pkl_file_path,
        'path_to_annotated_data': config.path_to_annotated_data,
        'experiment_name': config.experiment_name,
        'hop_length': hop_length,
        'n_mels': config.n_mels,
        'fmax': config.fmax,
        'limit_number_of_files_flag': config.limit_number_of_files_flag,
        'max_number_train': config.max_number_train,
        'max_number_val': config.max_number_val,
        'sample_rate': config.sample_rate,
        'input_dimension_width': config.input_dimension_width,
        'number_of_classes': config.number_of_classes,
        'layers': param['layers']
    }
    # Define the K-fold cross validatior
    kfold = KFold(n_splits=config.num_folds, shuffle=True)
    fold_no = 1
    #Think of something for the next splitting method
    input = [f.split('.wav')[0] for f in os.listdir(config.path_to_annotated_data) if ".wav" in f]
    #[print (i) for i in input] 
    evaluation = 1
    for trainindex, testindex in kfold.split(input):
        print (trainindex, testindex)
        #place K-fold here
        #print(initial_classification_model_input_parameters)
        print("Training experiment: {} on files from: {}".format(config.experiment_name, config.experiment_directory))
        emg_class = InitialClassificationModel(**initial_classification_model_input_parameters)
        emg_class.create_train_val_test(trainindex, testindex, class_balance_flag=config.class_balance_flag)
        # emg_class.create_train_val_test(class_balance_flag=config.class_balance_flag)
#TODO Zie hier 0 files in map, emg_class.train_files = 0, config.batch_size = 64
        print("Training model.")
        #print (int(np.floor(config.batch_size)))#zelf toegevoegd
        print("Total labelled segments. Rest: {}, contraction: {}, needle: {}".format(
            len(os.listdir(config.experiment_directory+'/' + str(param['sample_time']) + "/Rest/")), len(os.listdir(config.experiment_directory + '/' + str(param['sample_time']) + "/Contraction/")),
            len(os.listdir(config.experiment_directory + '/' +str(param['sample_time']) + "/Needle/")),len(os.listdir(config.experiment_directory + '/' +str(param['sample_time']) + "/Empty/"))))#TODO verandert
        print("Training on {} files, validating on {}".format(len(emg_class.train_files), len(emg_class.val_files)))

        checkpoint_filepath = config.model_weights_path + config.experiment_name + "_weights."+ str(fold_no)+"-{epoch:02d}-{val_loss:2f}_{val_accuracy:2f}.hdf5"

        tensorboard_callback = TensorBoard(log_dir=config.working_directory + "Experimentation/Tensorboard_log/")

        model_checkpoint_callback = ModelCheckpoint(
            filepath= checkpoint_filepath,
            monitor='val_accuracy',
            save_weights_only=False,
            save_best_only=False,
            verbose=1
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1,
            patience=50,
            verbose=1
        )

        reduce_lr_on_plateau_callback = ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=6,
            verbose=1
        )

        csv_logger = CSVLogger(
            filename=config.csv_log_file_path + config.experiment_name + ".csv",
            separator=','
        )

        dataset = tensorflow.data.Dataset.from_generator(
            emg_class.dense_net_training_generator,
            (tensorflow.float32, tensorflow.int16),
            output_shapes=(tensorflow.TensorShape([config.n_mels, config.input_dimension_width, 3]), 
                           tensorflow.TensorShape([3]))
        )#TODO verandert
              
        dataset = dataset.shuffle(buffer_size=int(np.floor(len(emg_class.train_files) / config.batch_size)))
        dataset = dataset.batch(config.batch_size)
        dataset = dataset.prefetch(tensorflow.data.AUTOTUNE)

        dataset_val = tensorflow.data.Dataset.from_generator(
            # ambda: make_training_generator_callable(training_generator(train_files)),
            emg_class.dense_net_validation_generator,
            (tensorflow.float32, tensorflow.int16),
            output_shapes=(tensorflow.TensorShape([config.n_mels, config.input_dimension_width, 3]), 
                           tensorflow.TensorShape([3]))#TODO Verandert
        )
        dataset_val = dataset_val.batch(config.batch_size)
        dataset_val = dataset_val.prefetch(tensorflow.data.AUTOTUNE)

        emg_class.build_transfer_learning_based_model()
        model = emg_class.model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(param['learning_rate']), metrics=['accuracy'])
        # if augmentation_flag:
        #     train_steps = 3* hoger
        history = model.fit(dataset.repeat(), validation_data=dataset_val.repeat(count=1), epochs=config.number_of_epochs,
                            verbose=1,
                            steps_per_epoch=3*int(np.floor(len(emg_class.train_files) / param['batch_size'])),
                            validation_steps=int(np.floor(len(emg_class.val_files) / param['batch_size'])),
                            callbacks=[model_checkpoint_callback, early_stop_callback, reduce_lr_on_plateau_callback,
                                    tensorboard_callback, SanityCheckCallback(emg_class.test_files, fold_no, config, param, 100),
                                    csv_logger])
        #model.save(checkpoint_filepath)        
        evaluation = evaluation+1
        plot_history(history, 
                    path=(config.model_weights_path + config.experiment_name + 'fold'+ str(fold_no) +'batch_size' +
                          str(param['batch_size']) +'layers' +str(param['layers']) +'sample_time' +str(param['sample_time']) +".png"))
        fold_no = fold_no+1
        #Get the lowest validation loss of the training epochs
        validation_loss=np.amin(history.history['val_loss'])
        print('Best validation loss of epoch: ', validation_loss)
        
        if config.hyperoptimization_flag==True:
            return {'loss': validation_loss,
                    'status': STATUS_OK,
                    'model': model,
                    'param':param}
    """     
    IN DEVELOPMENT: LOG PER KFOLD
    def k_fold_end(self, epoch, fold_no, logs=None):
        print("Sanity Check callback")
        for i in self.test_file_indices:
            data, label = self.test_generator_based_on_filename(self.test_files[i])
            prediction = self.model.predict(data)
            print("Evaluating: ", self.test_files[i])
            print("True label: ", label)
            print("Prediction: ", prediction)
            prediction_class = np.argmax(prediction, axis=1)
    """


if __name__ == '__main__':
    config = ModelTrainingOptions().parse()    

    if config.hyperoptimization_flag==True:
        #model will be trained with hyperparameteroptimalisation
        space = {'batch_size': hp.choice('batch_size',[64,128]),
                'learning_rate': hp.uniform('learning_rate', 0.001, 0.005),
                'layers': hp.choice('layers', [1,2,3,4]), # ammount of trainable layers
                'sample_time':hp.choice('sample_time', [0.5,1,2]), #TODO make this in the preprocessing (dus 3x opslaan en dan kiezen in het script)
                'config': hp.choice('config', [config])}
                #'dropout_prob': hp.uniform('dropout_prob', 0., 0.5),}
        trials = Trials()
        best = fmin(fn=train_initial_emg_classification_model,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=2,
                    trials=trials)
        best_model=trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        worst_model=trials.results[np.argmax([r['loss'] for r in trials.results])]['model']
        best_params=trials.results[np.argmin([r['loss'] for r in trials.results])]['param']
        worst_params=trials.results[np.argmax([r['loss'] for r in trials.results])]['param']
        print(best)
        #print("Best model: " + str(best_model))
        #print("Worst model: " + str(worst_model))
        print("Best parameters: " + 'batch_size: ' + str(best_params['batch_size']) + 'learning rate: ' + 
              str(best_params['learning_rate'])+ 'layers: ' + str(best_params['layers'])+ 'sample_time: ' + 
              str(best_params['sample_time']))
        print("Worst parameters: " + 'batch_size: ' + str(worst_params['batch_size']) + 'learning rate: ' + 
              str(worst_params['learning_rate'])+ 'layers: ' + str(worst_params['layers'])+ 'sample_time: ' + 
              str(worst_params['sample_time']))

        #make k-fold crossvalidation possible, the return function (line 222) prevents it and needs to be only used in hyperparameter optimalisation
        best_params['config']['hyperoptimization_flag']=False  
        train_initial_emg_classification_model(best_params)

    else:
        #model will be trained with previously set parameters
        param = {'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'layers': 4,
        'sample_time':config.sample_time,
        'config': [config]}
        train_initial_emg_classification_model(param)

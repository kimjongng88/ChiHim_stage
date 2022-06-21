"""
This script defines all default values for the models. 
Included values:
Filepaths, data input parameters, flags and model parameters
"""
import argparse
import datetime
import numpy as np


class ModelTrainingOptions:
    """
    A class used to parse options for training a model.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialised = False

    def initialise(self):
        # Filepaths #############TODO Aanpassen
        self.parser.add_argument('--working_directory', type=str,
                                 default="H:/Downloads/Stage/",
                                 help='The working directory for all the files.')
        self.parser.add_argument('--log_file_path', type=str, default="Experimentation/Experiment_log_files/",
                                 help='The directory in which the log files are stored.')
        self.parser.add_argument('--model_weights_directory', type=str,
                                 default="Experimentation/Model_weights/",
                                 help='Filepath extension to where the model weights are stored.')
        self.parser.add_argument('--pkl_file_path', type=str, default="Experimentation/pkl_files/",
                                 help='Filepath extension to where pickle dumps are stored.')
        self.parser.add_argument('--path_to_annotated_data', type=str, default='Validation_signal_type/',#'a_subset/',,# #TODO veranderen voor soft-label?
                                 help='Filepath extension to where the annotated data is stored.')
        self.parser.add_argument('--path_to_predicted_data', type=str, default='prediction_smal/',
                                 help='Filepath extension to where the annotated data is stored.')
        self.parser.add_argument('--path_to_original_data', type=str, default='Exported_wav_emg/',
                                 help='Filepath extension to where the original data is stored.')
        self.parser.add_argument('--csv_log_file_path', type=str, default="Experimentation/CSV_log_files/",
                                 help='Filepath extension to where the CSV log files are to be stored.')

        # Data input parameters
        self.parser.add_argument('--max_number_train', type=int)
        self.parser.add_argument('--max_number_val', type=int)
        self.parser.add_argument('--num_folds', type=int, default=5, 
                                help='number of k-folds for cross-validation')
        self.parser.add_argument('--sample_time', type=float, default=2,
                                 help='Sample time for the input data fragments.')
        self.parser.add_argument('--sliding_window_step_time', type=float, default=0.1, # for quick testing, set back to 0.1 or 0.2 in experiments
                                 help='Time between successive extractions from an annnotated file')
        self.parser.add_argument('--sample_rate', type=int, default=44100,
                                 help='Signal sample rate.')
        self.parser.add_argument('--hop_length', type=int, default=512,
                                 help='Number of hops per time step in the mel spectrogram.')
        self.parser.add_argument('--n_mels', type=int, default=128,
                                 help='Number of mels used to create mel spectrogram.')
        self.parser.add_argument('--fmax', type=int, default=10000,
                                 help='Maximum frequency represented in the mel spectrogram.')
        self.parser.add_argument('--number_of_classes', type=int, default=3,
                                 help='added for signal_type_model')


        # Flags
        self.parser.add_argument('--dataset_generation_override_flag',
                                 type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True,
                                 help='Flag to indicate whether an existing dataset with the same should be '
                                      'overwritten or not.')
        self.parser.add_argument('--base_layers_trainable_flag',
                                 type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=False,
                                 help='Flag to indicate whether the base layers in a transfer learning based model'
                                      ' should be trainable or not.')
        self.parser.add_argument('--limit_number_of_files_flag',
                                 type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=False,
                                 help='Flag to indicate whether the number of training and validation files'
                                      'should be limited or not.')
        self.parser.add_argument('--class_balance_flag',
                                 type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True,
                                 help='Flag to indicate whether the number of training files should be resampled so '
                                      'that all classes have an equal representation in the training set.')
        self.parser.add_argument('--hyperoptimization_flag',
                                 type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True,
                                 help='Flag to indicate whether hyperparameter optimization is applied or not.')
        # Model parameters
        self.parser.add_argument('--experiment_name', type=str, default='validations',
                                 help='The name for the current experiment.')
        self.parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training the model.')
        self.parser.add_argument('--learning_rate', type=float, default=1e-3,
                                 help='The learning rate for training the model.')
        self.parser.add_argument('--number_of_epochs', type=int, default=2,
                                 help='The number of epochs a model is trained for.')
        self.parser.add_argument('--extraction_target_length', type=float, default=2,
                                 help='Length in seconds of period required.')        
        # # Final Evaluation
        # self.parser.add_argument('--final_evaluation_flag', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        #                          help='A flag that is used to generate a simple dataset for the final evaluation.')
        # self.parser.add_argument('--copy_directory', default=None, type=str, help='Directory to copy train and validation files from.')
        # self.parser.add_argument('--crossvalidation_flag', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        #                          help='A flag that is used to generate a crossvalidation dataset for the final evaluation.')
        # self.parser.add_argument('--crossvalidation_type', default=None, type=int)

    def parse(self):
        if not self.initialised:
            self.initialise()
        self.opt = self.parser.parse_args()

        # Update filepaths to reflect the correct base working directory
        self.opt.log_file_path = self.opt.working_directory + self.opt.log_file_path
        self.opt.model_weights_path = self.opt.working_directory + self.opt.model_weights_directory
        self.opt.experiment_directory = self.opt.working_directory + "Experimentation/Experiments/" + self.opt.experiment_name 
        self.opt.pkl_file_path = self.opt.working_directory + self.opt.pkl_file_path
        self.opt.path_to_annotated_data = self.opt.working_directory + self.opt.path_to_annotated_data
        self.opt.path_to_predicted_data = self.opt.working_directory + self.opt.path_to_predicted_data
        self.opt.path_to_original_data = self.opt.working_directory + self.opt.path_to_original_data
        self.opt.csv_log_file_path = self.opt.working_directory + self.opt.csv_log_file_path

        # Update the input size for the model based on the given data input parameters
        self.opt.input_dimension_width = np.int(np.ceil(self.opt.sample_time * self.opt.sample_rate / self.opt.hop_length))

        # Create a log file
        time = str(datetime.datetime.now())
        time_stamp = time[8:10] + time[5:7] + time[0:4] + "_" + time[11:13] + time[14:16]
        self.opt.save_name = str(time_stamp) + "_" + str(self.opt.experiment_name) +\
                              "_" + str(self.opt.number_of_epochs) + "epochs"

        # Write a log file.
        with open(self.opt.log_file_path + str(self.opt.save_name) + ".txt", "w") as a:
            a.write("Date: {} \n".format(time_stamp))
            a.write("Logfile for experiment: {} \n".format(self.opt.experiment_name))
            a.write(" \n")
            a.write("Dataset creation options:\n")
            a.write("-------------------------\n")
            a.write('Experiment directory \t :{}\n'.format(self.opt.experiment_directory))
            a.write('Path to original data \t: {}\n'.format(self.opt.path_to_original_data))
            a.write('Path to annotated data \t: {}\n'.format(self.opt.path_to_annotated_data))
            a.write('Override flag is set to \t: {}\n'.format(self.opt.dataset_generation_override_flag))
            a.write("\n")
            a.write("Input data options: \n")
            a.write('Sample time \t\t: {}\n'.format(self.opt.sample_time))
            a.write('Sliding window step time \t : {}\n'.format(self.opt.sliding_window_step_time))
            a.write("Sample rate \t\t: {}\n".format(self.opt.sample_rate))
            a.write("Number of mels \t\t; {}\n".format(self.opt.n_mels))
            a.write("Maximum frequency for mel-spect \t: {}\n".format(self.opt.fmax))
            #a.write("Input dimension width \t: {}\n".format(self.opt.input_dimension_width))
            a.write('\n')
            a.write("Experiment options:\n")
            a.write("------------------- \n")
            a.write("Batch size \t\t: {} \n".format(self.opt.batch_size))
            a.write("Learning rate \t\t: {}\n".format(self.opt.learning_rate))
            a.write("Number of epochs \t: {}\n".format(self.opt.number_of_epochs))
            a.write('Base layer trainable flag is set to \t: {}\n'.format(self.opt.base_layers_trainable_flag))

        return self.opt

import tensorflow
import pickle
import os
import pandas as pd
import pyloudnorm as pyln
import librosa
from tqdm import tqdm
import shutil
from tensorflow.keras.models import load_model
import collections
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# GPU options
config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)

# Data read imports
import numpy as np
from scipy.io import wavfile


class EmgAnnotationPrediction:
    """
    Class that handles the prediction of .wav EMG files based on a model that is trained to classify these files into
    rest, contraction and needle movement.
    """
    def __init__(self, model_weights_name, annotation_path, destination_path, batch_size=32, loudness_target=-26.0,
                 n_mels=128, fmax=10000, sample_rate=44100, input_dimension_width=173,
                 sliding_window_size=0.1, sample_time=2, base_path='H:/Downloads/Stage/',#r'L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/',
                 print_flag=False, destination_folder_override_flag=True):
        """
        @param model_weights_name: str
            .hdf5 name of the model that is to be loaded for the annotation.
        @param annotation_path: str
            Folder name of where the to be annotated files are stored.
        @param destination_path: str
            Folder name of where the predicted labels are to be stored.
        @param batch_size: int
            Number of segments to batch together for prediction. Default (experimentally optimised) is 32.
        @param loudness_target: float
            Target dB for the normalising of sound levels. Default (inspired by paper) is -26.0 dB
        @param n_mels: int
            Number of mels used in creating the mel spectrogram. Default (inspired by paper) is 128.
        @param fmax: int
            Maximum frequency represented in the mel spectrogram. Default (inspired by paper) is 8000.
        @param sample_rate: int
            Rate at which the original file is sampled. Default (inspired by paper) is 44100.
        @param input_dimension_width: int
            Given a certain amount of hop_length (parameter used in creation of mel spectrogram) this will be a fixed
            value for a segment of a certain length. Mainly used here as a sanity check. Default is set to 173. Check
            model_training_options for a calculation.
        @param sliding_window_size: float
            Interval at which the data segments are cut up. Default is set to 0.1s here as this would represent a
            refresh rate of 10Hz for real-time predictions.
        @param sample_time: int
            Length (in seconds of time) of every sample. Default (inspired by paper) is set to 2s.
        @param base_path: str
            Path in which the entire experiment folder is located. Added as convenience should it ever move.
        @param print_flag: Boolean
            Flag to indicate whether certain (semi-debug) prints should be printed.
        @param destination_folder_override_flag : Boolean
            Flag to indicate whether an existing folder in self.destination_path should be overridden.
        """
        # Paths
        self.base_path = base_path
        self.annotation_path = self.base_path + annotation_path #Comment, voor full path bij upload file
        #self.annotation_path = annotation_path
        self.destination_path = self.base_path + destination_path
        self.model_weights_path = self.base_path + model_weights_name
        

        # Model
        self.model = load_model(self.model_weights_path)

        # Files
        self.annotated_files = [self.annotation_path + f for f in os.listdir(self.annotation_path) if '.wav' in f]
        #print(self.annotation_path , "path") -> dit klopt
        # (Pre-)processing parameters
        self.loudness_target = loudness_target
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.input_dimension_width = input_dimension_width
        self.sliding_window_size = sliding_window_size
        self.sample_time = sample_time

        # Flags
        self.print_flag = print_flag
        self.destination_folder_override_flag = destination_folder_override_flag
        self.predict_all_labels()

    def predict_all_labels(self):
        print(self.destination_path, "asdfasdf")
        """
        Function to loop over all files in self.annotated_files, cut them into batches using
        self.generate_batched_predictions and then saving in self.destination_path.
        """
        if os.path.exists(self.destination_path) and not self.destination_folder_override_flag:
            print("A folder already exists in {} and the destination_folder_override_flag is set to {}. "
                  "Please run again with different parameters if this wasn't the intended "
                  "result.".format(self.destination_path, self.destination_folder_override_flag))
        elif os.path.exists(self.destination_path) and self.destination_folder_override_flag:
            print("A folder already exists in {} and the destination_folder_override_flag is set to {}. "
                  "Removing the folder and creating a new one "
                  "instead.".format(self.destination_path, self.destination_folder_override_flag))
            #shutil.rmtree(self.destination_path, ignore_errors=True)
            #os.makedirs(self.destination_path)
            self._predict_labels()
        else:
            print("Creating new folder in {}.".format(self.destination_path))
            self._predict_labels()

    def _predict_labels(self):
        """
        Generate and save predictions for files in self.annotated_files.
        @return:
        """
        for filename in tqdm(self.annotated_files):
            data = self.preprocess_input(filename)
            print(data)
            predicted_labels, chances, certain_labels75, certain_labels85, certain_labels95 = self.generate_batched_predictions(data, filename)
            base_name = os.path.basename(filename).split(".wav")[0]
            
            np.save(os.path.join(self.annotation_path, base_name + "_time_step_" + str(self.sliding_window_size) + ".npy"),
                    chances)
            print("_predict_labels ok")
            #np.save(base_name + "chances_time_step_" + str(self.sliding_window_size) + ".npy",
            #        chances)

            #np.save(base_name + "certain75_time_step_" + str(self.sliding_window_size) + ".npy",
            #        certain_labels75)
            #np.save(base_name + "certain85_time_step_" + str(self.sliding_window_size) + ".npy",
            #        certain_labels85)
            #np.save(base_name + "certain95_time_step_" + str(self.sliding_window_size) + ".npy",
            #        certain_labels95)


    def preprocess_input(self, filename):
        """
        Function to preprocess a .wav file in a similar fashion as was done when the model was trained.

        Returns:
            loudness_normalized_audio : original audio file with loudness normalized to self.loudness_target.
        """
        # Load data
        fs, data = wavfile.read(filename)
        data = data.astype('float')

        # measure the loudness first
        meter = pyln.Meter(fs)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)

        # loudness normalize audio to -26 dB LUFS
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, self.loudness_target)

        return loudness_normalized_audio

    def generate_batched_predictions(self, data, filename):
        """
        Function that generates predictions for a .wav file load as data. The data is sampled every self.sample_time
        seconds which is then combined into self.batch_size sized batches.

        Background
        ----------
        The choice for batching is made because the model.predict call in Tensorflow 2.x has a significant overhead.
        This makes it very inefficient to loop through the samples without batching. If at a later point a 'real-time'
        application is desired it could be worth investigating converting the model to Tensorflow-Lite, increasing the
        spatial resolution from the now default 0.1s to say 0.5s or simply accepting that there is ~0.2-0.5s processing
        time for generating a prediction on a 2s segment (given that a model is loaded into memory).

        Parameters:
            data : loaded .wav file. In this case it has been normalised using self.preprocess_input.

        Returns:
            predicted_labels : np array that contains predictions for the input data sampled every self.sample_time s.
        """
        # Actual number of samples to take from the data
        number_of_samples_per_step = self.sample_time * self.sample_rate
        if self.print_flag:
            print("Data length: ", len(data) / 44100)
            print("Max: ", np.max(data))
            print("Min: ", np.min(data))

        # Ensure prediction is synchronised with the time. This does make the first time_sample seconds of predictions
        # effectively worthless, although this is unavoidable.
        data = np.append(np.zeros(number_of_samples_per_step), data)
        data_iterations = np.arange(0, len(data) - number_of_samples_per_step,
                                    int(np.floor(self.sliding_window_size * self.sample_rate)))
        if self.print_flag:
            print("Number of iteration steps: ", len(data_iterations))

        predict_list = []
        # Loop through the j batches that can be created from data_iterations.
        for j in range(0, int(np.ceil(len(data_iterations) / self.batch_size))):
            data_batch = []
            # Loop through all i samples in the batch j
            for i in range(j * self.batch_size, (j + 1) * self.batch_size):
                # Ensure that if the batch tries to fetch a sample that is out of bounds it won't produce an error.
                # All zero's are returned (and later removed) instead.
                if i >= len(data_iterations):
                    d = np.zeros((self.sample_time * self.sample_rate))
                else:
                    # Slice appropriate time segment.
                    d = data[data_iterations[i]:data_iterations[i] + self.sample_time * self.sample_rate]
                data_batch.append(d)
            # Generate and store predictions based on the batch.
            input_data = self.test_generator_batch(data_batch)
            predictions = self.model.predict(input_data)
            predict_list.append(predictions)

        # Reshape to a [n, 3] format where n is the number of data_iterations. Remove all predictions corresponding
        # to zero inputs.
        pred_list = np.asarray(predict_list)
        pred_list = pred_list.reshape([pred_list.shape[0] * pred_list.shape[1], pred_list.shape[2]])
        pred_list = pred_list[:len(data_iterations)]


        labels75 = []
        labels85 = []
        labels95 = []
        for x in range(len(pred_list)):
            if pred_list[x][0]>0.75: #rest
                prediction = 1
            elif pred_list[x][1] > 0.75: #contraction
                prediction = 2
            elif pred_list[x][2] > 0.75: #needle
                prediction = 3
            else:
                prediction = 0
            labels75.append(prediction)

        for x in range(len(pred_list)):
            if pred_list[x][0]>0.85: #rest
                prediction = 1
            elif pred_list[x][1] > 0.85: #contraction
                prediction = 2
            elif pred_list[x][2] > 0.85: #needle
                prediction = 3
            else:
                prediction = 0
            labels85.append(prediction)

        for x in range(len(pred_list)):
            if pred_list[x][0]>0.95: #rest
                prediction = 1
            elif pred_list[x][1] > 0.95: #contraction
                prediction = 2
            elif pred_list[x][2] > 0.95: #needle
                prediction = 3
            else:
                prediction = 0
            labels95.append(prediction)
        '''    
        counter = collections.Counter(labels)
        print(collections.Counter(labels))
        uncertain = counter[0.0]
        percentage = (uncertain/len(labels))*100
        print('Percentage uncertain of ' + str(filename) + 'is' + str(percentage) + '%')
        '''
        # Return predicted labels
        predicted_labels = np.argmax(pred_list, axis=1)+1
        print("generate_batched_predictions ok")
        return predicted_labels, pred_list, labels75, labels85, labels95

    def test_generator_batch(self, data_batch):
        """
        Function that batches the individual files in data_batch to form an input to self.model.

        Parameters:
            data_batch : batched preprocess input files.

        Returns:
            mel_spects : mel spectrograms based on the batched input files.
        """
        mel_spects = []
        for data in data_batch:
            # Create mel spectrogram, cast to np.array and transform to dB.
            mel_spect = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=self.n_mels, fmax=self.fmax)
            mel_spect = np.asarray(mel_spect).reshape(self.n_mels, self.input_dimension_width)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # Normalise input
            max_mel_spect = np.max(mel_spect)
            min_mel_spect = np.min(mel_spect)
            mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)

            # sanity check
            if np.max(mel_spect) > 1:
                print("ERROROROROR")

            mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
            mel_spect = mel_spect.reshape([self.n_mels, self.input_dimension_width, 3])
            mel_spects.append(mel_spect)
        mel_spects = np.asarray(mel_spects).reshape([self.batch_size, self.n_mels, self.input_dimension_width, 3])
        return mel_spects

if __name__ == '__main__':
    input_dictionary = {
        'model_weights_name': 'test.hdf5',
        'annotation_path': 'test_evaluation/',
        'destination_path': 'Predicted_labels/'
    }
    emg_annotation_prediction = EmgAnnotationPrediction('test1_[0.6744086742401123]_[0.3707664906978607]0.004289946753700525.hdf5','test_evaluation/','Predicted_labels/')
    #print(emg_annotation_prediction)
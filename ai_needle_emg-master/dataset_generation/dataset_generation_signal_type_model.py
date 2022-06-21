import numpy as np
from scipy.io import wavfile
import shutil
import os
import pyloudnorm as pyln
import pandas as pd
from tqdm import tqdm
import librosa

class DatasetGenerationInitialModel:

    def __init__(self, base_working_path, experiment_directory,
                 path_to_annotated_data, path_to_original_data, dataset_generation_override_flag, hop_length, n_mels, fmax, input_dimension_width, sample_time,
                 sliding_window_step_time, loudness_normalise_level=-26.0, border_flag=False, sample_rate=44100
                 ):
        """
        @param base_working_path: str
            Directory path to the root folder of the project.
        @param experiment_directory: str
            Directory where the experiment files are to be stored.
        @param path_to_annotated_data: str
            Directory in which the annotated files are stored.
        @param path_to_original_data: str
            Directory in which the original data is stored.
        @param dataset_generation_override_flag: Boolean
            Flag to indicate whether an existing experiment_directory should be overwritten or not.
        @param sample_time: float
            Time per extracted segment. Inherits default from model_training_options.
        @param sliding_window_step_time: float
            Sliding window size. Inherits default from model_training_options.
        @param loudness_normalise_level: float
            Level to normalise audio to in db.
        @param border_flag: Boolean
            Flag to indicate whether edge cases (border) should be included or not.
        @param sample_rate: int
            Rate original signal was sampled at. Inherits default from model_training_options.
        """
        # Filepaths
        self.base_working_path = base_working_path
        self.path_to_annotated_data = path_to_annotated_data
        self.path_to_original_data = path_to_original_data
        self.experiment_directory = experiment_directory

        # Input data parameters
        self.sample_time = sample_time
        self.sliding_window_step_time = sliding_window_step_time
        self.loudness_normalise_level = loudness_normalise_level
        self.sample_rate = sample_rate

        #librosa parameters
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = fmax 
        self.input_dimension_width = input_dimension_width
        # Flags
        self.dataset_generation_override_flag = dataset_generation_override_flag
        self.border_flag = border_flag

    def create_dataset(self, needle_flag=False):
        """
        Create the dataset directory structure and populate it with data. A dataset_generation_override_flag is used
        to prevent accidental deletion of a folder. Might want to update this with symlinks later.
        """
        if os.path.exists(self.experiment_directory) and self.dataset_generation_override_flag:
            print("A directory already exists in {}, removing it and creating a new one.".format(self.experiment_directory))
            shutil.rmtree(self.experiment_directory, ignore_errors=True)
            os.makedirs(self.experiment_directory)
            os.makedirs(self.experiment_directory + "Rest/")
            os.makedirs(self.experiment_directory + "Contraction/")
            os.makedirs(self.experiment_directory + "Needle/")
            self.compute_input_data_from_annotated_segments()
        elif os.path.exists(self.experiment_directory) and not self.dataset_generation_override_flag:
            print("A directory already exists in {} and the override flag is set to {}. If this was not the"
                  "desired result run again with different parameters".format(self.experiment_directory,
                                                                              self.dataset_generation_override_flag))
        else:
            print("Creating new directory in {}.".format(self.experiment_directory))
            os.makedirs(self.experiment_directory)
            os.makedirs(self.experiment_directory + "Rest/")
            os.makedirs(self.experiment_directory + "Contraction/")
            os.makedirs(self.experiment_directory + "Needle/")
            self.compute_input_data_from_annotated_segments(needle_flag)

    def normalise_db(self, data, fs):
        """
        Normalise the dB of a given volume to self.loudness_normalise_level
        @param data: np.array
            Array of floats representing the .wav file.
        @param fs: int
            Sample rate of the signal.
        @return: loudness_normalized_audio: np.array
            Array of floats representing the .wav file normalised in dB.
        """
        # measure the loudness first
        meter = pyln.Meter(fs)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)

        # loudness normalize audio to self.loudness_normalise_level
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, self.loudness_normalise_level)
        return loudness_normalized_audio

    def compute_input_data_from_annotated_segments(self, needle_flag=False):
        """
        Use the annotated segments to create input data with length self.sample_time which is cut out at intervals
        of self.sliding_window_step_time apart.
        :return:
        """
        print("Creating input data from annotated segments.")
        # Grabbing all .csv files (i.e. the annotated segments)
        annotated_file_list = \
            [self.path_to_annotated_data + f for f in os.listdir(self.path_to_annotated_data) if ".csv" in f]

        # Loop through all annotated segments
        for annotated_file in tqdm(annotated_file_list):
            filename_base = os.path.basename(annotated_file).split(".csv")[0]
            string_zonder = filename_base
            string_zonder = string_zonder.replace("CV","") 
            string_zonder = string_zonder.replace("DH","") 
            string_zonder = string_zonder.replace("LW","") 
            string_zonder = string_zonder.replace("WVP","") 
           # original_data_path = self.path_to_annotated_data + filename_base + ".wav"
            original_data_path = self.path_to_annotated_data + string_zonder + ".wav"
            print(original_data_path)

            fs, data = wavfile.read(original_data_path) #TODO 
            samplerate = 44100
            data = data.astype('float')
            data = self.normalise_db(data, fs)
            annotations = np.array(pd.read_csv(annotated_file)).flatten()

            # Cut out 0.5s segments based on labels (this time is variable based on self.sample_time)
            number_of_samples_per_step = np.int(np.floor(self.sample_time * samplerate))
            sliding_window_size = np.int(np.floor(self.sliding_window_step_time * samplerate))
            data_iterations = np.arange(0, len(data)-number_of_samples_per_step, sliding_window_size)
 
            # Loop through all cut out segments
            for i in range(len(data_iterations)):
                ann = annotations[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]
                d = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]
                ann_set = set(ann)

                # Ensure that a cut out segment is indeed of only one class type (i.e. rest, contraction or needle)
                # and that it is long enough (to prevent edge cases).
                if len(d) == number_of_samples_per_step:
                    # Create mel-spectrogram
                    mel_spect = librosa.feature.melspectrogram(y=d, sr=self.sample_rate, n_mels=self.n_mels, fmax=self.fmax, hop_length=self.hop_length)
                    mel_spect = np.asarray(mel_spect).reshape(self.n_mels, self.input_dimension_width)
                    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

                    # Normalise input
                    max_mel_spect = np.max(mel_spect)
                    min_mel_spect = np.min(mel_spect)
                    mel_save = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)
                    if len(ann_set) == 1:
                        if ann[0] == 1:
                            np.save(self.experiment_directory + "Rest/" + filename_base + "_" +
                                    str(data_iterations[i]) + ".npy", mel_save)
                        elif ann[0] == 2:
                            np.save(self.experiment_directory + "Contraction/" + filename_base + "_" +
                                    str(data_iterations[i]) + ".npy", mel_save)
                        elif ann[0] == 3:
                            if needle_flag:
                                dif = np.max(d) - np.min(d)
                                if dif > 1000:
                                    np.save(self.experiment_directory + "Needle/" + filename_base + "_" +
                                            str(data_iterations[i]) + ".npy", mel_save)
                            else:
                                np.save(self.experiment_directory + "Needle/" + filename_base + "_" +
                                        str(data_iterations[i]) + ".npy", mel_save)
                        elif ann[0] == 0:
                            #np.save(self.experiment_directory + "Empty/" + filename_base + "_" +
                                    #str(data_iterations[i]) + ".npy", mel_save)
                            pass                
                    # Alternative is that a data segment rests at the edge of two or three different segments
                    # In this case a priority label is applied going from needle > contraction > rest
                    # If 50% or more of a segment is of a certain category in this priority list it gets this label
                    elif len(ann_set) == 2 or len(ann_set) == 3 and self.border_flag:
                        len_rest = len(ann[ann == 1])
                        len_cont = len(ann[ann == 2])
                        len_noanno = len(ann[ann==0])
                        needle_movement = 3 * np.ones(np.int(np.floor(0.25 * self.sample_rate * self.sample_time)))
                        needle_boolean = self.annotation_array_contains_needle_movement(ann, needle_movement)
                        if needle_boolean:
                            np.save(self.experiment_directory + "Needle/" + filename_base + "_" +
                                    str(data_iterations[i]) + ".npy", mel_save)
                        elif len_noanno/number_of_samples_per_step>0.5:# more then 50% no annotation = no mel to save
                            #np.save(self.experiment_directory + "Empty/" + filename_base + "_" +
                                    #str(data_iterations[i]) + ".npy", mel_save)
                            pass
                        else:
                            if len_rest > len_cont:
                                np.save(self.experiment_directory + "Rest/" + filename_base + "_" +
                                        str(data_iterations[i]) + ".npy", mel_save)
                            else:
                                np.save(self.experiment_directory + "Contraction/" + filename_base + "_" +
                                        str(data_iterations[i]) + ".npy", mel_save)

    @staticmethod
    def annotation_array_contains_needle_movement(target_array, target_sequence):
        """
        Added [0::4410] slicing to significantly speed up evaluation.
        @param target_array:
        @param target_sequence:
        @return:
        """
        target_array = target_array[0::4410]
        target_sequence = target_sequence[0::4410]
        return_bool = False
        for i in range(0, len(target_array) - len(target_sequence) + 1):
            if np.array_equal(target_sequence, target_array[i:i+len(target_sequence)]):
                return_bool = True
        return return_bool

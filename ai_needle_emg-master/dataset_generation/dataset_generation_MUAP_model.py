import numpy as np
from scipy.io import wavfile
import pandas as pd
import shutil
import os
import pyloudnorm as pyln
from tqdm import tqdm
from scipy.io.wavfile import write

class DatasetGenerationMUAPModel:

    def __init__(self, base_working_path, experiment_directory, extraction_target_length,
                 path_to_predicted_data, path_to_original_data, dataset_generation_override_flag, sample_time=2,
                 sliding_window_step_time=0.1, loudness_normalise_level=-26.0, border_flag=False, sample_rate=44100,
                 emg_notebook_path=r"Database.xlsx", pkl_file_path="pkl_files/"):
        """
        @param base_working_path: str
            Directory path to the root folder of the project.
        @param experiment_directory: str
            Directory where the experiment files are to be stored.
        @param extraction_target_length: float
            Length in seconds of the rest period required after each needle movement to match.
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
        @param emg_notebook_path: str
            Filename for the xlsx file containing data annotations.
        @param pkl_file_path: str
            Path where pickle files are stored. Inherits default from model_training_options.
        """
        # Filepaths
        self.base_working_path = base_working_path
        self.path_to_predicted_data = path_to_predicted_data
        self.path_to_original_data = path_to_original_data
        self.experiment_directory = experiment_directory
        self.pkl_file_path = self.base_working_path + "Experimentation/" + pkl_file_path
        self.emg_notebook_path = self.base_working_path + "/" + emg_notebook_path

        # Files
        self.annotated_files = [self.path_to_predicted_data + f for f in os.listdir(self.path_to_predicted_data)]
        self.files_and_annotations_dictionary = {}
        for file in self.annotated_files:
            self.files_and_annotations_dictionary[file] = np.load(file)

        # Input data parameters
        self.sample_rate = sample_rate
        self.extraction_target_length = extraction_target_length  # Compensate for 2s time prediction
        self.sample_time = sample_time
        self.sliding_window_step_time = sliding_window_step_time
        self.loudness_normalise_level = loudness_normalise_level
        self.time_step_size_annotated_data = float(self.annotated_files[0].split("_time_step_")[1].split(".npy")[0])
        self.samples_to_meet = np.int(np.floor((self.extraction_target_length) / self.time_step_size_annotated_data))

        # Flags
        self.dataset_generation_override_flag = dataset_generation_override_flag
        self.border_flag = border_flag

        # Pre-initialisation
        self.all_files_with_a_finding = None
        self.matching_files_with_a_finding = None
        self.matching_files_without_a_finding = None
        self.annotated_files_with_a_matching_sequence = []
        self.files_and_parsed_annotation_data_dictionary = {}
        for file in self.annotated_files:
            self.files_and_parsed_annotation_data_dictionary[file] = []

        # Initialisation
        self._parse_emg_reports()
        self._parse_annotated_files()
        self._match_overlapping_files()

    def _parse_emg_reports(self):
        """
        Parse EMG reports from a .xlsx file. The parsed results are stored in a dataframe and dumpled as a .pkl file.
        @return:
        """
        if os.path.exists(self.pkl_file_path + "Extracted_df_from_emg_notebook.pkl"):
            df = pd.read_pickle(self.pkl_file_path + "Extracted_df_from_emg_notebook.pkl")
        else:
            # Load excel file
            df = pd.read_excel(self.emg_notebook_path)
            filenames = df['Filename'].tolist()

            df_findings = df['FreeTextFindings (use EMGSummaryConfig to translate)'].str.split('_x0008_', expand=True)
            df_findings['Filename'] = filenames
            df_findings.columns = ['Index', 'Ins', 'Fibr.', 'Pos.', 'Fasc.', 'Duur', 'Poly', 'Max', 'Recrutisering',
                                   'Ampl.', 'Comm', 'Filename']
            #df.drop(columns=['Name', 'Unnamed: 1', 'FreeTextFindings (use EMGSummaryConfig to translate)',
            #                 'binary findings (use EMGSummaryConfig to translate)', 'MRL raw file location'], inplace=True)
            df = df.merge(df_findings, on='Filename', how='inner')
            df.drop(columns=['Index'], inplace=True)
            df.to_pickle(self.pkl_file_path + "Extracted_df_from_emg_notebook.pkl")

        # Parse positive sharp waves and fibrillations
        search_tags_breed = ['Border', 'Breed','Br. en smal']
        search_tags_normal = ['Normaal']
        search_tags_smal = ['Smal', 'Norm en smal']
        search_tags_unknown = ['Br. en smal']
        brede_files = df[df['Duur'].isin(search_tags_breed)]['Filename'].tolist()
        smalle_files = df[df['Duur'].isin(search_tags_smal)]['Filename'].tolist()
        normale_files = df[df['Duur'].isin(search_tags_normal)]['Filename'].tolist()

        self.normaal = list(set(normale_files))
        self.breed = list(set(brede_files))
        self.smal = list(set(smalle_files))

    def _parse_annotated_files(self):
        """
        This function looks at a folder of annotated files to determine the presence of specific segments of EMG
        signals. In this case the matching segment consists of a needle movement followed by a period of rest. This
        period is governed by self.extraction_target_length. In this case an assumption is made that the frequency
        at which the files were annotated is 10 Hz. A needle movement is currently defined as a sequence of 5 events,
        of which at least 4 have to have been classified as a needle movement. Then the next 5 segments are examined, of
        which at least 4 have to have been classified as rest. If this condition is met the subsequent 35 (assuming the
        target extraction length is 4s) segments are examined and here once again at least 80% has to have been
        classified as being rest.

        ## Design choices:
        - Is a single needle movement prediction sufficient or should there be more than a number in a certain timeframe?
        - Is 80% of samples being classified as rest enough to say that a period is rest?
        - Is 4s target extraction length realistic? Initial experimentation with 6s showed poorer results as fewer
        segments could be extracted.
        @return:
        """
        for file in self.annotated_files:
            annotation = self.files_and_annotations_dictionary[file]
            n = 0

            # Slide through the annotated file and search for a match
            while n < len(annotation[:-self.samples_to_meet + 1]):

                an = annotation[n]
                # We only need data classified as contraction
                # This is defined as at least 80% of samples in 2s having a contraction prediction
                if an == 1:
                    contraction_selection = annotation[n+1:n+200]
                    number_of_contraction_predictions = contraction_selection[contraction_selection == 1]
                    contraction_predictions = len(number_of_contraction_predictions) / 200
                    if contraction_predictions >= 0.75:
                        stored = self.files_and_parsed_annotation_data_dictionary[file]
                        self.files_and_parsed_annotation_data_dictionary[file] = stored + ([np.arange(n+1, n+self.samples_to_meet+1)])
                        n = n + 200       
                n = n + 1

        for key in self.files_and_parsed_annotation_data_dictionary.items():
            if self.files_and_parsed_annotation_data_dictionary[str(key[0])] == []:
                pass
            else:
                self.annotated_files_with_a_matching_sequence.append(str(key[0]))

    def _match_overlapping_files(self):
        """
        Function to match the files that have a finding based on the excel notebook included in self.emg_notebook_path
        and files that have a segment that corresponds to the self.extraction_target_length criterion. These are
        included in the class lists self.all_files_with_a_finding and self.annotated_files_with_a_matching_sequence
        respectively.

        If the self.fib_pos_flag is set to True, a further distinction in the self.matching_files_with_a_finding is
        made. These are then split according to them being either a fibrillation (fib) or a positive sharp wave (pos).
        @return:
        """
        basenames_matching_sequence = [os.path.basename(f).split("_time")[0] for f in self.annotated_files_with_a_matching_sequence]
        self.matching_files_breed = list(set(basenames_matching_sequence) - set(self.breed))
        self.matching_files_smal = list(set(basenames_matching_sequence) - set(self.smal))
        self.matching_files_normaal = list(set(basenames_matching_sequence) - set(self.normaal))
        
    def _convert_annotation_index_to_time(self, annotation_list):
        """
        Function which grabs the first and final data point from an annotation list and converts those back to realtime
        indices such that the original data can be sliced.
        @param annotation_list: list
            List containing indices pointing to annotation data at which a certain condition was met.
        @return: begin_sample, end_sample: int
            Initial and final real-time data indices corresponding to the points given by the annotation list.
        """
        begin = annotation_list[0]
        end = annotation_list[-1]

        begin_sample = np.int(np.floor(begin * self.time_step_size_annotated_data * self.sample_rate) - self.sample_time * self.sample_rate)
        end_sample = np.int(np.floor((end + 1) * self.time_step_size_annotated_data * self.sample_rate))
        begin_sample = np.max([0, begin_sample])
        return begin_sample, end_sample

    def extract_and_save_relevant_segments(self, target_files, breed_flag, small_flag):
        """
        Method to extract and save segments from the annotated data. The method checks how many relevant segments are
        found and loops through these to extract self.sample_time length segments moving self.sliding_window_step_time
        each time.

        @param target_files: list
            Files to loop through
        @param findings_flag: Boolean
            Flag to indicate whether these are 'findings' or 'no findings'
        """
        for f in tqdm(target_files):
            file_path_annotation = self.path_to_predicted_data + str(f) + "_time_step_0.01.npy"
            found_segments = self.files_and_parsed_annotation_data_dictionary[file_path_annotation]
            original_file_path = self.path_to_original_data + str(f) + ".wav"

            if os.path.exists(original_file_path):
                normalised_file_path = self.path_to_original_data[:-1] + "_normalised/" + str(f) + ".wav"
                #if os.path.exists(normalised_file_path) and old_normalise_flag:
                #    # If a normalised version exists load this
                #    fs, data = wavfile.read(normalised_file_path)
                #    data = np.asarray([float(i) for i in data])
                #else:
                #    # Else create a new one and save it.
                #    print("Creating new file in {}".format(normalised_file_path))
                fs, data = wavfile.read(original_file_path)
                    #data = np.asarray([float(i) for i in data])
                #data = self.normalise_db(data.astype(np.int16), fs)
                    #wavfile.write(normalised_file_path, 44100, data.astype(np.int16))
                # Cut out 2s segments based on labels (this time is variable based on self.sample_time)
                number_of_samples_per_step = np.int(np.floor(self.sample_time * self.sample_rate))
                sliding_window_size = np.int(np.floor(self.sliding_window_step_time * self.sample_rate))

                if len(found_segments) == 1:
                    begin_sample, end_sample = self._convert_annotation_index_to_time(found_segments[0])
                    if end_sample - begin_sample == np.int(np.floor(self.extraction_target_length * self.sample_rate)):
                        data_segment = data[begin_sample:end_sample]
                    else:
                        difference = np.int(np.floor((self.extraction_target_length + 2)* self.sample_rate)) - (end_sample - begin_sample)
                        data_segment = np.append(np.zeros(difference), data[begin_sample:end_sample])
                    if len(data_segment) == number_of_samples_per_step:
                        if breed_flag:
                            write(self.experiment_directory + "breed/" + str(f) + "_" + str(
                                begin_sample) + ".wav", 44100, data_segment)
                        elif small_flag:
                            np.save(self.experiment_directory + "smal/" + str(f) + "_" + str(
                                begin_sample) + ".npy", data_segment)                            
                        else:
                            np.save(self.experiment_directory + "Normaal/" + str(f) + "_" + str(
                                begin_sample) + ".npy", data_segment)
                    else:
                        data_iterations = np.arange(0, len(data_segment) - number_of_samples_per_step, sliding_window_size)

                        # Loop through all cut out segments
                        for i in range(len(data_iterations)):
                            sub_segment = data_segment[data_iterations[i]:data_iterations[i] + number_of_samples_per_step]
                            if breed_flag:
                                write(self.experiment_directory + "breed/" + str(f) + "_" + str(
                                    begin_sample) + ".wav", 44100, data_segment)
                            elif small_flag:
                                np.save(self.experiment_directory + "smal/" + str(f) + "_" + str(
                                    data_iterations[i] + begin_sample) + ".npy", sub_segment)
                            else:
                                np.save(self.experiment_directory + "Normaal/" + str(f) + "_" + str(
                                    data_iterations[i] + begin_sample) + ".npy", sub_segment)
                else:
                    for found_segment in found_segments:
                        begin_sample, end_sample = self._convert_annotation_index_to_time(found_segment)
                        data_segment = data[begin_sample:end_sample]
                        if len(data_segment) == number_of_samples_per_step:
                            if breed_flag:
                                write(self.experiment_directory + "breed/" + str(f) + "_" + str(
                                    begin_sample) + ".wav", 44100, data_segment)
                            elif small_flag:
                                np.save(self.experiment_directory + "smal/" + str(f) + "_" + str(
                                    begin_sample) + ".npy", data_segment)                            
                            else:
                                np.save(self.experiment_directory + "Normaal/" + str(f) + "_" + str(
                                    begin_sample) + ".npy", data_segment)
                        else:
                            data_iterations = np.arange(0, len(data_segment) - number_of_samples_per_step,
                                                        sliding_window_size)

                            # Loop through all cut out segments
                            for i in range(len(data_iterations)):
                                sub_segment = data_segment[
                                              data_iterations[i]:data_iterations[i] + number_of_samples_per_step]
                                if breed_flag:
                                    write(self.experiment_directory + "breed/" + str(f) + "_" + str(
                                        begin_sample) + ".wav", 44100, sub_segment)
                                elif small_flag:
                                    np.save(self.experiment_directory + "smal/" + str(f) + "_" + str(
                                        data_iterations[i] + begin_sample) + ".npy", sub_segment)
                                else:
                                    np.save(self.experiment_directory + "Normaal/" + str(f) + "_" + str(
                                        data_iterations[i] + begin_sample) + ".npy", sub_segment)
            else:
                print("File does not exist: {}".format(original_file_path))

    def create_dataset(self):
        """
        Create the dataset directory structure and populate it with data. A dataset_generation_override_flag is used
        to prevent accidental deletion of a folder. Might want to update this with symlinks later.
        """
        if os.path.exists(self.experiment_directory) and self.dataset_generation_override_flag:
            print("A directory already exists in {}, removing it and creating a new one.".format(self.experiment_directory))
            shutil.rmtree(self.experiment_directory, ignore_errors=True)
            os.makedirs(self.experiment_directory)
            os.makedirs(self.experiment_directory + "Breed/")
            os.makedirs(self.experiment_directory + "Normaal/")
            os.makedirs(self.experiment_directory + "Smal/")
            self.extract_and_save_relevant_segments(self.matching_files_breed, breed_flag=True, small_flag=False)
            self.extract_and_save_relevant_segments(self.matching_files_smal, breed_flag=False, small_flag=True)
            self.extract_and_save_relevant_segments(self.matching_files_normaal, breed_flag=False, small_flag=False)

        elif os.path.exists(self.experiment_directory) and not self.dataset_generation_override_flag:
            print("A directory already exists in {} and the override flag is set to {}. If this was not the "
                  "desired result run again with different parameters".format(self.experiment_directory,
                                                                              self.dataset_generation_override_flag))
        else:
            print("Creating new directory in {}.".format(self.experiment_directory))
            os.makedirs(self.experiment_directory)
            os.makedirs(self.experiment_directory + "Breed/")
            os.makedirs(self.experiment_directory + "Normaal/")
            os.makedirs(self.experiment_directory + "Smal/")
            self.extract_and_save_relevant_segments(self.matching_files_breed, breed_flag=True, small_flag=False)
            self.extract_and_save_relevant_segments(self.matching_files_smal, breed_flag=False, small_flag=True)
            self.extract_and_save_relevant_segments(self.matching_files_normaal, breed_flag=False, small_flag=False)

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


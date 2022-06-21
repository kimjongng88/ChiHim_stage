import tensorflow
import pickle
import os
import pandas as pd
import pyloudnorm as pyln
import librosa
from tqdm import tqdm
import shutil
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import collections
import matplotlib.pyplot as plt
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
    def __init__(self, model_weights_name_anno, model_weights_name_MUAP, annotation_path, destination_path, batch_size=32, loudness_target=-26.0,
                 n_mels=128, fmax=5000, sample_rate=44100, input_dimension_width=173,
                 sliding_window_size=0.1, sample_time=2, base_path=r"L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/",
                 print_flag=False, destination_folder_override_flag=False):
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
        self.annotation_path = self.base_path + annotation_path
        self.destination_path = self.base_path + destination_path
        self.model_weights_path_anno = self.base_path + "Results_linux/" + model_weights_name_anno
        self.model_weights_path_MUAP = self.base_path + "Results_linux/" + model_weights_name_MUAP
        self.emg_notebook_path = self.base_path + 'Database.xlsx'
        # Model
        self.model_anno = load_model(self.model_weights_path_anno)
        self.model_MUAP = load_model(self.model_weights_path_MUAP)
        # Files
        self.annotated_files = [self.annotation_path + f for f in os.listdir(self.annotation_path) if '.wav' in f]

        # (Pre-)processing parameters
        self.loudness_target = loudness_target
        self.batch_size = batch_size
        self.n_mels = n_mels
        self.fmax = fmax
        self.sample_rate = sample_rate
        self.input_dimension_width = input_dimension_width
        self.sliding_window_size = sliding_window_size
        self.sample_time = sample_time
        self.breed_segments = 0
        self.smal_segments = 0
        self.normaal_segments = 0
        # Flags
        self.print_flag = print_flag
        self.destination_folder_override_flag = destination_folder_override_flag
        # load labels
        self.load_labels_from_table()
        #predict labels
        self.predict_all_labels()

    def predict_all_labels(self):
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
        y_pred = []
        y_true = []
        not_predicted=0
        fivelabel = []
        percentages=[]
        for filename in tqdm(self.annotated_files):
            data = self.preprocess_input(filename)
            base_name = os.path.basename(filename).split(".wav")[0]
            perc_breed, perc_smal, perc_normaal, true_label, tabellabel = self.generate_batched_predictions(data, base_name)
            if tabellabel == 0 :
                print('File: ' +str(base_name) + 'has no true label or not enough contraction fragments')
                not_predicted = not_predicted+1
            elif tabellabel ==7:
                pass
            else:
                prediction = [perc_breed, perc_smal, perc_normaal]
                percentage =[filename, true_label, perc_breed*100, perc_smal*100, perc_normaal*100]#true_abel for 5 categories
                percentages.append(percentage)
                predictions = np.argmax(prediction)+1
                y_pred.append(predictions)
                y_true.append(tabellabel)
                fivelabel.append(true_label)

        print('Not predicted files: ' + str(not_predicted))
        print("evaluationmetric argmax: ", metrics.classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        plotpercentages=pd.DataFrame(percentages, columns = ['database','MUAP type','Prolonged','Shortened','Normal'])
        plotpercentages=plotpercentages.sort_values(by=['MUAP type'])
        breed = plotpercentages[plotpercentages['MUAP type']==1]
        smal = plotpercentages[plotpercentages['MUAP type']==2]
        smalennormal = plotpercentages[plotpercentages['MUAP type']==3]
        normaal = plotpercentages[plotpercentages['MUAP type']==4]
        border = plotpercentages[plotpercentages['MUAP type']==5]

        # small=plotpercentages[smal]
        # breedd=plotpercentages[breed]
        # normaall=plotpercentages[normaal]
        # borderr=plotpercentages[border]
        # normsm=plotpercentages[smalennormal]
        normsm=smalennormal.sort_values(by='Shortened')
        borderr=border.sort_values(by='Normal')
        small=smal.sort_values(by='Shortened')
        breedd=breed.sort_values(by='Prolonged')
        normaall=normaal.sort_values(by='Normal')
        sorted = small.append(normsm)
        sorted = sorted.append(normaall)
        sorted = sorted.append(borderr)
        sorted = sorted.append(breedd)
        print(self.breed_segments)
        print(self.normaal_segments)
        print(self.smal_segments)
        sorted.to_csv('validationset.csv')
        sorted.plot(x='MUAP type', kind='barh', stacked = True, title = 'Distribution of shortened, prolonged and normal data parts per group', ylabel= 'Distribution in percentage', width = 1, color = ['lightcoral','whitesmoke','lightskyblue'])
        plt.show()

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

    def generate_batched_predictions(self, data, base_name):
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

        MUAP_type = []
        Contraction_count = 0
        # Loop through the j batches that can be created from data_iterations.
        for j in range(0, int(np.ceil(len(data_iterations) / self.batch_size))):
            # Loop through all i samples in the batch j
            for i in range(j * self.batch_size, (j + 1) * self.batch_size):
                # Ensure that if the batch tries to fetch a sample that is out of bounds it won't produce an error.
                # All zero's are returned (and later removed) instead.
                if i >= len(data_iterations):
                    d = np.zeros((self.sample_time * self.sample_rate))
                else:
                    # Slice appropriate time segment.
                    d = data[data_iterations[i]:data_iterations[i] + self.sample_time * self.sample_rate]

                # Create mel spectrogram, cast to np.array and transform to dB.
                mel_spect = librosa.feature.melspectrogram(y=d, sr=self.sample_rate, n_mels=self.n_mels, fmax=10000)
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
                mel_spect = mel_spect.reshape([1, self.n_mels, self.input_dimension_width, 3])

                # Generate predictions.
                predictions = self.model_anno.predict(mel_spect)

                # Reshape to a [n, 3] format where n is the number of data_iterations. Remove all predictions corresponding
                # to zero inputs.
                predictions = np.asarray(predictions)
                if predictions[0][0]>0.75: #rest
                    signal_type = 1
                elif predictions[0][1] > 0.75: #contraction
                    signal_type = 2
                elif predictions[0][2] > 0.75: #needle
                    signal_type = 3
                else:
                    signal_type = 0
                
                if signal_type == 2:
                    Contraction_count = Contraction_count +1
                    # Create mel spectrogram, cast to np.array and transform to dB.
                    mel_spect2 = librosa.feature.melspectrogram(y=d, sr=self.sample_rate, n_mels=self.n_mels, fmin=500, fmax=5000)
                    mel_spect2 = np.asarray(mel_spect2).reshape(self.n_mels, self.input_dimension_width)
                    mel_spect2 = librosa.power_to_db(mel_spect2, ref=np.max)

                    # Normalise input
                    max_mel_spect2 = np.max(mel_spect2)
                    min_mel_spect2 = np.min(mel_spect2)
                    mel_spect2 = (mel_spect2 - min_mel_spect2) / (max_mel_spect2 - min_mel_spect2)

                    # sanity check
                    if np.max(mel_spect2) > 1:
                        print("ERROROROROR")

                    mel_spect2 = np.stack((mel_spect2, mel_spect2, mel_spect2), axis=2)
                    mel_spect2 = mel_spect2.reshape([1, self.n_mels, self.input_dimension_width, 3])   
                    MUAP_prediction = self.model_MUAP.predict(mel_spect2)
                    MUAP_type.append(MUAP_prediction)


        if Contraction_count > 1:        
        # Reshape to a [n, 3] format where n is the number of data_iterations. Remove all predictions corresponding
        # to zero inputs.
            MUAP_type = np.asarray(MUAP_type)
            MUAP_type = MUAP_type.reshape([MUAP_type.shape[0] * MUAP_type.shape[1], MUAP_type.shape[2]])
            MUAP_type = MUAP_type[:len(data_iterations)]
            predicted_labels = np.argmax(MUAP_type, axis=1)+1
            #print(predicted_labels)
            print('number of contraction segments= '+ str(Contraction_count))

            # Return predicted labels
            perc_breed = len(predicted_labels[predicted_labels ==1])/len(predicted_labels)
            perc_smal = len(predicted_labels[predicted_labels ==2])/len(predicted_labels)
            perc_normaal = len(predicted_labels[predicted_labels ==3])/len(predicted_labels)

            # Return predicted labels
            print(base_name)
            print('breed: ' + str(perc_breed))
            print('smal: ' + str(perc_smal))
            print('normaal: '+ str(perc_normaal))
            if base_name in self.breed:
                true_label = 1
                tabellabel =1
                print('True label is breed')
                self.breed_segments = self.breed_segments + Contraction_count
            elif base_name in self.smal:
                true_label = 2
                tabellabel=2
                print('True label is smal')
                self.smal_segments = self.smal_segments + Contraction_count
            elif base_name in self.normaleensmal:
                true_label = 3
                tabellabel=2
                print('True label is normaal en smal')
                self.smal_segments = self.smal_segments + Contraction_count
            elif base_name in self.normaal:
                true_label = 4
                tabellabel=3
                print('True label is normaal')
                self.normaal_segments = self.normaal_segments + Contraction_count
            elif base_name in self.border:
                true_label = 5
                tabellabel=3
                print('True label is border')
                self.normaal_segments = self.normaal_segments + Contraction_count
            else:
                true_label = 0
                print('No true label')
                tabellabel=7
            return perc_breed, perc_smal, perc_normaal, true_label, tabellabel
        else: 
            return 0, 0, 0, 0, 0

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

    def load_labels_from_table(self):
        """
        Parse EMG reports from a .xlsx file. The parsed results are stored in a dataframe and dumpled as a .pkl file.
        @return:
    """
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
        #df.to_pickle(self.pkl_file_path + "Extracted_df_from_emg_notebook.pkl")

        # Parse positive breed smal normaal
        search_tags_breed = ['Breed']
        search_tags_normal = ['Normaal']
        search_tags_border = ['Border']
        search_tags_smal = ['Smal']
        search_tags_normalandsmal = ['Norm. en smal', 'norm. en smal']
        search_tags_unknown = ['Br. en smal']
        brede_files = df[df['Duur'].isin(search_tags_breed)]['Filename'].tolist()
        smalle_files = df[df['Duur'].isin(search_tags_smal)]['Filename'].tolist()
        normale_files = df[df['Duur'].isin(search_tags_normal)]['Filename'].tolist()
        border_files = df[df['Duur'].isin(search_tags_border)]['Filename'].tolist()
        normaleensmal_files = df[df['Duur'].isin(search_tags_normalandsmal)]['Filename'].tolist()

        self.normaal = list(set(normale_files))
        self.breed = list(set(brede_files))
        self.smal = list(set(smalle_files))
        self.border = list(set(border_files))
        self.normaleensmal = list(set(normaleensmal_files))
if __name__ == '__main__':
    input_dictionary = {
        'model_weights_name_signal_type': 'test4_weights.51-0.202162_0.943182.hdf5',
        'model_weights_name_MUAP': 'MUAPnotransfer_weights.1-27-0.664719_0.634095.hdf5',
        'annotation_path': 'Exported_wav_emg/',
        'destination_path': 'Predicted_labels/'
    }
    emg_annotation_prediction = EmgAnnotationPrediction('SignalTypeamsgrad2_weights.2-02-0.094379_0.975811.hdf5','MUAPfinalsaveset_weights.2-11-0.815962_0.611400.hdf5','MUAPVALIDATION/','Predictedlabels_validation_set/')
    
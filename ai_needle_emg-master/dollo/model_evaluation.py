import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import librosa


class ModelEvaluation:
    """
    Class that handles the evaluation of trained models.
    """

    def __init__(self, experiment_name, pkl_file_path, model_weights, sample_rate, n_mels,
                 fmax, input_dimension_width, weights_only_flag=False):
        """
        :param experiment_name: name of the experiment that is to be evaluated.
        :param pkl_file_path: file path to where pickle dumps are stored.
        :param model_weights_path_extension: file path to where all model weights are stored.
        """
        self.experiment_name = experiment_name
        self.pkl_file_path = pkl_file_path
        self.model_weights_path = model_weights
        self.weights_only_flag = weights_only_flag

        # Pre-initialisation
        self.train_files = None
        self.val_files = None
        self.test_files = None
        self.model = None

        # Data input parameters
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmax = fmax
        self.input_dimension_width = input_dimension_width

        # Load files
        print("Loading files...")
        self._load_files()
        print("Done!")
        print("Loading model...")
        self._load_model()
        print("Done!")

    def _load_files(self):
        """
        Read dumped pickle files.
        :return:
        """
        # Training files
        train_file_path_name = self.pkl_file_path + self.experiment_name + "_training_files.pkl"
        with open(train_file_path_name, 'rb') as f:
            self.train_files = pickle.load(f)

        # Validation files
        val_file_path_name = self.pkl_file_path + self.experiment_name + "_validation_files.pkl"
        with open(val_file_path_name, 'rb') as f:
            self.val_files = pickle.load(f)

        # Testing files
        test_file_path_name = self.pkl_file_path + self.experiment_name + "_testing_files.pkl"
        with open(test_file_path_name, 'rb') as f:
            self.test_files = pickle.load(f)

    def _load_model(self):
        """
        Load saved model.
        """
        self.model = load_model(self.model_weights_path)

    def evaluate(self, evaluation_target):
        """
        Evaluate all files in a specific file list. Results are stored in a dataframe and pickle dumped.
        :param evaluation_target: one of 'train', 'val' or 'test'. Used to indicate which set of files is to
        be evaluated.
        """
        if evaluation_target == 'train':
            target_list = self.train_files
        elif evaluation_target == 'val':
            target_list = self.val_files
        else:
            target_list = self.test_files

        filenames = []
        true_labels = []
        predicted_labels = []
        rest_predictions = []
        contraction_predictions = []
        needle_predictions = []

        for n, f in enumerate(target_list):
            filenames.append(f)
            data, label = self.test_generator_based_on_filename(f)
            prediction = self.model.predict(data)
            true_labels.append(np.argmax(label))
            rest_predictions.append(prediction[0][0])
            contraction_predictions.append(prediction[0][1])
            needle_predictions.append(prediction[0][2])
            predicted_labels.append(np.argmax(prediction))
            print("Evaluating {} out of {}.".format(n + 1, len(target_list)), end='\r')

        df = pd.DataFrame(data=[filenames, true_labels, predicted_labels, rest_predictions,
                                contraction_predictions, needle_predictions])
        df = df.T
        df.columns = ['filename', 'true_label', 'predicted_label', 'predicted_rest',
                      'predicted_contraction', 'predicted_needle']
        df_pkl_save_name = self.pkl_file_path + "Evaluation_" + str(evaluation_target) + "_" + self.experiment_name + ".pkl"
        df.to_pickle(df_pkl_save_name)
        return df

    def test_generator_based_on_filename(self, f):
        data = np.load(f)
        data = np.asarray([np.float(i) for i in data])
        mel_spect = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=self.n_mels,
                                                   fmax=self.fmax)
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
        if 'Rest' in f:
            label = to_categorical(0, num_classes=3)
        elif 'Contraction' in f:
            label = to_categorical(1, num_classes=3)
        elif 'Needle' in f:
            label = to_categorical(2, num_classes=3)
        label = label.reshape([1, 3])
        return mel_spect, label

        # from sklearn.metrics import classification_report
        #
        # true_labels_vec = [np.argmax(f) for f in true_labels]
        # predicted_labels_vec = [np.argmax(f) for f in predicted_labels]
        #
        # print(classification_report(true_labels_vec, predicted_labels_vec))

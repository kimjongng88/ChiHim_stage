# Check number of files
import random
import os
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import librosa
from PIL import Image
from utils.data_augmentation import Skew, Distort


class MUAPClassificationModel:

    def __init__(self, experiment_directory, base_layers_trainable_flag, pkl_file_path, experiment_name,
                 hop_length, n_mels, fmax, limit_number_of_files_flag, max_number_train, max_number_val,
                 sample_rate, input_dimension_width, number_of_classes):
        """
        @param experiment_directory: str
            Directory in which the files pertaining to the experiment are stored.
        @param base_layers_trainable_flag: Boolean
            Flag to indicate whether the base layers in a transfer learning model should be trainable.
            Inherits default from model_training_options.
        @param pkl_file_path: str
            File path to where .pkl files are stored. Inherits default from model_training_options.
        @param experiment_name: str
            Name of the current experiment.
        @param hop_length: int
            Hop length used to construct mel spectrograms. Inherits default from model_training_options.
        @param n_mels: int
            Number of mels used to construct mel spectrograms. Inherits default from model_training_options.
        @param fmax: int
            Maximum frequency represented in mel spectrograms. Inherits default from model_training_options.
        @param limit_number_of_files_flag: Boolean
            Flag used during initial testing to allow for fewer files to be used in training.
            Inherits default from model_training_options.
        @param max_number_train: int
            If limit_number_of_files_flag is True, this would indicate to how many training files the training should
            be limited. Inherits default from model_training_options.
        @param max_number_val: int
            If limit_number_of_files_flag is True, this would indicate to how many validation files the training should
            be limited. Inherits default from model_training_options.
        @param sample_rate: int
            Sample rate for the original files. Inherits default from model_training_options.
        @param input_dimension_width: int
            Width of constructed mel spectrograms. Inherits default from model_training_options.
        @param number_of_classes: int
            Number of classes in the dataset.
        """
        self.experiment_directory = experiment_directory
        self.pkl_file_path = pkl_file_path
        self.experiment_name = experiment_name

        # Pre-initialisation
        self.model = None
        self.train_files = None
        self.val_files = None
        self.test_files = None

        # Input data parameters
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmax = fmax
        self.input_dimension_width = input_dimension_width

        # Model training parameters
        self.max_number_train = max_number_train
        self.max_number_val = max_number_val
        self.number_of_classes = number_of_classes

        # Flags
        self.base_layers_trainable_flag = base_layers_trainable_flag
        self.limit_number_of_files_flag = limit_number_of_files_flag

        # Create dataset
        self.create_train_val_test()

        # Data augmentation
        # TODO: internalise augmentation parameters?
        self.distort = Distort(0.25, 16, 16, 1)
        self.skew = Skew(0.25, 'TILT', 0.5)

    def build_transfer_learning_based_model(self):
        """
        Build a model based on transfer learning. Uncomment (or add) whichever model instance you deem fit.
        @return: Keras model
            Keras model instance.
        """
        m = tf.keras.applications.InceptionResNetV2(include_top=False,
                                              weights='imagenet',
                                              input_shape=(self.n_mels, self.input_dimension_width, 3),
                                              pooling='avg')
        # m = tf.keras.applications.DenseNet201(include_top=False,
        #                                       weights='imagenet',
        #                                       input_shape=(self.n_mels, self.input_dimension_width, 3),
        #                                       pooling='avg')
        # m = tf.keras.applications.NASNetLarge(include_top=False,
        #                                       weights=None,#'imagenet',
        #                                       input_shape=(self.n_mels, self.input_dimension_width, 3),
        #                                       pooling='avg')
        for layer in m.layers:
            layer.trainable = self.base_layers_trainable_flag

        x = m.output
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
        outputs = Dense(self.number_of_classes, activation='softmax')(x)
        model_transfer = tf.keras.Model(inputs=m.input, outputs=outputs)
        self.model = model_transfer

    @staticmethod
    def _balance_classes(file_list, max_len_train):
        """
        Function to balance the classes used for training such that each is seen an equal number of times.
        @param file_list : list
            The list of files that is to be balanced.
        @param max_len_train : int
            The amount of times the class which is most prevalent occurs.
        @return: file_list : list
            A list of files that is resampled to max_len_train amount of files.
        """
        if len(file_list) < max_len_train:
            length_difference = max_len_train - len(file_list)
            files_to_duplicate = np.random.randint(len(file_list), size=length_difference)
            for f in files_to_duplicate:
                file_list.append(file_list[f])
        return file_list

    def create_train_val_test(self, train_size=0.7, val_size=0.2, test_only_flag=False, class_balance_flag=True):
        """
        Create the training, validation and testing sets. These are based on the files present in /Findings/
        and /No_findings/ from the experiment directory. Several flags can be used to alter the desired result.
        @param train_size: float
            Fraction of all files to be used for training. Default is 0.7.
        @param val_size: float
            Fraction of all files to be used for validation. Default is 0.2.
        @param test_only_flag: Boolean
            Flag to indicate whether (if True) no validation set should be generated. Default is False.
        @param class_balance_flag: Boolean
            Flag to indicate whether the classes should be balanced such that each class is as prevalent as the
            most commonly occurring one. Default is True
        @return:
        """
        breed_files = [self.experiment_directory + "Breed/" + f for f in os.listdir(self.experiment_directory + "Breed/")]
        smal_files = [self.experiment_directory + "Smal/" + f for f in os.listdir(self.experiment_directory + "Smal/")]
        normaal_files = [self.experiment_directory + "Normaal/" + f for f in os.listdir(self.experiment_directory + "Normaal/")]

        random.shuffle(breed_files)
        random.shuffle(smal_files)
        random.shuffle(normaal_files)

        if class_balance_flag:
            # Slice out train files
            breed_train = breed_files[:np.int(np.floor(train_size * len(breed_files)))]
            smal_train = smal_files[:np.int(np.floor(train_size * len(smal_files)))]
            normaal_train = normaal_files[:np.int(np.floor(train_size * len(normaal_files)))]

            # Balance classes
            amount_of_training_files = [len(breed_train), len(smal_train), len(normaal_train)]
            max_len_train = amount_of_training_files[np.argmax(amount_of_training_files)]

            breed_train = self._balance_classes(breed_train, max_len_train)
            smal_train = self._balance_classes(smal_train, max_len_train)
            normaal_train = self._balance_classes(normaal_train, max_len_train)

            # Slice out validation files
            breed_val = breed_files[np.int(np.floor(train_size * len(breed_files))):np.int(
                np.floor(train_size * len(breed_files) + val_size * len(breed_files)))]
            smal_val = smal_files[np.int(np.floor(train_size * len(smal_files))):np.int(
                np.floor(train_size * len(smal_files) + val_size * len(smal_files)))]
            normaal_val = normaal_files[np.int(np.floor(train_size * len(normaal_files))):np.int(
                np.floor(train_size * len(normaal_files) + val_size * len(normaal_files)))]
            # Slice out test files
            breed_test = \
                breed_files[np.int(np.floor(train_size * len(breed_files) + val_size * len(breed_files))):]
            smal_test = \
                smal_files[np.int(np.floor(train_size * len(smal_files) +
                                                  val_size * len(smal_files))):]
            normaal_test = \
                normaal_files[np.int(np.floor(train_size * len(normaal_files) +
                                                  val_size * len(normaal_files))):]
            # Assemble
            train_files = breed_train + smal_train + normaal_train
            val_files = breed_val + smal_val + normaal_val
            test_files = breed_test + smal_test + normaal_test
        else:
            # Slice out train files
            breed_train = breed_files[:np.int(np.floor(train_size * len(breed_files)))]
            smal_train = smal_files[:np.int(np.floor(train_size * len(smal_files)))]
            normaal_train = normaal_files[:np.int(np.floor(train_size * len(normaal_files)))]

            # Slice out validation files
            breed_val = breed_files[np.int(np.floor(train_size * len(breed_files))):np.int(
                np.floor(train_size * len(breed_files) + val_size * len(breed_files)))]
            smal_val = smal_files[np.int(np.floor(train_size * len(smal_files))):np.int(
                np.floor(train_size * len(smal_files) + val_size * len(smal_files)))]
            normaal_val = normaal_files[np.int(np.floor(train_size * len(normaal_files))):np.int(
                np.floor(train_size * len(normaal_files) + val_size * len(normaal_files)))]

            # Slice out test files
            breed_test = \
                breed_files[np.int(np.floor(train_size * len(breed_files) + val_size * len(breed_files))):]
            smal_test = \
                smal_files[np.int(np.floor(train_size * len(smal_files) +
                                                  val_size * len(smal_files))):]
            normaal_test = \
                normaal_files[np.int(np.floor(train_size * len(normaal_files) +
                                                  val_size * len(normaal_files))):]
            # Assemble
            train_files = breed_train + smal_train + normaal_train
            val_files = breed_val + smal_val + normaal_val
            test_files = breed_test + smal_test + normaal_test

        # Shuffle
        random.shuffle(train_files)
        random.shuffle(val_files)
        random.shuffle(test_files)

        if self.limit_number_of_files_flag:
            self.train_files = train_files[:self.max_number_train]
            self.val_files = val_files[:self.max_number_val]
            self.test_files = test_files
        else:
            self.train_files = train_files#[:max_number_train]
            self.val_files = val_files#[:max_number_val]
            self.test_files = test_files

        # Dump to pickle for referencing in either evaluation or in general.
        self._dump_list_to_pickle(self.train_files, 'training_files')
        self._dump_list_to_pickle(self.val_files, 'validation_files')
        self._dump_list_to_pickle(self.test_files, 'testing_files')

    def _dump_list_to_pickle(self, target_list, list_type):
        """
        Dump a generated list of files (for training, validation or testing) to a pickle so it can be referenced later.
        :param target_list: one of self.train_files, self.val_files or self.test_files
        :param list_type: one of 'training_files', 'validation_files' or 'testing_files'
        """
        with open(self.pkl_file_path + self.experiment_name + "_" + str(list_type) + ".pkl", 'wb') as f:
            pickle.dump(target_list, f)

    def dense_net_training_generator(self, augmentation_flag=True):
        """
        Training generator. Reads a .npy file and converts this into a mel-spectrogram. The spectrogram is normalised
        and, depending on augmentation_flag data augmentation is applied. The label is parsed from the filename.
        :param augmentation_flag: Boolean flag to indicate whether or not data augmentation should be applied.
        Default is set to True.
        """
        for f in self.train_files:
            # Load data and convert to float (required for melspectrogram)
            data = np.load(f)
            data = np.asarray([np.float(i) for i in data])

            # Create mel-spectrogram
            mel_spect = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=self.n_mels, fmax=self.fmax,
                                                       hop_length=self.hop_length)
            mel_spect = np.asarray(mel_spect).reshape(self.n_mels, self.input_dimension_width)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # Normalise input
            max_mel_spect = np.max(mel_spect)
            min_mel_spect = np.min(mel_spect)
            mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)

            # Apply image augmentation to the generated mel spectograms.
            if augmentation_flag:
                # Load as PIL Image for augmentation
                im = Image.fromarray(np.uint8(mel_spect*255))
                augmentation_option = np.random.randint(3)
                if augmentation_option == 0:
                    # Do nothing, pass image as is
                    pass
                elif augmentation_option == 1:
                    # Apply skew
                    im = self.skew.perform_operation(im)
                elif augmentation_option == 2:
                    # Apply distortion
                    im = self.distort.perform_operation(im)
                # Convert back to numpy and normalise.
                mel_spect = np.array(im) / 255.0

            # Stack for model input
            mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
            if 'Breed' in f:
                label = to_categorical(0, num_classes=3)
            elif 'Smal' in f:
                label = to_categorical(1, num_classes=3)
            elif 'Normaal' in f:
                label = to_categorical(2, num_classes=3)
            yield mel_spect, label

    def dense_net_validation_generator(self):
        """
        Validation generator. Reads a .npy file and converts this into a mel-spectrogram. The spectrogram is normalised.
        The label is parsed from the filename.
        """
        for f in self.val_files:
            # Load data and convert to float (required for melspectrogram)
            data = np.load(f)
            data = np.asarray([np.float(i) for i in data])

            # Create mel-spectrogram
            mel_spect = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=self.n_mels, fmax=self.fmax,
                                                       hop_length=self.hop_length)
            mel_spect = np.asarray(mel_spect).reshape(self.n_mels, self.input_dimension_width)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # Normalise input
            max_mel_spect = np.max(mel_spect)
            min_mel_spect = np.min(mel_spect)
            mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)

            # Stack for model input
            mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
            if 'Breed' in f:
                label = to_categorical(0, num_classes=3)
            elif 'Smal' in f:
                label = to_categorical(1, num_classes=3)
            elif 'Normaal' in f:
                label = to_categorical(2, num_classes=3)
            yield mel_spect, label
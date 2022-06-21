import tensorflow
from models.MUAP_model import MUAPClassificationModel
import random
import numpy as np
import librosa
from options.model_training_options import ModelTrainingOptions
from dataset_generation.powertemporaryfile import DatasetGenerationMUAPModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.models import load_model

# limit Tensorflow spam
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# For reproducibility
random.seed(42)

# Example usage
# --experiment_name EMG_classification_test
# --sample_time 0.5
# --sliding_window_step_time 0.2
# --dataset_generation_override_flag False
# --number_of_classes 2

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)


class SanityCheckCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, test_files, config, num_test_files=5):
        super(SanityCheckCallback, self).__init__()
        self.test_files = test_files
        self.config = config
        self.num_test_files = num_test_files
        self.test_file_indices = np.random.randint(len(self.test_files), size=self.num_test_files)

    def on_epoch_end(self, epoch, logs=None):
        print("Sanity Check callback")
        for i in self.test_file_indices:
            data, label = self.test_generator_based_on_filename(self.test_files[i])
            prediction = self.model.predict(data)
            print("Evaluating: ", self.test_files[i])
            print("True label: ", label)
            print("Prediction: ", prediction)

    def test_generator_based_on_filename(self, f):
        data = np.load(f)
        data = np.asarray([np.float(i) for i in data])
        mel_spect = librosa.feature.melspectrogram(y=data, sr=self.config.sample_rate, n_mels=self.config.n_mels, fmax=self.config.fmax)
        mel_spect = np.asarray(mel_spect).reshape(self.config.n_mels, self.config.input_dimension_width)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        # Normalise input
        max_mel_spect = np.max(mel_spect)
        min_mel_spect = np.min(mel_spect)
        mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)

        # sanity check
        if np.max(mel_spect) > 1:
            print("ERROROROROR")

        mel_spect = np.stack((mel_spect, mel_spect, mel_spect), axis=2)
        mel_spect = mel_spect.reshape([1, self.config.n_mels, self.config.input_dimension_width, 3])
        if 'Breed' in f:
            label = to_categorical(0, num_classes=3)
        elif 'Normaal' in f:
            label = to_categorical(1, num_classes=3)
        elif 'Smal':
            label = to_categorical(2, num_classes=3)
        label = label.reshape([1, 3])
        return mel_spect, label


def train_MUAP_classification_model(config):
    dataset_generation_input_parameters = {
        'base_working_path': config.working_directory,
        'experiment_directory': config.experiment_directory,
        'path_to_predicted_data': config.path_to_predicted_data,
        'path_to_original_data': config.path_to_original_data,
        'dataset_generation_override_flag': config.dataset_generation_override_flag,
        'sample_time': config.sample_time,
        'sliding_window_step_time': config.sliding_window_step_time,
        'extraction_target_length': config.extraction_target_length
    }

    print("Attempting to create dataset in: {}".format(config.experiment_directory))
    dataset_generation = DatasetGenerationMUAPModel(**dataset_generation_input_parameters)
    dataset_generation.create_dataset()
'''
    MUAP_classification_model_input_parameters = {
        'experiment_directory': config.experiment_directory,
        'base_layers_trainable_flag': config.base_layers_trainable_flag,
        'pkl_file_path': config.pkl_file_path,
        'experiment_name': config.experiment_name,
        'hop_length': config.hop_length,
        'n_mels': config.n_mels,
        'fmax': config.fmax,
        'limit_number_of_files_flag': config.limit_number_of_files_flag,
        'max_number_train': config.max_number_train,
        'max_number_val': config.max_number_val,
        'sample_rate': config.sample_rate,
        'input_dimension_width': config.input_dimension_width,
        'number_of_classes': config.number_of_classes
    }

    print("Training experiment: {} on files from: {}".format(config.experiment_name, config.experiment_directory))
    MUAP_emg_classification_model = MUAPClassificationModel(**MUAP_classification_model_input_parameters)
    MUAP_emg_classification_model.create_train_val_test(class_balance_flag=config.class_balance_flag)

    print("Training model.")
    print("Total labelled segments. Breed: {}, Smal: {}, Normaal: {}".format(
        len(os.listdir(config.experiment_directory + "Breed/")), len(os.listdir(config.experiment_directory + "Smal/")),
        len(os.listdir(config.experiment_directory + "Normaal/"))))
    print("Training on {} files, validating on {}".format(len(MUAP_emg_classification_model.train_files), len(MUAP_emg_classification_model.val_files)))

    
    checkpoint_filepath = config.model_weights_path + config.experiment_name + "_weights.{epoch:02d}-{val_loss:2f}-{val_accuracy:2f}.hdf5"

    tensorboard_callback = TensorBoard(log_dir=config.working_directory + "Experimentation/Tensorboard_log/")

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=50,
        verbose=1
    )

    reduce_lr_on_plateau_callback = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=4,
        verbose=1
    )

    csv_logger = CSVLogger(
        filename=config.csv_log_file_path + config.experiment_name + ".csv",
        separator=','
    )

    dataset = tensorflow.data.Dataset.from_generator(
        MUAP_emg_classification_model.dense_net_training_generator,
        (tensorflow.float32, tensorflow.int16),
        output_shapes=(tensorflow.TensorShape([config.n_mels, config.input_dimension_width, 3]), tensorflow.TensorShape([config.number_of_classes]))
    )
    dataset = dataset.shuffle(buffer_size=np.int(np.floor(len(MUAP_emg_classification_model.train_files) / config.batch_size)))
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tensorflow.data.AUTOTUNE)

    dataset_val = tensorflow.data.Dataset.from_generator(
        # ambda: make_training_generator_callable(training_generator(train_files)),
        MUAP_emg_classification_model.dense_net_validation_generator,
        (tensorflow.float32, tensorflow.int16),
        output_shapes=(tensorflow.TensorShape([config.n_mels, config.input_dimension_width, 3]), tensorflow.TensorShape([config.number_of_classes]))
    )
    dataset_val = dataset_val.batch(config.batch_size)
    dataset_val = dataset_val.prefetch(tensorflow.data.AUTOTUNE)

    MUAP_emg_classification_model.build_transfer_learning_based_model()
    model = MUAP_emg_classification_model.model
    #model = load_model("C:/Users/olivi/Documents/AMC/Experimentation/Model_weights/OLD_Rest_model_4s_edges_05_slide_weights.04-0.456921-0.805519.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(config.learning_rate), metrics=['accuracy'])
    # if augmentation_flag:
    #     train_steps = 3* hoger
    history = model.fit(dataset.repeat(), validation_data=dataset_val.repeat(count=1), epochs=config.number_of_epochs,
                        verbose=1,
                        steps_per_epoch=3*np.int(np.floor(len(MUAP_emg_classification_model.train_files) / config.batch_size)),
                        validation_steps=np.int(np.floor(len(MUAP_emg_classification_model.val_files) / config.batch_size)),
                        callbacks=[model_checkpoint_callback, early_stop_callback, reduce_lr_on_plateau_callback,
                                   tensorboard_callback, SanityCheckCallback(MUAP_emg_classification_model.test_files, config, 15),
                                   csv_logger])
    model.save(checkpoint_filepath)
'''

if __name__ == '__main__':

    config = ModelTrainingOptions().parse()
    train_MUAP_classification_model(config)

# AMC EMG Classification 

This is the README for the AMC EMG Classification repository. 
The scripts in this repository are configured
to be run from command line with specific options. All options are specified in the options/model_training_options.py 
file. For a significant number of these options defaults are included. Where applicable, an executable script includes example
settings for the command line options. 

## Initialisation
Install requirements from requirements.txt. Note that the project is implemented in Tensorflow 2.X. 
```bash
pip install -r ai_needle_emg-master/requirements.txt
```

## File hierarchy
The scripts expect a certain file hierarchy. Everything sits in a certain base folder and from here the following hierarchy is used:\
/ AMC_EMG_Classification _The code repository_\
/ Exported_wav_EMG  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     _The folder containing the original .wav data_\
/ Manual_annotation &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _A folder with .npy files containing manual annotations_\
/ Experimentation\
/ -- Experiments &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _Files used for the experiments are stored here_ \
/ -- Model_weights &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Trained model weights are stored here_\
/ -- CSV_log_file &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Log files produced by the CSV logging callback are stored here_\
/ -- pkl_files &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Directory for all pickle dumps_\
/ -- Experiment_log_files &nbsp;&nbsp;&nbsp;&nbsp; _Log files containing hyperparameters per experiment are stored here_\
Whenever a model_trainings_options.py instance is called a check is performed to see whether all folders are present. All the folder names can of course be changed. For compatibility the defaults should be updated in the model_training_options.py file. 

## Training a model

```bash
python3 train_findings_model.py --experiment_name EMG_classification_test --sample_time 0.5 --sliding_window_step_time 0.2 --dataset_generation_override_flag False --number_of_classes 2
```

python3 ai_needle_emg-master/train_signal_type_model_hyperparam.py --experiment_name EMG_classification_test --sample_time 0.5 --sliding_window_step_time 0.2 --dataset_generation_override_flag False --number_of_classes 2

python3 ai_needle_emg-master/train_signal_type_model_hyperparam.py --experiment_name EMG_classification_test --sample_time 0.5 --sliding_window_step_time 0.2 --dataset_generation_override_flag True --number_of_classes 2

**Things to keep in mind**
- Garbage collection is not really implemented. Disk space can fill up quickly as many experiments are performed and model weights are stored. 
  The reason that all experiment files are generated and stored on training is mainly because the sliding window step size variation means that 
  training sets vary.
- Included in the folder is a jupyter notebook which includes some examples on evaluation and visualisation. 



emg_class heeft 0 files, oplossen

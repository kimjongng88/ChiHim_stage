import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
# from utils.data_augmentation import Skew, Distort
from PIL import Image
import pyloudnorm as pyln
import wave
import os 
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
emglab = pd.read_excel("L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/ai_needle_emg/emglab.xlsx")
musclenames = emglab['Patient'].values
unique_muscles = np.unique(musclenames)

for name in tqdm(unique_muscles):
    Filenames = emglab[emglab['Patient'].isin([name])]['Filename'].tolist()
    long_data_trace = []
    for files in Filenames:
        filepath = "L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/EMGlabDatabaseWav/" + files + '.wav'
        fs, data = wavfile.read(filepath)
        
        long_data_trace.append(data)
        long_data_trace.append(np.zeros(fs))
        
    long_data_trace = np.hstack(long_data_trace)

    savename = "L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/EMGlabDatabaseMuscle/"+name +'.wav'
    wavfile.write(savename, fs, long_data_trace.astype(np.int16))
    # wavfiles = wave.open(savename, 'wb')
    # wavfiles.setnchannels(1)
    # wavfiles.setsampwidth(2)
    # wavfiles.setframerate(fs)
    # wavfiles.writeframes(long_data_trace)
    # wavfiles.close()



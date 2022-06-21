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
import tqdm
from scipy.io import wavfile

filepath =r"L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/EMGlabDatabaseBin/"
savefilepath =r"L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/EMGlabDatabaseWav/"
binfiles = [filepath + f for f in os.listdir(filepath) if '.bin' in f]
basenames = []

for file in binfiles:
    base_name = os.path.basename(file).split(".bin")[0]
    data = np.fromfile(file, dtype='int16')
    basenames.append(base_name)
    
    data = data.astype('float')
    data *= 13.1070

    savename = savefilepath+base_name +'.wav'
    samplerate = 23437.5

    wavfile = wave.open(savename, 'wb')
    wavfile.setnchannels(1)
    wavfile.setsampwidth(2)
    wavfile.setframerate(samplerate)
    wavfile.writeframes(data)
    wavfile.close()

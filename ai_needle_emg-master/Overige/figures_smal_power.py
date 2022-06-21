import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from os.path import dirname
import os
from utils.data_augmentation import Skew, Distort
from PIL import Image
import pyloudnorm as pyln
from tqdm import tqdm

filepath = askopenfilename(
            filetypes=[("Npy Files", "*.npy"), ("All Files", "*.*")])

mypath = dirname(filepath)
smal_file_list = \
[mypath +'/' + f for f in os.listdir(mypath) if ".npy" in f]
number_of_plots = 10
for smal_file in tqdm(smal_file_list):
    picturepath = str(smal_file).split(".npy")[0]
    data = np.load(smal_file)
    samplerate = 44100
    fig, ax= plt.subplots(number_of_plots,figsize=(60,30))
    for x in range(number_of_plots):        
        ax[x].plot(np.arange(0, len(data)/number_of_plots) / samplerate, data[x*int(len(data)/number_of_plots):(x+1)*int(len(data)/number_of_plots)], color='k')
        ax[x].set_ylim(-200,200)
        fig.tight_layout()
    plt.savefig(picturepath +'.png')
    plt.close()

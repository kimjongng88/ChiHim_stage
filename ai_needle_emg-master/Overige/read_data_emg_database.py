import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
from PIL import Image
import pyloudnorm as pyln

# Open the binary file for reading
filepath =r"L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/EMGlabDatabaseBin"

binfiles = [filepath + f for f in os.listdir(filepath) if '.bin' in f]
file = open("C:/Users/debor/Downloads/test/N2001A01AP52.bin", "rb")

# Read the first five numbers into a list

data = list(file.read())
# Print the list
#print(data)
data = np.asarray(data)
data = data.astype('float')
data = data[1000:21000]
# Close the file
file.close()
samplerate = 10000 #bron uit de emgdatabase
def normalise_db(data, fs):
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
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -26)
    return loudness_normalized_audio

distort = Distort(0.25, 16, 16, 1) #same as in the models
skew = Skew(0.25, 'TILT', 0.5) #same as in the models


fig, ax= plt.subplots(1,1)
ax.plot(np.arange(0, len(data)) / samplerate, data, color='k')
ax.set_ylim(-200,200)
plt.show()

data = normalise_db(data, samplerate)
input_dimensiont_width =  np.int(np.ceil(2* samplerate / 512))

spect = librosa.stft(y=data)
spect = librosa.amplitude_to_db(spect, ref=np.max)
fig,ax=plt.subplots()
img = librosa.display.specshow(spect, sr=samplerate, hop_length=512, x_axis='time', y_axis='linear', ax=ax)
fig.colorbar(img, ax=ax)
plt.xlabel("Time")
plt.ylabel("Frequency")

mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate , n_mels=128, fmax=5000, fmin=500, hop_length=512)
#mel_spect = np.asarray(mel_spect).reshape(128, input_dimensiont_width)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
fig,ax=plt.subplots()
img = librosa.display.specshow(mel_spect, sr=samplerate, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(img, ax=ax)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()


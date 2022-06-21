import ffmpeg
import librosa
import librosa.display
import pyloudnorm as pyln
from ffmpeg_normalize import FFmpegNormalize
from scipy.io import wavfile
from tkinter.filedialog import askopenfilename, asksaveasfilename
import matplotlib.pyplot as plt
import numpy as np

filepath = askopenfilename(
            filetypes=[("Wav Files", "*.wav"), ("All Files", "*.*")])
data, fs = librosa.load(filepath)
# measure the loudness first
meter = pyln.Meter(fs)  # create BS.1770 meter
loudness = meter.integrated_loudness(data)
print(loudness)
loudness_normalise_level=-26
# loudness normalize audio to self.loudness_normalise_level
loudness_normalized_audio = pyln.normalize.loudness(data, loudness, loudness_normalise_level)

loudness = meter.integrated_loudness(loudness_normalized_audio)
print(loudness)

length = len(data)/fs
time = np.linspace(0., length, data.shape[0])

plt.plot(time, loudness_normalized_audio)
plt.show()


mel_spect = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=13, fmax=8000)
#mel_spect = np.asarray(mel_spect).reshape(self.config.n_mels, self.config.input_dimension_width)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

# Normalise input
max_mel_spect = np.max(mel_spect)
min_mel_spect = np.min(mel_spect)
mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)
librosa.display.specshow(mel_spect, sr=fs)
plt.show() 
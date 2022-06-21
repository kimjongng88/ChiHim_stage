# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:42:49 2021

@author: debor
"""
# Data read imports
from pandas.core.frame import DataFrame
from typing import ClassVar
from matplotlib import axes
import numpy as np
from scipy.io import wavfile
from scipy import signal
import sys
# Visualisation imports
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from os.path import dirname, isfile
import tkinter as tk
import sys
import pandas as pd
from collections import Counter

#mypath='L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/ai_needle_emg/Data'
filepath = askopenfilename(
            filetypes=[("Wav Files", "*.wav"), ("All Files", "*.*")])
datafile = str(filepath).split("/")[-1].split(".wav")[0]
mypath = dirname(filepath)
datapath= mypath+'/'+datafile+'.wav'
annotationWVP = mypath+'/'+datafile+'WVP.csv'
annotationCV = mypath+'/'+datafile+'CV.csv'
annotationDH = mypath+'/'+datafile+'DH.csv'
annotationLW = mypath+'/'+datafile+ 'LW.csv'
annotationfinal = mypath+'/'+datafile+'.csv'
annotationpred = mypath+ '/'+datafile+'prednew.csv'
annotationsliding = mypath+ '/'+datafile+'annonew.csv'
#annotationauto = mypath+'/'+datafile+'autofill.csv'



samplerate,data = wavfile.read(datapath)
data=-data

sos = signal.butter(10, 12,'hp', fs=samplerate, output='sos')
data = signal.sosfilt(sos, data)

length = len(data)/samplerate
time = np.linspace(0., length, data.shape[0])

##Load annotations if exist
if isfile(annotationWVP):
    WVP = np.array(pd.read_csv(annotationWVP)).flatten()
else:
    pass
if isfile(annotationCV):
    CV = np.array(pd.read_csv(annotationCV)).flatten()
else: pass

if isfile(annotationDH):
    DH = np.array(pd.read_csv(annotationDH)).flatten()
else:
    pass

if isfile(annotationLW):
    LW = np.array(pd.read_csv(annotationLW)).flatten()
else:
    pass

if isfile(annotationpred):
    pred = np.array(pd.read_csv(annotationpred)).flatten()
else:
    pass

if isfile(annotationfinal):
    finalannotation = np.array(pd.read_csv(annotationfinal)).flatten()
else:
    pass

if isfile(annotationsliding):
    slidingannotation = np.array(pd.read_csv(annotationsliding)).flatten()
else:
    pass

ex1=WVP
ex2=DH
ex3=LW
ex4 =CV
'''
ex3=finalannotation
ex4 = slidingannotation

ex5 = pred

ex1=DH
ex2=pred
ex3=pred
ex4=pred
ex5=pred
'''
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, sharex=True, gridspec_kw={'height_ratios': [6,1,1,1,1]})

ax1.plot(time, data)
ax1.axes.plot(time, data, label = 'EMG data', color='b' )
ax1.set_ylim(-200,200)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (µV)")

first_it = ex1[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex1[0])
for n, i in enumerate(ex1):
    if n == len(ex1) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex1[n + 1])
index_list = [0] + index_list
index_list.append((len(ex1) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax2.add_collection(lc)

first_it = ex2[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex2[0])
for n, i in enumerate(ex2):
    if n == len(ex2) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex2[n + 1])
index_list = [0] + index_list
index_list.append((len(ex2) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax3.add_collection(lc)

first_it = ex3[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex3[0])
for n, i in enumerate(ex3):
    if n == len(ex3) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex3[n + 1])
index_list = [0] + index_list
index_list.append((len(ex3) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax4.add_collection(lc)


first_it = ex4[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex4[0])
for n, i in enumerate(ex4):
    if n == len(ex4) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex4[n + 1])
index_list = [0] + index_list
index_list.append((len(ex4) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax5.add_collection(lc)
'''

first_it = ex5[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex5[0])
for n, i in enumerate(ex5):
    if n == len(ex5) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex5[n + 1])
index_list = [0] + index_list
index_list.append((len(ex5) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax6.add_collection(lc)

first_it = ex6[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex6[0])
for n, i in enumerate(ex6):
    if n == len(ex6) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex6[n + 1])
index_list = [0] + index_list
index_list.append((len(ex6) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax7.add_collection(lc)

first_it = ex7[0]  # labels[0]
index_list = []
label_list = []
label_list.append(ex7[0])
for n, i in enumerate(ex7):
    if n == len(ex7) - 1:
        pass
    else:
        if i == first_it:
            pass
        else:
            first_it = i
            index_list.append((n - 1) / 44100)
            label_list.append(ex7[n + 1])
index_list = [0] + index_list
index_list.append((len(ex7) - 1) / 44100)

points_t = np.array([np.asarray(index_list), 0.5 * np.ones(len(index_list))]).T.reshape(-1, 1, 2)
segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)

cmap = ListedColormap(['k', 'mediumpurple', 'palegreen', 'lightskyblue','salmon'])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
lc = LineCollection(segments_t, cmap=cmap, norm=norm)
lc.set_array(np.asarray(label_list))
lc.set_linewidth(75)
line = ax8.add_collection(lc)
'''
plt.show()
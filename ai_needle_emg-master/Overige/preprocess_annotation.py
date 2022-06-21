from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os.path import dirname, isfile
from tkinter.filedialog import askopenfilename, asksaveasfilename
from collections import Counter
from tqdm import tqdm

filepath = askopenfilename(
            filetypes=[("Wav Files", "*.wav"), ("All Files", "*.*")])

manual_anno_path= str(filepath).split("/Annotations_signal_type")[0] +'/Data'
mypath = dirname(filepath)
annotated_file_list = \
    [mypath +'/' + f for f in os.listdir(mypath) if ".wav" in f]
#datapath= mypath+'/'+datafile+'.wav'

def needleplusnonanalysable(data):
    data= np.where(data>3,3,data)
    return data

for annotated_file in tqdm(annotated_file_list):
    datafile = str(annotated_file).split("/")[-1].split(".wav")[0]
    mypath = str(annotated_file).split(datafile)[0]

    annotationWVP = mypath+'/'+datafile+'WVP.csv'
    annotationCV = mypath+'/'+datafile+'CV.csv'
    annotationDH = mypath+'/'+datafile+'DH.csv'
    annotationLW = mypath+'/'+datafile+'LW.csv'
#annotationauto = mypath+'/'+datafile+'autofill.csv'
    if 'WVP' in locals():
        del WVP
    if 'DH' in locals():
        del DH
    if 'CV' in locals():
        del CV
    if 'LW' in locals():
        del LW

    if 'ex1' in locals():
        del ex1
    if 'ex2' in locals():
        del ex2
    if 'ex3' in locals():
        del ex3
    if 'ex4' in locals():
        del ex4
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
    #auto = np.array(pd.read_csv(annotationauto)).flatten()
    ## Looking for existing annotations
    if 'WVP' in locals():
        ex1 = WVP
        if 'CV' in locals():
            ex2 = CV
            if 'DH' in locals():
                ex3 = DH
                if 'LW' in locals():
                    ex4 = LW
                    print('Four annotations for this file:' + datafile)
                else:
                    print('Three annotations for this file:' + datafile)  
            elif 'LW' in locals():
                ex3 = LW
                print('Three annotations for this file:' + datafile)
            else: 
                print('Two annotations for this file:' + datafile)
        elif 'DH' in locals():
            ex2 = DH
            if 'LW' in locals():
                ex3 = LW
                print('Three annotations for this file:' + datafile)
            else: 
                print('Two annotations for this file:' + datafile)
        elif 'LW' in locals():
            ex2 = LW
            print('Two annotations for this file:' + datafile)
        else: 
            print('One annotations for this file:' + datafile)

    elif 'CV' in locals():
        ex1 = CV
        if 'DH' in locals():
            ex2 = DH
            if 'LW' in locals():
                ex3 = LW
                print('Three annotations for this file:' + datafile)
            else:
                print('Two annotations for this file:' + datafile)  
        elif 'LW' in locals():
            ex2 = LW
            print('Two annotations for this file:' + datafile)
        else: 
            print('One annotations for this file:' + datafile)
    elif 'DH' in locals():
        ex1 = DH
        if 'LW' in locals():
            ex2 = LW
            print('Two annotations for this file:' + datafile)
    elif 'LW' in locals():
        ex1 = LW
        print('One annotations for this file:' + datafile)
    else:
        print('No annotations for this file:' + datafile)
    annotation={}

    annotation={}

    if 'ex4' in locals():
        ex4 = needleplusnonanalysable(ex4)

    if 'ex2' in locals():
        ex2 = needleplusnonanalysable(ex2)

    if 'ex4' in locals():
        ex4 = needleplusnonanalysable(ex4)

    if 'ex1' in locals():
        ex1 = needleplusnonanalysable(ex1)

    if 'ex4' in locals():
        for x in range(len(ex1)):
            annotation[x]=ex1[x],ex2[x], ex3[x], ex4[x]
    elif 'ex3' in locals():
        for x in range(len(ex1)):
            annotation[x]=ex1[x], ex2[x], ex3[x]
    elif 'ex2' in locals():
        for x in range(len(ex1)):
            annotation[x]=ex1[x],ex2[x]
    else:
        pass
    

    ##calculating majority vote: Tie is no annotations, otherwise majority wins.
    if ('ex1' in locals()) and ('ex2' in locals()): # only calculate when two or more annotations
        finalannotations=np.zeros([1,len(ex1)])
        for x in range (len(ex1)):
            count=Counter(annotation[x])
            top_two = count.most_common(2)
            if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
                # It is a tie
                finalannotations[0,x] = 0
            else:
                finalannotations[0,x] = top_two[0][0]
            #print(top_two[0][0])

        finalannotations=np.array(finalannotations).flatten()
        finalannotations=pd.DataFrame(finalannotations)

        finalpath=manual_anno_path+'/'+datafile+'.csv'
        finalannotations.to_csv(finalpath, index= False)
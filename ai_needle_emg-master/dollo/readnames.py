import random
import os

def readnames(names, experiment_directory):
    rest_files = []
    needle_files = []
    contraction_files = []  
    empty_files = []  
#look for all files with trainingname in name
    for x in range(len(names)):
        rest = []
        needle = []
        contraction = []
        rest = [experiment_directory + "Rest/"+ f for f in os.listdir(experiment_directory + "Rest/") if str(names[x]) in f]
        contraction = [experiment_directory + "Contraction/" + f for f in os.listdir(experiment_directory + "Contraction/") if str(names[x]) in f]
        needle = [experiment_directory + "Needle/" + f for f in os.listdir(experiment_directory + "Needle/") if str(names[x]) in f]

        rest_files = rest_files + rest
        needle_files = needle_files + needle
        contraction_files = contraction_files + contraction
    random.shuffle(rest_files)
    random.shuffle(contraction_files)
    random.shuffle(needle_files)
    return  rest_files, contraction_files, needle_files

from os import path
import numpy as np
from scipy import signal
import pandas as pd
from os import listdir
from os.path import isfile, join
path_to_prediction= ("C:/Users/debor/OneDrive/Documents/TG/Stages/Afstudeerstage/Python/Predictionforvalidation/")
save_path="C:/Users/debor/OneDrive/Documents/TG/Stages/Afstudeerstage/Python/Validationannotations/"
predicted_files = [f for f in listdir(path_to_prediction) if isfile(join(path_to_prediction,f))]
predicted_files = pd.DataFrame(predicted_files, columns=['filename'])
path_to_prediction = path_to_prediction + predicted_files.filename
save_filename=predicted_files.filename.str[:-18]

for f in range(len(path_to_prediction)):
   prediction = np.load(path_to_prediction[f])
   #upsample_prediction=signal.resample(prediction,4410*len(prediction))
   prediction_new=[]
   for x in range(len(prediction)-10):
      to_append=(prediction[x+10])*np.ones(4410) 
      prediction_new=np.append(prediction_new, to_append)
   saveas = save_path+save_filename+'prednew.csv'
   prediction_new=pd.DataFrame(prediction_new)
   prediction_new.to_csv(saveas[f], index= False)
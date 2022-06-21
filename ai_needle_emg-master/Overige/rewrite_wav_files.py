#rewrite wav files
#Author: Deborah Hubers

from os import listdir
from os.path import isfile, join
import pandas as pd
mypath = 'L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/Experimentation/Experiments'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = pd.DataFrame(onlyfiles, columns=['filenames'])
onlyfiles = onlyfiles.sort_values(by=['filenames'])
onlyfiles = onlyfiles.reset_index(drop=True)
directories = mypath+'/'+onlyfiles

#make string that needs to replace faulty string
with open(r'L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/ai_needle_emg/Data/0030_R_tib_ant_1_1.wav', 'rb') as file_good:
    byte_good = file_good.read().hex()
replace_str = byte_good[0:46] # first 16 bytes contain information that is specific for each datafile and must not be replaced

#loop through all data and replace the first 60 bytes
for x in range (len(directories)):
    file = directories.filenames[x]
    print(file)
    with open(file, 'rb') as file_bad:
        byte_bad = file_bad.read().hex()
    print(byte_bad[:60])
    print(byte_good[:60])
    source_str = byte_bad[0:46]
    with open(file, 'rb') as f:
        content = f.read().hex()
    #print(source_str + " old wav file`:       ", source_str in content)
    content = content.replace(source_str, replace_str)
    with open(file, 'wb') as f:
        f.write(bytes.fromhex(content))
    with open(file, 'rb') as f:
        new_content = f.read().hex()
    #print(source_str + " in `new wav file`:", source_str in new_content)
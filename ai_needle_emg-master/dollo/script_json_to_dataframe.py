import json
from pathlib import Path
import re

from numpy import NaN
import pandas as pd

# collect Json files
source_folder = '2swavfiles'
json_files = Path(source_folder).glob('*.json')

# initalize empty dataframe
all_data = pd.DataFrame()

# loop over found json files
for f in json_files:
    # open single json file
    with open(f) as fo:
        json_data = json.load(fo)
        # only load data if 'annotations' are present.
        if 'annotations' in json_data:
            # load json data to dataframe
            df = pd.DataFrame.from_dict(json_data['annotations'])
            # Add column with filename
            df[['file']] = str(f)
            df[['frame_start']] = str(re.search('[0-9]+.json$',str(f))[0][:-5])
            # drop column with the dates.
            df.drop(columns='changed', inplace=True)
            # convert to another format "file, annotator1, annotator2 etc."
            file_data = pd.pivot_table(df,values='annotation', columns=['user'], index=['file','frame_start'])
            all_data = all_data.append(file_data)

print(all_data)
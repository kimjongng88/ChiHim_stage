import pandas as pd

df = pd.read_excel('C:/Users/debor/OneDrive/Documents/TG/Stages/Afstudeerstage/Python/Database_included_smal_only.xlsx')
filenames = df['Filename'].tolist()

df_findings = df['FreeTextFindings (use EMGSummaryConfig to translate)'].str.split('_x0008_', expand=True)
df_findings['Filename'] = filenames
df_findings.columns = ['Index', 'Ins', 'Fibr.', 'Pos.', 'Fasc.', 'Duur', 'Poly', 'Max', 'Recrutisering',
                        'Ampl.', 'Comm', 'Filename']
#df.drop(columns=['Name', 'Unnamed: 1', 'FreeTextFindings (use EMGSummaryConfig to translate)',
#                 'binary findings (use EMGSummaryConfig to translate)', 'MRL raw file location'], inplace=True)
df = df.merge(df_findings, on='Filename', how='inner')
df.drop(columns=['Index'], inplace=True)

df.to_excel('C:/Users/debor/OneDrive/Documents/TG/Stages/Afstudeerstage/Python/Database_columns.xlsx', index=False)
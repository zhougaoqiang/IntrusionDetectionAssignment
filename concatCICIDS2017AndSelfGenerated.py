#mapping file
import pandas as pd

filenames = ['concatenated_rename_2Label',
             'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX']

df1 = pd.read_csv('./data/' + filenames[0] + '.csv')
df2 = pd.read_csv('./data/' + filenames[1] + '.csv')

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()
df2 = df2.drop('Fwd Header Length.1',  axis=1)
#make sure the column is same
df1 = df1[df2.columns.tolist()]
concatenated_df = pd.concat([df1,df2], ignore_index=True)
concatenated_df.to_csv('./data/combinedCICIDS2017andSelfGenerated.csv', index=False)
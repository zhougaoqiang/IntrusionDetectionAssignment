import pandas as pd

df = pd.read_csv('./data/concatenated_rename.csv')
df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace from column names
df['Label'] = df['Label'].apply(lambda x: 'PortScan' if 'Portscan' in x else x)
df.to_csv('./data/concatenated_rename_2Label.csv', index=False)
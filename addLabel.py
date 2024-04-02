import pandas as pd;

import sys

# Check if there are enough arguments provided
if len(sys.argv) < 4:
  print("Error: Please provide both filename and IP address as arguments.")
  exit(1)

# Assign arguments to variables
filename = sys.argv[1]
ip = sys.argv[2]
portscantype = sys.argv[3]

print(filename)
print(ip)
print(portscantype)

df = pd.read_csv(filename+'.csv')
df.columns = df.columns.str.strip()
portscantype = 'Portscan('+ portscantype + ')'
print(portscantype)
df['Label'] = df['src_ip'].apply(lambda x: portscantype if x == ip else 'BENIGN')
print(df['Label'].unique())

filename = filename + '_labeled.csv'
df.to_csv(filename, index=False)
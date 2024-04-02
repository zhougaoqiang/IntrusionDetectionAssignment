import pandas as pd
import matplotlib.pyplot as plt
import re

# are_columns_equal = (df['Fwd Header Length'] == df['Fwd Header Length.1']).all()
# print("Are 'Fwd Header Length' and 'Fwd Header Length.1' identical across all rows?:", are_columns_equal)

def remove_special_characters(text):
    # 使用正则表达式去除特殊字符
    return re.sub(r'[\\/:\*\?\"\<\>\|]', '', text)

def drawPictures(file, savePath, isCIC):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    print(df.columns.tolist())
    
    if (isCIC == True) :
        #remove duplicate column
        df = df.drop('Fwd Header Length.1', axis=1)
        #remove useless column
        df = df.drop('Destination Port', axis=1)
    else :
        #remove useless column
        df = df.drop(['src_ip','dst_ip','src_port','Destination Port','protocol','timestamp'], axis=1)
    
        # Combined condition for dropping columns with "IAT" or "Bluk"
    columns_to_drop = [col for col in df.columns if ('IAT' in col or 'Bulk' in col)]
    df = df.drop(columns_to_drop, axis=1)  # Drop matching columns
    columns_to_drop = ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Down/Up Ratio']
    df = df.drop(columns_to_drop, axis=1)  # Drop matching columns
    print(df.columns.tolist())
    
    print(df.shape)
    print(df['Label'].unique())

    for column in df.columns:
       plt.figure(figsize=(6, 2))
       plt.scatter(df[column], df['Label'], c='blue', alpha=0.4)
       plt.xlabel(column)
       plt.ylabel('Label')
       plt.title(f'{column} vs. Label')
       plt.grid(True)
       filename = remove_special_characters(column)
       plt.savefig(f'./{savePath}/{filename}_vs_Label.png')
       plt.close()
    
drawPictures('./data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Review_Feature_CICIDS2017', True)
drawPictures('./data/concatenated_rename_2Label.csv', 'Review_Feature_SelfGenerated', False)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import time


def removeColumns(df, columns):
    for col in columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
        # else :
        #     print(f'{col} not in columns')
    return df

def clearFeatures(df):
    # CICIDS2017 have
    df = removeColumns(df, ['Fwd Header Length.1'])
    #remove specific features
    df = removeColumns(df, ['src_ip','dst_ip','src_port','Destination Port','protocol','timestamp'])
    #remove network traffic releated  features;
    columns_to_drop = [col for col in df.columns if ('IAT' in col or 'Bulk' in col)]
    df = df.drop(columns_to_drop, axis=1)  # Drop matching columns
    #remove the columns that depends on the network congestion conditions
    columns_to_drop = ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Down/Up Ratio']
    df = removeColumns(df, columns_to_drop)
    #remove the columns based on the PCA visualization method
    columns_to_drop = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'min_seg_size_forward']
    df = removeColumns(df, columns_to_drop)
    # print(df.columns.tolist())
    return df

def getProcessedData(filename, keep2LableOnly):
    df = pd.read_csv(filename)
    # Remove any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip() 
    # data cleaning
    df = clearFeatures(df)
    # Remove empty data row
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)  
    # if keep2LableOnly == True :
    #     df['Label'] = df['Label'].apply(lambda x: 'PortScan' if 'Portscan' in x else x)
    
    return df

classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(), 
    "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(100,)),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

def trainData(trainFile, testFile, needValidation):
    df = getProcessedData(trainFile, True)
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

    features = df.drop('Label', axis=1)
    label = df['Label']
    # Split the dataset into training and validation sets
    if needValidation :
        train_features, test_features, train_label, test_label = train_test_split(features, label, test_size=0.7, random_state=100)
    else :
        train_features = features
        train_label = label
    
    test_df = getProcessedData(testFile, True)
    test_df['Label'] = label_encoder.transform(test_df['Label'])  # Use the same encoder as the training data
    X_test = test_df.drop('Label', axis=1)
    X_test = X_test.reindex(columns=train_features.columns)
    y_test = test_df['Label']
    
    print('_______________________________________________________________________')
    print('_______________________________________________________________________')
    for name, clf in classifiers.items():
        #train the classifer
        start_time = time.time()
        clf.fit(train_features, train_label)
        # Predict on validation set
        if needValidation :
            predictions_val = clf.predict(test_features)
            print(f"{name} Classifier (Validation):")
            print(classification_report(test_label, predictions_val, target_names=label_encoder.classes_, digits=7))
    
        # Predict on test set
        predictions_test = clf.predict(X_test)
        print(f"{name} Classifier (Test):")
        print(classification_report(y_test, predictions_test, target_names=label_encoder.classes_, digits=7))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{name} Classifier Elapsed time: {elapsed_time}")
        print('_______________________________________________________________________')
        print('_______________________________________________________________________')

def printSplitLine(trainingfile, targetfile):
    print('#######################################################################')
    print(f"####{trainingfile}  => {targetfile} ####")
    print('#######################################################################')

#################################################
#setting
#################################################
CICIDS2017 = './data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv' #the test result are all 0
selfGenerated = './data/concatenated_rename_2Label.csv'
combinedFile = './data/combinedCICIDS2017andSelfGenerated.csv'
testFileProviedByProf = './data/sample4_labeled_renamed.csv'
# needValidation = True 
needValidation = True

#################################################
### enable what you want
#################################################
##train with CICIDS2017, test with prof test file.
printSplitLine(CICIDS2017, testFileProviedByProf)
trainData(CICIDS2017, testFileProviedByProf, needValidation)

##train with self data, test with prof test file
printSplitLine(selfGenerated, testFileProviedByProf)
trainData(selfGenerated, testFileProviedByProf, needValidation)

##train with combined both CICIDS2017 and selfGenerated, test with prof test file
printSplitLine(combinedFile, testFileProviedByProf)
trainData(combinedFile, testFileProviedByProf, needValidation)


#################################################
##train with CICIDS2017, test with self data,
printSplitLine(CICIDS2017, selfGenerated)
trainData(CICIDS2017, selfGenerated, needValidation)

##train with self data, test with CICIDS2017
printSplitLine(selfGenerated, CICIDS2017)
trainData(selfGenerated, CICIDS2017, needValidation)
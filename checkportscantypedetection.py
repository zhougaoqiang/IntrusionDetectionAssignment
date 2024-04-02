import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


def removeColumns(df, columns):
    for col in columns:
        if col in df.columns:
            df = df.drop(col, axis=1)
        else :
            print(f'{col} not in columns')
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
    print(df.columns.tolist())
    return df

def getProcessedData(filename):
    df = pd.read_csv(filename)
    # Remove any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip() 
    # data cleaning
    df = clearFeatures(df)
    # Remove empty data row
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(), 
    "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(100,)),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Evaluation function to save predictions and true labels to a file
def savePredictions(classifier_name, classifier, X_test, y_test):
    predictions = classifier.predict(X_test)  # Predict on test set
    
    # Create a DataFrame with true labels and predictions
    results_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Label': predictions
    })
    
    # Save to CSV file
    results_df.to_csv(f'./testFilePortScanTypeDetectResult/{classifier_name}_predictions.csv', index=False)

def trainData(trainFile, testFile):
    df = getProcessedData(trainFile)
    features = df.drop('Label', axis=1)
    label = df['Label']
    train_features = features
    train_label = label
    
    test_df = getProcessedData(testFile)
    X_test = test_df.drop('Label', axis=1)
    X_test = X_test.reindex(columns=train_features.columns)
    y_test = test_df['Label']
    
    for name, clf in classifiers.items():
        #train the classifer
        clf.fit(train_features, train_label)
        savePredictions(name, clf, X_test, y_test)

#################################################
#setting
#################################################
selfGenerated = './data/concatenated_rename.csv'
testFileProviedByProf = './data/sample4_labeled_renamed.csv'

#################################################
trainData(selfGenerated, testFileProviedByProf)

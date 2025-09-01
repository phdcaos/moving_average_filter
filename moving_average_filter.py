import wfdb
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder



'''first we will define wich daata we want to load
luckily MIT-BIH already provides a file with all record names

this funtion reads the RECORDS file and returns a lsit of the record names present in the database'''

def load_record_names(record_file_path):
    with open(record_file_path, 'r') as file:
        record_names = file.read().splitlines()
    return record_names

#this function loads both the record and the annotation for said racord and returns them as a list
def load_database_data(record_name, db_path='database'):
    record = wfdb.rdrecord(f'{db_path}/{record_name}')
    annotation = wfdb.rdann(f'{db_path}/{record_name}','atr')
    signal = [record, annotation]
    return signal



'''These functions slpits the records into single heart beats using the annotation data
it returns a list of wfdb record objects, each containing a single heartbeat
'''
def split_records(signals):
    all_records = []

    window  = 200
    half_window = window /2

    for i in range(len(signals)):
        record = signals[i][0]
        annotation = signals[i][1]
        peak_locations = annotation.sample

        start = int(peak_locations[i] - half_window)
        end = int(peak_locations[i] + half_window)  

        if start >= 0 or end <= len(record.p_signal):
            segment = record.p_signal[start:end]
            
            signal = [segment, annotation]
            all_records.append(signal)


    
    return all_records


def split_annotations(signals):
    all_labels = []

    for i in range(len(signals)):
        annotation = signals[i][1]
        labels = annotation.symbol
        all_labels.append(labels)

    return all_labels


'''This funtion creates a dictionary with the lables as keys and the records and annotations as values'''
def create_signal_dic(keys, values):
    
    signal_dic = {}

    for key, value in zip(keys, values):
        signal_dic[key] = value
    
    return signal_dic

if __name__ == "__main__":
    
    data_path = 'database/'
    #reads the RECORDS file and stores the record names in a list
    names = load_record_names(f'{data_path}/RECORDS')
    
    #creates the dictionary that will recieve the signals and the annotations
    separated_beats ={}

    #defyning the window size for each heartbeat
    window = 260
    half_window = window / 2

    #Loop for processing each record
    for record_name in names:

        #gets a record and its annotation
        record = wfdb.rdrecord(f'{data_path}/{record_name}')
        annotation = wfdb.rdann(f'{data_path}/{record_name}','atr')

        #gets the first channel of the record
        signals = record.p_signal[:,0]

        #Annotation processing loop
        for i in range(len(annotation.sample)):
            #gets the label of the heartbeat
            label = annotation.symbol[i]

            #gets the R peak location   
            r_peak = annotation.sample[i]  

            #gets the endpoint and startpoint of the heartbeat window
            start = int(r_peak - half_window)
            end = int(r_peak + half_window)

            #checks if the window is within the signal bounds
            if 0 <= start < end <= len(signals):
                segment = signals[start:end]

            #adds the label as a key to a dictionary if it doesn't already exists
                if label not in separated_beats:
                    separated_beats[label] = []
                separated_beats[label].append(segment)

    min_samples = 150
    
    #removes labels with less than min_samples heartbeats

    final_count_beats = {
        label: records for label, records in separated_beats.items()
        if len(records) >= min_samples
    }
    '''
    if 'V' in final_count_beats:

        beat = final_count_beats['V'][1]

        plt.figure(figsize=(12,6))
        plt.plot(beat)
        plt.title('Example of a Normal Heartbeat (V)')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    '''
    #now we will generate 150 random indexes for each label to choose wich ones with be filtered
    number_of_samples = 150
    
    random_beats ={
        label: random.sample(records, number_of_samples)
        for label, records in final_count_beats.items() 

    }

    #now we will aplly a moving average filter
    filtered_beats = {}
    filter_average_window = 3


    for label, records in random_beats.items():
        
        #temporary list for filtered records of this class
        filtered_records = []

        #Iterates each class record
        for record in records:
            #Converts the record to a pandas series
            record_series = pd.Series(record)

            #Applies the moving average filter the min_periods guarantees that the filter will aply even to the first values
            filtered_beat = record_series.rolling(window = filter_average_window, min_periods = 1).mean()

            #Adds the the filtered beat to the temporary list converting it back to numpy array
            filtered_records.append(filtered_beat.values)
        
        #Adds the filtered records to the final dcitionary
        filtered_beats[label] = filtered_records
     
     #Now we will aplly the Smoteen algorithm to balance the dataset
    
    #prepare data for the Smoteen algorithm
    x_records = []
    Y_labels = []

    for label, records in filtered_beats.items():
        x_records.extend(records)
        Y_labels.extend([label] * len(records))

    x_records = np.array(x_records)
    Y_labels = np.array(Y_labels)

    #map the string labels to integers
    all_labels = list(filtered_beats.keys())
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    y_encoded = label_encoder.transform(Y_labels)

    #Verify the mapping
    for original_label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f'Original label: {original_label} -> Encoded label: {encoded_label}')

    #Aplly the Smoteen algorithm
    sampling_strategy = {
                        label_encoder.transform([label])[0]: 5000 
                        for label in all_labels}
    smoteenn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    
    
    #Adjusting the data to 2D
    if x_records.ndim == 1:
        x_records = np.expand_dims(x_records, axis=-1)
    
    x_resampled, Y_resampled = smoteenn.fit_resample(x_records, y_encoded)

    
    x_final, y_final = shuffle(x_resampled, Y_resampled, random_state=42)

    print(Counter(y_final))
    
print(f"Formato do conjunto de dados final (x_final): {x_final.shape}")
print(f"Formato dos rótulos finais (y_final): {y_final.shape}")







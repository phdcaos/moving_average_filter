#pip install -r requirements.txt
import wfdb
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
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
    window = 360 # TESTING THE WINDOW SIZE AS 1 SECOND 
    window_begining = int(window*0.3) #DEFINING 30% OF THE SAMPLES TO BE BEFORE THE R PEAK
    window_end = int(window*0.6) #DEFINING THE REMAINING 70% TO BE AFTER THE R PEAK

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
            start = int(r_peak - window_begining)
            end = int(r_peak + window_end)

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
   
    
    #now we will generate 150 random indexes for each label to choose wich ones with be filtered
    number_of_samples = 150
    
    random_beats ={
        label: random.sample(records, number_of_samples)
        for label, records in final_count_beats.items() 

    }




    #TRYING TO APPLY SMOTE BEFORE THE FILTER
    x_raw = []
    y_raw = []

    #after definyng the raw variables we associate the beats to the x and the labels to the y a number of times equal to the number of beats in each key, 150 in this case

    for label, beats in random_beats.items():
        x_raw.extend(beats)
        y_raw.extend([label]*len(beats))

    #then, we convert it into an numpy array
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)

    #now we will encode the symbols, we will get each key from the random_beats dictionary
    all_labels = list(random_beats.keys())

    #now we will create and train the label_encoder with all labels it will find
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    #now we will get a nuumpy list of all the encoded symbols
    y_encoded = label_encoder.transform(y_raw)

    #checking if it worked
    
    print("Labels encoded:")
    for original_label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"'{original_label}' to {encoded_label}")


    #now we will apply the SMOTE 
    #first we define the sampling strategy
    sampling_strategy ={
        label_encoder.tranform([label])[0]:5000
        for label in all_labels
    }
    #now we create the smote object
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42) 
    
    #the smote expects 2d data, this will adjust that
    if x_raw.ndim == 1:
        x_raw = np.expand_dims(x_raw, axis=-1)

    #Appling the transformation, this function will learn the class distribuition and create new samples
    #TRYING TO APPLY SMOTE BEFORE THE FILTER
    x_raw = []
    y_raw = []

    #after definyng the raw variables we associate the beats to the x and the labels to the y a number of times equal to the number of beats in each key, 150 in this case

    for label, beats in random_beats.items():
        x_raw.extend(beats)
        y_raw.extend([label]*len(beats))

    #then, we convert it into an numpy array
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)

    #now we will encode the symbols, we will get each key from the random_beats dictionary
    all_labels = list(random_beats.keys())

    #now we will create and train the label_encoder with all labels it will find
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    #now we will get a nuumpy list of all the encoded symbols
    y_encoded = label_encoder.transform(y_raw)

    #checking if it worked
    
    print("Labels encoded:")
    for original_label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"'{original_label}' to {encoded_label}")


    #now we will apply the SMOTE 
    #first we define the sampling strategy
    sampling_strategy ={
        label_encoder.transform([label])[0]:5000
        for label in all_labels
    }
    #now we create the smote object
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42) 
    
    #the smote expects 2d data, this will adjust that
    if x_raw.ndim == 1:
        x_raw = np.expand_dims(x_raw, axis=-1)

    #Appling the transformation, this function will learn the class distribuition and create new samples
    x_resampled, y_resampled = smote.fit_resample(x_raw, y_encoded)

    #now we will create a dictionary with the SMOTE created beats
    #getting back the original labels
    dic_labels = label_encoder.inverse_transform(y_resampled)

    #creating the dictionary
    smote_beats = {}

    for i, original_label in enumerate(dic_labels):
        #the beat is the i linne from x_resampled
        new_beat = x_resampled[i]

        if original_label not in smote_beats:
            smote_beats[original_label] =[]
        smote_beats[original_label].append(new_beat)
    

    #now we convert ir to a numpy array
    for label in smote_beats:
        smote_beats[label] = np.array(smote_beats[label])


    #now we will aplly a moving average filter
    filtered_beats = {}
    filter_average_window = 5


    for label, records in smote_beats.items():
        
        #temporary list for filtered records of this class
        filtered_records = []

        #Iterates each class record
        for record in records:
            #Converts the record to a pandas series
            record_series = pd.Series(record)

            #Applies the moving average filter the min_periods guarantees that the filter will aply even to the first values
            filtered_beat = record_series.rolling(window = filter_average_window).mean()

            #Adds the the filtered beat to the temporary list converting it back to numpy array
            filtered_records.append(filtered_beat.values)
        
        #Adds the filtered records to the final dcitionary
        filtered_beats[label] = filtered_records
     
     #Now we will aplly the Smoteen algorithm to balance the dataset

   
    for key in filtered_beats:
        beats = [key]
        plt.figure(figsize=(12,6))
        plt.plot(beats)
        plt.title(f'Unfiltered beats of {key} label: ' )
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
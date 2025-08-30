import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''first we will define wich daata we want to load
luckily MIT-BIH already provides a file with all record names

this funtion reads the RECORDS file and returns a lsit of the record names present in the database'''

def load_record_names(record_file_path):
    with open(record_file_path, 'r') as file:
        record_names = file.read().splitlines()
    return record_names


def load_database_data(record_name, db_path='database'):
    record = wfdb.rdrecord(f'{db_path}/{record_name}')
    annotation = wfdb.rdann(f'{db_path}/{record_name}','atr')
    signal = [record, annotation]
    return signal

if __name__ == "__main__":
    #recieves the names that will be used to load the data into a list
    data_path = 'database/'
    signals = []
    
    names = load_record_names(f'{data_path}/RECORDS')

    for name in names:
        signals.append(load_database_data(name, db_path=data_path))

    for i in range(len(signals)):
        wfdb.plot_wfdb(signals[i][0], annotation=signals[i][1], title=f'Record {names[i]}', time_units='seconds')       




    
    
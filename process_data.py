import os
import numpy as np
import sys
from scipy.io import wavfile
import pandas as pd

from helper_code import *
from ecg_extract import *

data_folder = 'data/training_data'

patient_ids = find_data_folders(data_folder)
patient_num = len(patient_ids)
data_list = []

for i in range(patient_num):
    print("Patient No ", i+1, "/", patient_num)
    metadata = load_challenge_data(data_folder, patient_ids[i])
    record_files = find_recording_files(data_folder, patient_ids[i])
    signal, chanels, sampling_frequency = load_recording_data(record_files[0])   ## Only the first electrocardiogram
    newData = {
        "ID": patient_ids[i],
        "Age": get_age(metadata),
        "Sex": get_sex(metadata),
        "Height": get_height(metadata),
        "Weight": get_weight(metadata),
        "Mean": get_ecg_mean(signal),
        "RMS": get_ecg_rms(signal),
        "Std": get_ecg_std(signal),
        "HeartRate": get_ecg_heart_rate(signal),
        "Sdnn": get_ecg_sdnn(signal),
        "Rmssd": get_ecg_rmssd(signal),
        "Pnn50": get_ecg_pnn50(signal),
        "Murmur": get_murmur(metadata)
    }
    data_list.append(newData, ignore_index=True)

data = pd.DataFrame(data_list)
data = data.set_index('ID')
print(data)

csv_file_path = "data.csv"
data.to_csv(csv_file_path, index=False)
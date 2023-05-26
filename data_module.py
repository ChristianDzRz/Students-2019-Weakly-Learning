# -*- coding: utf-8 -*-
# title           :dataModule.py
# description     :Class dedicated to extract the data from the Dcase Dataset
# author          :Christian Diaz
# date            :2019/04/17
# version         :0.1
# usage           :python pyscript.py
# notes           :
# python_version  :3.6
# ==============================================================================

# Import the modules needed to run the script.
import os
import librosa
import pandas as pd
from tqdm import tqdm


class DataBase:
    def __init__(self, mode, config):
        self.config = config['data_module']
        # Root folder of Data
        db_path = self.config['db_path']
        self.mode = mode
        self.classes = {'Speech': 0, 'Dog': 1, 'Cat': 2,
                        'Alarm_bell_ringing': 3, 'Dishes': 4, 'Frying': 5,
                        'Blender': 6, 'Running_water': 7, 'Vacuum,cleaner': 8,
                        'Electric_shaver_toothbrush': 9}

        # Modes corresponds to each of the Datasets that can be used:
        # 1. Weak label (train_weak): filename and which sounds are present
        # in the sample
        # 2. Strong label(test): filename, onset, offset,
        # class of sound present in the sample
        # 3. Unlabel in domain (train_unlabel_in_domain): filename
        # 4. Unlabel out of domain (train_unlabel_out_of_domain):
        # filename [more classes of sounds than 10 considered]

        # Set the data Directory according to the mode chosen
        self.data_dir = os.path.join(db_path, 'audio',
                                     self.mode.split('_')[0] + '/' + '_'.join(
                                        self.mode.split('_')[1:]))
        # Set .csv path according to the path chosen
        self.csv_path = os.path.join(db_path, 'metadata',
                                     mode.split('_')[0] + '/' +
                                     '%s.csv' % "{csv}".format(
                                        csv='_'.join(mode.split('_')[1:])
                                        if mode.split('_')[1:] != []
                                        else mode.split('_')[0]))

        if self.mode is 'others':
            self.create_others_csv()
        self.csv_file = pd.read_csv(self.csv_path, sep='\t')

    # Returns object with all frame indices
    def csv_index(self):
        return self.csv_file.index

    # Get data from each of the files, id corresponds to the row in csv file
    def items(self, idx):
        filename = self.csv_file['filename'][idx]
        exists = os.path.isfile(os.path.join(self.data_dir, filename))
        if exists:
            # Open audio file
            audio, sr = librosa.load(os.path.join(self.data_dir, filename))
            if self.mode in ['train_weak', 'test']:
                label = self.csv_file['event_label'][idx]
                if self.mode is 'test':
                    onset = self.csv_file['onset'][idx]
                    offset = self.csv_file['offset'][idx]
            return audio, sr

        else:
            print("File: " + filename + " is missing")

# Creates .csv from file the files inside the "others" folder
    def create_others_csv(self):
        df = pd.DataFrame({"filename": [], "event_label": [],
                           "onset": [], "offset": []})
        for file_name in tqdm(os.listdir(self.data_dir)):
            df = df.append({'filename': file_name}, ignore_index=True)
            df.to_csv(self.csv_path, header='filename', sep="\t", index=False)

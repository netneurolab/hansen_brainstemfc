"""
load in timeseries
"""

import numpy as np
import pandas as pd

path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/data/brainstem_fc/timeseries/"

parc = 400

info = pd.read_csv(path+'data/region_info_Schaefer'
                   + str(parc) + '.csv')
labels = info.query("structure == 'brainstem' or structure == 'cortex'")['labels']
subjects = np.arange(1, 22)
subjects = np.delete(subjects, 8)  # remove subject 9

timeseries = np.zeros((len(labels), 630, len(subjects)))  # region x time x subj

for s, subj in enumerate(subjects):
    for l, lab in enumerate(labels):
        if "7Networks" in lab:
            timeseries[-parc:, :, s] = np.genfromtxt(datapath+'sub_{}_schaefer'.format(str(subj))
                                                     + str(parc) + '.txt')
            break
        timeseries[l, :, s] = np.genfromtxt(datapath+'sub_{}_{}.txt'.format(str(subj), lab))

np.save(datapath+'timeseries_brainstem_schaefer'
        + str(parc) + '.npy',
        timeseries)

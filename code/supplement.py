"""
Supplement
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools.plotting import plot_point_brain
import scipy.io
from scipy.stats import spearmanr
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from palettable.colorbrewer.sequential import PuBuGn_9
import random


"""
set up
"""

path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/\
proj_brainstem/data/"

parc = 400

# load FC matlab file
fc_matlab = scipy.io.loadmat(datapath+'brainstem_fc/parcellated/Schaefer'
                             + str(parc)
                             + '/mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO']

# load region info file
info = pd.read_csv(path+'data/region_info_Schaefer'
                   + str(parc) + '.csv',
                   index_col=0)

# handy indices
idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_bc = np.concatenate((idx_bstem, idx_ctx))


"""
reliability of FC
"""

nsplit = 100
rho = dict([])
rho['fc'] = np.zeros((nsplit, ))
rho['bstem_hubs'] = np.zeros((nsplit, ))
rho['ctx_hubs'] = np.zeros((nsplit, ))
mask = np.triu(np.ones(len(idx_bc)), 1) > 0
for i in range(nsplit):
    print(i)
    idxA = np.array(random.sample(range(fc.shape[2]), 10))
    idxB = np.array(list(set(range(fc.shape[2])).difference(set(idxA))))
    fcA = np.mean(fc[:, :, idxA], axis=2)
    fcB = np.mean(fc[:, :, idxB], axis=2)
    rho['fc'][i] = spearmanr(fcA[np.ix_(idx_bc, idx_bc)][mask],
                             fcB[np.ix_(idx_bc, idx_bc)][mask])[0]
    rho['bstem_hubs'][i] = spearmanr(np.sum(fcA[np.ix_(idx_bstem, idx_ctx)], axis=1),
                                     np.sum(fcB[np.ix_(idx_bstem, idx_ctx)], axis=1))[0]
    rho['ctx_hubs'][i] = spearmanr(np.sum(fcA[np.ix_(idx_ctx, idx_bstem)], axis=1),
                                   np.sum(fcB[np.ix_(idx_ctx, idx_bstem)], axis=1))[0]

fig, ax = plt.subplots()
sns.violinplot(data=np.array(list(rho.values())).T)
ax.set(ylabel="spearman r")
ax.set_xticklabels(rho.keys())
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/violin_splithalf.eps')



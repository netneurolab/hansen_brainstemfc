"""
Supplement
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from scipy.stats import spearmanr
import pandas as pd
from palettable.colorbrewer.sequential import PuBuGn_9
import random


"""
set up
"""

# path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
# datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/\
# proj_brainstem/data/"
path = '/home/jhansen/gitrepos/hansen_brainstemfc/'
datapath='/home/jhansen/data-2/brainstem/'

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


"""
Replication in 3T
"""

timeseries_3T = np.load(datapath + "brainstem_fc/ave_tc_mcgill_3T/timeseries_brainstem3T_schaefer400.npy")
nnodes, ntime, nsubj = timeseries_3T.shape
fc3T = np.zeros((nnodes, nnodes, nsubj))
for s in range(nsubj):
    fc3T[:, :, s] = np.corrcoef(timeseries_3T[:, :, s])
fc3T = np.mean(fc3T, axis=2)

# plot heatmap and histograms
fig, ax = plt.subplots()
sns.heatmap(fc3T,
            vmin=0,
            vmax=np.max(abs(fc3T)),
            cmap=PuBuGn_9.mpl_colormap,
            square=True,
            xticklabels=False,
            yticklabels=False,
            rasterized=True)
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_histogram_fc3T.eps')

fclist = [fc3T[:len(idx_bstem), :len(idx_bstem)],
          fc3T[:len(idx_bstem), -parc:],
          fc3T[-parc:, -parc:]]
fig, ax = plt.subplots()

for i, f in enumerate(fclist):
    sns.kdeplot(f[np.triu_indices(len(f), k=1)],
                ax=ax)
ax.set_xlabel("FC")
ax.legend(["bstem only", "bstem to ctx", "ctx only"])
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/kdeplot_fc3T.eps')

# correlate with 7T data
fc7T = np.mean(fc, axis=2)
fc7T = fc7T[np.ix_(idx_bc, idx_bc)]

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax = ax.ravel()
# within cortex
mask = np.triu(np.ones(parc), 1) > 0
x = fc7T[-parc:, -parc:][mask]
y = fc3T[-parc:, -parc:][mask]
ax[0].scatter(x, y, s=0.6, c='#de7eaf', rasterized=True)
ax[0].set_xlabel('7T ctx')
ax[0].set_ylabel('3T ctx')
rho, pval = spearmanr(x, y)
ax[0].set_title('r = ' + str(rho)[:5] + ', p = ' + str(pval)[:5])

# within brainstem
mask = np.triu(np.ones(len(idx_bstem)), 1) > 0
x = fc7T[:len(idx_bstem), :len(idx_bstem)][mask]
y = fc3T[:len(idx_bstem), :len(idx_bstem)][mask]
ax[1].scatter(x, y, s=0.6, c='#007060', rasterized=True)
ax[1].set_xlabel('7T bstem')
ax[1].set_ylabel('3T bstem')
rho, pval = spearmanr(x, y)
ax[1].set_title('r = ' + str(rho)[:5] + ', p = ' + str(pval)[:5])

# brainstem --> cortex
x = fc7T[:len(idx_bstem), -parc:].flatten()
y = fc3T[:len(idx_bstem), -parc:].flatten()
ax[2].scatter(x, y, s=0.6, c='#6d91cb', rasterized=True)
ax[2].set_xlabel('7T bstem-ctx')
ax[2].set_ylabel('3T bstem-ctx')
rho, pval = spearmanr(x, y)
ax[2].set_title('r = ' + str(rho)[:5] + ', p = ' + str(pval)[:5])

fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/scatter_3Tvs7T.eps')

# within subject
rhobstem = np.zeros((nsubj, ))
rhoctx = np.zeros((nsubj, ))
fc3Tbstem = fc3T[:58, :58, :]
fc3Tctx = fc3T[-parc:, -parc:, :]
fc7Tbstem = fc[:58, :58, :]
fc7Tctx = fc[np.ix_(idx_ctx, idx_ctx, np.arange(fc.shape[2]))]
for i in range(nsubj):
    rhobstem[i] = spearmanr(fc3Tbstem[:, :, i][np.triu_indices(len(idx_bstem), 1)],
                            fc7Tbstem[:, :, i][np.triu_indices(len(idx_bstem), 1)])[0]
    rhoctx[i] = spearmanr(fc3Tctx[:, :, i][np.triu_indices(parc, 1)],
                          fc7Tctx[:, :, i][np.triu_indices(parc, 1)])[0]

# plot
fig, ax = plt.subplots()
for i in range(nsubj):
    ax.scatter(rhobstem[i], rhoctx[i])
    ax.text(rhobstem[i], rhoctx[i], s=str(i+1), fontsize=8)
ax.set_xlabel('within-brainstem 3T vs 7T spearmanr')
ax.set_ylabel('within-cortex 3T vs 7T spearmanr')

# check all the group averages
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
sns.heatmap(fc3T[-parc:, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[0, 0],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc7T[-parc:, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[1, 0],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc3T[:58, :58],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[0, 1],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc7T[:58, :58],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[1, 1],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc3T[:58, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[0, 2],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc7T[:58, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[1, 2],
            xticklabels=False, yticklabels=False)
ax[0, 0].set_ylabel('3T')
ax[1, 0].set_ylabel('7T')
ax[0, 0].set_title('within cortex')
ax[0, 1].set_title('within brainstem')
ax[0, 2].set_title('brainstem-cortex')
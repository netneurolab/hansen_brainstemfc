
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools.plotting import plot_point_brain
import scipy.io
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from palettable.colorbrewer.sequential import PuBuGn_9


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
fc = fc_matlab['C_BSwithHO_mean']

# load region info file
info = pd.read_csv(path+'data/region_info_Schaefer'
                   + str(parc) + '.csv',
                   index_col=0)

# handy indices
idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_bc = np.concatenate((idx_bstem, idx_ctx))

"""
toy brainstem plot
"""

plt.ion()
fig = plot_point_brain(data=np.concatenate((2*np.ones(len(idx_bstem), ),
                                            np.ones(len(idx_ctx), ))),
                       coords=info.query('structure == "brainstem" or \
                                         structure == "cortex"')
                       [['x', 'y', 'z']].values,
                       cmap=PuBuGn_9.mpl_colormap,
                       edgecolor=None,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(2.4, 2.4))
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_toy.eps')

"""
parcel size + tSNR
"""

# tSNR whole brain
fig = plot_point_brain(data=info.query("structure == 'brainstem'or \
                                       structure == 'cortex'")['tSNR'],
                       coords=info.query('structure == "brainstem" or \
                                         structure == "cortex"')
                       [['x', 'y', 'z']].values,
                       cmap=PuBuGn_9.mpl_colormap,
                       edgecolor=None, cbar=True,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(2.4, 2.4))
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_tsnr.eps')

# tSNR brainstem only
fig = plot_point_brain(data=info.query("structure == 'brainstem'")['tSNR'],
                       coords=info.query('structure == "brainstem"')
                       [['x', 'y', 'z']].values,
                       cmap=PuBuGn_9.mpl_colormap,
                       edgecolor=None,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(2.4, 2.4))
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_tsnr_bstem.eps')

# parcel size brainstem
fig = plot_point_brain(data=info.query("structure == 'brainstem'")['nvoxels'],
                       coords=info.query('structure == "brainstem"')
                       [['x', 'y', 'z']].values,
                       cmap=PuBuGn_9.mpl_colormap,
                       edgecolor=None,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(2.4, 2.4))
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_nvoxels_bstem.eps')

# compare
fig, ax = plt.subplots()
ax.scatter(info.query("structure == 'brainstem'")['nvoxels'],
           info.query("structure == 'brainstem'")['tSNR'])
ax.scatter(info.query("structure == 'cortex'")['nvoxels'],
           info.query("structure == 'cortex'")['tSNR'])
ax.set_xlabel('parcel size')
ax.set_ylabel('tSNR')
ax.legend(['brainstem', 'cortex'])
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/scatter_tsnr.eps')


"""
Heatmaps and histograms
"""

idx1 = info.query("structure == 'brainstem' | structure == 'cortex'").sort_values(by=['structure', 'hemisphere', 'rsn']).index.values
idx2 = info.query("structure == 'brainstem'").sort_values(by=['hemisphere']).index.values
idx3 = info.query("structure == 'cortex'").sort_values(by=['hemisphere', 'rsn']).index.values

fclist = [fc[np.ix_(idx1, idx1)],  # cortex + brainstem
          fc[np.ix_(idx2, idx2)],  # brainstem only
          fc[np.ix_(idx2, idx3)]   # brainstem --> cortex
          ]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig2, ax = plt.subplots()
for i, f in enumerate(fclist):
    sns.heatmap(f,
                vmin=0,
                vmax=np.max(abs(f)),
                cmap=PuBuGn_9.mpl_colormap,
                square=True,
                xticklabels=False,
                yticklabels=False,
                ax=axs[0, i],
                rasterized=True)
    sns.histplot(f[np.triu_indices(len(f), k=1)],
                 ax=axs[1, i])
    axs[1, i].set_xlabel('fc')
    sns.kdeplot(f[np.triu_indices(len(f), k=1)],
                ax=ax)
axs[0, 0].set_xlabel("both")
axs[0, 0].set_ylabel("both")
axs[0, 1].set_xlabel("brainstem")
axs[0, 1].set_ylabel("brainstem")
axs[0, 2].set_xlabel("cortex")
axs[0, 2].set_ylabel("brainstem")
ax.set_xlabel("FC")
ax.legend(["both", "bstem only", "bstem to ctx"])
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_histogram_fc.eps')
fig2.savefig(path+'figures/eps/Schaefer' + str(parc) + '/kdeplot_fc.eps')


"""
Distance
"""

eu = squareform(pdist(info[['x', 'y', 'z']].values))
fig, ax = plt.subplots()
ax.set_xlabel('euclidean distance')
ax.set_ylabel('fc')
ax.scatter(eu[np.ix_(idx_ctx, idx_ctx)][np.triu_indices(len(idx_ctx), k=1)],
           fc[np.ix_(idx_ctx, idx_ctx)][np.triu_indices(len(idx_ctx), k=1)],
           s=1, edgecolors=None, c='#de7eaf', rasterized=True
           )  # or 'orange' ?
ax.scatter(eu[np.ix_(idx_ctx, idx_bstem)].flatten(),
           fc[np.ix_(idx_ctx, idx_bstem)].flatten(),
           s=1, edgecolors=None, c='cornflowerblue', rasterized=True
           )
ax.scatter(eu[np.ix_(idx_bstem, idx_bstem)][np.triu_indices(len(idx_bstem),
                                                            k=1)],
           fc[np.ix_(idx_bstem, idx_bstem)][np.triu_indices(len(idx_bstem),
                                                            k=1)],
           s=1, edgecolors=None, c='darkgreen', rasterized=True
           )
ax.legend(['within ctx', 'within bstem', 'ctx-bstem'])
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/scatter_distance.eps')
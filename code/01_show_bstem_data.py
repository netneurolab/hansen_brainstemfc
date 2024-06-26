
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools.plotting import plot_point_brain, plot_fsaverage
from netneurotools.datasets import fetch_schaefer2018
import scipy.io
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import ttest_ind
from palettable.colorbrewer.sequential import PuBuGn_9


"""
set up
"""

path = '/home/jhansen/gitrepos/hansen_brainstemfc/'
datapath = '/home/jhansen/data-2/brainstem/'

parc = 400
annot = fetch_schaefer2018('fsaverage')[str(parc) + 'Parcels7Networks']

# load FC matlab file (not on github, too big)
fc_matlab = scipy.io.loadmat(datapath+'brainstem_fc/parcellated/Schaefer'
                             + str(parc)
                             + '/mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']
np.save(path
        + 'data/brainstemfc_mean_corrcoeff_full_Schaefer{}.npy'.format(parc),
        fc)

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
parcel size + tSNR (supplement)
"""

# tSNR whole brain
fig = plot_point_brain(data=info.query("structure == 'brainstem' or \
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

# tSNR cortical surface only
brain = plot_fsaverage(data=info.query("structure == 'cortex'")['tSNR'],
                       lhannot=annot.lh, rhannot=annot.rh,
                       vmin=info['tSNR'].min(),
                       vmax=info['tSNR'].max(),
                       colormap=PuBuGn_9.mpl_colormap,
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe",
                                 'line_width': 4.0})
brain.save_image(path+'figures/eps/Schaefer' + str(parc)
                 + '/surface_ctx_tsnr.eps')

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

idx1 = info.query("structure == 'brainstem' | structure == 'cortex'").\
    sort_values(by=['structure', 'hemisphere', 'rsn']).index.values
idx2 = info.query("structure == 'brainstem'").\
    sort_values(by=['hemisphere']).index.values
idx3 = info.query("structure == 'cortex'").\
    sort_values(by=['hemisphere', 'rsn']).index.values

# FC heatmaps
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, f in enumerate([fc[np.ix_(idx1, idx1)],     # cortex + brainstem
                       fc[np.ix_(idx2, idx2)],     # brainstem only
                       fc[np.ix_(idx2, idx3)]]):   # brainstem --> cortex
    sns.heatmap(f,
                vmin=0,
                vmax=np.max(abs(f)),
                cmap=PuBuGn_9.mpl_colormap,
                square=True,
                xticklabels=False,
                yticklabels=False,
                ax=axs[i],
                rasterized=True)
axs[0].set_xlabel("both")
axs[0].set_ylabel("both")
axs[1].set_xlabel("brainstem")
axs[1].set_ylabel("brainstem")
axs[2].set_xlabel("cortex")
axs[2].set_ylabel("brainstem")
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_fc.eps')

# FC histograms
fig, ax = plt.subplots()
for i, f in enumerate([fc[np.ix_(idx2, idx2)],     # brainstem only
                       fc[np.ix_(idx2, idx3)],     # brainstem --> cortex
                       fc[np.ix_(idx3, idx3)]]):   # cortex only
    if f.shape[0] == f.shape[1]:  # if square, plot upper triangle
        sns.kdeplot(f[np.triu_indices(len(f), k=1)],
                    ax=ax)
    else:  # else, plot all elements
        sns.kdeplot(f.flatten(), ax=ax)
ax.set_xlabel("FC")
ax.legend(["bstem only", "bstem to ctx", "ctx only"])
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/kdeplot_fc.eps')

# t-test: is bstem-ctx fc > bstem-bstem fc?
t, p = ttest_ind(fc[np.ix_(idx2, idx3)].flatten(),
                 fc[np.ix_(idx2, idx2)][np.triu_indices(len(idx2), k=1)],
                 equal_var=False)
# Ttest_indResult(statistic=33.91973991912895, pvalue=4.8001996581462265e-198)

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

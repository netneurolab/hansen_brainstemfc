
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.datasets import fetch_atlas_schaefer_2018
from scipy.stats import pearsonr, spearmanr, zscore
from palettable.colorbrewer.sequential import PuBuGn_9
from netneurotools import datasets
from netneurotools.plotting import plot_point_brain, plot_fsaverage
from netneurotools.stats import gen_spinsamples
from neuromaps.datasets import fetch_annotation
from neuromaps.parcellate import Parcellater
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
import scipy.io
import pandas as pd
import math


def get_rmssd(ts):
    n = len(ts) - 1
    return math.sqrt(np.sum(np.array([(ts[i+1] - ts[i])**2 for i in range(n)])) / (2*n))


def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared


def corr_spin(x, y, spins, nspins):
    rho, _ = spearmanr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = spearmanr(x[spins[:, i]], y)

    pval = (1 + sum(abs(null) > abs(rho))) / (nspins + 1)
    return rho, pval, null


"""
set up
"""

path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/\
proj_brainstem/data/"

parc = 400

schaefer = fetch_atlas_schaefer_2018(n_rois=parc)
annot = datasets.fetch_schaefer2018('fsaverage')[str(parc) + 'Parcels7Networks']

# make spins
# nnodes = len(schaefer['labels'])
# coords = np.genfromtxt(path+'data/coords/Schaefer'
#                        + str(parc) + '_coords.txt')[:, 1:]
# hemiid = np.zeros((nnodes, ))
# hemiid[:int(nnodes/2)] = 1
# nspins = 10000
# spins = gen_spinsamples(coords, hemiid, n_rotate=nspins, method='hungarian', seed=1234)

# load spins
spins = np.load(path+'data/spins_schaefer' + str(parc) + '_hungarian.npy')
nspins = spins.shape[1]

# load FC matlab file
fc_matlab = scipy.io.loadmat(datapath+'brainstem_fc/parcellated/Schaefer'
                             + str(parc) + '/mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO_mean']

# load region info file
info = pd.read_csv(path+'data/region_info_Schaefer'
                   + str(parc) + '.csv', index_col=0)

# handy indices
idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_bc = np.concatenate((idx_bstem, idx_ctx))

"""
Calculate hubness ("strength" or "weighted degree")
"""

strength = dict([])
strength['bstem_bstem'] = np.sum(fc[np.ix_(idx_bstem, idx_bstem)], axis=1)
strength['bstem_ctx'] = np.sum(fc[np.ix_(idx_bstem, idx_ctx)], axis=1)
strength['bstem_all'] = np.sum(fc[np.ix_(idx_bstem, np.concatenate((idx_bstem, idx_ctx)))], axis=1)
strength['ctx_ctx'] = np.sum(fc[np.ix_(idx_ctx, idx_ctx)], axis=1)
strength['ctx_bstem'] = np.sum(fc[np.ix_(idx_ctx, idx_bstem)], axis=1)
strength['ctx_all'] = np.sum(fc[np.ix_(idx_ctx, np.concatenate((idx_bstem, idx_ctx)))], axis=1)

"""
brainstem --> cortex strength
"""

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(projection='3d'))
fc_bstem = fc[np.ix_(idx_bstem, idx_bstem)]
fc_bstem_flat = fc_bstem[np.triu_indices(len(fc_bstem), k=1)]
thresh = np.flipud(np.sort(fc_bstem_flat))[int(np.floor(0.05 * len(fc_bstem_flat)))]
edges = np.where(np.triu(fc_bstem, k=1) > thresh)
coords = info.query('structure == "brainstem"')[['x', 'y', 'z']]
for edge_i, edge_j in zip(edges[0], edges[1]):
    x1 = coords.values[edge_i, 0]
    x2 = coords.values[edge_j, 0]
    y1 = coords.values[edge_i, 1]
    y2 = coords.values[edge_j, 1]
    z1 = coords.values[edge_i, 2]
    z2 = coords.values[edge_j, 2]
    ax.plot([x1, x2], [y1, y2], [z1, z2],
            linewidth=1, c='k', alpha=0.5, zorder=0)
ax.scatter(*coords.T.values,
           s=strength["bstem_ctx"]**1.3,  # 1.2 for schaefer400, 1.8 for schaefer100
           c=strength["bstem_ctx"],
           cmap=PuBuGn_9.mpl_colormap,
           edgecolors=None)
ax.axis('off')
ax.view_init(0, -90)
scaling = np.array([ax.get_xlim(),
                    ax.get_ylim(),
                    ax.get_zlim()])
ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/pointbrain_bstem_hub_network_coronal.eps')
ax.view_init(0, 180)
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/pointbrain_bstem_hub_network_sag.eps')
ax.view_init(90, 180)
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/pointbrain_bstem_hub_network_axial.eps')

# compare to tSNR
r, p = spearmanr(info.query("structure == 'brainstem'")['tSNR'], strength['bstem_ctx'])

"""
cortex --> brainstem strength
"""

brain = plot_fsaverage(data=strength['ctx_bstem'],
                       lhannot=annot.lh, rhannot=annot.rh,
                       colormap=PuBuGn_9.mpl_colormap,
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/eps/Schaefer' + str(parc) + '/surface_ctx_bstem_strength.eps')

# cortex --> cortex strength
brain = plot_fsaverage(data=strength['ctx_ctx'],
                       lhannot=annot.lh, rhannot=annot.rh,
                       colormap=PuBuGn_9.mpl_colormap,
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/eps/Schaefer' + str(parc) + '/surface_ctx_ctx_strength.eps')

# compare
fig, ax = plt.subplots()
r, p, _ = corr_spin(strength['ctx_bstem'], strength['ctx_ctx'], spins, nspins)
ax.scatter(strength['ctx_bstem'], strength['ctx_ctx'])
ax.set_xlabel('ctx-bstem strength')
ax.set_ylabel('ctx-ctx strength')
ax.set_title('r = ' + str(r)[:5] + ', pspin = ' + str(p)[:5])
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/scatter_ctx_strength.eps')

"""
laminar networks
"""

me = np.genfromtxt(path + 'data/mesulam_schaefer400.csv')
ve = np.genfromtxt(path + 'data/voneconomo_schaefer400.csv')

ve_names = ['pm', 'assoc1', 'assoc2', 'pss', 'ps', 'lim', 'ins']
ve_order = np.array([7, 6, 2, 3, 4, 1, 5])
me_names = ['plmb', 'het', 'uni', 'idio']

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
sns.violinplot(x=me, y=strength['ctx_bstem'],  color=".8", ax=axs[0])
axs[0].set_xticklabels(me_names)
axs[0].set_ylabel('ctx weighted degree')
sns.violinplot(x=ve, y=strength['ctx_bstem'], order=ve_order, color=".8", ax=axs[1])
axs[1].set_xticklabels([ve_names[i-1] for i in ve_order])
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/violin_mesulam+voneconomo.eps')


"""
MEG dynamics
"""

# fetch and parcellate meg maps
megmaps = fetch_annotation(source='hcps1200', den='4k')
s_parc = (path + 'data/Schaefer' + str(parc) + '_L.4k.label.gii',
          path + 'data/Schaefer' + str(parc) + '_R.4k.label.gii')
parcellater = Parcellater(s_parc, 'fsLR')
megmaps_parc = dict([])
for (src, desc, space, den) in megmaps.keys():
    megmaps_parc[desc] = parcellater.fit_transform(megmaps[(src, desc, space, den)], 'fsLR')

# plot correlation coefficients
rhopspin = np.array([corr_spin(strength['ctx_bstem'],
                               megmaps_parc[desc],
                               spins,
                               nspins)[:2] for desc in megmaps_parc.keys()])
rhopspin[:, 1] = multipletests(rhopspin[:, 1], method='fdr_bh')[1]
reorderidx = np.array([2, 5, 0, 1, 3, 4, 6])  # order meg maps by freq
fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.barh(np.arange(len(rhopspin)), abs(rhopspin[reorderidx, 0]), tick_label=[list(megmaps_parc.keys())[i] for i in reorderidx])
ax.plot(0.1 * np.ones((np.sum(rhopspin[reorderidx, 1] < 0.05), )),
        np.arange(len(rhopspin))[np.where(rhopspin[reorderidx, 1] < 0.05)[0]],
        '*', c='k')
ax.set_xlabel('spearman r with ctx --> bstem strength')
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/barh_megmaps.eps')

# plot individual scatter plots
fig, ax = plt.subplots(1, len(rhopspin), figsize=(20, 3))
ax = ax.ravel()
for i, desc in enumerate(megmaps_parc.keys()):
    ax[reorderidx[i]].scatter(megmaps_parc[desc], strength['ctx_bstem'], s=3)
    ax[reorderidx[i]].set_xlabel(desc)
    ax[reorderidx[i]].set_ylabel('weighted degree')
    ax[reorderidx[i]].set_title('r = ' 
                                + str(rhopspin[i, 0])[:5] 
                                + ', pspin = ' 
                                + str(rhopspin[i, 1])[:5])
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/scatter_megmaps.eps')

# plot MEG surfaces
for desc in megmaps_parc.keys():
    brain = plot_fsaverage(data=megmaps_parc[desc],
                           lhannot=annot.lh, rhannot=annot.rh,
                           colormap=PuBuGn_9.mpl_colors,
                           vmin=megmaps_parc[desc].min(),
                           vmax=megmaps_parc[desc].max(),
                           views=['lat', 'med'],
                           data_kws={'representation': "wireframe"})
    brain.save_image(path+'figures/eps/Schaefer' + str(parc) + '/surface_' + desc + '.eps')


"""
receptors
"""

recpath = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/github/hansen_receptors/data/PET_parcellated/scale" + str(parc) + "/"

rec_cols = ['5HT1a_cumi_hc8_beliveau',
            '5HT1b_p943_hc65_gallezot',
            '5HT2a_cimbi_hc29_beliveau',
            '5HT4_sb20_hc59_beliveau',
            '5HT6_gsk_hc30_radhakrishnan',
            '5HTT_dasb_hc100_beliveau',
            'A4B2_flubatine_hc30_hillmer',
            'CB1_omar_hc77_normandin',
            'D2_flb457_hc55_sandiego',
            'DAT_fepe2i_hc6_sasaki',
            'GABAa-bz_flumazenil_hc16_norgaard',
            'H3_cban_hc8_gallezot',
            'M1_lsn_hc24_naganawa',
            'mGluR5_abp_hc28_dubois',
            'MU_carfentanil_hc39_turtonen',
            'NAT_MRB_hc77_ding',
            'NMDA_ge179_hc29_galovic',
            'VAChT_feobv_hc18_aghourian_sum']

receptor_ctx = dict([])
for rec in rec_cols:
    receptor_ctx[rec] = np.genfromtxt(recpath+rec+'.csv', delimiter=',')
receptor_ctx = pd.DataFrame(data=receptor_ctx, index=info.query('structure == "cortex"')['labels'])

# bstem --> ctx
rsq = np.zeros((len(fc), ))
for i in range(len(fc)):
    if i < len(idx_bstem):
        rsq[i] = get_reg_r_sq(zscore(receptor_ctx), zscore(fc[i, idx_ctx]))
    else:
        rsq[i] = get_reg_r_sq(zscore(receptor_ctx), zscore(fc[i, idx_ctx]))

fig = plot_point_brain(data=rsq[idx_bstem],
                       coords=info.query("structure == 'brainstem'")[['x', 'y', 'z']].values,
                       size=strength['bstem_ctx']**0.9,  # 0.9 for schaefer400, 1.2 for schaefer100
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(2.4, 2.4),
                       cmap=PuBuGn_9.mpl_colormap, cbar=True,
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/pointbrain_receptor_rsq.eps')

plt.figure()
sns.kdeplot(rsq[idx_ctx], color='orange')
sns.kdeplot(rsq[idx_bstem], color='darkgreen')
plt.xlabel('rsq')
plt.legend(['cortex', 'brainstem'])
plt.tight_layout()
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/kdeplot_receptor_rsq.eps')


# ctx --> bstem
rsq = np.zeros((len(fc), ))
for i in range(len(fc)):
    rsq[i] = get_reg_r_sq(zscore(receptor_bstem[rec_cols], nan_policy='omit'), zscore(fc[i, idx_bstem]))

fig = plot_point_brain(data=rsq[idx_bstem],
                       coords=info.query("structure == 'brainstem'")[['x', 'y', 'z']].values,
                       size=strength['bstem_ctx']**0.9,  # 0.9 for schaefer400, 1.2 for schaefer100
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(2.4, 2.4),
                       cmap=PuBuGn_9.mpl_colormap, cbar=True,
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/pointbrain_receptor_rsq.eps')

plt.figure()
sns.kdeplot(rsq[idx_ctx], color='orange')
sns.kdeplot(rsq[idx_bstem], color='darkgreen')
plt.xlabel('rsq')
plt.legend(['cortex', 'brainstem'])
plt.tight_layout()
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/kdeplot_receptor_rsq.eps')
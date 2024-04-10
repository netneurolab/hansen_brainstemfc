"""
Supplement
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from scipy.stats import spearmanr
import pandas as pd
import random
from netneurotools.plotting import plot_point_brain, plot_fsaverage
from netneurotools.datasets import fetch_schaefer2018
from palettable.colorbrewer.sequential import (PuBuGn_9,
                                               PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from matplotlib.colors import LinearSegmentedColormap


def coefvar(x, axis):
    """
    coefficient of variation
    """
    return np.std(x, axis=axis) / np.mean(x, axis=axis)


def scale_values(values, vmin, vmax, axis=None):
    s = (values - values.min(axis=axis)) /\
        (values.max(axis=axis) - values.min(axis=axis))
    s = s * (vmax - vmin)
    s = s + vmin
    return s


def corrcoefspearman(matrix):
    """
    calculate spearman correlatoin of pairs of rows in `matrix`
    """
    n = matrix.shape[0]
    corrs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corrs[i, j] = spearmanr(matrix[i, :], matrix[j, :])[0]
    return corrs


"""
set up
"""

path = '/home/jhansen/gitrepos/hansen_brainstemfc/'
datapath = '/home/jhansen/data-2/brainstem/'

parc = 400
annot = fetch_schaefer2018('fsaverage')[str(parc) + 'Parcels7Networks']

# load FC matlab file
fc_matlab = scipy.io.loadmat(datapath+'brainstem_fc/parcellated/Schaefer'
                             + str(parc)
                             + '/mean_corrcoeff_full.mat')
fc = fc_matlab['C_BSwithHO']
timeseries_7T = np.load(datapath + 'brainstem_fc/timeseries/'
                        + 'timeseries_brainstem_schaefer{}.npy'.format(parc))

# thresholded fc
fcthresh = fc.copy()
fcthresh[fc_matlab['connectom_final'] == 0] = 0

# load region info file
info = pd.read_csv(path+'data/region_info_Schaefer'
                   + str(parc) + '.csv',
                   index_col=0)

# handy indices
idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_bc = np.concatenate((idx_bstem, idx_ctx))

# make diverging colourmap
teals = PuBuGn_4.mpl_colors
teals.reverse()
reds = PuRd_4.mpl_colors
teals.extend(reds)
teals[0] = PuBuGn_8.mpl_colors[-1]
del teals[4]
cmap = LinearSegmentedColormap.from_list('cmap', teals, N=256)


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
    rho['bstem_hubs'][i] = spearmanr(np.sum(fcA[np.ix_(idx_bstem, idx_ctx)],
                                            axis=1),
                                     np.sum(fcB[np.ix_(idx_bstem, idx_ctx)],
                                            axis=1))[0]
    rho['ctx_hubs'][i] = spearmanr(np.sum(fcA[np.ix_(idx_ctx, idx_bstem)],
                                          axis=1),
                                   np.sum(fcB[np.ix_(idx_ctx, idx_bstem)],
                                          axis=1))[0]

fig, ax = plt.subplots()
sns.violinplot(data=np.array(list(rho.values())).T)
ax.set(ylabel="spearman r")
ax.set_xticklabels(rho.keys())
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/violin_splithalf.eps')


"""
Replication in 3T
"""

timeseries_3T = np.load(datapath + "brainstem_fc/ave_tc_mcgill_3T/"
                        + "timeseries_brainstem3T_schaefer400.npy")
nnodes, ntime, nsubj = timeseries_3T.shape
fc3T = np.zeros((nnodes, nnodes, nsubj))
for s in range(nsubj):
    fc3T[:, :, s] = np.corrcoef(timeseries_3T[:, :, s])
fc3Tavg = np.mean(fc3T, axis=2)

# plot heatmap and histograms
fig, ax = plt.subplots()
sns.heatmap(fc3Tavg,
            vmin=0,
            vmax=np.max(abs(fc3Tavg)),
            cmap=PuBuGn_9.mpl_colormap,
            square=True,
            xticklabels=False,
            yticklabels=False,
            rasterized=True)
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/heatmap_histogram_fc3T.eps')

fclist = [fc3Tavg[:len(idx_bstem), :len(idx_bstem)],
          fc3Tavg[:len(idx_bstem), -parc:],
          fc3Tavg[-parc:, -parc:]]
fig, ax = plt.subplots()

for i, f in enumerate(fclist):
    sns.kdeplot(f[np.triu_indices(len(f), k=1)],
                ax=ax)
ax.set_xlabel("FC")
ax.legend(["bstem only", "bstem to ctx", "ctx only"])
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/kdeplot_fc3T.eps')

# correlate with 7T data
fc7Tavg = np.mean(fc, axis=2)
fc7Tavg = fc7Tavg[np.ix_(idx_bc, idx_bc)]

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax = ax.ravel()
# within cortex
mask = np.triu(np.ones(parc), 1) > 0
x = fc7Tavg[-parc:, -parc:][mask]
y = fc3Tavg[-parc:, -parc:][mask]
ax[0].scatter(x, y, s=0.6, c='#de7eaf', rasterized=True)
ax[0].set_xlabel('7T ctx')
ax[0].set_ylabel('3T ctx')
rho, pval = spearmanr(x, y)
ax[0].set_title('r = ' + str(rho)[:5] + ', p = ' + str(pval)[:5])

# within brainstem
mask = np.triu(np.ones(len(idx_bstem)), 1) > 0
x = fc7Tavg[:len(idx_bstem), :len(idx_bstem)][mask]
y = fc3Tavg[:len(idx_bstem), :len(idx_bstem)][mask]
ax[1].scatter(x, y, s=0.6, c='#007060', rasterized=True)
ax[1].set_xlabel('7T bstem')
ax[1].set_ylabel('3T bstem')
rho, pval = spearmanr(x, y)
ax[1].set_title('r = ' + str(rho)[:5] + ', p = ' + str(pval)[:5])

# brainstem --> cortex
x = fc7Tavg[:len(idx_bstem), -parc:].flatten()
y = fc3Tavg[:len(idx_bstem), -parc:].flatten()
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
    rhobstem[i] = spearmanr(fc3Tbstem[:, :, i][np.triu_indices(len(idx_bstem),
                                                               1)],
                            fc7Tbstem[:, :, i][np.triu_indices(len(idx_bstem),
                                                               1)])[0]
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
sns.heatmap(fc3Tavg[-parc:, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[0, 0],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc7Tavg[-parc:, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[1, 0],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc3Tavg[:58, :58],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[0, 1],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc7Tavg[:58, :58],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[1, 1],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc3Tavg[:58, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[0, 2],
            xticklabels=False, yticklabels=False)
sns.heatmap(fc7Tavg[:58, -parc:],
            cmap=PuBuGn_9.mpl_colormap,
            square=True, ax=ax[1, 2],
            xticklabels=False, yticklabels=False)
ax[0, 0].set_ylabel('3T')
ax[1, 0].set_ylabel('7T')
ax[0, 0].set_title('within cortex')
ax[0, 1].set_title('within brainstem')
ax[0, 2].set_title('brainstem-cortex')

"""
subject-level analyses
"""

# loop through and plot individual data for each subject
fig, axs = plt.subplots(4, 5, figsize=(20, 10))
axs = axs.ravel()
for i in range(nsubj):
    # FC
    sns.heatmap(fc[:, :, i][np.ix_(idx_bc, idx_bc)],
                cmap=PuBuGn_9.mpl_colormap,
                square=True, ax=axs[i],
                vmin=0, vmax=0.9,
                xticklabels=False, yticklabels=False,
                rasterized=True)
    axs[i].set_title('subj' + str(i+1))

    # brainstem hubs
    bhub = np.sum(fc[:, :, i][np.ix_(idx_bstem, idx_ctx)], axis=1)
    fig2 = plot_point_brain(bhub,
                            coords=info.query("structure == 'brainstem'")
                            [['x', 'y', 'z']].values,
                            size=scale_values(bhub, 2, 100) ** 1.3,
                            vmin=np.min(bhub), vmax=np.max(bhub),
                            views_orientation='horizontal',
                            views=['coronal_rev', 'sagittal', 'axial'],
                            views_size=(5, 5),
                            cmap=PuBuGn_9.mpl_colormap, cbar=True,
                            edgecolor=None)
    fig2.savefig(path+'figures/eps/Schaefer'
                 + str(parc) + '/pointbrain_brainstem_hub_subj' + str(i+1)
                 + '.eps')

    # cortex hubs
    chub = np.sum(fc[:, :, i][np.ix_(idx_bstem, idx_ctx)], axis=0)
    fig3 = plot_fsaverage(data=chub,
                          lhannot=annot.lh, rhannot=annot.rh,
                          colormap=PuBuGn_9.mpl_colormap,
                          vmin=np.min(chub), vmax=np.max(chub),
                          views=['lat', 'med'],
                          data_kws={'representation': "wireframe",
                                    'line_width': 4.0})
    fig3.save_image(path+'figures/eps/Schaefer'
                    + str(parc) + '/surface_ctxhub_subj' + str(i+1) + '.eps')

fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_fc_indiv.eps')

# standard deviation and coefficient of variation of FC
fcstd = np.std(fc[np.ix_(idx_bc, idx_bc)], axis=2)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(fcstd, cmap=PuBuGn_9.mpl_colormap, square=True,
            vmin=0, vmax=np.max(abs(fcstd)),
            xticklabels=False, yticklabels=False,
            ax=axs[0], rasterized=True)
sns.heatmap(coefvar(fc[np.ix_(idx_bc, idx_bc)], axis=2),
            cmap=PuBuGn_9.mpl_colormap, square=True,
            xticklabels=False, yticklabels=False,
            vmin=0, vmax=1, ax=axs[1], rasterized=True)
axs[0].set_title('std')
axs[1].set_title('std/mean')
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_fc_std_cv.eps')

# standard deviation of brainstem hubs
bstem_hub_std = np.std(np.array([np.sum(fc[:, :, subj][np.ix_(idx_bstem,
                                                              idx_ctx)],
                                        axis=1)
                                 for subj in range(fc.shape[2])]), axis=0)

fig = plot_point_brain(bstem_hub_std,
                       coords=info.query("structure == 'brainstem'")
                       [['x', 'y', 'z']].values,
                       size=scale_values(bstem_hub_std, 2, 100) ** 1.3,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(5, 5),
                       cmap=PuBuGn_9.mpl_colormap,
                       cbar=True,
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_brainstem_hub_std.eps')

# standard deviation of cortical hubs
ctx_hub_std = np.std(np.array([np.sum(fc[:, :, subj][np.ix_(idx_bstem,
                                                            idx_ctx)],
                                      axis=0)
                               for subj in range(fc.shape[2])]), axis=0)
brain = plot_fsaverage(data=ctx_hub_std,
                       lhannot=annot.lh, rhannot=annot.rh,
                       colormap=PuBuGn_9.mpl_colormap,
                       vmin=ctx_hub_std.min(),
                       vmax=ctx_hub_std.max(),
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe",
                                 'line_width': 4.0})
brain.save_image(path+'figures/eps/Schaefer'
                 + str(parc) + '/surface_ctxhub_std.eps')

# subject-level features (to plot pairwise correlations)
subj_fc_ctx = np.array([fc[:, :, i][np.ix_(idx_ctx, idx_ctx)][
    np.triu_indices(len(idx_ctx), 1)] for i in range(nsubj)])
subj_fc_bstem = np.array([fc[:, :, i][np.ix_(idx_bstem, idx_bstem)][
    np.triu_indices(len(idx_bstem), 1)] for i in range(nsubj)])
subj_fc_bc = np.array([fc[:, :, i][np.ix_(idx_bstem, idx_ctx)].flatten()
                       for i in range(nsubj)])
subj_bstem_ctx_strength = np.array([fc[:, :, i][np.ix_(idx_bstem,
                                                       idx_ctx)].sum(axis=1)
                                    for i in range(nsubj)])
subj_ctx_bstem_strength = np.array([fc[:, :, i][np.ix_(idx_ctx,
                                                       idx_bstem)].sum(axis=1)
                                    for i in range(nsubj)])

mats = [subj_fc_ctx, subj_fc_bstem, subj_fc_bc,
        subj_bstem_ctx_strength, subj_ctx_bstem_strength]
matnames = ['fc-ctx', 'fc-bstem', 'fc-bc', 'hubs-bstem', 'hubs-ctx']

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(len(mats)):
    matcorr = corrcoefspearman(mats[i])
    sns.stripplot(x=i, y=matcorr[np.triu_indices(nsubj, 1)], ax=ax,
                  jitter=0.2, alpha=0.5)
ax.set_xticklabels(matnames)
ax.set_ylabel('spearman r')
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/stripplot_subjcorrs.eps')


"""
# regress fc
fc_reg = np.zeros((len(idx_bstem), len(fc), nsubj))
for subj in range(nsubj):
    for i in range(len(fc)):
        subjstr = np.sum(fc[:, :, subj][np.ix_(idx_bstem, idx_ctx)], axis=1)
        fc_reg[:, i, subj] = np.squeeze(regress_out(subjstr.reshape(-1, 1),
                                                    fc[idx_bstem, i, subj].
                                                    reshape(-1, 1)))

subj_bstem_fcregsim = np.array([np.corrcoef(fc_reg[:, idx_ctx, subj])[
    np.triu_indices(len(idx_bstem), 1)] for subj in range(nsubj)])

subj_ctx_fcregsim = np.array([np.corrcoef(fc_reg[:, idx_ctx, subj].T)[
    np.triu_indices(len(idx_ctx), 1)] for subj in range(nsubj)])

# subject-specific community detection oh boy
out = Parallel(n_jobs=40)(delayed(community_detection)(
    np.corrcoef(fc_reg[:, idx_ctx, subj]),
    [x/10.0 for x in range(1, 61, 1)]) for subj in range(nsubj))

# saved as a .pkl dictionary; load it in:
with open(path + 'results/Schaefer400/community_detection/'
          + 'community_detection_subj.pkl', 'rb') as f:
    out = pickle.load(f)

# for each subject, find the gamma that maximizes zrand with group consensus
grpcon = np.load(path+'results/Schaefer400'
                 + '/community_detection/assignments_bstem.npy')[27, :]

gamma_idx = np.zeros((nsubj, ))
for subj in range(nsubj):
    # calculate zrand index of consensus assignments
    # for each gamma and group consensus
    zrands = np.array([zrand(out['subj' + str(subj+1)][0][:, g], grpcon)
                       for g in range(out['subj' + str(subj+1)][0].shape[1])])
    gamma_idx[subj] = np.argmax(zrands)

# plot each community assignment
for subj in range(nsubj):
    fig = plot_point_brain(out[subj][0][int(gamma_idx[subj]), :],
                           coords=info.query("structure == 'brainstem'")
                           [['x', 'y', 'z']].values,
                           size=np.sum(np.mean(fc, axis=2)[
                               np.idx_(idx_bstem, idx_ctx)]) ** 1.2,
                           views_orientation='horizontal',
                           views=['coronal_rev', 'sagittal', 'axial'],
                           views_size=(5, 5),
                           cmap='Accent', cbar=True,
                           edgecolor=None)

# get modularity of each individual with consensus assignment
q = np.zeros((nsubj, len(np.unique(grpcon))))
for subj in range(nsubj):
    q[subj, :] = get_modularity_sig(np.corrcoef(fc_reg[:, idx_ctx, subj]),
                                  grpcon,
                                  2.8)


# gradient analysis
gradients = [diffusion_map_embed(np.corrcoef(fc_reg[:, idx_ctx, subj].T),
                                 no_dims=1, alpha=0.5)[0]
             for subj in range(nsubj)]
subj_gradients = np.array(gradients).squeeze()
"""

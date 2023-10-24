
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools.plotting import (plot_point_brain,
                                    plot_mod_heatmap,
                                    plot_fsaverage)
from netneurotools.datasets import fetch_schaefer2018
from netneurotools.stats import gen_spinsamples
from nilearn.datasets import fetch_atlas_schaefer_2018
import scipy.io
from scipy.stats import spearmanr
import pandas as pd
from palettable.colorbrewer.sequential import (PuBuGn_9,
                                               PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


def regress_out(x, y):
    """
    remove the effect of a out of b
    """
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    yhat = lin_reg.predict(x)
    resid = y - yhat
    return resid


def corr_spin(x, y, spins, nspins):
    rho, _ = spearmanr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = spearmanr(x[spins[:, i]], y)

    pval = (1 + sum(abs(null) > abs(rho))) / (nspins + 1)
    return rho, pval, null


def diffusion_map_embed(X, no_dims=5, alpha=1, sigma=1):
    X = X - np.min(X)
    X = X / np.max(X)
    sum_X = np.sum(X ** 2, axis=1)
    K = np.exp(-1 * (sum_X.T + (sum_X.T - 2 * (X @ X.T).T).T) / (2 * sigma ** 2))

    p = np.sum(K, axis=0, keepdims=True).T
    K = np.divide(K, (p @ p.T) ** alpha)
    p = np.sqrt(np.sum(K, axis=0, keepdims=True)).T
    K = K / (p @ p.T)
    u, s, v = np.linalg.svd(K, full_matrices=False)
    from sklearn.utils.extmath import svd_flip
    u, v = svd_flip(u, v)
    u_norm = np.divide(u, u[:, 0].T)
    mapped_X = u_norm[:, 1:no_dims + 1]
    return mapped_X, u, s, v


"""
set up
"""

path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/\
proj_brainstem/data/"

parc = 400

# schaefer surface
schaefer = fetch_atlas_schaefer_2018(n_rois=parc)
annot = fetch_schaefer2018('fsaverage')[str(parc) + 'Parcels7Networks']

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
info['rsn'] = pd.Categorical(info['rsn'], categories=['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'])
info.reindex

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
regress out dominant FC pattern
"""

str_bstem_ctx = np.sum(fc[np.ix_(idx_bstem, idx_ctx)], axis=1)
fc_reg = np.zeros((len(idx_bstem), len(fc)))
for i in range(len(fc)):
    fc_reg[:, i] = np.squeeze(regress_out(str_bstem_ctx.reshape(-1, 1),
                                          fc[idx_bstem, i].reshape(-1, 1)))


"""
gradient of FC connectivity
"""

ax = plot_mod_heatmap(np.corrcoef(fc_reg[:, idx_ctx].T),
                      info.query("structure == 'cortex'")['rsn'],
                      cmap=cmap, vmin=-1, vmax=1,
                      xlabels=np.unique(info.query("structure == 'cortex'")['rsn']),
                      ylabels=np.unique(info.query("structure == 'cortex'")['rsn']),
                      rasterized=True)
ax.set_title('ctx-->bstem FC similarity')
plt.tight_layout()
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_fcregcorr.eps')

grad = 0  # gradient to focus on

# cortical FC gradient
fc_grad, u, s, v = diffusion_map_embed(fc[np.ix_(idx_ctx, idx_ctx)],
                                       no_dims=3, alpha=0.5)

# brainstem gradient
fc_grad_bstem, u, s, v = diffusion_map_embed(np.corrcoef(fc_reg[:, idx_ctx].T),
                                no_dims=3, alpha=0.5)

plt.figure()
h = sns.jointplot(x=fc_grad[:, 0], y=fc_grad_bstem[:, 0])
h.set_axis_labels("functional hierarchy", "brainstem gradient")
h.figure.tight_layout()
h.figure.savefig(path+'figures/eps/Schaefer' + str(parc) + '/jointplot_gradients.eps')

brain = plot_fsaverage(data=fc_grad_bstem[:, 0],
                   lhannot=annot.lh, rhannot=annot.rh,
                   colormap=cmap,
                   vmin=-np.max(np.abs(fc_grad_bstem[:, 0])),
                   vmax=np.max(np.abs(fc_grad_bstem[:, 0])),
                   views=['lat', 'med'],
                   data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/eps/Schaefer'
                 + str(parc) + '/surface_fc_grad_dmebstem.eps')

# brainstem profiles
data = np.sum(fc_reg[:, idx_ctx[fc_grad_bstem[:, 0] < 0]], axis=1)
fig = plot_point_brain(data,
                       coords=info.query('structure == "brainstem"')[['x', 'y', 'z']].values,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       cbar=True, size=str_bstem_ctx,
                       cmap=PuBuGn_9.mpl_colormap, edgecolor=None)
fig.suptitle('gradient negative')
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_bstemfc_ctxgrad'
            + str(grad+1) + '_neg.eps')

data = np.sum(fc_reg[:, idx_ctx[fc_grad_bstem[:, 0] > 0]], axis=1)
fig = plot_point_brain(data,
                       coords=info.query('structure == "brainstem"')[['x', 'y', 'z']].values,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       cbar=True, size=str_bstem_ctx,
                       cmap=PuBuGn_9.mpl_colormap, edgecolor=None)
fig.suptitle('gradient positive')
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_bstemfc_ctxgrad'
            + str(grad+1) + '_pos.eps')

# community detection (supplement)

gamma_range = [x/10.0 for x in range(1, 61, 1)]

assignments_ctx = np.load(path+'results/community_detection/Schaefer'
                            + str(parc) +'/assignments_ctx.npy').T

# plot heatmap with modules
idx = 5  # gamma = 0.6
fig = plot_mod_heatmap(data=np.corrcoef(fc_reg[:, idx_ctx].T),
                       communities=assignments_ctx[idx, :],
                       cmap=cmap, vmin=-0.9, vmax=0.9,
                       rasterized=True)
plt.title('gamma = ' + str(gamma_range[idx]))
plt.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/heatmap_ctx_communities_'
            + str(idx) + '.eps')

# plot communities on cortex
brain = plot_fsaverage(data=assignments_ctx[idx, :],
                       lhannot=annot.lh, rhannot=annot.rh,
                       colormap=cmap,
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/eps/Schaefer'
                 + str(parc) + '/surface_ctx_communities_gamma_'
                 + str(idx) + '.eps')

# community-specific brainstem FC patterns
for i in np.unique(assignments_ctx[idx, :]):
    data = np.sum(fc_reg[:, idx_ctx[assignments_ctx[idx, :] == i]], axis=1)
    fig = plot_point_brain(data,
                           coords=info.query("structure == 'brainstem'")[['x', 'y', 'z']].values,
                           size=str_bstem_ctx ** 1.2,
                           views_orientation='horizontal',
                           views=['coronal_rev', 'sagittal', 'axial'],
                           views_size=(5, 5),
                           cmap=PuBuGn_9.mpl_colormap, cbar=True,
                           edgecolor=None)
    fig.savefig(path+'figures/eps/Schaefer'
                + str(parc) + '/pointbrain_ctx_community_'
                + str(i) + '_gamma_' + str(idx) + '.eps')

# mean-variance plot and number of communities
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(m, c='red')
ax[0].set_ylabel('mean', c='red')
ax[0].tick_params(axis='y', labelcolor='red')
ax[0].set_xticks(np.arange(-1, 60, 10))
xticklabels = gamma_range[9::10]
xticklabels.insert(0, 0.0)
ax[0].set_xticklabels(xticklabels)
ax2=ax[0].twinx()
ax2.plot(v, c='blue')
ax2.set_ylabel('var', c='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax[0].vlines(x=5, ymin=np.min(m), ymax=np.max(m))
ax[0].set_xlabel('gamma')
ax[1].plot(np.max(assignments_ctx, axis=1))
ax[1].set_ylabel("number of communities")
ax[1].set_xticks(np.arange(-1, 60, 10))
ax[1].set_xticklabels(xticklabels)
ax[1].set_xlabel('gamma')
ax[1].vlines(x=5, ymin=0, ymax=42)
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/plot_community_meanvar_ctx.eps')
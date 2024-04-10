"""
extension to the diencephalon/subcortex
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools.plotting import (plot_point_brain,
                                    plot_mod_heatmap,
                                    plot_fsaverage)
from netneurotools.datasets import fetch_schaefer2018
import scipy.io
import pandas as pd
from palettable.colorbrewer.sequential import PuBuGn_9
from enigmatoolbox.plotting import plot_subcortical
from netneurotools.modularity import consensus_modularity
from sklearn.linear_model import LinearRegression
from palettable.colorbrewer.sequential import (PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


def regress_out(x, y):
    """
    remove the effect of a out of b
    """
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    yhat = lin_reg.predict(x)
    resid = y - yhat
    return resid


def community_detection(A, gamma_range):
    nnodes = len(A)
    ngamma = len(gamma_range)
    consensus = np.zeros((nnodes, ngamma))
    qall = []
    zrand = []
    i = 0
    for g in gamma_range:
        print(g)
        consensus[:, i], q, z = consensus_modularity(A, g, B='negative_asym')
        qall.append(q)
        zrand.append(z)
        i += 1
    return (consensus, qall, zrand)


def diffusion_map_embed(X, no_dims=5, alpha=1, sigma=1):
    X = X - np.min(X)
    X = X / np.max(X)
    sum_X = np.sum(X ** 2, axis=1)
    K = np.exp(-1 * (sum_X.T + (sum_X.T - 2 * (X @ X.T).T).T) /
               (2 * sigma ** 2))

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

path = '/home/jhansen/gitrepos/hansen_brainstemfc/'
datapath = '/home/jhansen/data-2/brainstem/'

parc = 400
annot = fetch_schaefer2018('fsaverage')[str(parc) + 'Parcels7Networks']

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
# freesurfer subcortex
idx_subc = info.query("structure == 'subcortex'\
                      and\
                      labels.str.contains('Cereb-Ctx') == False").index.values
# brainstem navigator diencephalic nuclei
idx_dien = info.query("structure == 'diencephalon'").index.values
idx_bd = np.concatenate((idx_bstem, idx_dien))
idx_notctx = np.concatenate((idx_bstem, idx_dien, idx_subc))

# make diverging colourmap
teals = PuBuGn_4.mpl_colors
teals.reverse()
reds = PuRd_4.mpl_colors
teals.extend(reds)
teals[0] = PuBuGn_8.mpl_colors[-1]
del teals[4]
cmap = LinearSegmentedColormap.from_list('cmap', teals, N=256)

"""
plot the FC matrix
"""

orderidx = info.sort_values(by=['structure', 'hemisphere', 'rsn']).index.values
fig, ax = plt.subplots()
sns.heatmap(fc[np.ix_(orderidx, orderidx)],
            vmin=0, vmax=1,
            cmap=PuBuGn_9.mpl_colormap,
            square=True,
            xticklabels=False,
            yticklabels=False,
            rasterized=True)

"""
hubs
"""

strength = dict([])
strength['bstem_subc'] = np.sum(fc[np.ix_(idx_bstem, idx_subc)], axis=1)
strength['subc_bstem'] = np.sum(fc[np.ix_(idx_subc, idx_bstem)], axis=1)
strength['bd_bstem'] = np.sum(fc[np.ix_(idx_bd, idx_bstem)], axis=1)

# diencephalon --> brainstem
fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(projection='3d'))
fc_bd = fc[np.ix_(idx_bd, idx_bd)]  # brainstem + diencephalon FC
fc_bd_flat = fc_bd[np.triu_indices(len(fc_bd), k=1)]
thresh = np.flipud(np.sort(fc_bd_flat))[int(np.floor(0.05 * len(fc_bd_flat)))]
edges = np.where(np.triu(fc_bd, k=1) > thresh)
coords = info.query('structure == "brainstem" or\
                     structure == "diencephalon"')[['x', 'y', 'z']]
for edge_i, edge_j in zip(edges[0], edges[1]):
    x1 = coords.values[edge_i, 0]
    x2 = coords.values[edge_j, 0]
    y1 = coords.values[edge_i, 1]
    y2 = coords.values[edge_j, 1]
    z1 = coords.values[edge_i, 2]
    z2 = coords.values[edge_j, 2]
    ax.plot([x1, x2], [y1, y2], [z1, z2],
            linewidth=1, c='k', alpha=0.5, zorder=0)
# 1.2 for schaefer400, 1.8 for schaefer100
ax.scatter(*coords.T.values,
           s=strength["bd_bstem"]**2.1,
           c=strength["bd_bstem"],
           cmap=PuBuGn_9.mpl_colormap,
           edgecolors=None)
ax.axis('off')
ax.view_init(0, -90)
scaling = np.array([ax.get_xlim(),
                    ax.get_ylim(),
                    ax.get_zlim()])
ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))
plt.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/pointbrain_bstemdien_hub_network_coronal.eps')
ax.view_init(0, 180)
plt.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/pointbrain_bstemdien_hub_network_sag.eps')
ax.view_init(90, 180)
plt.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/pointbrain_bstemdien_hub_network_axial.eps')

# subcortex --> brainstem
enigma_reorder_idx = np.array([0, 1, 5, 2, 3, 4, 6, 13, 12, 8, 11, 10, 9, 7])
plot_subcortical(strength['subc_bstem'][enigma_reorder_idx],
                 ventricles=False, cmap='PuBuGn',
                 size=(1800, 600), screenshot=True, transparent_bg=True,
                 filename=path+'figures/eps/Schaefer' + str(parc)
                 + '/surfacesubc_bstemhubs.png')


"""
communities
"""

# regress out dominant subcortical connectivity pattern
dom = np.sum(fc[np.ix_(idx_notctx, idx_ctx)], axis=1)
fc_reg = np.zeros((len(idx_notctx), len(fc)))
for i in range(len(fc)):
    fc_reg[:, i] = np.squeeze(regress_out(dom.reshape(-1, 1),
                                          fc[idx_notctx, i].reshape(-1, 1)))

# community detection
gamma_range = [x/10.0 for x in range(1, 61, 1)]
# consensus, qall, zrand = community_detection(np.corrcoef(fc_reg[:, idx_ctx]),
#                                              gamma_range)

assignments = np.load(path + 'results/Schaefer' + str(parc)
                      + '/community_detection_wholebrain/assignments.npy')

idx = 18  # gamma = 1.9

fig = plot_mod_heatmap(data=np.corrcoef(fc_reg[:, idx_ctx]),
                       communities=assignments[idx, :],
                       cmap=cmap, vmin=-0.9, vmax=0.9,
                       rasterized=True)
plt.title('gamma = ' + str(gamma_range[idx]))
plt.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/heatmap_notctx_communities_' + str(idx) + '.eps')

# plot communities on brainstem + diencephalon
fig = plot_point_brain(assignments[idx, :67],  # don't plot fs subcortex
                       coords=info.query("structure == 'brainstem' or\
                                         structure == 'diencephalon'")[
                                             ['x', 'y', 'z']].values,
                       size=dom[:67], vmin=1, vmax=5,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(5, 5),
                       cmap='Accent', cbar=True,
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_notctx_communities_'
            + str(idx) + '.eps')

# plot communities on freesurfer subcortex
qualcm = mpl.colors.ListedColormap(['#666666', '#196534',
                                    '#6d91cb', '#faa51b'])
mpl.colormaps.register(qualcm, name="qual_cm")

plot_subcortical(assignments[idx, 67:][enigma_reorder_idx],
                 ventricles=False, cmap='qual_cm',
                 size=(1800, 600), screenshot=True, transparent_bg=True,
                 filename=path+'figures/eps/Schaefer' + str(parc)
                 + '/surfacesubc_communities_' + str(idx) + '.png')

# community-specific cortical FC pattern
for i in np.unique(assignments[idx, :]):
    data = np.sum(fc_reg[np.ix_(assignments[idx, :] == i, idx_ctx)], axis=0)
    brain = plot_fsaverage(data=data,
                           lhannot=annot.lh, rhannot=annot.rh,
                           colormap=cmap,
                           vmax=np.max(np.abs(data)),
                           vmin=-np.max(np.abs(data)),
                           views=['lat', 'med'],
                           data_kws={'representation': "wireframe",
                                     'line_width': 4.0})
    brain.save_image(path+'figures/eps/Schaefer'
                     + str(parc) + '/surface_ctx_community_'
                     + str(i) + '_gamma_' + str(idx) + '_withsubc.eps')

# make latex table
nrows = np.max([np.sum(assignments[idx, :] == i)
                for i in range(1, max(assignments[idx, :]).astype(int) + 1)])
community_nuclei = dict.fromkeys(['grey', 'green', 'blue', 'yellow'])
for i, key in enumerate(community_nuclei.keys()):
    community_nuclei[key] = []
    nuclei_names = info.iloc[idx_notctx]['labels'][
        assignments[idx, :] == i+1].values
    for row in range(nrows):
        try:
            community_nuclei[key].append(nuclei_names[row])
        except IndexError:
            community_nuclei[key].append(" ")
pd.DataFrame(data=community_nuclei).to_latex(
    path + 'results/Schaefer' + str(parc)
    + '/community_detection/comunity_regions_wholebrain_latex.txt',
    index=False)

"""
gradient
"""

# plot matrix
ax = plot_mod_heatmap(np.corrcoef(fc_reg[:, idx_ctx].T),
                      info.query("structure == 'cortex'")['rsn'],
                      cmap=cmap, vmin=-1, vmax=1,
                      xlabels=np.unique(info.query("structure == 'cortex'")
                                        ['rsn']),
                      ylabels=np.unique(info.query("structure == 'cortex'")
                                        ['rsn']),
                      rasterized=True)
ax.set_title('ctx-->bstem FC similarity')
plt.tight_layout()
plt.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/heatmap_fcregcorr-notctx.eps')

# brainstem gradient
fc_grad_notctx = diffusion_map_embed(np.corrcoef(fc_reg[:, idx_ctx].T),
                                     no_dims=3, alpha=0.5)[0]

brain = plot_fsaverage(data=fc_grad_notctx[:, 0],
                       lhannot=annot.lh, rhannot=annot.rh,
                       colormap=cmap,
                       vmin=-np.max(np.abs(fc_grad_notctx[:, 0])),
                       vmax=np.max(np.abs(fc_grad_notctx[:, 0])),
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe"})
brain.save_image(path+'figures/eps/Schaefer'
                 + str(parc) + '/surface_fc_grad_dme-bstem-dien-subc.eps')

# negative brainstem + diencephalon + subcortex profile
data = np.sum(fc_reg[:, idx_ctx[fc_grad_notctx[:, 0] < 0]], axis=1)
fig = plot_point_brain(data[:67],
                       coords=info.query('structure == "brainstem" or\
                                          structure == "diencephalon"')[
                                              ['x', 'y', 'z']].values,
                       vmin=np.min(data), vmax=np.max(data),
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       cbar=True, size=dom[:67],
                       cmap=PuBuGn_9.mpl_colormap, edgecolor=None)
fig.suptitle('gradient negative')
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_notctx_ctxgrad1_neg.eps')

plot_subcortical(data[67:][enigma_reorder_idx],
                 ventricles=False, cmap='PuBuGn',
                 color_range=(np.min(data), np.max(data)),
                 size=(1800, 600), screenshot=True, transparent_bg=True,
                 filename=path+'figures/eps/Schaefer' + str(parc)
                 + '/surfacesubc_grad_neg.png')

# positive brainstem + diencephalon + subcortex profile
data = np.sum(fc_reg[:, idx_ctx[fc_grad_notctx[:, 0] > 0]], axis=1)
fig = plot_point_brain(data[:67],
                       coords=info.query('structure == "brainstem" or\
                                          structure == "diencephalon"')[
                                              ['x', 'y', 'z']].values,
                       vmin=np.min(data), vmax=np.max(data),
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       cbar=True, size=dom[:67],
                       cmap=PuRd_4.mpl_colormap, edgecolor=None)
fig.suptitle('gradient positive')
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_notctx_ctxgrad1_pos.eps')

plot_subcortical(data[67:][enigma_reorder_idx],
                 ventricles=False, cmap='PuRd',
                 color_range=(np.min(data), np.max(data)),
                 size=(1800, 600), screenshot=True, transparent_bg=True,
                 filename=path+'figures/eps/Schaefer' + str(parc)
                 + '/surfacesubc_grad_pos.png')

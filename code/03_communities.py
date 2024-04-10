
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from netneurotools.plotting import (plot_point_brain,
                                    plot_mod_heatmap,
                                    plot_fsaverage)
from netneurotools.datasets import fetch_schaefer2018
from netneurotools.stats import get_dominance_stats
from nilearn.datasets import fetch_atlas_schaefer_2018
import scipy.io
from scipy.stats import spearmanr, zscore
import pandas as pd
from palettable.colorbrewer.sequential import (PuBuGn_9,
                                               PuBuGn_4,
                                               PuRd_4,
                                               PuBuGn_8)
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from netneurotools.modularity import consensus_modularity
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import squareform, pdist
import pickle


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
        consensus[:, i], q, z = consensus_modularity(A, g, B='negative_asym')
        qall.append(q)
        zrand.append(z)
        i += 1
    return (consensus, qall, zrand)


def scale_values(values, vmin, vmax, axis=None):
    s = (values - values.min(axis=axis)) /\
        (values.max(axis=axis) - values.min(axis=axis))
    s = s * (vmax - vmin)
    s = s + vmin
    return s


def corr_spin(x, y, spins, nspins):
    rho, _ = spearmanr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = spearmanr(x[spins[:, i]], y)

    pval = (1 + sum(abs(null) > abs(rho))) / (nspins + 1)
    return rho, pval, null


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


def cv_slr_distance_dependent(X, y, coords, train_pct=.75, metric='rsq'):
    '''
    cross validates linear regression model using distance-dependent method.
    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    train_pct (between 0 and 1), percent of observations in training set
    metric = {'rsq', 'corr'}
    '''

    P = squareform(pdist(coords, metric="euclidean"))
    train_metric = np.zeros((len(y)))
    test_metric = np.zeros((len(y)))

    for i in range(len(y)):
        distances = P[i, :]  # for every node
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(train_pct * len(coords)))]
        test_idx = idx[int(np.floor(train_pct * len(coords))):]

        mdl = LinearRegression()
        mdl.fit(X[train_idx, :], y[train_idx])
        if metric == 'rsq':
            # get r^2 of train set
            train_metric[i] = get_reg_r_sq(X[train_idx, :], y[train_idx])

        elif metric == 'corr':
            rho, _ = spearmanr(mdl.predict(X[train_idx, :]), y[train_idx])
            train_metric[i] = rho

        yhat = mdl.predict(X[test_idx, :])
        if metric == 'rsq':
            # get r^2 of test set
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1-(1-r_squared)*((len(y[test_idx]) - 1) /
                                                  (len(y[test_idx]) -
                                                   X.shape[1]-1))
            test_metric[i] = adjusted_r_squared

        elif metric == 'corr':
            rho, _ = spearmanr(yhat, y[test_idx])
            test_metric[i] = rho

    return train_metric, test_metric


def get_reg_r_pval(X, y, spins, nspins):
    emp = get_reg_r_sq(X, y)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(X[spins[:, s], :], y)
    return (1 + sum(null > emp))/(nspins + 1)


def get_interactional_dominance(x, y, idx):
    return get_reg_r_sq(x, y) - get_reg_r_sq(np.delete(x, idx, axis=1), y)


"""
set up
"""

path = '/home/jhansen/gitrepos/hansen_brainstemfc/'
datapath = '/home/jhansen/data-2/brainstem/'

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

# show that it's a dominant pattern
str_bstem_ctx = np.sum(fc[np.ix_(idx_bstem, idx_ctx)], axis=1)

plt.ion()
rho = np.array([spearmanr(str_bstem_ctx, fc[i, idx_bstem])[0] for i in idx_bc])
plt.figure()
sns.kdeplot(rho, label='all')
sns.kdeplot(rho[:len(idx_bstem)], label='brainstem')
sns.kdeplot(rho[len(idx_bstem):], label='cortex')
plt.legend()
plt.xlabel('spearman r')
plt.tight_layout()
plt.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/kdeplot_rho_fc_strength.eps')

# plot rhos on cortical surface and brainstem
brain = plot_fsaverage(data=rho[len(idx_bstem):],
                       lhannot=annot.lh, rhannot=annot.rh,
                       colormap=PuBuGn_9.mpl_colormap,
                       vmax=np.max(rho[len(idx_bstem):]),
                       vmin=np.min(rho[len(idx_bstem):]),
                       views=['lat', 'med'],
                       data_kws={'representation': "wireframe",
                                 'line_width': 4.0})
brain.save_image(path+'figures/eps/Schaefer' + str(parc)
                 + '/surface_ctx_rho_bstemhubs.eps')

fig = plot_point_brain(rho[:len(idx_bstem)],
                       coords=info.query("structure == 'brainstem'")
                       [['x', 'y', 'z']].values,
                       size=str_bstem_ctx ** 1.2,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(5, 5),
                       cmap=PuBuGn_9.mpl_colormap, cbar=True,
                       vmin=np.min(rho[:len(idx_bstem)]),
                       vmax=np.max(rho[:len(idx_bstem)]),
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/pointbrain_bstem_rho_bstemhubs.eps')

# regression
fc_reg = np.zeros((len(idx_bstem), len(fc)))
for i in range(len(fc)):
    fc_reg[:, i] = np.squeeze(regress_out(str_bstem_ctx.reshape(-1, 1),
                                          fc[idx_bstem, i].reshape(-1, 1)))

# plot fc_reg
fig, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(fc_reg[:, idx_ctx],
            vmin=-np.max(abs(fc_reg[:, idx_ctx])),
            vmax=np.max(abs(fc_reg[:, idx_ctx])),
            cmap=cmap, square=True,
            xticklabels=False, yticklabels=False,
            rasterized=True)
plt.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_fcreg.eps')

# plot regression example
fig, ax = plt.subplots(2, 1, figsize=(5, 5))
ax[0].plot(fc[info.query("labels == 'IC_r'").index, idx_bstem])
# pick Vis_2 for schaefer100
ax[0].plot(fc[info.query("labels == '7Networks_LH_Vis_26'").index, idx_bstem])
ax[0].set_xlabel('brainstem regions')
ax[0].set_ylabel('FC')
ax[0].legend(['IC_r', 'Vis'])
ax[1].plot(fc_reg[:, info.query("labels == 'IC_r'").index])
ax[1].plot(fc_reg[:, info.query("labels == '7Networks_LH_Vis_26'").index])
ax[1].set_xlabel('brainstem regions')
ax[1].set_ylabel('FC')
ax[1].legend(['IC_r', 'Vis'])
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/plot_toy_spaceseries.eps')

"""
community detection
"""

gamma_range = [x/10.0 for x in range(1, 61, 1)]
# consensus, qall, zrand = community_detection(np.corrcoef(fc_reg[:, idx_ctx]),
#                                              gamma_range)

assignments_bstem = np.load(path+'results/Schaefer' + str(parc)
                            + '/community_detection/assignments_bstem.npy')
zrand = np.load(path+'results/Schaefer' + str(parc)
                + '/community_detection/Zrand_bstem.npy')

# show that the community detection is appropriate
m = np.zeros((len(gamma_range), ))
v = np.zeros((len(gamma_range), ))
for i in range(len(gamma_range)):
    m[i] = np.mean(zrand[i, :])
    v[i] = np.std(zrand[i, :] ** 2)

# mean-variance plot and number of communities
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(m, c='red')
ax[0].set_ylabel('mean', c='red')
ax[0].tick_params(axis='y', labelcolor='red')
ax[0].set_xticks(np.arange(-1, 60, 10))
xticklabels = gamma_range[9::10]
xticklabels.insert(0, 0.0)
ax[0].set_xticklabels(xticklabels)
ax2 = ax[0].twinx()
ax2.plot(v, c='blue')
ax2.set_ylabel('var', c='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax[0].vlines(x=18, ymin=np.min(m), ymax=np.max(m))
ax[0].vlines(x=21, ymin=np.min(m), ymax=np.max(m))
ax[0].vlines(x=27, ymin=np.min(m), ymax=np.max(m))
ax[0].set_xlabel('gamma')
ax[1].plot(np.max(assignments_bstem, axis=1))
ax[1].set_ylabel("number of communities")
ax[1].set_xticks(np.arange(-1, 60, 10))
ax[1].set_xticklabels(xticklabels)
ax[1].set_xlabel('gamma')
ax[1].vlines(x=18, ymin=2, ymax=24)
ax[1].vlines(x=21, ymin=2, ymax=24)
ax[1].vlines(x=27, ymin=2, ymax=24)
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/plot_community_meanvar.eps')

# plot heatmap with modules
idx = 27   # gamma = 2.8; also show gamma = 1.9 and gamma = 2.2;
fig = plot_mod_heatmap(data=np.corrcoef(fc_reg[:, idx_ctx]),
                       communities=assignments_bstem[idx, :],
                       cmap=cmap, vmin=-0.9, vmax=0.9,
                       rasterized=True)
plt.title('gamma = ' + str(gamma_range[idx]))
plt.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/heatmap_bstem_communities_'
            + str(idx) + '.eps')

# plot communities on brainstem
fig = plot_point_brain(assignments_bstem[idx, :],
                       coords=info.query("structure == 'brainstem'")
                       [['x', 'y', 'z']].values,
                       size=str_bstem_ctx ** 1.5,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(5, 5),
                       cmap='Accent', cbar=True,
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_bstem_communities_'
            + str(idx) + '.eps')

"""
community-specific cortical FC patterns
"""

for i in np.unique(assignments_bstem[idx, :]):
    summap = np.sum(fc_reg[np.ix_(idx_bstem[assignments_bstem[idx, :] == i],
                                  idx_ctx)], axis=0)
    brain = plot_fsaverage(data=summap,
                           lhannot=annot.lh, rhannot=annot.rh,
                           colormap=cmap,
                           vmax=np.max(np.abs(summap)),
                           vmin=-np.max(np.abs(summap)),
                           views=['lat', 'med'],
                           data_kws={'representation': "wireframe",
                                     'line_width': 4.0})
    brain.save_image(path+'figures/eps/Schaefer'
                     + str(parc) + '/surface_ctx_community_'
                     + str(i) + '_gamma_' + str(idx) + '.eps')

    # going in the supplement
    varmap = np.var(fc_reg[np.ix_(idx_bstem[assignments_bstem[idx, :] == i],
                                  idx_ctx)], axis=0)
    brain = plot_fsaverage(data=varmap,
                           lhannot=annot.lh, rhannot=annot.rh,
                           colormap=PuBuGn_9.mpl_colormap,
                           vmax=0.000475,
                           vmin=0,
                           views=['lat', 'med'],
                           data_kws={'representation': "wireframe",
                                     'line_width': 4.0})
    brain.save_image(path+'figures/eps/Schaefer'
                     + str(parc) + '/surface_ctx_community_'
                     + str(i) + '_gamma_' + str(idx) + '_var.eps')

# how correlated is each brainstem nucleus with its community's fc pattern
rhos = np.zeros(58)
for i in range(len(idx_bstem)):
    samelabel = np.where(assignments_bstem[idx, :]
                         == assignments_bstem[idx, i])[0]
    summap = np.sum(fc_reg[np.ix_(samelabel, idx_ctx)], axis=0)
    rhos[i] = spearmanr(fc_reg[i, idx_ctx], summap)[0]

fig = plot_point_brain(rhos,
                       coords=info.query("structure == 'brainstem'")
                       [['x', 'y', 'z']].values,
                       size=str_bstem_ctx ** 1.2,
                       views_orientation='horizontal',
                       views=['coronal_rev', 'sagittal', 'axial'],
                       views_size=(5, 5),
                       cmap=PuBuGn_9.mpl_colormap, cbar=True,
                       vmin=0, vmax=1,
                       edgecolor=None)
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/pointbrain_bstem_community_rhos.eps')


"""
neurosynth decoding
"""

nsynth = pd.read_csv(path+'data/neurosynth/atl-schaefer2018_res-'
                     + str(parc) + '_neurosynth.csv',
                     index_col=0)

for commID in np.unique(assignments_bstem[idx, :]):
    print(commID)
    data = np.sum(fc_reg[np.ix_(idx_bstem[assignments_bstem[idx, :] == commID],
                                idx_ctx)], axis=0)
    rho = np.array([spearmanr(data, nsynth[key])[0] for key in nsynth.keys()])
    sortidx = np.argsort(rho)
    fig, ax = plt.subplots(figsize=(5, 4))
    nbar = 12
    pspin = np.array([corr_spin(data, nsynth[key], spins, nspins)[1]
                      for key in nsynth.keys().values[sortidx][-nbar:]])
    pspin = multipletests(pspin, method='fdr_bh')[1]
    ax.barh(np.arange(nbar),
            np.sort(rho)[-nbar:],
            tick_label=nsynth.keys().values[sortidx][-nbar:])
    ax.plot(0.1 * np.ones((np.sum(pspin < 0.05), )),
            np.arange(nbar)[np.where(pspin < 0.05)[0]],
            '*', c='k')
    ax.set_xlabel('spearman r')
    fig.tight_layout()
    fig.savefig(path+'figures/eps/Schaefer'
                + str(parc) + '/bar_community_'
                + str(commID) + '_nsynth_gamma_' + str(idx) + '.eps')

# also do it for the strength map (supplement)
str_ctx_bstem = np.sum(fc[np.ix_(idx_bstem, idx_ctx)], axis=0)

rho = np.array([spearmanr(str_ctx_bstem, nsynth[key])[0]
                for key in nsynth.keys()])
sortidx = np.argsort(rho)
fig, ax = plt.subplots(figsize=(5, 4))
nbar = 12
ax.barh(np.arange(nbar),
        np.sort(rho)[-nbar:],
        tick_label=nsynth.keys().values[sortidx][-nbar:])
ax.set_xlabel('spearman r')
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer'
            + str(parc) + '/bar_nsynth_ctxstrength.eps')

# also show heatmap of all correlations
rho = []
for i, commID in enumerate(np.unique(assignments_bstem[idx, :])):
    print(commID)
    data = np.sum(fc_reg[np.ix_(idx_bstem[assignments_bstem[idx, :] == commID],
                                idx_ctx)], axis=0)
    rho.append(np.array([spearmanr(data, nsynth[key])[0]
                         for key in nsynth.keys()]))
rho = np.array(rho)
fig, ax = plt.subplots(figsize=(20, 5))
sns.heatmap(rho, cmap=cmap, square=True,
            vmin=-np.max(abs(rho)), vmax=np.max(abs(rho)),
            xticklabels=nsynth.keys(),
            yticklabels=np.unique(assignments_bstem[idx, :]),
            linewidths=.5, ax=ax)
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/heatmap_nsynth_rhos.eps')


"""
receptor decoding
"""

recpath = '/home/jhansen/gitrepos/hansen_receptors/data/PET_parcellated/scale'\
     + str(parc) + '/'

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
receptor_ctx = pd.DataFrame(data=receptor_ctx,
                            index=info.query('structure == "cortex"')
                            ['labels'])

ncommun = np.max(assignments_bstem[idx, :]).astype(int)

model_metrics = dict([])
train_metric = np.zeros([len(idx_ctx), ncommun])
test_metric = np.zeros(train_metric.shape)
model_pval = np.zeros((ncommun, ))

for i in np.unique(assignments_bstem[idx, :]):
    print(i)
    fcmap = np.sum(fc_reg[np.ix_(idx_bstem[assignments_bstem[idx, :] == i],
                                 idx_ctx)], axis=0)
    model_metrics['community' + str(i)] =\
        get_dominance_stats(zscore(receptor_ctx.values), zscore(fcmap))[0]
    with open(path + 'results/Schaefer' + str(parc)
              + '/dominance_analysis/model_metrics_community'
              + str(i) + '.pkl', 'wb') as f:
        pickle.dump(model_metrics['community' + str(i)], f)
    # cross validate the model
    # train_metric[:, int(i)-1], test_metric[:, int(i)-1] = \
    #     cv_slr_distance_dependent(zscore(receptor_ctx.values),
    #                               zscore(fcmap),
    #                               info.query("structure == 'cortex'")
    #                               [['x', 'y', 'z']].values
    #                               , .75, metric='corr')
    # get model pval
    # model_pval[int(i)-1] = get_reg_r_pval(zscore(receptor_ctx.values),
    #                                zscore(fcmap),
    #                                spins, nspins)

dominance = np.zeros((ncommun, receptor_ctx.shape[1]))
for i in range(len(model_metrics)):
    tmp = model_metrics['community' + str(i+1) + '.0']
    dominance[i, :] = tmp["total_dominance"]

model_pval = multipletests(model_pval, method='fdr_bh')[1]

np.save(path+'results/Schaefer' + str(parc)
        + '/dominance_analysis/dominance.npy', dominance)
np.save(path+'results/Schaefer' + str(parc)
        + '/dominance_analysis/dominance_cv_train.npy', train_metric)
np.save(path+'results/Schaefer' + str(parc)
        + '/dominance_analysis/dominance_cv_test.npy', test_metric)

# dominance heatmap
fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(dominance / np.sum(dominance, axis=1)[:, None],
            xticklabels=[rec.split('_')[0] for rec in rec_cols],
            yticklabels=model_metrics.keys(),
            cmap=PuBuGn_9.mpl_colormap, square=True,
            linewidths=.5, ax=ax)
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/heatmap_dominance.eps')

# rsq
fig, ax = plt.subplots()
ax.barh(np.arange(ncommun), np.sum(dominance, axis=1),
        tick_label=np.arange(1, ncommun+1))
ax.set_xlabel('adjusted Rsq')
ax.set_ylabel('community')
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/bar_dominance.eps')

# plot cross validation
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.violinplot(data=train_metric, ax=ax1)
sns.violinplot(data=test_metric, ax=ax2)
ax1.set(ylabel='train set correlation', ylim=(-1, 1))
ax1.set_xticklabels(np.arange(1, ncommun+1))
ax2.set_xticklabels(np.arange(1, ncommun+1))
ax2.set(ylabel='test set correlation', ylim=(-1, 1))
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc) + '/violin_crossval.eps')

# also do it for the strength map (supplement)
dominance = get_dominance_stats(zscore(receptor_ctx.values),
                                zscore(str_ctx_bstem))[0]

fig, ax = plt.subplots()
sns.heatmap((dominance['total_dominance'] /
            dominance['full_r_sq']).reshape(1, -1),
            xticklabels=[rec.split('_')[0] for rec in rec_cols],
            yticklabels=['ctxhubs'],
            cmap=PuBuGn_9.mpl_colormap, square=True,
            linewidths=.5, ax=ax)
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/heatmap_dominance_ctxstrength.eps')

# look at interactional and individual dominance (supplement)

interactional_dominance = dict([])
individual_dominance = dict([])
dominance = np.load(path+'results/Schaefer' + str(parc)
                    + '/dominance_analysis/dominance.npy')

for i in np.unique(assignments_bstem[idx, :]):
    fcmap = np.sum(fc_reg[np.ix_(idx_bstem[assignments_bstem[idx, :] == i],
                                 idx_ctx)], axis=0)
    X = zscore(receptor_ctx.values)
    Y = zscore(fcmap)
    dom = np.array([get_interactional_dominance(X, Y, rec)
                    for rec in range(receptor_ctx.shape[1])])
    interactional_dominance['community' + str(i)] = dom

    dom = np.array([get_reg_r_sq(X[:, rec].reshape(-1, 1), Y)
                    for rec in range(receptor_ctx.shape[1])])
    individual_dominance['community' + str(i)] = dom

fig, axs = plt.subplots(2, 1, figsize=(10, 5))
for i, dom in enumerate([interactional_dominance, individual_dominance]):
    sns.heatmap(np.array(list(dom.values())) /
                np.sum(dominance, axis=1).reshape(-1, 1),
                xticklabels=[rec.split('_')[0] for rec in rec_cols],
                yticklabels=np.unique(assignments_bstem[idx, :]),
                cmap=PuBuGn_9.mpl_colormap, square=True,
                linewidths=.5, ax=axs[i])
    axs[i].set_title('interactional' if i == 0 else 'individual')
fig.tight_layout()
fig.savefig(path+'figures/eps/Schaefer' + str(parc)
            + '/heatmap_other_dominances.eps')


"""
print regions in community
"""

for i in range(1, ncommun + 1):
    np.savetxt(path+'results/Schaefer' + str(parc)
               + '/community_detection/community_regions_' + str(i) + '.txt',
               info.query("structure == 'brainstem'")['labels']
               [assignments_bstem[idx, :] == i].values,
               delimiter=" ",
               newline="\n",
               fmt="%s")

# save as tex table
nrows = np.max([np.sum(assignments_bstem[idx, :] == i)
                for i in range(1, ncommun + 1)])
community_nuclei = dict.fromkeys(['green', 'yellow', 'pink', 'blue', 'grey'])
for i, key in enumerate(community_nuclei.keys()):
    community_nuclei[key] = []
    nuclei_names = info.query("structure == 'brainstem'")['labels'][
        assignments_bstem[idx, :] == i+1].values
    for row in range(nrows):
        try:
            community_nuclei[key].append(nuclei_names[row])
        except IndexError:
            community_nuclei[key].append(" ")
pd.DataFrame(data=community_nuclei).to_latex(
    path + 'results/Schaefer' + str(parc)
    + '/community_detection/comunity_regions_latex.txt',
    index=False)


"""
count how many bilateral nuclei are split
"""

data = {'labels': info.query("structure == 'brainstem'")['labels'],
        'assignments_bstem': assignments_bstem[idx, :]}
data = pd.DataFrame(data)

same_count = 0
different_count = 0

# iterate through each label
for label in data['labels']:
    # check if the label has '_r' or '_l'
    if label.endswith('_r'):
        counterpart = label[:-2] + '_l'
        counterpart_assignment = data.loc[info['labels'] == counterpart,
                                          'assignments_bstem'].values[0]
        current_assignment = data.loc[info['labels'] == label,
                                      'assignments_bstem'].values[0]
        if counterpart_assignment == current_assignment:
            same_count += 1
        else:
            different_count += 1
            print(label)

print(same_count, "/", same_count + different_count, "together")

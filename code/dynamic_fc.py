"""
dynamic FC
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

path = '/home/jhansen/gitrepos/hansen_brainstemfc/'
datapath='/home/jhansen/data-2/brainstem/'

parc = 400

# load region info file
info = pd.read_csv(path+'data/region_info_Schaefer'
                   + str(parc) + '.csv', index_col=0)

# handy indices
idx_bstem = info.query("structure == 'brainstem'").index.values
idx_ctx = info.query("structure == 'cortex'").index.values
idx_bc = np.concatenate((idx_bstem, idx_ctx))

# load in the timeseries
ts = np.load(datapath + 'brainstem_fc/timeseries/timeseries_brainstem_schaefer'
             + str(parc) + '.npy')

zts = zscore(ts[-parc:, :, :], axis=1)  # zscore timeseries over time
u, v = np.where(np.triu(np.ones(parc), 1))
dfc = np.multiply(zts[u, :, :], zts[v, :, :])
# put back into matrix so it's (node x node x time x subj)
dfcmat = np.zeros((parc, parc, ts.shape[1], ts.shape[2]))
for t in range(dfc.shape[1]):
    # this is surely an awful and time+space intensive way of doing this sorry
    dfcmat[np.triu_indices(parc, 1)[0], np.triu_indices(parc, 1)[1], t, :] = dfc[:, t, :]
    dfcmat[:, :, t, :] = dfcmat[:, :, t, :] + dfcmat[:, :, t, :].transpose(1, 0, 2)

pc = np.zeros((dfcmat.shape[2], ))
for t in range(len(pc)):
    pc[t] = np.mean(participation_coef(dfcmat[:, :, t, 0], info.query("structure == 'cortex'")['rsn']))


mdfc = np.median(abs(dfc), axis=0)

# neuromodulatory nuclei (van den Brink et al 2019 Front Hum Neurosci):
# locus coeruleus, substantia nigra (pars compacta), ventral tegmental area,
# pedunculopontine nuclei, basal forebrain, laterodorsal tegmental area,
# raphe nuclei, tuberomammilary nucleus of hypothalamus

nmod_nuclei = ["MnR", "RMg", "ROb", "RPa", "CLi_RLi", "DR_2020", "PMnR", # raphe nuclei
               "SN_subregion2_l", "SN_subregion2_r",                     # substantia nigra: pars compacta
               "LDTg_CGPn_l", "LDTg_CGPn_r",                             # laterodorsal tegmental nucleus
               "LC_l", "LC_r",                                           # locus coeruleus
               "VTA_PBP_l", "VTA_PBP_r",                                 # ventral tegmental area
               "PTg_l", "PTg_r"]                                         # pedunculotegmental nuclei
nmod_idx = np.array(info[info['labels'].isin(nmod_nuclei)].index)

for subj in range(20):
    fig, ax = plt.subplots()
    nmodcorrs = np.array([spearmanr(ts[node, :, subj], mdfc[:, subj])[0] for node in nmod_idx])
    notnmodcorrs = np.array([spearmanr(ts[node, :, subj], mdfc[:, subj])[0] for node in np.setdiff1d(idx_bstem, nmod_idx)])
    sns.violinplot([nmodcorrs, notnmodcorrs], inner='points', ax=ax)
    ax.set_xticklabels(['neuromods', 'not'])
    ax.set_ylabel('corr')
    ax.set_title('subj' + str(subj))

## correlations with receptors?
zts = zscore(ts, axis=1)
u, v = np.where(np.ones((len(idx_bstem), len(idx_ctx))))
dfc = np.multiply(zts[u, :, :], zts[v + len(idx_bstem), :, :])
dfc = np.reshape(dfc, (len(idx_bstem), parc, ts.shape[1], ts.shape[2]))

# 
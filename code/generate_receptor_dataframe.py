"""
generate brainstem receptor dataframe
"""


import numpy as np
import pandas as pd
from neuromaps.parcellate import Parcellater
from neuromaps.datasets import fetch_atlas
import nibabel as nib
from os.path import exists

path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/data/"
atlaspath_bstem = datapath + "BrainstemNavigator/0.9/2a.BrainstemNucleiAtlas_MNI/labels_thresholded_binary_0.35/"
atlaspath_dien = datapath + "BrainstemNavigator/0.9/2b.DiencephalicNucleiAtlas_MNI/labels_thresholded_binary_0.35/"
receptpath = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_receptors/github/hansen_receptors/"


nuclei_names = list(np.loadtxt(datapath+'brainstem_fc/labels_acronyms_list.txt', dtype=str))
nuclei_names_nav = list(np.loadtxt(datapath+'brainstem_fc/labels_acronyms_list_fordisplay_BrainstemNavigatornomenclature.txt', dtype=str))

for i, name in enumerate(nuclei_names_nav):
    if name.split('_')[-1] == 'l':
        nuclei_names_nav.append(name[:-1] + 'r')
        nuclei_names.append(nuclei_names[i][:-1] + 'r')
nuclei_names.extend(['STh_subregion1_l', 'STh_subregion2_l',
                     'HTH',
                     'STh_subregion2_r', 'STh_subregion1_r',
                     'LG_l', 'LG_r', 'MG_l', 'MG_r'])
nuclei_names_nav.extend(['STh1_l', 'STh2_l', 'HTH', 'STh2_r', 'STh1_r',
                         'LG_l', 'LG_r', 'MG_l', 'MG_r'])
np.savetxt(path+'data/brainstem_fc/labels_BrainstemNavigator.csv',
           nuclei_names_nav, delimiter=',', fmt='%s')
np.savetxt(path+'data/brainstem_fc/labels_dataframes.csv',
           nuclei_names, delimiter=',', fmt='%s')


receptors_nii = [receptpath+'data/PET_nifti_images/5HT1a_way_hc36_savli.nii',
                 receptpath+'data/PET_nifti_images/5HT1a_cumi_hc8_beliveau.nii',
                 receptpath+'data/PET_nifti_images/5HT1b_az_hc36_beliveau.nii',
                 receptpath+'data/PET_nifti_images/5HT1b_p943_hc22_savli.nii',
                 receptpath+'data/PET_nifti_images/5HT1b_p943_hc65_gallezot.nii.gz',
                 receptpath+'data/PET_nifti_images/5HT2a_cimbi_hc29_beliveau.nii',
                 receptpath+'data/PET_nifti_images/5HT2a_alt_hc19_savli.nii',
                 receptpath+'data/PET_nifti_images/5HT2a_mdl_hc3_talbot.nii.gz',
                 receptpath+'data/PET_nifti_images/5HT4_sb20_hc59_beliveau.nii',
                 receptpath+'data/PET_nifti_images/5HT6_gsk_hc30_radhakrishnan.nii.gz',
                 receptpath+'data/PET_nifti_images/5HTT_dasb_hc100_beliveau.nii',
                 receptpath+'data/PET_nifti_images/5HTT_dasb_hc30_savli.nii',
                 receptpath+'data/PET_nifti_images/A4B2_flubatine_hc30_hillmer.nii.gz',
                 receptpath+'data/PET_nifti_images/CB1_omar_hc77_normandin.nii.gz',
                 receptpath+'data/PET_nifti_images/CB1_FMPEPd2_hc22_laurikainen.nii',
                 receptpath+'data/PET_nifti_images/D1_SCH23390_hc13_kaller.nii',
                 receptpath+'data/PET_nifti_images/D2_fallypride_hc49_jaworska.nii',
                 receptpath+'data/PET_nifti_images/D2_flb457_hc37_smith.nii.gz',
                 receptpath+'data/PET_nifti_images/D2_flb457_hc55_sandiego.nii.gz',
                 receptpath+'data/PET_nifti_images/D2_raclopride_hc7_alakurtti.nii',
                 receptpath+'data/PET_nifti_images/DAT_fpcit_hc174_dukart_spect.nii',
                 receptpath+'data/PET_nifti_images/DAT_fepe2i_hc6_sasaki.nii.gz',
                 receptpath+'data/PET_nifti_images/GABAa-bz_flumazenil_hc16_norgaard.nii',
                 receptpath+'data/PET_nifti_images/GABAa_flumazenil_hc6_dukart.nii',
                 receptpath+'data/PET_nifti_images/H3_cban_hc8_gallezot.nii.gz',
                 receptpath+'data/PET_nifti_images/M1_lsn_hc24_naganawa.nii.gz',
                 receptpath+'data/PET_nifti_images/mGluR5_abp_hc22_rosaneto.nii',
                 receptpath+'data/PET_nifti_images/mGluR5_abp_hc28_dubois.nii',
                 receptpath+'data/PET_nifti_images/mGluR5_abp_hc73_smart.nii',
                 receptpath+'data/PET_nifti_images/MU_carfentanil_hc204_kantonen.nii',
                 receptpath+'data/PET_nifti_images/MU_carfentanil_hc39_turtonen.nii',
                 receptpath+'data/PET_nifti_images/NAT_MRB_hc77_ding.nii.gz',
                 receptpath+'data/PET_nifti_images/NAT_MRB_hc10_hesse.nii',
                 receptpath+'data/PET_nifti_images/NMDA_ge179_hc29_galovic.nii.gz',
                 receptpath+'data/PET_nifti_images/VAChT_feobv_hc3_spreng.nii',
                 receptpath+'data/PET_nifti_images/VAChT_feobv_hc4_tuominen.nii',
                 receptpath+'data/PET_nifti_images/VAChT_feobv_hc5_bedard_sum.nii',
                 receptpath+'data/PET_nifti_images/VAChT_feobv_hc18_aghourian_sum.nii']

bstem_receptor_den = dict([])
for receptor in receptors_nii:
    atlaspath = atlaspath_bstem  # start with the brainstem nuclei
    print(receptor)
    name = receptor.split('/')[-1].split('.')[0]  # get nifti file name
    density = np.zeros((len(nuclei_names_nav), ))
    for n, nucleus in enumerate(nuclei_names_nav):
        if nucleus == "HTH":
            density[n] = np.nan
            continue
        if nucleus == "STh1_l":
            atlaspath = atlaspath_dien  # we're into the diencephalic nuclei
        atlas = atlaspath + nucleus + '.nii.gz'
        parcellater = Parcellater(atlas, 'MNI152')
        try:
            density[n] = np.squeeze(parcellater.fit_transform(receptor, 'MNI152', True))
        except ValueError:
            density[n] = np.nan
        bstem_receptor_den[name] = density

dataframe = pd.DataFrame(data=bstem_receptor_den, index=nuclei_names)
dataframe.to_csv(path+'results/brainstem_receptor_density.csv')







"""
quality check
"""

from math import comb
import matplotlib.pyplot as plt
import itertools
from scipy.stats import spearmanr

receptor_names = np.unique(np.array([name.split('_')[0] for name in dataframe.columns]))
for substring in receptor_names:
    map_idx = [i for i, flag in enumerate(dataframe.columns.str.contains(substring)) if flag]
    ncomb = comb(len(map_idx), 2)
    if ncomb == 0:
        continue
    fig, axs = plt.subplots(1, ncomb, figsize=(4*ncomb, 4))
    for i, idx in enumerate(itertools.combinations(map_idx, 2)):
        x = dataframe.iloc[:, idx[0]]
        y = dataframe.iloc[:, idx[1]]
        r, p = spearmanr(x, y, nan_policy='omit')
        if len(map_idx) > 2:
            ax = axs[i]
        else:
            ax = axs
        ax.scatter(x, y)
        ax.set_xlabel(dataframe.columns[idx[0]])
        ax.set_ylabel(dataframe.columns[idx[1]])
        ax.set_title('r=' + str(r)[:5])
    fig.tight_layout()
    fig.savefig(path+'figures/png/scatter_receptor_bstem_qc_' + substring + '.png')



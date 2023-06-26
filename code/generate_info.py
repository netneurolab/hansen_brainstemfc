"""
generate region_info.csv
"""

import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from netneurotools import datasets
import pandas as pd
import nibabel as nib
from neuromaps.parcellate import Parcellater

path = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/"
datapath = "C:/Users/justi/OneDrive - McGill University/MisicLab/proj_brainstem/data/"

parc = 400  # number of cortical nodes

schaefer = fetch_atlas_schaefer_2018(n_rois=parc)
annot = datasets.fetch_schaefer2018('fsaverage')[str(parc) + 'Parcels7Networks']

"""
set up dataframe
"""

info = dict([])
info['labels'] = np.loadtxt(datapath+'brainstem_fc/parcellated/Schaefer'
                            + str(parc) + '/all_labels_mcgill'
                            + str(parc) + '.txt', dtype=str)
info = pd.DataFrame(data=info)

## set up structure type
ctx_idx = np.arange(75, parc + 75)
bstem_idx = np.arange(58)
dien_idx = np.arange(58, 67)
subc_idx = np.concatenate((np.arange(67, 75),
                           np.arange(parc + 75, parc + 83)))
structure = np.zeros((len(info['labels']), )).astype(str)
structure[ctx_idx] = 'cortex'
structure[bstem_idx] = 'brainstem'
structure[dien_idx] = 'diencephalon'
structure[subc_idx] = 'subcortex' 
info['structure'] = pd.Categorical(structure, ["cortex", "subcortex", "brainstem", "diencephalon"])

## set up hemisphere
hemisphere = np.zeros((len(info['labels']), )).astype(str)
# cortex + subcortex
hemisphere[info['labels'].str.contains('_lh|L-|LH|_l')] = 'L'
hemisphere[info['labels'].str.contains('_rh|R-|RH|_r')] = 'R'
hemisphere[hemisphere == '0.0'] = 'M'  # medial; not bilateral structure
info['hemisphere'] = pd.Categorical(hemisphere, ["L", "R", "M"])

## set up RSN networks for cortical regions
rsn = [info['labels'][i].split('_')[2]
         if info['structure'][i] == 'cortex' 
         else ''
         for i in range(len(info['labels']))]
info['rsn'] = pd.Categorical(rsn, ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'])

## coordinates
coords_bstem = np.loadtxt(datapath+'coords/brainstem_coords.txt')
coords_bstem = coords_bstem * [-1, -1, 1]   # LPS to RAS coordinates
coords_bstem_labels = np.loadtxt(datapath+'coords/brainstem_coords_labels.txt', dtype=str)
coords = pd.DataFrame(data=coords_bstem, index=coords_bstem_labels, columns=['x', 'y', 'z'])
todrop = set(coords_bstem_labels).difference(set(info.query("structure == 'brainstem' or structure == 'diencephalon'")['labels']))
coords.drop(index=todrop, inplace=True)

coords_ctx = np.genfromtxt(datapath+'coords/Schaefer' + str(parc) + '_coords.txt')[:, -3:]
coords = pd.concat([coords, pd.DataFrame(data=coords_ctx,
                                         index=info['labels'][info['structure'] == 'cortex'],
                                         columns=['x', 'y', 'z'])])

coords_subc = pd.read_csv(datapath+'coords/subcortex_coords.csv',
                          index_col=0,
                          delimiter=',')
# rename sighhh
newname = []
for name in list(coords_subc.index):
    namesplit = name.split('-')
    hem = namesplit[0][0]
    roi = namesplit[1].capitalize()
    if roi == 'Thalamusproper':
        roi = 'Thalamus-Proper'
    elif roi == 'Accumbensarea':
        roi = 'Accumbens-area'
    newname.append(hem + '-' + roi)
mapper = {list(coords_subc.index)[i] : newname[i] for i in range(len(newname))}
toadd = set(info['labels'][info['structure'] == 'subcortex']).difference(set(newname))
nans = np.empty((1, 3))
nans[:] = np.nan
coords_subc = pd.concat([coords_subc, pd.DataFrame(data=np.repeat(nans, 2, axis=0),
                                                   index=list(toadd),
                                                   columns=['x', 'y', 'z'])]).rename(mapper=mapper)

coords = pd.concat([coords, coords_subc])
info = pd.merge(info, coords, left_on='labels', right_index=True)

## count number of voxels
ctximg = nib.load(schaefer['maps']).get_fdata()
nvoxels_ctx = np.array([np.sum(ctximg == i) for i in range(1, parc + 1)])
nvoxels = pd.DataFrame(data=nvoxels_ctx,
                       index=info['labels'][info['structure'] == 'cortex'].values,
                       columns=['nvoxels'])
nans = np.empty((len(subc_idx), ))
nans[:] = np.nan
nvoxels = pd.concat([nvoxels, pd.DataFrame(data=nans,
                                           index=info['labels'][info['structure'] == 'subcortex'],
                                           columns=['nvoxels'])])

atlaspath_bstem = datapath + "BrainstemNavigator/0.9/2a.BrainstemNucleiAtlas_MNI/labels_thresholded_binary_0.35/"
atlaspath_dien = datapath + "BrainstemNavigator/0.9/2b.DiencephalicNucleiAtlas_MNI/labels_thresholded_binary_0.35/"
nuclei_names_nav = np.loadtxt(datapath+'brainstem_fc/labels_BrainstemNavigator.csv', dtype=str)
nuclei_names = np.loadtxt(datapath+'brainstem_fc/labels_dataframes.csv', dtype=str)
nvoxels_bstem = np.zeros((len(nuclei_names_nav), ))
atlaspath = atlaspath_bstem
for n, nucleus in enumerate(nuclei_names_nav):
    if nucleus == "HTH":
        nvoxels_bstem[n] = np.nan
        continue
    if nucleus == "STh1_l":
        atlaspath = atlaspath_dien
    atlas = atlaspath + nucleus + '.nii.gz'
    bstemimg = nib.load(atlas).get_fdata()
    nvoxels_bstem[n] = np.sum(bstemimg == 1)

nvoxels = pd.concat([nvoxels, pd.DataFrame(data=nvoxels_bstem,
                                           index=nuclei_names,
                                           columns=['nvoxels'])])

info = pd.merge(info, nvoxels, left_on='labels', right_index=True)

## tSNR map
tsnr = datapath + 'brainstem_fc/TSNRv2_ave.nii'

parcellater = Parcellater(schaefer['maps'], 'MNI152')
tsnr_ctx = np.squeeze(parcellater.fit_transform(tsnr, 'MNI152', True))

tsnr_bstem = np.zeros((len(nuclei_names_nav)))
atlaspath = atlaspath_bstem
for n, nucleus in enumerate(nuclei_names_nav):
    if nucleus == "HTH":
        tsnr_bstem[n] = np.nan
        continue
    if nucleus == "STh1_l":
        atlaspath = atlaspath_dien  # we're into the diencephalic nuclei
    atlas = atlaspath + nucleus + '.nii.gz'
    parcellater = Parcellater(atlas, 'MNI152')
    try:
        tsnr_bstem[n] = np.squeeze(parcellater.fit_transform(tsnr, 'MNI152', True))
    except ValueError:
        tsnr_bstem[n] = np.nan

tsnr_df = pd.DataFrame(data=tsnr_ctx,
                       index=info.query("structure == 'cortex'")['labels'].values,
                       columns=['tSNR'])
tsnr_df = pd.concat([tsnr_df, pd.DataFrame(data=nans,
                                           index=info['labels'][info['structure'] == 'subcortex'],
                                           columns=['tSNR'])])
tsnr_df = pd.concat([tsnr_df, pd.DataFrame(data=tsnr_bstem,
                                           index=nuclei_names,
                                           columns=['tSNR'])])

info = pd.merge(info, tsnr_df, left_on='labels', right_index=True)

## save out
info.to_csv(path+'data/region_info_Schaefer' + str(parc) + '.csv')
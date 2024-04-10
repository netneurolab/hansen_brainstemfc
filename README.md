# Integrating brainstem and cortical functional architectures
This repository contains code and data in support of "Integrating brainstem and cortical functional architectures", now up on [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.26.564245v1).
All code was written in Python.
Below, I describe all the foldes and files in details.

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
A description of each file follows (in an order that complements the manuscript):
- [generate_info.py](code/generate_info.py) will create the `info` dataframe that stores information about each parcel, including the parcel's name, structure (brainstem, cortex, diencephalon, subcortex), the hemisphere (right, left, medial), parcel centroid coordinates, the number of voxels mapped to the parcel (i.e. volume), and the tSNR of the parcel. Parcels in the `info` dataframe are in the same order as in the $483\times 483$ fc matrix I use in the analyses.
- [01_show_bstem_data.py](code/01_show_bstem_data.py) covers Figure 1 and supplementary Figure S2 where I plot some basic features of brainstem and cortical fc.
- [02_hubs.py](code/02_hubs.py) covers Figure 2, supplementary Figures S3, and S22, where I calculates the weighted degree of brainstem-to-cortex and cortex-to-brainstem fc (also called brainstem-to-cortex and cortex-to-brainstem hubs). This is also where I look at MEG-derived patterns of neuro-oscillatory dynamics.
- [03_communities.py](code/03_communities.py) covers Figure 3, Figure 4, and supplementary Figures S5--S11, S21 where I look at communities of brainstem nuclei that are similarly functionally connected with the cortex. Then I do a neurosynth decoding (Figure 3d) and a receptor decoding (Figure 4).
- [04_gradients.py](code/04_gradients.py) covers Figure 5 and supplementary Figure S12 where I look at how cortical regions are similarly functionally connected with the brainstem.
- [05_subcortex.py](code/05_subcortex.py) covers supplementary Figure S13 where I replicate the findings in 14 FreeSurfer subcortical structures + 8 Brainstem Navigator diencephalic nuclei + the hypothalamus.
- [supplement.py](code/supplement.py) covers supplementary Figures S14 (split-half analysis), S16 (replication in 3 Tesla data), S17 (individual variability), and S18 (individual connectomes).

## `data`
The [data](data/) folder contains data files used for the analyses.

The most important file is [brainstemfc_mean_corrcoeff_full.npy](data/brainstemfc_mean_corrcoeff_full.npy), which is a $483\times 483$ functional connectivity matrix representing the average functional connectivity (Pearson's r) across 20 healthy individuals.
This data was collected by the [Brainstem Imaging Lab](https://brainstemimaginglab.martinos.org/) (led by Marta Bianciardi) and first presented in [Cauzzo & Singh et al 2022](https://www.sciencedirect.com/science/article/pii/S1053811922000544?via%3Dihub) and [Singh & Cauzzo et al 2022](https://www.sciencedirect.com/science/article/pii/S1053811921011368).
Please be sure to cite these two papers if you use the brainstem functional connectivity data!

I also make use of the [Brainstem Navigator](https://www.nitrc.org/projects/brainstemnavig/) which defines, in MNI152 space, parcels for 58 brainstem nuclei and 8 diencephalic nuclei.

## `results`
The [results](results/) folder contains some saved outputs from my scripts, including outputs for the community detection and dominance analysis.

## `manuscript`
The [manuscript](manuscript/) folder contains the PDF of the preprint.

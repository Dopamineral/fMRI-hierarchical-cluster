import os
import numpy as np
import nibabel as nib
import seaborn as sns 
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


file = "<-insert file path here->"
img = nib.load(file)
sub_img = img.slicer[30:50,70:90,35:50,:] #prefrontal


img_data = sub_img.get_fdata()
img_reshaped = img_data.reshape(6000,1200)
img_normalized = normalize(img_reshaped)
img_movmean = uniform_filter1d(img_reshaped,
                               size=1,
                               axis=1)


g = sns.clustermap(img_movmean,
                col_cluster=False,
                cmap='rocket',
                z_score=0,
                figsize=(50,50),
                metric='correlation')

ax = g.ax_heatmap
ax.set_xlabel('time (samples)')
ax.set_ylabel('clustered voxels')

# sns.clustermap(img_normalized,
#                 figsize=(50,50),
#                 metric='correlation')

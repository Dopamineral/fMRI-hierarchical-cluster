import numpy as np
import nibabel as nib
import seaborn as sns 
import pandas as pd

atlas_file = "<-insert atlas nii path here ->"
file = "<- insert scan nii path here ->" 
legend_file = "<- insert atlas legend txt/csv here ->"

img = nib.load(file)
atlas = nib.load(atlas_file)
legend = pd.read_csv(legend_file,sep='(',header=None)

df_max = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float") #make df size of areas x timepoints
df_mean = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float") #make df size of areas x timepoints

atlas_data = atlas.get_fdata() #load atlas values

for ii in range(img.shape[-1]): #loop over timepoints
# for ii in range(50):
    print(f'timepoint {ii}')
    sub_img = img.slicer[:,:,:,ii] 
    
     
    img_data = sub_img.get_fdata() #load vol values
    
    for jj in range(legend.shape[0]): #loop over areas
    # for jj in range(50):
        area_number = jj 
        area_index = np.where(atlas_data==area_number)
        area_values = img_data[area_index]
        
        max_value = np.max(area_values)
        mean_value = np.mean(area_values)
        
        df_max.iloc[jj,ii] = max_value
        df_mean.iloc[jj,ii] = mean_value


figsize_x = round(img.shape[-1]/10)
g = sns.clustermap(df_mean,
                col_cluster=False,
                cmap='rocket',
                z_score=0,
                figsize=(figsize_x,25),
                metric='correlation',
                yticklabels=1,
                xticklabels=10)

ax = g.ax_heatmap
ax.set_xlabel('time (samples)')
ax.set_ylabel('brain areas')
ax.tick_params(left=True, top=True, labelleft=True, labeltop=True, right=False, labelright=False, bottom=False, labelbottom=False)




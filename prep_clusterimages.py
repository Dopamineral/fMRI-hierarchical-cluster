import numpy as np
import nibabel as nib
import seaborn as sns 
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

atlas_file = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas_2mm.nii.gz"
file = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_rfMRI_REST1_LR.nii" 
legend_file = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas.txt"

img = nib.load(file)
atlas = nib.load(atlas_file)
legend = pd.read_csv(legend_file,sep='(',header=None)

df_max = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float") #make df size of areas x timepoints
df_mean = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float") #make df size of areas x timepoints
df_min = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float")
df_std = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float")
atlas_data = atlas.get_fdata() #load atlas values

img_shape = img.shape[-1]
for ii in range(img_shape): #loop over timepoints
# for ii in range(50):
    print(f'running timepoint {ii:04d} of {img_shape:04d}')
    sub_img = img.slicer[:,:,:,ii] 
    
     
    img_data = sub_img.get_fdata() #load vol values
    
    for jj in range(legend.shape[0]): #loop over areas
    # for jj in range(50):
        area_number = jj 
        area_index = np.where(atlas_data==area_number)
        area_values = img_data[area_index]
        
        max_value = np.max(area_values)
        mean_value = np.mean(area_values)
        min_value = np.min(area_values)
        std_value = np.std(area_values)
        
        df_max.iloc[jj,ii] = max_value
        df_mean.iloc[jj,ii] = mean_value
        df_min.iloc[jj,ii] = min_value
        df_std.iloc[jj,ii] = std_value
        


# figsize_x = round(img.shape[-1]/10)
# g = sns.clustermap(df_mean,
#                 col_cluster=False,
#                 cmap='rocket',
#                 z_score=0,
#                 figsize=(figsize_x,25),
#                 metric='correlation',
#                 yticklabels=1,
#                 xticklabels=10)

# ax = g.ax_heatmap
# ax.set_xlabel('time (samples)')
# ax.set_ylabel('brain areas')
# ax.tick_params(left=True, top=True, labelleft=True, labeltop=True, right=False, labelright=False, bottom=False, labelbottom=False)

def standard_scale_data(data):
    'scales data between 0 and 1'
    return (data - np.min(data)) / (np.max(data) - np.min(data))

min_vals = df_min.to_numpy()
max_vals = df_max.to_numpy()
mean_vals = df_mean.to_numpy()
std_vals = df_std.to_numpy()

#normalize these values across axis=1 before passing to make_rgb_value
from sklearn.preprocessing import normalize



def make_rgb_image(rdata,gdata,bdata,kron_val=3):
    '''makes rgb image from input data, scales everything to 0-255'''
        
    r = standard_scale_data(rdata)*255
    g = standard_scale_data(gdata)*255
    b = standard_scale_data(bdata)*255
    
    if kron_val > 0:
        #scale up image x3 with kronecker product in numpy:
        n = kron_val
        r = np.kron(r, np.ones((n,n)))
        g = np.kron(g, np.ones((n,n)))
        b = np.kron(b, np.ones((n,n)))
        
    
    rgb = np.dstack((r,g,b)).astype(np.uint8)
    
    img_rgb = Image.fromarray(rgb)
    return img_rgb

# img_rgb.save('rgb_image_language.png')


#%% image from correlations
max_corr = df_max.T.corr()
mean_corr = df_mean.T.corr()
std_corr = df_std.T.corr()

plt.figure(figsize=(29,25))
ax = sns.heatmap(max_corr,yticklabels=1,xticklabels=1)
sns.heatmap(mean_corr)
sns.heatmap(std_corr)


test_img = make_rgb_image(max_corr,mean_corr,std_corr)
test_img.save('corr_test.png')

#%% Create sliding window
df = df_max
window_size = 50
ii=0
iimax = df.shape[1] - window_size
def combine_corr_ts(df,ii,window_size):
   
    data_norm = normalize(df,axis=1)
    data_standard = standard_scale_data(data_norm)
    
    data_windowed = data_standard[:,ii:ii+window_size]
  
    
    df_windowed = df.iloc[:,ii:ii+window_size]
    data_corr = df_windowed.T.corr()
    data_corr = data_corr.to_numpy()
    data_corr_standard = standard_scale_data(data_corr)
    
    data_out = np.concatenate((data_corr_standard,data_windowed),axis=1)
    
    return data_out

for ii in range(0,iimax,5):
    max_image_data = combine_corr_ts(df_max,ii,window_size)
    mean_image_data = combine_corr_ts(df_mean,ii,window_size)
    std_image_data = combine_corr_ts(df_std,ii,window_size)
    
    
    fig, axs = plt.subplots(2,3,figsize = (15,6))
    
    g1 = sns.heatmap(max_image_data,ax=axs[0,0],cmap = 'rocket')
    g2 = sns.heatmap(mean_image_data,ax=axs[0,1],cmap='rocket')
    g3 = sns.heatmap(std_image_data,ax=axs[0,2],cmap='rocket')
    
    axs[0,0].set_title('Max Intensity')
    axs[0,1].set_title('Mean Intensity')
    axs[0,2].set_title('Std of Intensity')
    
    image_rgb = make_rgb_image(max_image_data,mean_image_data,std_image_data)
    axs[1,1].imshow(image_rgb)
    axs[1,0].axis('off')
    axs[1,2].axis('off')
    plt.show()
    

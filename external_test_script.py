from prep_clusterimages import ImagePrepper
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

loader = ImagePrepper(working_directory="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/",
                      SUBJECT=1,
                      atlas_file="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas_2mm.nii.gz",
                      file_rs="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_rfMRI_REST1_LR.nii",
                      file_tb1="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_LANGUAGE_LR.nii",
                      file_tb2="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_MOTOR_LR.nii",
                      legend_file="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas.txt")

# loader.output_images(loader.file_tb1,'images_tb',file_specifier='lang',step_size=5)
# loader.output_images(loader.file_tb2,'images_tb',file_specifier='mot',step_size=5)
# loader.output_images(loader.file_rs,'images_rs',file_specifier='rs',step_size=10)

df_max, df_mean_tb2, df_min, df_std = loader.dfs_from_nii(loader.file_tb2)
df_max, df_mean_tb1, df_min, df_std = loader.dfs_from_nii(loader.file_tb1)
df_max, df_mean_rs, df_min, df_std = loader.dfs_from_nii(loader.file_tb1)


def compare_mean_windowed_corr(df,window_size):
    tsize = df.shape[1]
    corr_size = df.shape[0]
    timewindow = tsize-window_size
    result_matrix = np.zeros([corr_size,corr_size,timewindow])
    
    print('Calculating for timewindow')
    for ii in tqdm(range(tsize-window_size),position=0):
        test = loader.combine_corr(df, ii, window_size)
        result_matrix[:,:,ii] = test
    
    complete_corr = np.array(df.T.corr())
    mean_corr = np.mean(result_matrix,axis=2)
    
    result_corr = mean_corr - complete_corr
    return result_corr


result_corr_tb2 = compare_mean_windowed_corr(df_mean_tb2,50)

sns.heatmap(result_corr_tb2)

fig, axs = plt.subplots(1,3,figsize=(32,8))

axs[0].set_title('tb1')
sns.heatmap(result_corr,ax=axs[0])

axs[1].set_title('tb2')
sns.heatmap(result_corr,ax=axs[1])

axs[2].set_title('rs')
sns.heatmap(result_corr,ax=axs[2])

plt.show()

legend = loader.get_legend()

brain_areas_short = legend.iloc[:,0]
brain_areas_long = legend.iloc[:,1]

indices = pd.Series([str(x) for x in range(132)])
brain_areas = brain_areas_short + '(' + brain_areas_long + ' ' +indices



g = sns.clustermap(result_corr_tb2,
               yticklabels=brain_areas_short,
               xticklabels = brain_areas_short,
               figsize=(30,25),
               )
ax = g.ax_heatmap
ax.set_xlabel('time (samples)')
ax.set_ylabel('clustered voxels')
ax.tick_params(left=True, top=True, labelleft=True, labeltop=True, right=False, labelright=False, bottom=False, labelbottom=False)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

def investigate_single_correlations(area1,area2):
    A1 = df_mean.iloc[area1,:]
    A2 = df_mean.iloc[area2,:]
    
    A1_norm = loader.standard_scale_data(A1)
    A2_norm = loader.standard_scale_data(A2)
    
    complete_corr = np.array(df_mean_tb2.T.corr())
    area_complete_corr = complete_corr[area1,area2]
    area_corr = A1_norm.corr(A2_norm)
    
    print(f'complete_corr for areas {area1} and {area2}: {area_complete_corr:.04f}, area_corr:{area_corr:.04f}')
    
    fig= plt.figure()
    p1, = plt.plot(A1_norm)
    p2, = plt.plot(A2_norm)
    plt.title(f'full signal areas {area1} and {area2}. Corr = {area_corr:.04f}')
    plt.legend([p1,p2],[f'area {area1}',f'area {area2}'])

    return A1_norm, A2_norm, area_corr

A1,A2, area_corr = investigate_single_correlations(29, 32)


window_size = 50
corr_list = []

for ii in range(len(A1)-window_size):
    A1_sample = A1.iloc[ii:ii+window_size]
    A2_sample = A2.iloc[ii:ii+window_size]
    
    A1_norm = loader.standard_scale_data(A1_sample)
    A2_norm = loader.standard_scale_data(A2_sample)
    
    sample_corr = A1_norm.corr(A2_norm)
    corr_list.append(sample_corr)
    mean_corr = np.mean(corr_list)
    
    fig= plt.figure()
    p1, = plt.plot(A1_norm)
    p2, = plt.plot(A2_norm)
    plt.title(f'sample signal over windowsize {window_size} Corr = {sample_corr:.04f} \n mean correlation so far:{mean_corr}')
    plt.show()
    
mean_corr = np.mean(corr_list)

result_corr = mean_corr - area_corr
print(f'mean corr: {mean_corr:.04f}, area corr: {area_corr:.04f}  resulting corr: {result_corr:.04f}')

    



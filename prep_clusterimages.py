import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
import os
from sklearn.preprocessing import normalize
from tqdm import tqdm

# working_directory = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/" 
# SUBJECT = 1

# atlas_file = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas_2mm.nii.gz"
# file_rs = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_rfMRI_REST1_LR.nii" 
# file_tb1 = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_LANGUAGE_LR.nii"
# file_tb2 = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_MOTOR_LR.nii"
# legend_file = "C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas.txt"


class ImagePrepper():
    def __init__(self, working_directory,SUBJECT,atlas_file,file_rs,file_tb1,file_tb2,legend_file):
        self.working_directory = working_directory
        self.SUBJECT = SUBJECT 
    
        self.atlas_file = atlas_file 
        self.file_rs = file_rs
        self.file_tb1 = file_tb1
        self.file_tb2 = file_tb2
        self.legend_file = legend_file
    
    def get_legend(self):
        legend = pd.read_csv(self.legend_file,sep='(',header=None)
        return legend
    
    def dfs_from_nii(self,file):
        ''' creates dfs for each brain area in a certain atlas / legend conformation
            does this for a file: .nii
        '''
        file = file
        atlas_file = self.atlas_file
        legend_file = self.legend_file
        
            
        img = nib.load(file)
        atlas = nib.load(atlas_file)
        atlas_data = atlas.get_fdata() #load atlas values
        legend = pd.read_csv(legend_file,sep='(',header=None)
        
        df_max = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float") #make df size of areas x timepoints
        df_mean = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float") #make df size of areas x timepoints
        df_min = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float")
        df_std = pd.DataFrame(index=legend[0],columns=range(img.shape[-1]),dtype="float")
        
        
        img_shape = img.shape[-1]
        print(f'extracting from timepoints of {img_shape:04d} from subject:{self.SUBJECT}')
        for ii in tqdm(range(img_shape),position=0): #loop over timepoints
        # for ii in range(50):
            # print(f'extracting from timepoint {ii:04d} of {img_shape:04d} from subject:{self.SUBJECT}')
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
                
        return df_max, df_mean, df_min, df_std
            
    
    
    def standard_scale_data(self,data):
        'scales data between 0 and 1'
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    
    def make_rgb_image(self,rdata,gdata,bdata,kron_val=3):
        '''makes rgb image from input data, scales everything to 0-255'''
            
        r = self.standard_scale_data(rdata)*255
        g = self.standard_scale_data(gdata)*255
        b = self.standard_scale_data(bdata)*255
        
        if kron_val > 0:
            #scale up image x3 with kronecker product in numpy:
            n = kron_val
            r = np.kron(r, np.ones((n,n)))
            g = np.kron(g, np.ones((n,n)))
            b = np.kron(b, np.ones((n,n)))
            
        
        rgb = np.dstack((r,g,b)).astype(np.uint8)
        
        img_rgb = Image.fromarray(rgb)
        return img_rgb
    
    def combine_corr_ts(self,df,ii,window_size):
        '''combines correlation images and time series into one big RGB composite
        returns RGB image object'''
       
        data_norm = normalize(df,axis=1)
        data_standard = self.standard_scale_data(data_norm)
        
        data_windowed = data_standard[:,ii:ii+window_size]
      
        
        df_windowed = df.iloc[:,ii:ii+window_size]
        data_corr = df_windowed.T.corr()
        data_corr = data_corr.to_numpy()
        data_corr_standard = self.standard_scale_data(data_corr)
        
        data_out = np.concatenate((data_corr_standard,data_windowed),axis=1)
        
        return data_out
    
    def combine_ts(self,df,ii,window_size):
        '''combines correlation images and time series into one big RGB composite
        returns RGB image object'''
       
        data_norm = normalize(df,axis=1)
        data_standard = self.standard_scale_data(data_norm)
        
        data_windowed = data_standard[:,ii:ii+window_size]
        
        return data_windowed
    
    def combine_corr(self,df,ii,window_size):
        '''combines correlation images and time series into one big RGB composite
        returns RGB image object'''
       
        data_norm = normalize(df,axis=1)
        data_standard = self.standard_scale_data(data_norm)
        
        data_windowed = data_standard[:,ii:ii+window_size]
      
        df_windowed = df.iloc[:,ii:ii+window_size]
        data_corr = df_windowed.T.corr()
        data_corr = data_corr.to_numpy()
        
        return data_corr
    
    def output_images(self,file,img_path,file_specifier='',window_size=50,step_size=25):
        ''' outputs image objects to png files in the defined file directory
        add a custom file specifier if you want to add multiple images to the same folder'''
        
        #LOAD DFS HERE - Takes Long
        df_max, df_mean, df_min, df_std =  self.dfs_from_nii(file)
        
        
        wd = self.working_directory
        os.chdir(wd)
        IMAGE_DIR_PATH = img_path
        if not os.path.exists(IMAGE_DIR_PATH):
            print(f'making images directory:{IMAGE_DIR_PATH}')
            os.mkdir(IMAGE_DIR_PATH)
        
        os.chdir(IMAGE_DIR_PATH)
        
        
        window_size = window_size
        ii=0
        iimax = df_max.shape[1] - window_size
        
       
        
        #Output the images
        print( f' saving IMG_subject{self.SUBJECT:03d}_{file_specifier} - INDEX.png')
        for ii in tqdm(range(0,iimax,step_size)):
            max_image_data = self.combine_corr_ts(df_max,ii,window_size)
            mean_image_data = self.combine_corr_ts(df_mean,ii,window_size)
            std_image_data = self.combine_corr_ts(df_std,ii,window_size)
            
            image_rgb = self.make_rgb_image(max_image_data,mean_image_data,std_image_data)
            
            img_string = f'IMG_subject{self.SUBJECT:03d}_{file_specifier}{ii:04d}.png'
            
            # print(f'saving {img_string}')
            image_rgb.save(img_string)
            
            #UNCOMMENT IF YOU WNAT TO SEE PLOTS
            # fig, axs = plt.subplots(2,3,figsize = (15,6))
            
            # g1 = sns.heatmap(max_image_data,ax=axs[0,0],cmap = 'rocket')
            # g2 = sns.heatmap(mean_image_data,ax=axs[0,1],cmap='rocket')
            # g3 = sns.heatmap(std_image_data,ax=axs[0,2],cmap='rocket')
            
            # axs[0,0].set_title('Max Intensity')
            # axs[0,1].set_title('Mean Intensity')
            # axs[0,2].set_title('Std of Intensity')
            
            
            # axs[1,1].imshow(image_rgb)
            # axs[1,0].axis('off')
            # axs[1,2].axis('off')
            # plt.show()
        os.chdir(wd)
        
        
        
def main():
    pass
    # loader = ImagePrepper(working_directory="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/",
    #                       SUBJECT=2,
    #                       atlas_file="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas_2mm.nii.gz",
    #                       file_rs="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas_2mm.nii.gz",
    #                       file_tb1="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_LANGUAGE_LR.nii",
    #                       file_tb2="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_MOTOR_LR.nii",
    #                       legend_file="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas.txt")
    
   
    # # df_max, df_mean, df_min, df_std =  loader.dfs_from_nii(loader.file_tb1)
    # loader.output_images('images_tb',file_specifier='lang')
    
    
if __name__=='__main__':
    main()
    
    
    
    
        

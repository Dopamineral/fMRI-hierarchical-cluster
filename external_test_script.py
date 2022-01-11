from prep_clusterimages import ImagePrepper

loader = ImagePrepper(working_directory="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/",
                      SUBJECT=1,
                      atlas_file="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas_2mm.nii.gz",
                      file_rs="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_rfMRI_REST1_LR.nii",
                      file_tb1="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_LANGUAGE_LR.nii",
                      file_tb2="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/swausub-01_tfMRI_MOTOR_LR.nii",
                      legend_file="C:/Users/gebruiker/OneDrive - KU Leuven/Desktop/atlas.txt")

loader.output_images(loader.file_tb1,'images_tb',file_specifier='lang',step_size=5)
loader.output_images(loader.file_tb2,'images_tb',file_specifier='mot',step_size=5)
loader.output_images(loader.file_rs,'images_rs',file_specifier='rs',step_size=5)


# fMRI-hierarchical-cluster

Playing with visualizing fMRI data in 2D. 

Decompising 4D image into 2D (voxels x time) axes and then applying hierarchical clustering (metric=correlation) to the voxels. Pretty interesting.

Examples:
![prefrontal cluster](https://github.com/Dopamineral/fMRI-hierarchical-cluster/blob/main/examples/hierarchical_cluster_prefrontal.png)
![amygdala cluster](https://github.com/Dopamineral/fMRI-hierarchical-cluster/blob/main/examples/hierarchical_cluster_rs_amygdala.png)
For all image examples: x-axis = time (samples), y-axis = clustered voxels

# Hierarchical Cluster from Atlas
Same principle applies but extracting the mean or max values from a set of anatomical areas as defined by an atlas. 
The figure created will scale tot the time series of brainscan used.
![language](https://github.com/Dopamineral/fMRI-hierarchical-cluster/blob/main/examples%20cluster%20by%20atlas/Language.png)
![motor](https://github.com/Dopamineral/fMRI-hierarchical-cluster/blob/main/examples%20cluster%20by%20atlas/Motor.png)
![rest](https://github.com/Dopamineral/fMRI-hierarchical-cluster/blob/main/examples%20cluster%20by%20atlas/REST.png)

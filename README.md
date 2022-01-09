# fMRI-hierarchical-cluster

Playing with visualizing fMRI data in 2D. 

Decompising 4D image into 2D (voxels x time) axes and then clustering then applying hierarchical clustering (metric=correlation) to the voxels. Pretty interesting.

Example images can be found in examples folder. 
For all image examples: x-axis = time (samples), y-axis = clustered voxels

# Hierarchical Cluster from Atlas
Same principle applies but extracting the mean or max values from a set of anatomical areas as defined by an atlas. 
The figure created will scale tot the time series of brainscan used.

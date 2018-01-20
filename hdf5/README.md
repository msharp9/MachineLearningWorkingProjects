hdf5 databases are very useful to train large datasets.  Here I give some code to convert images into an hdf5 database.
It sizes the images to 256x256, a test and validation set of 15% of the data, and also calculates the mean of each channel.

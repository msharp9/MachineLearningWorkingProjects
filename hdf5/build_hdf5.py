# USAGE
# python build_hdf5.py --image_path .. --output output

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import json
import cv2
import os

# construct the argument parser, only need an image input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="path to the image files")
ap.add_argument("-o", "--output", required=True, help="path for outputs")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())

# grab the paths to the images
imagePaths = list(paths.list_images(args["image_path"]))
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# perform stratified sampling from the training set to build the testing split from the training data
split = train_test_split(imagePaths, labels,
	test_size=round(len(imagePaths)*0.15), stratify=labels)
(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the validation data
split = train_test_split(trainPaths, trainLabels,
	test_size=round(len(imagePaths)*0.15), stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5 files
datasets = [
	("train", trainPaths, trainLabels, os.path.join(args["output"], "train.hdf5")),
	("val", valPaths, valLabels, os.path.join(args["output"], "val.hdf5")),
	("test", testPaths, testLabels, os.path.join(args["output"], "test.hdf5"))]

# initialize the image pre-processor and the lists of RGB channel averages
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath, bufSize=args["buffer_size"])

	# initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()

	# loop over the image paths
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# load the image and process it
		image = cv2.imread(path)
		image = aap.preprocess(image)

		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the respective lists
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# add the image and label # to the HDF5 dataset
		writer.add([image], [label])
		pbar.update(i)

	# close the HDF5 writer
	pbar.finish()
	writer.close()

# construct a dictionary of averages, then serialize the means to a JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(os.path.join(args["output"], "train_mean.json"), "w")
f.write(json.dumps(D))
f.close()

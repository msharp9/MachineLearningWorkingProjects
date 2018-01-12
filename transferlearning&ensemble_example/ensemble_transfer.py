# USAGE
# python ensemble_transfer.py --output output --models models

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as pl
import argparse
import glob
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-m", "--models", required=True,
	help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=3,
	help="# of models to train")
args = vars(ap.parse_args())

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

# load the training and testing data, then scale it into the range [0, 1]
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
	height_shift_range=0.1, zoom_range=0.1,
	horizontal_flip=True, fill_mode="nearest")

# loop over the number of models to train
for i in np.arange(0, args["num_models"]):
	# initialize the optimizer and model
	print("[INFO] training model {}/{}".format(i + 1, args["num_models"]))

	# load the VGG16 network, ensuring the head FC layer sets are left off
	baseModel = VGG16(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))

	# initialize the new head of the network, a set of FC layers
	# followed by a softmax classifier
	headModel = FCHeadNet.build(baseModel, len(classNames), 256)

	# place the head FC model on top of the base model -- this will
	# become the actual model we will train
	model = Model(inputs=baseModel.input, outputs=headModel)

	# loop over all layers in the base model and freeze them so they
	# will *not* be updated during the training process
	for layer in baseModel.layers:
		layer.trainable = False

	# compile our model (this needs to be done after our setting our
	# layers to being non-trainable
	print("[INFO] compiling model...")
	opt = RMSprop(lr=0.001)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the head of the network for a few epochs (all other
	# layers are frozen) -- this will allow the new FC layers to
	# start to become initialized with actual "learned" values
	# versus pure random
	print("[INFO] training head...")
	model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
		validation_data=(testX, testY), epochs=20,
		steps_per_epoch=len(trainX) // 64, verbose=1)

	# now that the head FC layers have been trained/initialized, lets
	# unfreeze the final set of CONV layers and make them trainable
	for layer in baseModel.layers[15:]:
		layer.trainable = True

	# for the changes to the model to take affect we need to recompile
	# the model, this time using SGD with a *very* small learning rate
	print("[INFO] re-compiling model...")
	opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	# train the model again, this time fine-tuning *both* the final set
	# of CONV layers along with our set of FC layers
	print("[INFO] fine-tuning model...")
	model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
		validation_data=(testX, testY), epochs=50,
		steps_per_epoch=len(trainX) // 64, verbose=1)

	# save the model to disk
	print("[INFO] serializing model...")
	p = [args["models"], "model_{}.model".format(i)]
	model.save(os.path.sep.join(p))

	# evaluate the network
	predictions = model.predict(testX, batch_size=64)
	report = classification_report(testY.argmax(axis=1),
		predictions.argmax(axis=1), target_names=labelNames)

	# save the classification report to file
	p = [args["output"], "model_{}.txt".format(i)]
	f = open(os.path.sep.join(p), "w")
	f.write(report)
	f.close()

	# plot the training loss and accuracy
	p = [args["output"], "model_{}.png".format(i)]
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, 40), H.history["loss"],
		label="train_loss")
	plt.plot(np.arange(0, 40), H.history["val_loss"],
		label="val_loss")
	plt.plot(np.arange(0, 40), H.history["acc"],
		label="train_acc")
	plt.plot(np.arange(0, 40), H.history["val_acc"],
		label="val_acc")
	plt.title("Training Loss and Accuracy for model {}".format(i))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(os.path.sep.join(p))
	plt.close()

# Now we want to test our ensemble
# construct the path used to collect the models then initialize the models list
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelPaths))
# alternatively could have saved the models into the array as we created them
models = []

# loop over the model paths, loading the model, and adding it to the list of models
for (i, modelPath) in enumerate(modelPaths):
	print("[INFO] loading model {}/{}".format(i + 1, len(modelPaths)))
	models.append(load_model(modelPath))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions = []

# loop over the models
for model in models:
	# use the current model to make predictions on the testing data,
	# then store these predictions in the aggregate predictions list
	predictions.append(model.predict(testX, batch_size=64))

# average the probabilities across all model predictions, then show a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

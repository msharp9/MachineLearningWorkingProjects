AlexNet was a seminal architecture created in 2012. It achieved top-1 and top-5 test set error rates of 37.5% and 17.0% on ILSVRC-2010.  The next best network at the time achieved 45.7% and 25.7% error rates.

It took advantage of ReLU's and Dropouts, which were relatively new at the time.  It is also significant since it achieved great results with limited architecture, there are only 5 CNN layers (which was large at the time).

AlexNet Paper:
http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

It's important to note that the paper appears to have a typo as input images should be of size 227x227.


I trained a Dogs vs Cats model using AlexNet with the dataset found on kaggle: https://www.kaggle.com/c/dogs-vs-cats
I achieved a 89% accuracy.  Using the ten patch classification outlined in the paper I was able to increase that to 91.68%.
These aren't awesome results, just using simple transfer learning on ResNet50 gives better results.  AlexNet is useful to know as it was a pivoting architecture.
If you are looking to duplicate, I used the hdf5 code in this repository to prepare the data.

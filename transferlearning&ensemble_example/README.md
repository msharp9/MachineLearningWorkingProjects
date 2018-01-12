This code is a copy of the previous project, except with a twist.  We are going to incorporate ensemble learning into it as well.  We are also switching the dataset from flowers17 to cifar10 for simplicity.  Note: generally I would come up with a robust model then apply ensemble learning afterwards to make it more robust; however, this code is really just for practice/demonstration.

Transfer learning is process of taking an already trained model and applying into to a different dataset.  This example explores fine-tuning where the head of a model is replaced and retrained.  Generally requires a warm-up of the new FC head layers before back propagating to further layers.

I applied the VGG16 model trained on the ImageNet database to then be trained on the flowers17 dataset.

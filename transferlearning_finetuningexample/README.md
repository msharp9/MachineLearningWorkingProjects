Transfer learning is process of taking an already trained model and applying into to a different dataset.  This example explores fine-tuning where the head of a model is replaced and retrained.  Generally requires a warm-up of the new FC head layers before back propagating to further layers.

I applied the VGG16 model trained on the ImageNet database to then be trained on the flowers17 dataset.  Thanks to Dr. Adrian Rosebrock and pyimagesearch.  flowers17 dataset isn't included but can be found here: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/
flowers17.model is my personal output model. (too large to upload unfortunately)

inspect_model.py is a simple code to review an unknown model.
![Alt text](inspectmodel_VGG16.png?raw=true "Model Layers")

Accuracy after new FC layers have been "warmed-up"
![Alt text](warmup_scores.png?raw=true "Warmup Scores")

Final Model Evaluation
![Alt text](model_scores.png?raw=true "Model Scores")
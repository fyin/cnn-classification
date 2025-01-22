# cnn-classification
A practical LeNet-5 implementation, training and inference of convolutional neural network (CNN) for image classification.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n cnn-classification python=3.12`
* Activate the environment `conda activate cnn-classification`
* Install the dependencies `conda install --yes --file requirements.txt`

## Modules
* `model`: implements the LeNet-5 CNN model.
* `dataset`: load dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and create dataloader for batch training and validation.
* `config`: contains the configuration for the model and the training process.
* `train`: contains the code to train and validate the model. In the training, Kaiming normal distribution is used for weight initialization and Adam optimizer is used for parameter optimization.

## Training
  * Run script directly, `python train.py` 
  * All training and evaluation processes are logged to Tensorboard. Each model epoch checkpoint is saved for later inference.  
  * It was trained on mps device. After 10 epochs, the model accuracy is about 99%.

## Visualization
* Use Tensorboard to visualize the weights, weight gradients, loss and accuracy during training process.
  * Run `tensorboard --logdir runs/model_tb_logs`
  * Open `http://localhost:6006` in your browser

## Inference
* Run script directly, `python inference.py`
* Use the latest model checkpoint for inference, which predicts the class of the input image you provide.

## References
* https://www.youtube.com/watch?v=jDe5BAsT2-Y
* https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
* https://pytorch.org/get-started/locally/
* https://arxiv.org/pdf/1803.09820

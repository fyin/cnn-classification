# cnn-classification/lenet
A practical LeNet-5 implementation, training and inference of convolutional neural network (CNN) for image classification.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n cnn-classification-lenet python=3.12`
* Activate the environment `conda activate cnn-classification-lenet`
* Install the dependencies `conda install --yes --file lenet/requirements.txt && conda install --yes --file tests/requirements.txt`

## Modules
* `model`: implements the LeNet-5 CNN model.
* `dataset`: load dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and create dataloader for batch training and validation.
* `config`: contains the configuration for the model and the training process.
* `train`: contains the code to train and validate the model. In the training, Kaiming normal distribution is used for weight initialization and Adam optimizer is used for parameter optimization.
* `inference`: contains the code to predict the class of the input image.

## Training
* Run script directly, `python3 -m lenet.train` or just run train.py script in your IDE.
* All training and evaluation processes are logged to Tensorboard. Each model epoch checkpoint is saved for later deployment and inference.
    * To visualize the weights, weight gradients, loss and accuracy during training process in Tensorboard,
      * Run `tensorboard --logdir runs/lenet_tensorboard_logs` to start Tensorboard.
      * Open `http://localhost:6006` in your browser to view the Tensorboard logs.
* It was trained on mps device for 10 epochs. According to Tensorboard logs, after 5 epochs, 
  * model accuracy reaches ~99%, 
  * train loss keeps decreasing but eval loss starts to increase. To prevent from overfitting, early stopping is used. Checkpoint at epoch 5 is used for inference.
* Also tried to train the model using SGD optimizer to compare with Adam optimizer performance. As expected, Adam optimizer performs better, it trains faster to get the same accuracy. 

## Inference
* Run script directly, `python3 -m lenet.inference` or just run inference.py script in your IDE.
* Use the latest model checkpoint for inference, which predicts the class of the input image you provide.  

**Note**: Here just for a quick local prediction. For a real time inference, you need to deploy the model to a server.

## Run unit test
* Run `pytest tests/`

## References
* https://www.youtube.com/watch?v=jDe5BAsT2-Y
* https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
* https://pytorch.org/get-started/locally/
* https://arxiv.org/pdf/1803.09820

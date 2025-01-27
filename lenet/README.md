# cnn-classification/lenet
A practical LeNet-5 implementation, training and inference of convolutional neural network (CNN) for image classification.

## Dependency Management
Use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage the environment and 3rd party libraries.
All the required dependencies are put in requirements.txt.
* Create an environment `conda create -n cnn-classification-lenet python=3.12`
* Activate the environment `conda activate cnn-classification-lenet`
* Install the dependencies 
  * `conda install --yes --file lenet/requirements.txt && conda install --yes --file tests/requirements.txt`
  * `conda install -c conda-forge "ray-default`
  * `conda install -c conda-forge pyyaml`
  * `conda install -c conda-forge "ray-train`

## Modules
* `model`: implements the LeNet-5 CNN model.
* `dataset`: load dataset [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and create dataloader for batch training and validation.
* `train`: contains the code to train and validate the model. In the training, Kaiming normal distribution is used for weight initialization and Adam optimizer is used for parameter optimization.
* `ray_tune`: contains the code to automatically tune hyperparameters using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).
* `inference`: contains the code to predict the class of the input image.

## Training
* Run script directly, `python3 -m lenet.ray_tune` or just run ray_tune.py script in your IDE.
* Hyperparameter tuning results,
        
|Trial name | status | learning_rate | batch_size | optimizer | iter | total time (s) | train_loss | eval_loss | accuracy | 
        |----------|--------|---------------|-----------|-----------|-----|----------------|-----------|----------|----------|
        | train_model_e5590_00000 | TERMINATED | 0.000433557 | 16 | Adam | 8 | 480.026 | 0.0355927 | 0.0466648 | 98.5 |
        | train_model_e5590_00001 | TERMINATED | 0.000118775 | 32 | Adam | 1 | 42.4072 | 0.862882 | 0.315937 | 90.16 |
        | train_model_e5590_00002 | TERMINATED | 0.0013643 | 64 | Adam | 8 | 386.701 | 0.0456127 | 0.0453391 | 98.62 | 
        | train_model_e5590_00003 | TERMINATED | 0.000751662 | 16 | SGD | 4 | 421.861 | 0.0361178 | 0.0519034 | 98.58 | 
        | train_model_e5590_00004 | TERMINATED | 0.0027749 | 32 | SGD | 8 | 406.966 | 0.0310195 | 0.0533711 | 98.65 | 
        | train_model_e5590_00005 | TERMINATED | 0.000181391 | 64 | SGD | 1 | 29.0651 | 0.531067 | 0.165373 | 94.83 |

* From the tuning results, the best model is `train_model_e5590_00002` considering the lowest evaluation loss and highest accuracy and lower time cost. 
But train_model_e5590_00000 and train_model_e5590_00004  are also good models.
* All training and evaluation processes are logged to Tensorboard. Model epoch checkpoint is saved for later deployment and inference.
    * To visualize the loss and accuracy during training process in Tensorboard,
      * Run `tensorboard --logdir [your logging directory]` to start Tensorboard 
      * Open `http://localhost:6006` in your browser to view the Tensorboard logs.

## Inference
* Run script directly, `python3 -m lenet.inference` or just run inference.py script in your IDE.
* Use the best model checkpoint for inference, which predicts the class of the input image you provide.  

**Note**: Here just for a quick local prediction. For a real time inference, you need to deploy the model to a server.

## Run unit test
* Run `pytest tests/`

## References
* https://www.youtube.com/watch?v=jDe5BAsT2-Y
* https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
* https://pytorch.org/get-started/locally/
* https://docs.ray.io/en/latest/tune/index.html
* https://arxiv.org/pdf/1803.09820

---

# Neural Network

## Dipanshu's DA6401 - Assignment 1

This repository contains a Python script that trains a Feed Forward Neural Network using NumPy. The neural network is flexible and can be adjusted to work with various configurations. It's great for classifying datasets like MNIST or Fashion MNIST. You can easily customize it by adding different activation functions, loss functions, and other parameters as needed.

---

# Code-Structure

## Class `NeuralNetwork`

### Attributes:

- `dataset`: Choosen dataset for training ("mnist" or "fashion_mnist").
- `neuron`: Number of neurons in each hidden layer.
- `act_func`: Activation function used in the network.
- `out_act_func`: Activation function used in the network.
- `lossFunc`: Loss function used for training.
- `optimizer`: Optimizer used for updating network parameters.
- `iter`: Number of training epochs.
- `n`: Learning rate for optimization.
- `batch`: Batch size for training.
- `init`: Method for weight initialization.
- `hiddenlayers`:Number of hidden layers inside the network.
- `A`, `H`: Dictionary to store Pre-activation and Activation for the layers.
- `W`, `B`: Dictionary to store Weights and Biases for the layers.
- `dw`, `db`: Dictionary to store weights and biases gradients for the layers.


### Methods:
- `make_layers`: initializers the weights and baises with help of Xavier or random.
- `forward_pass`:forward propogation is done and prdiction is made.
- `backward_pass`: do backward propogation and compute gradient.
- `accuracy`: Accuracy and loss of the model is calculated.
- `fit_model`: Iterate through epochs and batches for training.
- `SGD`:Trains the model based on given parameters using sgd.
- `Momentum`:Trains the model based on given parameters using momentum.
- `NAG`:Trains the model based on given parameters using nag.
- `RMSProp`:Trains the model based on given parameters using rmsprop.
- `ADAM`:Trains the model based on given parameters using adam.
- `nADAM`:Trains the model based on given parameters using nadam.

## Features

- The neural network architecture is made flexible.
- You can pick from different activation functions like identity, sigmoid, tanh, or ReLU.
- There are multiple options for loss functions including Cross Entropy and Mean Squared Error.
- You have various optimization algorithms to choose from such as SGD, Momentum, NAG, RMSprop, Adam, and Nadam.
- You can initialize weights using either random values or the Xavier method.
-Plus, you can visualize your training metrics using Weights & Biases.
 
## Parameters

- `-d`, `--dataset`: Choose the dataset for training ("mnist" or "fashion_mnist").
- `-e`, `--epochs`: Number of training epochs.
- `-b`, `--batch_size`: Batch size for training.
- `-l`, `--loss`: Loss function for training ("mean_squared_error" or "cross_entropy").
- `-o`, `--optimizer`: Optimization algorithm ("sgd", "momentum", "nag", "rmsprop", "adam", "nadam").
- `-lr`, `--learning_rate`: Learning rate for optimization.
- `-m`: Momentum for Momentum and NAG optimizers.
- `-beta1`, `--beta1`: Beta1 parameter for Adam and Nadam optimizers.
- `-beta2`, `--beta2`: Beta2 parameter for Adam and Nadam optimizers.
- `-w_i`, `--weight_init`: Weight initialization method ("random" or "Xavier").
- `-nhl`, `--num_layers`: Number of hidden layers in the neural network.
- `-sz`, `--hidden_size`: Number of neurons in each hidden layer.
- `-a`, `--activation`: Activation function for hidden layers ("identity", "sigmoid", "tanh", "ReLU").
- `-cl`, `--console_log`: Log training metrics on Console (0: disable, 1: enable).
- `-wl`, `--wandb_log`: Log training metrics on Weights & Biases (0: disable, 1: enable).

# How to Train a Model

To train a model, run:

```bash
python train.py --wandb_entity myname --wandb_project myprojectname

# Neural Network - Doodle Trainer and Quick Test

## Overview

This repository contains a Python implementation of a neural network with the Sigmoid activation function and backpropagation training, utilizing the `numpy` library for numerical computations.

## Mini Projects

### Doodle Trainer

The `doodle_trainer.py` script trains the neural network on a doodle dataset, containing hand-drawn images of objects like cars, flowers, and animals. The neural network classifies these doodles into their respective categories.

### Quick Test

The `quick_test_attempt.py` script offers a quick test to check the neural network's learning progress. It includes a visual aid to show how the neural network's predictions change during training, using a simple, linearly separable dataset with two classes.

## Installation

To use this project, ensure you have Python 3.x and `pip` installed. Then, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/JibrilMamo/neural_network.git
    cd neural_network
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use this neural network for various tasks such as image classification and pattern recognition. Here's how:

1. **Train the neural network on a dataset:**

    Open `doodle_trainer.py` and modify the dataset paths, network architecture, and training parameters to suit your task.

2. **Run the doodle trainer script:**

    ```bash
    python doodle_trainer.py
    ```

    This script will train the neural network on the doodle dataset, displaying training progress and accuracy.

3. **Use the trained model for doodle classification:**

    After training, use the trained neural network to classify new doodles into their respective categories.

4. **Run the quick test script:**

    ```bash
    python quick_test_attempt.py
    ```

    This script will run a simple test, displaying the neural network's predictions as they change during training.



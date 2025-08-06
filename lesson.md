# Lesson: Introduction to the Keras Sequential API

This lesson provides a theoretical understanding of the concepts demonstrated in the Keras Sequential API application.

## 1. The Keras Sequential API

The **Sequential API** is the simplest way to get started with Keras. It allows you to build a neural network layer by layer, in a step-by-step fashion. It is ideal for most common network architectures but is not suitable for models with multiple inputs or outputs, or for models with complex, non-linear topologies.

### Key Concepts:

-   **Layers:** The fundamental building blocks of a neural network. Each layer performs a specific transformation on the data that passes through it.
-   **Units (Neurons):** Each layer is composed of a number of units, or neurons. In a `Dense` layer, every neuron is connected to every neuron in the previous layer.
-   **Activation Functions:** These functions determine the output of a neuron. They introduce non-linearity into the model, allowing it to learn complex patterns.
    -   **ReLU (Rectified Linear Unit):** A popular and effective activation function. It outputs the input directly if it is positive, and zero otherwise.
    -   **Sigmoid:** Often used in the output layer for binary classification. It squashes the output to a value between 0 and 1, which can be interpreted as a probability.

## 2. Compiling the Model

Before you can train a model, you need to configure its learning process. This is done with the `compile()` method, which takes three important arguments:

-   **Optimizer:** The algorithm that adjusts the weights of the network to minimize the loss. `adam` is a widely used optimizer that is efficient and requires little configuration.
-   **Loss Function:** This function measures how well the model is performing on the training data. The goal of training is to minimize this function.
    -   **`binary_crossentropy`:** Used for binary (two-class) classification problems.
-   **Metrics:** These are used to monitor the performance of the model during training and testing. `accuracy` is a common metric that calculates the percentage of correct predictions.

## 3. Training the Model

Training is the process of feeding the model with data and allowing it to learn the underlying patterns. This is done with the `fit()` method.

### Key Concepts:

-   **Epochs:** One epoch is a complete pass of the entire training dataset through the network. Training is typically done for multiple epochs to allow the model to learn effectively.
-   **Batch Size:** The training data is usually divided into smaller batches. The batch size is the number of samples that are processed before the model's weights are updated. Training in batches is more memory-efficient and can lead to faster convergence.

## 4. Evaluating the Model

After training, it's crucial to evaluate the model on data it has never seen before. This is done with the `evaluate()` method. This step helps you understand how well your model **generalizes** to new, unseen data.

## 5. Making Predictions

Once you have a trained and evaluated model, you can use it to make predictions on new data. The `predict()` method takes new data as input and returns the model's output. For a classification problem, this output can be interpreted as a class probability or a direct class prediction.

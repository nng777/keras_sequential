# Keras Sequential API Demonstration App

This application provides a simple, hands-on introduction to the Keras Sequential API, a fundamental component of building neural networks with TensorFlow.

It is designed for AI Engineering students to understand the basic workflow of creating, compiling, training, and evaluating a neural network.

## How to Run the App

1.  **Install Dependencies:**
    You will need to have TensorFlow installed. If you don't have it, you can install it using pip:
    ```bash
    pip install tensorflow
    ```

2.  **Execute the Script:**
    Run the `keras_sequential_api_app.py` script from your terminal:
    ```bash
    python keras_sequential_api_app.py
    ```

## What the App Demonstrates

The script is divided into five main steps, each with detailed comments to guide you through the process:

1.  **Define the Model:** Shows how to create a `Sequential` model and add layers to it.
2.  **Compile the Model:** Explains the purpose of the `compile()` method and its key arguments: optimizer, loss function, and metrics.
3.  **Train the Model:** Demonstrates how to train the model on sample data using the `fit()` method, and explains the concepts of epochs and batch size.
4.  **Evaluate the Model:** Shows how to evaluate the model's performance on unseen test data with the `evaluate()` method.
5.  **Make Predictions:** Illustrates how to use the trained model to make predictions on new data using the `predict()` method.

By running this application, you will see the output of each step, from the model summary to the final predictions, giving you a clear picture of the entire process.

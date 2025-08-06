# Homework: Build Your First Neural Network

**Objective:** Apply your understanding of the Keras Sequential API to build and train a simple neural network for a different type of problem.

## Your Task

Your task is to create a Python script that builds a neural network for a **regression problem**. In regression, the goal is to predict a continuous value instead of a class.

### Requirements:

1.  **Create a new Python script** named `my_first_regression_model.py`.

2.  **Generate Sample Data:**
    -   Create a training set of 100 samples with 5 features each (`X_train`).
    -   Create a corresponding set of target values (`y_train`) that are continuous numbers (e.g., `50 * np.random.rand(100, 1)`).

3.  **Build the Model:**
    -   Use the Keras `Sequential` API.
    -   Create a model with at least two `Dense` layers.
    -   The final layer should have a single unit and a `linear` activation function, which is suitable for regression.

4.  **Compile the Model:**
    -   Use the `adam` optimizer.
    -   For the loss function, use `mean_squared_error`, which is a common choice for regression problems.

5.  **Train and Evaluate:**
    -   Train the model for 20 epochs.
    -   Create a small test set to evaluate your model's performance.

**Bonus:** After training, use the model to make a prediction on a single new data sample and print the result.

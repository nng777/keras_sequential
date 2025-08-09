"""
The task is to create a Python script that builds a neural network for "a regression problem". In regression, the goal is to predict a continuous value instead of a class.

Requirements:
1.Create a new Python script named my_first_regression_model.py.
2.Generate Sample Data:
  2.1.Create a training set of 100 samples with 5 features each (X_train).
  2.2.Create a corresponding set of target values (y_train) that are continuous numbers (e.g., 50 * np.random.rand(100, 1)).
3.Build the Model:
  3.1.Use the Keras Sequential API.
  3.2.Create a model with at least two Dense layers.
  3.3.The final layer should have a single unit and a linear activation function, which is suitable for regression.
4.Compile the Model:
  4.1.Use the adam optimizer.
  4.2.For the loss function, use mean_squared_error, which is a common choice for regression problems.
5.Train and Evaluate:
  5.1.Train the model for 20 epochs.
  5.2.Create a small test set to evaluate your model's performance.
"""
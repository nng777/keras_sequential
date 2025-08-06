
import tensorflow as tf
from tensorflow import keras
import numpy as np

# --- 1. Introduction to Keras Sequential API ---
# The Sequential model is a linear stack of layers. It's the simplest way to build a model in Keras.
# You can create a Sequential model by passing a list of layer instances to the constructor.

print("--- Step 1: Define the Model ---")
# We define a simple neural network with three layers:
# - An input layer with 10 units and a ReLU activation function. The input_shape is (10,) because our data has 10 features.
# - A hidden layer with 8 units and a ReLU activation function.
# - An output layer with 1 unit and a sigmoid activation function, suitable for binary classification.

model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# You can also build a model by adding layers one by one using the .add() method:
# model = keras.Sequential()
# model.add(keras.layers.Dense(10, activation='relu', input_shape=(10,)))
# model.add(keras.layers.Dense(8, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))

# Display a summary of the model's architecture
model.summary()

# --- 2. Compile the Model ---
# Before training, we need to configure the learning process. This is done with the .compile() method.
# - Optimizer: The algorithm that updates the model's weights. 'adam' is a popular choice.
# - Loss Function: Measures how accurate the model is during training. 'binary_crossentropy' is used for binary (0 or 1) classification problems.
# - Metrics: Used to monitor the training and testing steps. We'll use 'accuracy' to see the percentage of correct predictions.

print("\n--- Step 2: Compile the Model ---")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Model compiled successfully.")

# --- 3. Prepare the Data ---
# For this example, we'll create some dummy data.
# In a real-world scenario, you would load your data from a file or a database.
# We'll create 100 samples with 10 features each.
X_train = np.random.rand(100, 10)
# The labels are either 0 or 1, representing two classes.
y_train = np.random.randint(2, size=(100, 1))

# We'll also create some test data to evaluate the model's performance on unseen data.
X_test = np.random.rand(50, 10)
y_test = np.random.randint(2, size=(50, 1))

print(f"\n--- Data Shapes ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# --- 4. Train the Model ---
# We train the model using the .fit() method.
# - Epochs: An epoch is one complete pass through the entire training dataset. We'll train for 10 epochs.
# - Batch Size: The number of samples processed before the model is updated. We'll use a batch size of 32.
# The training process will output the loss and accuracy for each epoch.

print("\n--- Step 3: Train the Model ---")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
print("Model training complete.")

# --- 5. Evaluate the Model ---
# After training, we evaluate the model's performance on the test data.
# This tells us how well the model generalizes to new, unseen data.
# The .evaluate() method returns the loss and any metrics we specified during compilation.

print("\n--- Step 4: Evaluate the Model ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 6. Make Predictions ---
# We can use the trained model to make predictions on new data.
# The .predict() method returns the model's output for each input sample.
# For our binary classification problem, the output is a probability between 0 and 1.
# We can threshold this probability (e.g., at 0.5) to get a class prediction (0 or 1).

print("\n--- Step 5: Make Predictions ---")
# Let's create some new data to make predictions on.
new_data = np.random.rand(5, 10)
predictions = model.predict(new_data)

print("Predictions on new data:")
for i, prediction in enumerate(predictions):
    predicted_class = 1 if prediction[0] > 0.5 else 0
    print(f"  Sample {i+1}: Probability={prediction[0]:.4f}, Predicted Class={predicted_class}")


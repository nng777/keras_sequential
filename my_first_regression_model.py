import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow import keras
from tensorflow.keras import layers


def main():
    """Build, train, evaluate and use a regression model for a prediction"""
    #Generate sample data
    #Training data: 100 samples, 5 features each
    X_train = np.random.rand(100, 5)
    y_train = 50 * np.random.rand(100, 1)

    #Test data: 20 samples, 5 features each
    X_test = np.random.rand(20, 5)
    y_test = 50 * np.random.rand(20, 1)

    #Build the model
    model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(5,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])

    #Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    #Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

    #Evaluate the model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Calculate R^2 as an accuracy metric for regression
    predictions = model.predict(X_test, verbose=0)
    ss_res = np.sum(np.square(y_test - predictions))
    ss_tot = np.sum(np.square(y_test - np.mean(y_test)))
    r2 = 1 - ss_res / ss_tot
    print(f"Test Loss: {loss:.4f} - Test Mean Absolute Error/ Mean Accuracy Value: {mae:.4f} - R2 Accuracy: {r2:.4f}")
    print("MAE near 0 for no error prediction, and R2 near 1.0 for perfect prediction.")

    #PredictionTest
    new_sample = np.random.rand(3, 5)
    prediction = model.predict(new_sample, verbose=0)
    print(f"Prediction for new sample: {prediction[0][0]:.4f}")


if __name__ == "__main__":
    main()

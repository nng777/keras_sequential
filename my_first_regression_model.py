import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow import keras
from tensorflow.keras import layers


def main() -> None:
    """Build, train, evaluate and use a regression model."""
    # --- Generate sample data ---
    # Training data: 100 samples, 5 features each
    X_train = np.random.rand(100, 5)
    y_train = 50 * np.random.rand(100, 1)

    # Test data: 20 samples, 5 features each
    X_test = np.random.rand(20, 5)
    y_test = 50 * np.random.rand(20, 1)

    # --- Build the model ---
    model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(5,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])

    # --- Compile the model ---
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    # --- Train the model ---
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # --- Evaluate the model ---
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} - Test MAE: {mae:.4f}")

    # --- Make a prediction ---
    new_sample = np.random.rand(1, 5)
    prediction = model.predict(new_sample, verbose=0)
    print(f"Prediction for new sample: {prediction[0][0]:.4f}")


if __name__ == "__main__":
    main()

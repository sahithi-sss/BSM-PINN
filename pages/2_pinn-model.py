import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

# Define Black-Scholes parameters
sigma = 0.2  # Volatility
r = 0.05     # Risk-free interest rate
K = 100      # Strike price
T = 1.0      # Time to maturity (1 year)

# Create the neural network model
def build_model():
    model = Sequential([
        Input(shape=(2,)),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(1, activation='linear')  # Output: Option price
    ])
    return model

# Define the Black-Scholes PDE residual loss function
def bs_pde_loss(model, S, t):
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([S, t])
            tape2.watch([S, t])
            V = model(tf.concat([S, t], axis=1))

        # Compute first-order derivatives
        V_t = tape1.gradient(V, t)
        V_S = tape1.gradient(V, S)

    # Compute second-order derivative
    V_SS = tape2.gradient(V_S, S)

    # Compute the Black-Scholes PDE residual
    residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V
    return tf.reduce_mean(tf.square(residual))

# Define the boundary condition loss
def boundary_loss(model):
    S_boundary = np.linspace(0, 200, 100).reshape(-1, 1)
    S_boundary = tf.convert_to_tensor(S_boundary, dtype=tf.float32)
    
    T_boundary = np.ones_like(S_boundary) * T
    V_terminal = tf.maximum(S_boundary - K, 0)  # Payoff function for European Call

    V_pred = model(tf.concat([S_boundary, T_boundary], axis=1))
    return tf.reduce_mean(tf.square(V_pred - V_terminal))

# Training function
def train_pinn(model, epochs=5000, lr=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        S_sample = tf.random.uniform((100, 1), minval=0, maxval=200, dtype=tf.float32)
        t_sample = tf.random.uniform((100, 1), minval=0, maxval=T, dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss_pde = bs_pde_loss(model, S_sample, t_sample)
            loss_boundary = boundary_loss(model)
            loss = loss_pde + loss_boundary  # Combined loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

    print("Training complete!")

# Build and train the PINN model
pinn_model = build_model()
train_pinn(pinn_model, epochs=5000)

# Generate test data
S_test = np.linspace(0, 200, 100).reshape(-1, 1)
t_test = np.ones_like(S_test) * 0.5  # Mid-time horizon (t = 0.5)

S_test_tensor = tf.convert_to_tensor(S_test, dtype=tf.float32)
t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)

# Predict option prices
V_pred = pinn_model(tf.concat([S_test_tensor, t_test_tensor], axis=1)).numpy()

# Plot the learned option price function
plt.plot(S_test, V_pred, label="Predicted Price (PINN)")
plt.plot(S_test, np.maximum(S_test - K, 0), '--', label="Intrinsic Value")
plt.xlabel("Stock Price (S)")
plt.ylabel("Option Price (V)")
plt.legend()
plt.title("Black-Scholes Option Pricing using PINNs")
plt.show()
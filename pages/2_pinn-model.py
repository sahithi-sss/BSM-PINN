import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.stats import norm

# Page configuration
st.set_page_config(
    page_title="PINN Model",
    page_icon="📈",
    menu_items={"Get Help": None, "Report a Bug": None, "About": None},
    initial_sidebar_state="expanded")

if st.sidebar.button("🏠 Home", use_container_width=True):
    st.switch_page("app.py")

# Navigation
selected_model = st.sidebar.selectbox(
    "Navigate to",
    ["Black Scholes Option Pricing Model", "Comparision of analytic vs PINN solution"],
    index=1
)

if selected_model == "Black Scholes Option Pricing Model":
    st.switch_page("pages/1_black-scholes-model.py")

st.title("Physics-Informed Neural Network (PINN) for Black-Scholes")

st.markdown("""***This page demonstrates the use of Physics-Informed Neural Networks (PINNs) to solve the Black-Scholes PDE. 
    The model learns to approximate the option price function by minimizing both the PDE residual and boundary conditions.***""")

# Define Black-Scholes parameters
sigma = 0.2  # Volatility
r = 0.05     # Risk-free interest rate
K = 100      # Strike price
T = 1.0      # Time to maturity (1 year)

# Create the neural network model
def build_model():
    model = Sequential([
        Dense(64, activation='tanh', input_shape=(2,)),
        BatchNormalization(),
        Dense(64, activation='tanh'),
        BatchNormalization(),
        Dense(64, activation='tanh'),
        BatchNormalization(),
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
    S_boundary = np.linspace(0, 150, 100).reshape(-1, 1)
    S_boundary = tf.convert_to_tensor(S_boundary, dtype=tf.float32)
    
    T_boundary = np.ones_like(S_boundary) * T
    V_terminal = tf.maximum(S_boundary - K, 0)  # Payoff function for European Call

    V_pred = model(tf.concat([S_boundary, T_boundary], axis=1))
    return tf.reduce_mean(tf.square(V_pred - V_terminal))

# Training function
def train_pinn(model, epochs=5000, initial_lr=0.001):
    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Create a placeholder for training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_history = []
    
    for epoch in range(epochs):
        S_sample = tf.random.uniform((100, 1), minval=0, maxval=150, dtype=tf.float32)
        t_sample = tf.random.uniform((100, 1), minval=0, maxval=T, dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss_pde = bs_pde_loss(model, S_sample, t_sample)
            loss_boundary = boundary_loss(model)
            # Equal weighting of losses
            loss = loss_pde + loss_boundary

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:  # Show more frequent updates
            loss_history.append(float(loss.numpy()))
            status_text.text(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
            progress_bar.progress(epoch / epochs)

    status_text.text("Training complete!")
    progress_bar.progress(1.0)
    return loss_history

# Initialize session state for storing static results
if 'static_results' not in st.session_state:
    # Load your pre-computed results
    data = np.load('static_pinn_results.npz')
    S_test = data['S_test']
    V_static = data['V_pred']
    st.session_state.static_results = (S_test, V_static)

# Pre-computed loss values for static display
static_loss_list = [
    (0, 281.740448),
    (500, 4.062307),
    (1000, 0.570410),
    (1500, 0.420799),
    (2000, 0.292110),
    (2500, 0.233636),
    (3000, 0.529828),
    (3500, 0.164196),
    (4000, 0.150151),
    (4500, 0.179937)
]

# State to control display
if 'show_dynamic' not in st.session_state:
    st.session_state.show_dynamic = False

if not st.session_state.show_dynamic:
    st.subheader("Pre-computed Results")
    # Plot static results
    plt.figure(figsize=(10, 6))
    plt.plot(st.session_state.static_results[0], st.session_state.static_results[1], label="Predicted Price (PINN)")
    plt.plot(st.session_state.static_results[0], np.maximum(st.session_state.static_results[0] - K, 0), '--', label="Intrinsic Value")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Price (V)")
    plt.legend()
    plt.title("Black-Scholes Option Pricing using PINNs")
    st.pyplot(plt.gcf())
    
    # Show static loss list
    st.markdown("**Epoch/Loss values:**")
    for epoch, loss in static_loss_list:
        st.text(f"Epoch {epoch}, Loss: {loss:.6f}")
    # Button to run model dynamically
    if st.button("Run Model Dynamically", use_container_width=True):
        st.session_state.show_dynamic = True
        st.rerun()
else:
    st.subheader("Training Progress")
    # Build and train the PINN model
    pinn_model = build_model()
    epochs = 5000
    loss_history = train_pinn(pinn_model, epochs=epochs)
    
    # Display final results
    st.success("Training complete!")
    
    # Generate and display comparison graph after training
    st.subheader("Comparison of PINN vs Analytical Solution")
    S_test = np.linspace(0, 150, 100).reshape(-1, 1)
    t_test = np.ones_like(S_test) * 0.5  # Mid-time horizon (t = 0.5)
    
    S_test_tensor = tf.convert_to_tensor(S_test, dtype=tf.float32)
    t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
    
    V_pred = pinn_model(tf.concat([S_test_tensor, t_test_tensor], axis=1)).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(S_test, V_pred, label="Predicted Price (PINN)")
    plt.plot(S_test, np.maximum(S_test - K, 0), '--', label="Intrinsic Value")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Price (V)")
    plt.legend()
    plt.title("Black-Scholes Option Pricing using PINNs")
    st.pyplot(plt.gcf())

st.markdown("---")
st.subheader("About PINN Model")
st.markdown("""
The Physics-Informed Neural Network (PINN) approach combines deep learning with the underlying physics of the Black-Scholes PDE.
Key features:
- Neural network learns to approximate the option price function
- Loss function incorporates both PDE residual and boundary conditions
- No need for labeled training data
- Can handle complex boundary conditions and non-linear PDEs
""")
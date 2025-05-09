import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

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
    
    # Create a placeholder for training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_history = []
    
    for epoch in range(epochs):
        S_sample = tf.random.uniform((100, 1), minval=0, maxval=200, dtype=tf.float32)
        t_sample = tf.random.uniform((100, 1), minval=0, maxval=T, dtype=tf.float32)

        with tf.GradientTape() as tape:
            loss_pde = bs_pde_loss(model, S_sample, t_sample)
            loss_boundary = boundary_loss(model)
            loss = loss_pde + loss_boundary  # Combined loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:  # Show more frequent updates
            loss_history.append(loss.numpy())
            status_text.text(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
            progress_bar.progress(epoch / epochs)
            
            # Display loss history
            loss_df = pd.DataFrame({
                'Epoch': range(0, epoch + 1, 100),
                'Loss': loss_history
            })
            st.line_chart(loss_df.set_index('Epoch'))

    status_text.text("Training complete!")
    progress_bar.progress(1.0)
    return loss_history

# Function to plot results
def plot_results(S_test, V_pred, V_analytical=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_test, V_pred, label="Predicted Price (PINN)")
    if V_analytical is not None:
        ax.plot(S_test, V_analytical, '--', label="Analytical Solution")
    ax.plot(S_test, np.maximum(S_test - K, 0), ':', label="Intrinsic Value")
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Option Price (V)")
    ax.legend()
    ax.set_title("Black-Scholes Option Pricing using PINNs")
    return fig

# Initialize session state for storing static results
if 'static_results' not in st.session_state:
    # Load your pre-computed results
    data = np.load('static_pinn_results.npz')
    S_test = data['S_test']
    V_static = data['V_pred']
    st.session_state.static_results = (S_test, V_static)

# Pre-computed loss values for static display
static_loss_list = [
    (0, 1703.023926),
    (500, 385.576721),
    (1000, 127.829712),
    (1500, 48.017029),
    (2000, 22.418875),
    (2500, 8.966265),
    (3000, 3.887328),
    (3500, 1.840979),
    (4000, 1.799492),
    (4500, 1.340811)
]

# State to control display
if 'show_dynamic' not in st.session_state:
    st.session_state.show_dynamic = False

if not st.session_state.show_dynamic:
    st.subheader("Pre-computed Results")
    # Plot static results
    fig_static = plot_results(
        st.session_state.static_results[0],
        st.session_state.static_results[1],
        np.maximum(st.session_state.static_results[0] - K, 0)
    )
    st.pyplot(fig_static)
    # Show static loss list
    st.markdown("**Epoch/Loss values:**")
    for epoch, loss in static_loss_list:
        st.text(f"Epoch {epoch}, Loss: {loss:.6f}")
    # Button to run model dynamically
    if st.button("Run Model Dynamically", use_container_width=True):
        st.session_state.show_dynamic = True
        st.experimental_rerun()
else:
    st.subheader("Training Progress")
    # Build and train the PINN model
    pinn_model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    epochs = 5000
    loss_list = []
    loss_placeholder = st.empty()
    for epoch in range(epochs+1):
        S_sample = tf.random.uniform((100, 1), minval=0, maxval=200, dtype=tf.float32)
        t_sample = tf.random.uniform((100, 1), minval=0, maxval=T, dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss_pde = bs_pde_loss(pinn_model, S_sample, t_sample)
            loss_boundary = boundary_loss(pinn_model)
            loss = loss_pde + loss_boundary
        grads = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))
        if epoch % 500 == 0 or epoch == epochs:
            loss_list.append((epoch, float(loss.numpy())))
            # Show the list as it grows
            loss_placeholder.text("\n".join([f"Epoch {e}, Loss: {l:.6f}" for e, l in loss_list]))
    loss_placeholder.text("\n".join([f"Epoch {e}, Loss: {l:.6f}" for e, l in loss_list]))
    st.success("Training complete!")
    # Show only the final graph
    S_test = np.linspace(0, 200, 100).reshape(-1, 1)
    t_test = np.ones_like(S_test) * 0.5
    S_test_tensor = tf.convert_to_tensor(S_test, dtype=tf.float32)
    t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
    V_pred = pinn_model(tf.concat([S_test_tensor, t_test_tensor], axis=1)).numpy()
    fig_final = plot_results(S_test, V_pred, np.maximum(S_test - K, 0))
    st.subheader("Final Model Output after 5000 Epochs")
    st.pyplot(fig_final)
    st.markdown("**X-axis:** Stock Price (S)")
    st.markdown("**Y-axis:** Option Price (V)")

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

st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none}
        section[data-testid="stSidebar"] > div:first-child {display: none}
        .main > div:first-child {display: none}
        button[kind="headerNoPadding"] {display: none}
        .st-emotion-cache-1dp5vir {display: none}
        [data-testid="stSidebarNav"] {display: none !important}
        .st-emotion-cache-16pwjcz {display: none}
    </style>
""", unsafe_allow_html=True)
import streamlit as st

st.set_page_config(
    page_title="Real-Time Option Pricing",
    page_icon="📈",
    menu_items={"Get Help": None, "Report a Bug": None, "About": None}
    )

# Create radio buttons for model selection
selected_model = st.radio(
    "Select an Options Pricing Model",
    ["Black-Scholes Model", "PINN Model Comparison with Analytical Solution"]
)

# Add a button to navigate to the selected model
if st.button("Go to Page", use_container_width=True):
    if selected_model == "Black-Scholes Model":
        st.switch_page("pages/1_black-scholes-model.py")
    elif selected_model == "PINN Model Comparison with Analytical Solution":
        st.switch_page("pages/2_pinn-model.py")